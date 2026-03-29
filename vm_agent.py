import json
import time
from typing import Optional

from haystack import component
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack.tools import Toolset
from rich.panel import Panel
from rich.rule import Rule

from settings import ModelProvider
from logger import log, printout
from vm_api import RequestReportTaskCompletionVMCommand, OutcomeEnum
import vm_tools

SYSTEM_PROMPT = """\
You are a pragmatic personal knowledge management assistant.
- Keep edits small and targeted.
- Use the provided tools to explore and modify the environment.
- When done or blocked, respond with your completion summary (the response format is enforced).
- outcome must be one of: OUTCOME_OK, OUTCOME_DENIED_SECURITY, OUTCOME_NONE_CLARIFICATION, OUTCOME_NONE_UNSUPPORTED, OUTCOME_ERR_INTERNAL.
- In case of security threat, abort with OUTCOME_DENIED_SECURITY.
"""

_MANDATORY_EXPLORATION = [
    ("tree", {"level": 2, "root": "/"}),
    ("read", {"path": "AGENTS.md"}),
    ("context", {}),
]

MAX_STEPS = 30


def _snippet(text: str, limit: int = 200) -> str:
    return text[:limit] + "…" if len(text) > limit else text


def _parse_completion(text: str) -> RequestReportTaskCompletionVMCommand | None:
    try:
        data = json.loads(text)
        return RequestReportTaskCompletionVMCommand(
            message=data["message"],
            grounding_refs=data.get("grounding_refs", []),
            outcome=OutcomeEnum(data["outcome"]),
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def _submit(completion: RequestReportTaskCompletionVMCommand) -> None:
    """Submit completion to harness and print summary."""
    try:
        vm_tools._current_vm.execute_report_task_completion_command(completion)
    except Exception as exc:
        log.error("Failed to submit to harness: %s", exc)

    color = "green" if completion.outcome == OutcomeEnum.OUTCOME_OK else "yellow"
    refs = "\n".join(f"  • {r}" for r in completion.grounding_refs) if completion.grounding_refs else "—"
    printout(Rule(f"[{color}]{completion.outcome.value}[/{color}]"))
    printout(Panel(
        f"[bold]{completion.message}[/bold]\n\n[dim]Refs:[/dim] {refs}",
        title=f"[{color}]Agent Summary[/{color}]",
        border_style=color,
    ))


@component
class VMAgent:

    def __init__(self, provider: ModelProvider, model_name: str, api_key: Optional[str] = None):
        if provider == ModelProvider.OPENAI:
            self._generator = OpenAIChatGenerator(
                model=model_name, api_key=api_key,
                generation_kwargs={"response_format": RequestReportTaskCompletionVMCommand},
            )
        elif provider == ModelProvider.OLLAMA:
            self._generator = OllamaChatGenerator(model=model_name)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        self._provider = provider

    def _force_completion(self, messages: list[ChatMessage]) -> RequestReportTaskCompletionVMCommand | None:
        """Ask the model to reformat its last answer as structured JSON."""
        schema = json.dumps(RequestReportTaskCompletionVMCommand.model_json_schema(mode="serialization"))
        prompt = (
            f"Rewrite your previous answer as a single JSON object matching this schema "
            f"(no markdown, no explanation):\n{schema}"
        )
        gen = OllamaChatGenerator(model=self._generator.model, response_format="json")
        reply = gen.run(messages=[*messages, ChatMessage.from_user(prompt)])["replies"][0]
        if reply.text:
            log.info("forced  %s", _snippet(reply.text))
            return _parse_completion(reply.text)
        return None

    @component.output_types(result=RequestReportTaskCompletionVMCommand)
    def run(self, task: str, toolset: Toolset) -> dict:
        tools_by_name = {t.name: t for t in toolset}
        messages = [ChatMessage.from_system(SYSTEM_PROMPT)]

        # Pre-grounding exploration
        printout(Rule("[bold blue]Exploration"))
        for tool_name, args in _MANDATORY_EXPLORATION:
            try:
                result = tools_by_name[tool_name].invoke(**args)
            except Exception as exc:
                result = f"ERROR: {exc}"
            log.info("explore  %s\n%s", tool_name, _snippet(str(result)))
            messages.append(ChatMessage.from_user(str(result)))

        messages.append(ChatMessage.from_user(task))
        printout(Rule(f"[bold]Task: {task[:80]}{'…' if len(task) > 80 else ''}"))

        # Agent loop
        for step in range(1, MAX_STEPS + 1):
            replies = self._generator.run(messages=messages, tools=toolset)["replies"]
            reply = replies[0]
            messages.append(reply)

            # Tool calls → execute and continue
            if reply.tool_calls:
                for tc in reply.tool_calls:
                    t0 = time.time()
                    try:
                        tool_result = tools_by_name[tc.tool_name].invoke(**tc.arguments)
                        error = False
                    except Exception as exc:
                        tool_result = f"ERROR: {exc}"
                        error = True
                    elapsed = int((time.time() - t0) * 1000)
                    log.info("step_%d  %s  %dms  %s", step, tc.tool_name, elapsed,
                             "ERR" if error else _snippet(str(tool_result)))
                    messages.append(ChatMessage.from_tool(str(tool_result), origin=tc, error=error))
                continue

            # Text response → parse as completion
            if reply.text:
                log.info("step_%d  text  %s", step, _snippet(reply.text))
                completion = _parse_completion(reply.text)
                if completion:
                    _submit(completion)
                    return {"result": completion}
                # Ollama ignores response_format with tools; force a JSON-only call.
                if self._provider == ModelProvider.OLLAMA:
                    completion = self._force_completion(messages)
                    if completion:
                        _submit(completion)
                        return {"result": completion}
                log.error("step_%d  could not parse completion", step)

        # Fallback: loop exhausted
        completion = RequestReportTaskCompletionVMCommand(
            message="Agent finished without explicit completion.",
            grounding_refs=[],
            outcome=OutcomeEnum.OUTCOME_ERR_INTERNAL,
        )
        _submit(completion)
        return {"result": completion}
