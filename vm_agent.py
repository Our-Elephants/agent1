import json
import time
from typing import Optional

from haystack import component
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from rich.panel import Panel
from rich.rule import Rule

from settings import ModelProvider
from logger import log, printout
from vm_api import VM, RequestReportTaskCompletionVMCommand, OutcomeEnum
from vm_tools import make_toolset

SYSTEM_PROMPT = """\
You are a pragmatic personal knowledge management assistant.
- Keep edits small and targeted.
- Use the provided tools to explore and modify the environment.
- When done or blocked, call the report_task_completion tool with your completion summary.
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


def _print_summary(completion: RequestReportTaskCompletionVMCommand) -> None:
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
            self._generator = OpenAIChatGenerator(model=model_name, api_key=Secret.from_token(api_key) if api_key else None)
        elif provider == ModelProvider.OLLAMA:
            self._generator = OllamaChatGenerator(model=model_name)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @component.output_types(result=RequestReportTaskCompletionVMCommand)
    def run(self, task: str, vm: VM) -> dict:
        toolset = make_toolset(vm)
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

            if reply.tool_calls:
                for tc in reply.tool_calls:
                    if tc.tool_name == "report_task_completion":
                        log.info("step_%d  report_task_completion", step)
                        args = tc.arguments
                        refs = args.get("grounding_refs") or []
                        if isinstance(refs, str):
                            refs = json.loads(refs) if refs.strip().startswith("[") else [refs]
                        completion = RequestReportTaskCompletionVMCommand(
                            message=args["message"],
                            grounding_refs=refs,
                            outcome=OutcomeEnum(args["outcome"]),
                        )
                        vm.execute_report_task_completion_command(completion)
                        _print_summary(completion)
                        return {"result": completion}

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

            if reply.text:
                log.info("step_%d  text  %s", step, _snippet(reply.text))

        # Fallback: loop exhausted
        completion = RequestReportTaskCompletionVMCommand(
            message="Agent finished without explicit completion.",
            grounding_refs=[],
            outcome=OutcomeEnum.OUTCOME_ERR_INTERNAL,
        )
        vm.execute_report_task_completion_command(completion)
        _print_summary(completion)
        return {"result": completion}
