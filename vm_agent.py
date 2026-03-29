import json
from typing import Optional

from haystack import component
from haystack.components.generators.chat import OpenAIResponsesChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from logger import RunLogger, Timer
from settings import ModelProvider
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


@component
class VMAgent:

    def __init__(self, provider: ModelProvider, model_name: str, api_key: Optional[str] = None, think: str = "high"):
        if provider == ModelProvider.OPENAI:
            gen_kwargs = {"reasoning_effort": think} if think else {}
            self._generator = OpenAIResponsesChatGenerator(
                model=model_name,
                api_key=Secret.from_token(api_key) if api_key else None,
                generation_kwargs=gen_kwargs,
            )
        elif provider == ModelProvider.OLLAMA:
            self._generator = OllamaChatGenerator(model=model_name, think=think or False)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @component.output_types(result=RequestReportTaskCompletionVMCommand)
    def run(self, task: str, vm: VM, logger: Optional[RunLogger] = None) -> dict:
        toolset = make_toolset(vm)
        tools_by_name = {t.name: t for t in toolset}
        messages = [ChatMessage.from_system(SYSTEM_PROMPT)]

        # Pre-grounding exploration
        for tool_name, args in _MANDATORY_EXPLORATION:
            with Timer() as t:
                try:
                    result = tools_by_name[tool_name].invoke(**args)
                    error = None
                except Exception as exc:
                    result = f"ERROR: {exc}"
                    error = str(exc)
            if logger:
                result_str = str(result)
                logger.log("explore",
                           tool_name=tool_name,
                           args=args,
                           duration_ms=t.ms,
                           result_size=len(result_str),
                           result_full=result_str,
                           error=error)
            messages.append(ChatMessage.from_user(str(result)))

        messages.append(ChatMessage.from_user(task))

        # Agent loop
        for step in range(1, MAX_STEPS + 1):
            if logger:
                logger.set_context(step=step)

            with Timer() as t:
                replies = self._generator.run(messages=messages, tools=toolset)["replies"]
            reply = replies[0]
            messages.append(reply)

            if logger:
                text = reply.text or ""
                meta = reply.meta or {}
                usage = meta.get("usage", {})
                logger.log("llm_response",
                           tool_call_count=len(reply.tool_calls) if reply.tool_calls else 0,
                           has_text=bool(text),
                           text=text if text else None,
                           message_count=len(messages),
                           prompt_tokens=usage.get("prompt_tokens"),
                           completion_tokens=usage.get("completion_tokens"),
                           duration_ms=t.ms)

            if reply.tool_calls:
                for tc in reply.tool_calls:
                    if tc.tool_name == "report_task_completion":
                        args = tc.arguments
                        refs = args.get("grounding_refs") or []
                        if isinstance(refs, str):
                            refs = json.loads(refs) if refs.strip().startswith("[") else [refs]
                        completion = RequestReportTaskCompletionVMCommand(
                            message=args["message"],
                            grounding_refs=refs,
                            outcome=OutcomeEnum(args["outcome"]),
                        )
                        if logger:
                            logger.log("task_completion",
                                       outcome=completion.outcome.value,
                                       message=completion.message,
                                       grounding_refs=completion.grounding_refs)
                        vm.execute_report_task_completion_command(completion)
                        return {"result": completion, "steps_used": step}

                    error_msg = None
                    with Timer() as t:
                        try:
                            tool_result = tools_by_name[tc.tool_name].invoke(**tc.arguments)
                            error = False
                        except Exception as exc:
                            tool_result = f"ERROR: {exc}"
                            error_msg = str(exc)
                            error = True

                    if logger:
                        result_str = str(tool_result)
                        logger.log("tool_call",
                                   tool_name=tc.tool_name,
                                   args=tc.arguments,
                                   duration_ms=t.ms,
                                   result_size=len(result_str),
                                   result_preview=result_str[:500],
                                   error=error_msg)

                    messages.append(ChatMessage.from_tool(str(tool_result), origin=tc, error=error))
                continue

        # Fallback: loop exhausted
        if logger:
            logger.log("loop_exhausted", level="WARN", steps_used=MAX_STEPS)

        completion = RequestReportTaskCompletionVMCommand(
            message="Agent finished without explicit completion.",
            grounding_refs=[],
            outcome=OutcomeEnum.OUTCOME_ERR_INTERNAL,
        )
        vm.execute_report_task_completion_command(completion)
        return {"result": completion, "steps_used": MAX_STEPS}
