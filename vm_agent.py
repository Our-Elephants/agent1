import json
import time

from haystack import component
from haystack.components.generators.chat import OpenAIResponsesChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from rich.rule import Rule

from models import ModelProvider
from logger import log, printout
from vm_api import VM, RequestReportTaskCompletionVMCommand, OutcomeEnum
from vm_tools import make_toolset
from prompt import load_sys_prompt
from utils import (
    build_generation_kwargs,
    snippet,
    print_summary,
    run_tool_sequence,
)


@component
class VMAgent:
    _MANDATORY_EXPLORATION = [
        ("tree", {"level": 2, "root": "/"}),
        ("read", {"path": "AGENTS.md"}),
        ("context", {}),
    ]
    def __init__(self, provider: ModelProvider, model_name: str, api_key: str, max_steps: int = 30):
        self.max_steps = max_steps
        self.provider = provider
        self.sys_prompt = load_sys_prompt()
        if provider == ModelProvider.OPENAI:
            self._generator = OpenAIResponsesChatGenerator(
                model=model_name,
                api_key=Secret.from_token(api_key),
            )
        elif provider == ModelProvider.OLLAMA:
            self._generator = OllamaChatGenerator(model=model_name)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @component.output_types(result=RequestReportTaskCompletionVMCommand)
    def run(self, task: str, vm: VM) -> dict:
        toolset = make_toolset(vm)
        tools_by_name = {t.name: t for t in toolset}
        messages = [ChatMessage.from_system(self.sys_prompt)]

        # Pre-grounding exploration
        printout(Rule("[bold blue]Exploration"))
        mandatory_tool_msgs = run_tool_sequence(
            self._MANDATORY_EXPLORATION,
            tools_by_name,
        )
        messages.extend(mandatory_tool_msgs)

        messages.append(ChatMessage.from_user(task))
        printout(Rule(f"[bold]Task: {task[:80]}{'…' if len(task) > 80 else ''}"))

        # Agent loop
        for step in range(1, self.max_steps + 1):
            generation_kwargs = build_generation_kwargs(self.provider)
            response = self._generator.run(
                messages=messages,
                tools=toolset,
                generation_kwargs=generation_kwargs,
            )
            reply = response["replies"][0]
            messages.append(reply)

            if tc := reply.tool_call:
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
                    print_summary(completion)
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
                            "ERR" if error else snippet(str(tool_result)))
                messages.append(ChatMessage.from_tool(str(tool_result), origin=tc, error=error))
            # TODO needed at all?
            if reply.text:
                log.info("step_%d  text  %s", step, snippet(reply.text))

        # Fallback: loop exhausted
        completion = RequestReportTaskCompletionVMCommand(
            message="Agent finished without explicit completion.",
            grounding_refs=[],
            outcome=OutcomeEnum.OUTCOME_ERR_INTERNAL,
        )
        vm.execute_report_task_completion_command(completion)
        print_summary(completion)
        return {"result": completion}
