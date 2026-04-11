import json
import posixpath

from haystack import component
from haystack.components.generators.chat import AzureOpenAIResponsesChatGenerator
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.utils import Secret
from haystack_integrations.components.generators.ollama import OllamaChatGenerator

from models import ModelProvider, ModelThinking
from vm_api import VM, RequestReportTaskCompletionVMCommand, OutcomeEnum
from vm_tools import make_toolset
from prompt import load_sys_prompt
from run_logger import RunLogger


@component
class VMAgent:
    _MANDATORY_EXPLORATION = [
        ("tree", {"level": 2, "root": "/"}),
        ("read", {"path": "AGENTS.md"}),
        ("context", {}),
    ]
    def __init__(
        self,
        provider: ModelProvider,
        model_name: str,
        api_key: str,
        azure_endpoint: str,
        logger: RunLogger,
        thinking: ModelThinking | None = None,
        max_steps: int = 30,
    ):
        self.max_steps = max_steps
        self.provider = provider
        self.thinking = thinking
        self.logger = logger
        self.sys_prompt = load_sys_prompt()
        if provider == ModelProvider.OPENAI:
            self._generator = AzureOpenAIResponsesChatGenerator(
                api_key=Secret.from_token(api_key),
                azure_endpoint=azure_endpoint,
                azure_deployment=model_name,
            )
        elif provider == ModelProvider.OLLAMA:
            self._generator = OllamaChatGenerator(
                model=model_name,
                think=thinking.value if thinking else None,
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @component.output_types(result=RequestReportTaskCompletionVMCommand)
    def run(self, task: str, vm: VM) -> dict:
        toolset = make_toolset(vm)
        tools_by_name = {t.name: t for t in toolset}
        messages = [ChatMessage.from_system(self.sys_prompt)]

        mandatory_tool_msgs = self.run_tool_sequence(
            self._MANDATORY_EXPLORATION,
            tools_by_name,
        )
        messages.extend(mandatory_tool_msgs)

        messages.append(ChatMessage.from_user(task))

        # Agent loop
        for _ in range(1, self.max_steps + 1):
            generation_kwargs = self.build_generation_kwargs(self.provider)
            response = self._generator.run(
                messages=messages,
                tools=toolset,
                generation_kwargs=generation_kwargs,
            )
            reply = response["replies"][0]
            if reply.reasoning:
                self.logger.log_reasoning(reply.reasoning.reasoning_text)
            messages.append(reply)

            if tc := reply.tool_call:
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
                    self.logger.log_task_completion(
                        message=completion.message,
                        outcome=completion.outcome.value,
                        grounding_refs=completion.grounding_refs,
                    )
                    vm.execute_report_task_completion_command(completion)
                    return {"result": completion}

                if self._is_blocked_delete(tc.tool_name, tc.arguments):
                    completion = RequestReportTaskCompletionVMCommand(
                        message="Denied unsafe delete request targeting protected root path or AGENTS.md.",
                        grounding_refs=[],
                        outcome=OutcomeEnum.OUTCOME_DENIED_SECURITY,
                    )
                    self.logger.log_task_completion(
                        message=completion.message,
                        outcome=completion.outcome.value,
                        grounding_refs=completion.grounding_refs,
                    )
                    vm.execute_report_task_completion_command(completion)
                    return {"result": completion}

                try:
                    tool_result = tools_by_name[tc.tool_name].invoke(**tc.arguments)
                    self.logger.log_task_tool_call(tc.tool_name, tc.arguments, tool_result)
                    error = False
                except Exception as exc:
                    tool_result = f"ERROR: {exc}"
                    error = True
                
                messages.append(ChatMessage.from_tool(str(tool_result), origin=tc, error=error))

        # Fallback: loop exhausted
        completion = RequestReportTaskCompletionVMCommand(
            message="Agent finished without explicit completion.",
            grounding_refs=[],
            outcome=OutcomeEnum.OUTCOME_ERR_INTERNAL,
        )
        self.logger.log_task_completion(
            message=completion.message,
            outcome=completion.outcome.value,
            grounding_refs=completion.grounding_refs,
        )
        vm.execute_report_task_completion_command(completion)
        return {"result": completion}
    
    def run_tool_sequence(
        self,
        tool_args_by_name: list[tuple[str, dict]],
        tools_by_name: dict,
    ) -> list[ChatMessage]:
        messages: list[ChatMessage] = []

        for index, (tool_name, args) in enumerate(tool_args_by_name):
            tool_id = f"fc_{index}"
            tool_call = ToolCall(
                tool_name,
                args,
                id=tool_id,
                extra={"call_id": tool_id},
            )

            try:
                messages.append(ChatMessage.from_assistant(tool_calls=[tool_call]))
                result = tools_by_name[tool_name].invoke(**args)
                self.logger.log_task_tool_call(tool_name, args, result)
            except Exception as exc:
                result = f"ERROR: {exc}"

            messages.append(ChatMessage.from_tool(str(result), origin=tool_call))

        return messages


    def build_generation_kwargs(self, provider: ModelProvider) -> dict:
        if provider == ModelProvider.OPENAI:
            kwargs = {
                "parallel_tool_calls": False,
                "tool_choice": "required",
            }
            if self.thinking:
                kwargs["reasoning"] = {"effort": self.thinking.value, "summary": "auto"}
            return kwargs
        return {}

    def _is_blocked_delete(self, tool_name: str, arguments: dict) -> bool:
        if tool_name != "delete":
            return False

        path = arguments.get("path")
        if not isinstance(path, str):
            return False

        normalized = posixpath.normpath(path.strip() or ".")
        return normalized in {"/", "/AGENTS.md", "AGENTS.md"}
