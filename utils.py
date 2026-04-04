from rich.panel import Panel
from rich.rule import Rule

from haystack.dataclasses import ChatMessage

from logger import log, printout
from models import ModelProvider
from vm_api import RequestReportTaskCompletionVMCommand, OutcomeEnum


def snippet(text: str, limit: int = 200) -> str:
    return text[:limit] + "…" if len(text) > limit else text


def print_summary(completion: RequestReportTaskCompletionVMCommand) -> None:
    color = "green" if completion.outcome == OutcomeEnum.OUTCOME_OK else "yellow"
    refs = "\n".join(f"  • {r}" for r in completion.grounding_refs) if completion.grounding_refs else "—"
    printout(Rule(f"[{color}]{completion.outcome.value}[/{color}]"))
    printout(Panel(
        f"[bold]{completion.message}[/bold]\n\n[dim]Refs:[/dim] {refs}",
        title=f"[{color}]Agent Summary[/{color}]",
        border_style=color,
    ))


def run_tool_sequence(
    tool_args_by_name: list[tuple],
    tools_by_name: dict,
) -> list[ChatMessage]:
    messages = []
    for tool_name, args in tool_args_by_name:
        try:
            result = tools_by_name[tool_name].invoke(**args)
        except Exception as exc:
            result = f"ERROR: {exc}"
        log.info("explore  %s\n%s", tool_name, snippet(str(result)))
        messages.append(ChatMessage.from_user(str(result)))

    return messages


def build_generation_kwargs(provider: ModelProvider) -> dict:
    if provider == ModelProvider.OPENAI:
        return {
            "tool_choice": "required",
            "parallel_tool_calls": False,
            "reasoning_effort": "high",
        }
    return {}
