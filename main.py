import sys
import textwrap

from bitgn.harness_connect import HarnessServiceClientSync
from bitgn.harness_pb2 import StatusRequest
from bitgn.vm.pcm_connect import PcmRuntimeClientSync
from connectrpc.errors import ConnectError
from rich.rule import Rule
from rich.text import Text

from logger import printout
from settings import Settings
from vm_agent import VMAgent
from vm_api import fetch_benchmark, Trial, VM


def main() -> None:
    task_filter = sys.argv[1:]
    settings = Settings()

    scores = []
    try:
        client = HarnessServiceClientSync(settings.benchmark_host)
        printout("Connecting to BitGN", client.status(StatusRequest()))

        benchmark = fetch_benchmark(client, settings.benchmark_id)
        printout(
            f"{benchmark.policy} benchmark: {benchmark.id} "
            f"with {len(benchmark.tasks)} tasks.\n[green]{benchmark.description}[/green]"
        )

        agent = VMAgent(settings.model_provider, settings.model_name, settings.model_api_token)

        for task in benchmark.tasks:
            if task_filter and task.task_id not in task_filter:
                continue

            printout(Rule(f"Starting task: {task.task_id}"))

            trial = Trial(client, benchmark.id, task.task_id)
            trial.start()

            printout(Text(trial.instruction, style="blue"))
            printout("-" * 80)

            try:
                pcm_client = PcmRuntimeClientSync(trial.harness_url)
                vm = VM(pcm_client)
                agent.run(task=trial.instruction, vm=vm)
            except Exception as exc:
                printout(Text(str(exc), style="red"))

            trial.end()
            if trial.score >= 0:
                scores.append((task.task_id, trial.score))
                style = "green" if trial.score == 1 else "red"
                explain = textwrap.indent("\n".join(trial.score_detail), " ")
                printout(f"\n[{style}]Score: {trial.score:0.2f}\n{explain}[/{style}]")

    except ConnectError as exc:
        printout(f"[red]{exc.code}: {exc.message}[/red]")
    except KeyboardInterrupt:
        printout("[red]Interrupted[/red]")

    if scores:
        for task_id, score in scores:
            style = "green" if score == 1 else "red"
            printout(f"{task_id}: [{style}]{score:0.2f}[/{style}]")

        total = sum(score for _, score in scores) / len(scores) * 100.0
        printout(f"FINAL: {total:0.2f}%")


if __name__ == "__main__":
    main()
