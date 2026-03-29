import sys
import traceback
import time
from datetime import datetime, timezone

from bitgn.harness_connect import HarnessServiceClientSync
from bitgn.harness_pb2 import StatusRequest
from bitgn.vm.pcm_connect import PcmRuntimeClientSync
from connectrpc.errors import ConnectError
from logger import RunLogger
from settings import Settings
from vm_agent import VMAgent
from vm_api import fetch_benchmark, Trial, VM


def main() -> None:
    task_filter = sys.argv[1:]
    settings = Settings()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    logger = RunLogger(run_id)

    scores = []
    run_start = time.perf_counter()
    try:
        client = HarnessServiceClientSync(settings.benchmark_host)
        client.status(StatusRequest())

        benchmark = fetch_benchmark(client, settings.benchmark_id)
        agent = VMAgent(settings.model_provider, settings.model_name, settings.model_api_token, settings.model_think)

        logger.log("run_start",
                    benchmark_id=benchmark.id,
                    task_count=len(benchmark.tasks),
                    model_provider=settings.model_provider.value,
                    model_name=settings.model_name,
                    model_think=settings.model_think)

        for task in benchmark.tasks:
            if task_filter and task.task_id not in task_filter:
                continue

            trial = Trial(client, benchmark.id, task.task_id)
            trial.start()
            logger.set_context(task_id=task.task_id, trial_id=trial.id)
            logger.log("task_start", instruction=trial.instruction, harness_url=trial.harness_url)
            task_start = time.perf_counter()
            outcome = None
            steps_used = 0

            try:
                pcm_client = PcmRuntimeClientSync(trial.harness_url)
                vm = VM(pcm_client)
                result = agent.run(task=trial.instruction, vm=vm, logger=logger)
                completion = result.get("result")
                if completion:
                    outcome = completion.outcome.value
                    steps_used = result.get("steps_used", 0)
            except Exception as exc:
                logger.log("task_error", level="ERROR",
                           error_type=type(exc).__name__,
                           error_message=str(exc),
                           traceback=traceback.format_exc())

            trial.end()
            task_duration = round(time.perf_counter() - task_start, 2)
            logger.log("task_end",
                        score=trial.score,
                        score_detail=trial.score_detail,
                        duration_s=task_duration,
                        steps_used=steps_used,
                        outcome=outcome)
            logger.clear_context("task_id", "trial_id")

            if trial.score >= 0:
                scores.append((task.task_id, trial.score))

    except ConnectError as exc:
        logger.log("run_error", level="ERROR", error_type="ConnectError", error_message=str(exc))
    except KeyboardInterrupt:
        logger.log("run_error", level="WARN", error_type="KeyboardInterrupt", error_message="Run interrupted by user")

    run_duration = round(time.perf_counter() - run_start, 2)
    logger.log("run_end", scores=scores, duration_s=run_duration)
    logger.write_summary({
        "benchmark_id": settings.benchmark_id,
        "model_provider": settings.model_provider.value,
        "model_name": settings.model_name,
        "scores": [{"task_id": tid, "score": s} for tid, s in scores],
        "duration_s": run_duration,
    })
    logger.close()


if __name__ == "__main__":
    main()
