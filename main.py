import sys

from bitgn.harness_connect import HarnessServiceClientSync
from bitgn.harness_pb2 import StatusRequest
from bitgn.vm.pcm_connect import PcmRuntimeClientSync
from connectrpc.errors import ConnectError

from models import Settings
from vm_agent import VMAgent
from run_logger import RunLogger
from vm_api import fetch_benchmark, Trial, VM


def main() -> None:
    task_filter = sys.argv[1:]
    settings = Settings()
    logger = RunLogger(
        settings.MODEL_NAME,
        reasoning_effort=settings.MODEL_THINKING.value if settings.MODEL_THINKING else None,
    )

    try:
        client = HarnessServiceClientSync(settings.BENCHMARK_HOST)
        logger.logger.info(f"Connected to BitGN: {client.status(StatusRequest())}")

        benchmark = fetch_benchmark(client, settings.BENCHMARK_ID)
        logger.log_benchmark_loaded(
            benchmark_id=benchmark.id, 
            benchmark_policy=benchmark.policy, 
            benchmark_description=benchmark.description,
            benchmark_tasks=len(benchmark.tasks)
        )
        agent = VMAgent(
            settings.MODEL_PROVIDER,
            settings.MODEL_NAME,
            settings.AZURE_OPENAI_API_KEY,
            settings.AZURE_OPENAI_ENDPOINT,
            logger=logger,
            thinking=settings.MODEL_THINKING,
        )

        for task in benchmark.tasks:
            if task_filter and task.task_id not in task_filter:
                continue

            trial = Trial(client, benchmark.id, task.task_id)
            trial.start()

            logger.log_task_started(
                id=task.task_id,
                task_preview=task.preview,
                task_hint=task.hint,
                instruction=trial.instruction
            )

            try:
                pcm_client = PcmRuntimeClientSync(trial.harness_url)
                vm = VM(pcm_client)
                agent.run(task=trial.instruction, vm=vm)
            except Exception as exception:
                logger.log_task_failed_with_exception(exception=exception)
                trial.end()
                logger.flush_task_log()
                continue

            trial.end()
            logger.log_task_scored(
                score=trial.score,
                score_detail=trial.score_detail
            )
            logger.flush_task_log()
    except ConnectError as exception:
        logger.logger.error("Connection error", exception)
    except KeyboardInterrupt:
        logger.logger.info("Interrupted")
    finally:
        logger.flush_summary()
        logger.teardown()

if __name__ == "__main__":
    main()
