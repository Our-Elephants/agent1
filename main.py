import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from bitgn.harness_connect import HarnessServiceClientSync
from bitgn.harness_pb2 import GetRunRequest, StartRunRequest, StatusRequest, SubmitRunRequest
from bitgn.vm.pcm_connect import PcmRuntimeClientSync
from connectrpc.errors import ConnectError

from models import Settings
from vm_agent import VMAgent
from run_logger import RunLogger
from vm_api import fetch_benchmark, Trial, VM


def _make_agent(settings: Settings, logger: RunLogger) -> VMAgent:
    return VMAgent(
        settings.MODEL_PROVIDER,
        settings.MODEL_NAME,
        settings.AZURE_OPENAI_API_KEY,
        settings.AZURE_OPENAI_ENDPOINT,
        logger=logger,
        thinking=settings.MODEL_THINKING,
    )


def _run_task(trial_id: str, task, benchmark_id: str, settings: Settings, logger: RunLogger) -> None:
    client = HarnessServiceClientSync(settings.BENCHMARK_HOST)
    trial = Trial(client, benchmark_id, task_id=task.task_id, trial_id=trial_id)

    try:
        trial.start(prod=True)
    except Exception as exception:
        logger.log_task_started(
            id=task.task_id,
            task_preview=task.preview,
            task_hint=task.hint,
            instruction="",
        )
        logger.log_task_failed_with_exception(exception)
        logger.flush_task_log()
        return

    logger.log_task_started(
        id=task.task_id,
        task_preview=task.preview,
        task_hint=task.hint,
        instruction=trial.instruction,
    )

    try:
        pcm_client = PcmRuntimeClientSync(trial.harness_url)
        vm = VM(pcm_client)
        agent = _make_agent(settings, logger)
        agent.run(task=trial.instruction, vm=vm)
    except Exception as exception:
        logger.log_task_failed_with_exception(exception)
    finally:
        try:
            trial.end()
            logger.log_task_scored(
                score=trial.score,
                score_detail=trial.score_detail,
            )
        except Exception as exception:
            logger.log_task_failed_with_exception(exception)
        finally:
            logger.flush_task_log()


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
        task_by_id = {task.task_id: task for task in benchmark.tasks}
        run = client.start_run(StartRunRequest(
            name="SGR NextStep Sample",
            benchmark_id=benchmark.id,
            api_key=settings.BENCH_API_KEY,
        ))

        try:
            run_state = client.get_run(GetRunRequest(run_id=run.run_id))
            trial_heads_by_id = {trial.trial_id: trial for trial in run_state.trials}
            selected_trials = []
            for trial_id in run.trial_ids:
                trial_head = trial_heads_by_id.get(trial_id)
                if trial_head is None:
                    logger.logger.warning(f"Run {run.run_id} did not include metadata for trial {trial_id}")
                    continue
                if task_filter and trial_head.task_id not in task_filter:
                    continue
                task = task_by_id.get(trial_head.task_id)
                if task is None:
                    logger.logger.warning(f"Benchmark {benchmark.id} did not include metadata for task {trial_head.task_id}")
                    continue
                selected_trials.append((trial_id, task))

            max_workers = min(settings.MAX_PARALLEL_TASKS, len(selected_trials)) or 1
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(_run_task, trial_id, task, benchmark.id, settings, logger)
                    for trial_id, task in selected_trials
                ]
                for future in as_completed(futures):
                    future.result()
        finally:
            client.submit_run(SubmitRunRequest(run_id=run.run_id, force=True))
    except ConnectError as exception:
        logger.logger.error("Connection error", exception)
    except KeyboardInterrupt:
        logger.logger.info("Interrupted")
    finally:
        logger.flush_summary()
        logger.teardown()

if __name__ == "__main__":
    main()
