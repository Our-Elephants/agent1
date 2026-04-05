from pathlib import Path
from dataclasses import dataclass, field
import json 
from datetime import datetime
from time import perf_counter
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

_LOGS_DIR = Path("logs")
_LOGS_DIR.mkdir(exist_ok=True)

@dataclass
class TaskLog:
    id: str
    preview: str
    hint: str
    instruction: str
    steps: list[tuple[str, dict, str]] = field(default_factory=list)
    is_failed: bool = False
    exception: Exception | None = None
    score: float | None = None
    score_detail: str | None = None

class RunLogger:
    def __init__(self, model: str):
        self.run_id = self._make_run_id()
        self.run_dir = _LOGS_DIR / self.run_id
        self.passed_file = self._make_passed_file()
        self.failed_file = self._make_failed_file()
        self.summary_file = self._make_summary_file()
        self.task_log_history = []
        self.current_task_log = None
        self.model = model
        self.time_started = perf_counter()
        self.logger = self._make_logger()

    def _make_run_id(self):
        total_runs = sum(1 for _ in _LOGS_DIR.iterdir())
        today = datetime.now().strftime("%d_%m_%Y")
        return f"{total_runs + 1:6f}__{today}" 

    def _make_passed_file(self):
        f = (self.run_dir / "passed.md").open()
        f.write(f"# Passed tasks of run {self.run_id}\n")
        f.flush()
        return f
    
    def _make_failed_file(self):
        f = (self.run_dir / "failed.md").open()
        f.write(f"# Failed tasks of run {self.run_id}\n")
        f.flush()
        return f
    
    def _make_summary_file(self):
        f = (self.run_dir / "summary.md").open()
        f.write(
            f"# Run {self.run_id}\n"
            f"- Model: {self.model}\n"
        )
        f.flush()
        return f
    
    def _make_logger(self):
        logger = logging.getLogger("agent")
        sh = logging.StreamHandler()
        fh = logging.FileHandler(str(self.run_dir / ".log"))
        fh.setLevel(logging.DEBUG)
        logger.addHandler(sh)
        logger.addHandler(fh)
        return logger

    def teardown(self):
        self.passed_file.close()
        self.failed_file.close()
        self.summary_file.close()

    def log_benchmark_loaded(self, benchmark_id, benchmark_policy, benchmark_description):
        self.summary_file.write(
            f"- Benchmark id: {benchmark_id}\n"
            f"- Benchmark policy: {benchmark_policy}\n"
            f"- Benchmark description: {benchmark_description}\n"
        )
        self.logger.info(f"Loaded benchmark: {benchmark_id}")

    def log_task_started(self, task_id: str, task_preview: str, task_hint: str, instruction: str):
        self.current_task_log = TaskLog(task_id, task_preview, task_hint, instruction)
        self.logger.info(f"Start task {task_id}: {task_preview}")

    def log_task_tool_call(self, tool_name: str, tool_args: dict, response: str):
        self.current_task_log.steps.append((tool_name, tool_args, response))
        self.logger.info(f"Tool called for task {self.current_task_log.task_id}: {tool_name}")

    def log_task_failed_with_exception(self, exception: Exception):
        self.current_task_log.is_failed = True
        self.current_task_log.exception = exception
        self.logger.warning(f"Task {self.current_task_log.task_id} failed with exception: {str(exception)}")

    def log_task_scored(self, score: float, score_detail: str):
        self.current_task_log.score = score
        self.current_task_log.score_detail = score_detail
        self.logger.info(f"Task {self.current_task_log.task_id} scored: {score}")
    
    def flush_task_log(self):
        f = self.failed_file if self.current_task_log.is_failed else self.passed_file
        f.write( 
            f"## Task {self.current_task_log.id}\n"
            f"### Preview\n{self.current_task_log.preview}"
            f"### Instruction\n{self.current_task_log.instruction}"
            f"### Hint\n{self.current_task_log.hint}"
            f"### Execution\n{self._make_execution_log(self.current_task_log)}\n"
            f"### Exception\n```{self.current_task_log.exception}```" if self.current_task_log.exception 
                else f"### Score\nValue: *{self.current_task_log.score}*\n{self.current_task_log.score_detail}\n"
            "---\n"
        )
        f.flush()
        self.task_log_history.append(self.current_task_log)
        self.current_task_log = None

    def _make_execution_log(self, task_log: TaskLog):
        execution_log = ""
        for tool_name, tool_args, tool_resp in task_log.steps:
            execution_log += f"#### Call {tool_name}\n" \
                f"```{json.dumps(tool_args)}```\n" \
                f"\n##### Response\n```{tool_resp}```\n"
        return execution_log
        

    def flush_summary(self):
        elapsed = perf_counter() - self.time_started
        tasks_count = len(self.task_log_history)
        passed_tasks_count = len(1 for t in self.task_log_history if not t.is_failed)
        failed_tasks_count = len(self.task_log_history) - passed_tasks_count
        self.summary_file.write(
            f"- Total tasks: {tasks_count}\n",
            f"- Passed tasks: {passed_tasks_count}\n",
            f"- Failed tasks: {failed_tasks_count}\n",
            f"- Success rate: {passed_tasks_count / tasks_count * 100:.2f}\n",
            f"- Elapsed: {elapsed:.0f}sec ({elapsed // 60}min {int(elapsed) % 60}sec)\n"
        )
        