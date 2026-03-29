import json
import os
import time
from datetime import datetime, timezone


class RunLogger:
    def __init__(self, run_id: str, log_dir: str = "logs"):
        self.run_id = run_id
        self._context: dict = {}
        run_dir = os.path.join(log_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        self._run_dir = run_dir
        self._file = open(os.path.join(run_dir, "run.jsonl"), "a", encoding="utf-8")

    def set_context(self, **fields) -> None:
        self._context.update(fields)

    def clear_context(self, *keys) -> None:
        for k in keys:
            self._context.pop(k, None)

    def log(self, event: str, level: str = "INFO", **data) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            **self._context,
            "event": event,
            "level": level,
            **data,
        }
        self._file.write(json.dumps(entry, default=str) + "\n")
        self._file.flush()
        self._print(entry)

    def write_summary(self, data: dict) -> None:
        path = os.path.join(self._run_dir, "summary.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"run_id": self.run_id, **data}, f, indent=2, default=str)

    def close(self) -> None:
        self._file.close()

    def _print(self, entry: dict) -> None:
        event = entry["event"]
        parts = [event]

        task_id = entry.get("task_id")
        if task_id:
            parts.append(f"[{task_id}]")

        step = entry.get("step")
        if step:
            parts.append(f"step={step}")

        tool = entry.get("tool_name")
        if tool:
            parts.append(tool)

        dur = entry.get("duration_ms")
        if dur is not None:
            parts.append(f"{dur}ms")

        msg_count = entry.get("message_count")
        if msg_count is not None:
            parts.append(f"msgs={msg_count}")

        prompt_t = entry.get("prompt_tokens")
        if prompt_t is not None:
            parts.append(f"in={prompt_t}")

        completion_t = entry.get("completion_tokens")
        if completion_t is not None:
            parts.append(f"out={completion_t}")

        err = entry.get("error")
        if err:
            parts.append(f"ERROR: {err}")

        score = entry.get("score")
        if score is not None:
            parts.append(f"score={score}")

        print(" | ".join(parts))


class Timer:
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.ms = round((time.perf_counter() - self._start) * 1000)
