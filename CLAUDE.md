# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BitGN Tools is an AI agent that runs benchmarks against a BitGN harness service. It connects to a remote VM environment via ConnectRPC (protobuf), executes tasks using an LLM-driven agent loop (Haystack), and reports scores.

## Commands

```bash
# Run (requires .env with BENCHMARK_HOST, BENCHMARK_ID, and optionally MODEL_* vars)
uv run python main.py

# Run specific tasks by ID
uv run python main.py <task_id> [task_id...]

# Install dependencies
uv sync
```

## Architecture

- **main.py** — Entry point. Connects to the harness, fetches a benchmark, iterates over tasks, creates a Trial/VM/Agent per task, and reports scores.
- **vm_agent.py** — `VMAgent` receives a `VM` instance, creates the toolset, runs mandatory exploration steps (tree, read AGENTS.md, context), then runs the agent loop. The loop intercepts `report_task_completion` tool calls to submit results and exit.
- **vm_api.py** — Data models (Pydantic `BaseModel` subclasses for each VM command), `VM` class that dispatches commands to a `PcmRuntimeClientSync` (ConnectRPC), `Trial` state machine, `VMResponseFormatter` for human-readable output.
- **vm_tools.py** — Haystack `@tool`-decorated functions wrapping VM operations: tree, find, search, ls, read, context, write, delete, mkdir, move. Also includes a noop `report_task_completion` tool (intercepted by the agent loop before invocation).
- **settings.py** — `pydantic-settings` config loaded from `.env`. Supports OpenAI and Ollama model providers.
- **logger.py** — Dual-output logging (terminal + file in `logs/`). `StepLogger` is a Haystack streaming callback that logs tool calls and results.
- **utils.py** — `TreeFormatter` for rendering tree structures.

## Configuration

Settings are loaded from `.env` via pydantic-settings (`settings.py`):

| Variable | Required | Description |
|---|---|---|
| `BENCHMARK_HOST` | Yes | BitGN harness endpoint (e.g. `https://api.bitgn.com`) |
| `BENCHMARK_ID` | Yes | Benchmark identifier (e.g. `bitgn/pac1-dev`) |
| `MODEL_PROVIDER` | No | `openai` (default) or `ollama` |
| `MODEL_NAME` | No | Model name (default: `gpt-5.4-mini`) |
| `MODEL_API_TOKEN` | No | API key for the model provider |
| `MODEL_THINK` | No | Thinking/reasoning effort level (default: `high`). OpenAI: `reasoning_effort` param (`none`, `low`, `medium`, `high`). Ollama: `think` param (`low`, `medium`, `high`). |

## Key Patterns

- All VM operations follow: Pydantic command model → `VM.execute_*` → protobuf RPC → `VMResponseFormatter.format()` → string for the LLM.
- The `bitgn-api-connectrpc-python` package (from buf.build registry) provides generated protobuf stubs (`bitgn.harness_pb2`, `bitgn.vm.pcm_pb2`) and ConnectRPC clients. Custom PyPI index `https://buf.build/gen/python` is configured in `pyproject.toml`.
- The agent loop has a hard cap of `MAX_STEPS = 30`. It exits early when the agent calls the `report_task_completion` tool.
- Trial lifecycle: `Trial.start()` → agent runs → `Trial.end()` returns score/detail.
- Logs are written to `logs/run_YYYYMMDD_HHMMSS.log` alongside terminal output.
- Python 3.14+ required. Uses `uv` for dependency management.
