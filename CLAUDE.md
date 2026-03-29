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
- **vm_agent.py** — `VMAgent` wraps a Haystack `Agent`. Before each task it runs mandatory exploration steps (tree, read AGENTS.md, context), then runs the agent loop with the task instruction. The agent exits on `report_task_completion`.
- **vm_api.py** — Data models (Pydantic `BaseModel` subclasses for each VM command), `VM` class that dispatches commands to a `PcmRuntimeClientSync` (ConnectRPC), `Trial` state machine, `VMResponseFormatter` for human-readable output.
- **vm_tools.py** — `VMToolset` (Haystack `Toolset`) that wraps VM operations as `@tool`-decorated functions the agent can call: tree, find, search, ls, read, context, write, delete, mkdir, move, report_task_completion.
- **settings.py** — `pydantic-settings` config loaded from `.env`. Supports OpenAI and Ollama model providers.
- **logger.py** — Dual-output logging (terminal + file in `logs/`). `StepLogger` is a Haystack streaming callback that logs tool calls and results.
- **utils.py** — `TreeFormatter` for rendering tree structures.

## Key Patterns

- All VM operations follow: Pydantic command model → `VM.execute_*` → protobuf RPC → `VMResponseFormatter.format()` → string for the LLM.
- The `bitgn-api-connectrpc-python` package (from buf.build registry) provides generated protobuf stubs (`bitgn.harness_pb2`, `bitgn.vm.pcm_pb2`) and ConnectRPC clients.
- Python 3.14+ required. Uses `uv` for dependency management.
