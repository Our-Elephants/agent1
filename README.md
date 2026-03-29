# BitGN Tools Agent

AI agent that runs benchmarks against a [BitGN](https://bitgn.com) harness service. Connects to a remote VM via ConnectRPC, executes tasks using an LLM-driven agent loop (Haystack), and reports scores.

## Setup

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Create a `.env` file:

```env
BENCHMARK_HOST=https://api.bitgn.com
BENCHMARK_ID=bitgn/pac1-dev

# Optional: defaults to openai / gpt-5.4-mini
MODEL_PROVIDER=ollama
MODEL_NAME=qwen3.5:9b
MODEL_API_TOKEN=sk-...
```

## Usage

```bash
uv run python main.py                # run all tasks
uv run python main.py t01 t03        # run specific tasks
```
