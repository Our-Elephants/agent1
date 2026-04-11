# BitGN Tools Agent

AI agent that runs benchmarks against a [BitGN](https://bitgn.com) harness service. Connects to a remote VM via ConnectRPC, executes tasks using an LLM-driven agent loop (Haystack), and reports scores.

## Setup

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Create a `.env` file with the required environment variables:

| Variable | Required | Default | Description |
|---|---|---|---|
| `BENCHMARK_HOST` | No | `https://api.bitgn.com` | BitGN harness endpoint |
| `BENCHMARK_ID` | No | `bitgn/pac1-dev` | Benchmark identifier |
| `MODEL_PROVIDER` | No | `openai` | `openai` or `ollama` |
| `MODEL_NAME` | No | `gpt-5.4` | Model name |
| `AZURE_OPENAI_API_KEY` | Yes | — | Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | Yes | — | Azure OpenAI endpoint URL |
| `MODEL_THINKING` | No | disabled | Reasoning effort: `low`, `medium`, or `high` |

```env
BENCHMARK_HOST=https://api.bitgn.com
BENCHMARK_ID=bitgn/pac1-dev

# Azure OpenAI
MODEL_PROVIDER=openai
MODEL_NAME=gpt-5.4
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/

# Optional: enable reasoning
# MODEL_THINKING=medium

# Ollama alternative
# MODEL_PROVIDER=ollama
# MODEL_NAME=qwen3.5:9b
```

## Usage

```bash
uv run python main.py                # run all tasks
uv run python main.py t01 t03        # run specific tasks
```
