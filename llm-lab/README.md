# llm-lab

Minimal LLM latency/cost instrumentation. Calls an OpenAI-compatible chat
completions endpoint, logs every run to CSV, and reports p50/p95 latency
per prompt.

## Setup

```bash
uv venv
uv pip install openai
cp .env.example .env   # fill in ANTHROPIC_API_KEY when you're ready to use it
```

Provider selection lives at the top of [call.py](call.py) — `PROVIDERS`
holds the `base_url`/`api_key`/`default_model` for each backend, and
`ACTIVE_PROVIDER` picks which one is used. Defaults to `"ollama"`
(`http://localhost:11434/v1`, `api_key="ollama"`). Switch to
`"anthropic"` once `ANTHROPIC_API_KEY` is set and Anthropic's
OpenAI-compatible endpoint is what you want to hit.

## Usage

```bash
uv run call.py
```

`runs.csv` is created on first write and is not committed. `main()`
runs every prompt in `PROMPTS` 5 times by default (pass `n=` to change
that).

## Status

Two pieces are intentionally left as `TODO` for you to implement:

- The timing capture inside `generate()` (currently `latency_ms` is a
  stubbed `None`).
- `percentile()` (currently `raise NotImplementedError`).

`main()` will run and log rows fine, but the p50/p95 summary at the end
will not work until both are filled in. Not run yet in this environment
since Ollama is still downloading.

## Retry/backoff

Retries only cover transient failures — timeouts, connection errors,
rate limits, and 5xx — up to 3 attempts with exponential backoff plus
jitter, since these are the failure modes where waiting and retrying
actually helps. Client errors (bad request, auth, 404) fail immediately
instead of burning retries on something that will never succeed.
