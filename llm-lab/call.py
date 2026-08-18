from __future__ import annotations

import csv
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from math import floor

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)


PROVIDERS: dict[str, dict[str, str]] = {
    # "ollama": {
    #     "base_url": "http://localhost:11434/v1",
    #     "api_key": "ollama",
    #     "default_model": "llama3.2:3b",
    # },
    "ollama": {
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",
            "default_model": "qwen2.5:1.5b",
        },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1",
        "api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
        "default_model": "claude-opus-5",
    },
}

ACTIVE_PROVIDER = "ollama"

REQUEST_TIMEOUT_S = 150.0
MAX_RETRIES = 3
BACKOFF_BASE_S = 1.0

RUNS_PER_PROMPT = 5
CSV_PATH = "runs2.csv"


RETRYABLE_EXCEPTIONS = (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError)


def get_client() -> OpenAI:
    cfg = PROVIDERS[ACTIVE_PROVIDER]
    return OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])


def generate(prompt: str, model: str) -> dict:
    client = get_client()
    ts = datetime.now(timezone.utc).isoformat()

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:

            start_time = time.perf_counter()
            latency_ms = None
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                timeout=REQUEST_TIMEOUT_S,
            )
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000

            choice = response.choices[0]
            usage = response.usage

            return {
                "text": choice.message.content,
                "model": response.model,
                "latency_ms": latency_ms,
                "tokens_in": usage.prompt_tokens if usage else None,
                "tokens_out": usage.completion_tokens if usage else None,
                "stop_reason": choice.finish_reason,
                "ts": ts,
            }

        except RETRYABLE_EXCEPTIONS as exc:
            last_error = exc
            if attempt == MAX_RETRIES:
                break
            sleep_s = BACKOFF_BASE_S * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            time.sleep(sleep_s)

        except APIStatusError as exc:
            last_error = exc
            break

    return {
        "text": f"ERROR after {MAX_RETRIES} attempt(s): {last_error}",
        "model": model,
        "latency_ms": latency_ms,
        "tokens_in": None,
        "tokens_out": None,
        "stop_reason": "error",
        "ts": ts,
    }


def log_row(row: dict, path: str) -> None:
    file_path = Path(path)
    write_header = not file_path.exists() or file_path.stat().st_size == 0

    with file_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


PROMPTS: list[dict[str, str]] = [
    {
        "id": "short_factual",
        "text": "What is the capital of France? Answer in one word.",
    },
    {
        "id": "long_form",
        "text": (
            "Write a roughly 200 word explanation of how DNS resolution "
            "works, covering recursive resolvers, root servers, TLD "
            "servers, and authoritative name servers, suitable for a "
            "junior engineer who has never worked with networking before."
        ),
    },
    {
        "id": "strict_json",
        "text": (
            "Return ONLY valid JSON (no markdown fences, no commentary) "
            'matching this exact shape: {"name": string, "age": number, '
            '"skills": string[]}. Populate it with a fictional software '
            "engineer."
        ),
    },
]


def percentile(values: list[float], p: float) -> float:
    sorted_values = sorted(values)
    rank = (p / 100) * (len(sorted_values) - 1)
    lower = floor(rank)
    upper = rank - lower
    if lower +1 < len(sorted_values):
        result = sorted_values[lower] + upper * (sorted_values[lower + 1] - sorted_values[lower])
    else:
        result = sorted_values[lower]

    return result



def main(n: int = RUNS_PER_PROMPT) -> None:
    model = PROVIDERS[ACTIVE_PROVIDER]["default_model"]
    latencies_by_prompt: dict[str, list[float]] = {p["id"]: [] for p in PROMPTS}

    for prompt in PROMPTS:
        for _ in range(n):
            result = generate(prompt["text"], model)
            row = {"prompt_id": prompt["id"], **result}
            log_row(row, CSV_PATH)
            if result["latency_ms"] is not None:
                latencies_by_prompt[prompt["id"]].append(result["latency_ms"])

    print(f"{'prompt_id':<15} {'p50_ms':>10} {'p95_ms':>10}")
    for prompt_id, latencies in latencies_by_prompt.items():
        if not latencies:
            print(f"{prompt_id:<15} {'n/a':>10} {'n/a':>10}")
            continue
        p50 = percentile(latencies, 50)
        p95 = percentile(latencies, 95)
        print(f"{prompt_id:<15} {p50:>10.1f} {p95:>10.1f}")


if __name__ == "__main__":
    main()