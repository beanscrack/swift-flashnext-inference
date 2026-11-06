"""Measure streamed chat latency against the locally running Swift server."""
import argparse
import json
import time
import urllib.request
from pathlib import Path


def request(base_url, label, prompt, max_tokens):
    payload = {
        "model": "swift",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first_token = None
    last_token = None
    usage = {}
    output = []
    with urllib.request.urlopen(req, timeout=600) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("error"):
                raise RuntimeError(chunk["error"])
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                content = delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content")
                if content:
                    now = time.perf_counter()
                    if first_token is None:
                        first_token = now
                    last_token = now
                    output.append(content)
    ended = time.perf_counter()
    count = usage.get("completion_tokens", 0)
    decode_seconds = (last_token - first_token) if first_token is not None and last_token is not None else 0
    return {
        "label": label,
        "thinking_enabled": False,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": count,
        "time_to_first_token_seconds": round(first_token - started, 4) if first_token is not None else None,
        "total_seconds": round(ended - started, 4),
        "decode_tokens_per_second": round((count - 1) / decode_seconds, 2) if count > 1 and decode_seconds > 0 else None,
        "output": "".join(output),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    args = parser.parse_args()
    cases = [
        ("warmup", "What is the capital of France? Answer with the city name only.", 16),
        ("short_prompt", "Explain how an LRU cache works, give a small Python example, and describe its time complexity.", 256),
        ("longer_prompt", "The following are independent application log records.\n" + "INFO request completed status=200 cache=hit duration_ms=12\n" * 128 + "\nSummarize these records and suggest three useful monitoring metrics.", 256),
    ]
    results = []
    for label, prompt, max_tokens in cases:
        result = request(args.base_url, label, prompt, max_tokens)
        results.append(result)
        print(json.dumps(result), flush=True)
    Path(__file__).with_name("latency-results.json").write_text(json.dumps(results, indent=2) + "\n")
