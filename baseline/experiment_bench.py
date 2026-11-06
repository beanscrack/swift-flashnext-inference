"""Matched single-request benchmarks for Swift inference experiments."""
import argparse
import datetime as dt
import hashlib
import json
import statistics
import time
import urllib.request
from pathlib import Path


def api(base, route, payload=None):
    req = urllib.request.Request(base + route, data=None if payload is None else json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as res:
        return res.read().decode()


def metrics(base):
    raw = api(base, "/metrics")
    values = {}
    for line in raw.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        key, value, *_ = line.split()
        key = key.split("{")[0]
        if key.endswith("_created"):
            continue
        try:
            values[key] = values.get(key, 0) + float(value)
        except ValueError:
            pass
    return raw, values


def request(base, item, sglang=False):
    payload = {"model": "swift", "prompt": item["prompt_ids"], "max_tokens": item["output_tokens"],
               "temperature": 0, "seed": 42, "ignore_eos": True, "stream": True,
               "stream_options": {"include_usage": True}}
    if sglang:
        payload["return_spec_tokens_details"] = True
        payload["return_cached_tokens_details"] = True
    req = urllib.request.Request(base + "/v1/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    first, last, usage, chunks, output, extensions = None, None, {}, 0, [], {}
    with urllib.request.urlopen(req, timeout=600) as res:
        for line in res:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                break
            data = json.loads(data)
            if data.get("error"):
                raise RuntimeError(data["error"])
            if data.get("usage"):
                usage = data["usage"]
            if data.get("sglext"):
                extensions.update(data["sglext"])
            for choice in data.get("choices", []):
                if choice.get("text"):
                    now = time.perf_counter()
                    if first is None:
                        first = now
                    last = now
                    chunks += 1
                    output.append(choice["text"])
    end = time.perf_counter()
    count = usage.get("completion_tokens", 0)
    if count != item["output_tokens"]:
        raise RuntimeError(f"Unexpected completion length: {count} vs {item['output_tokens']}")
    return {"case": item["label"], "prompt_sha256": hashlib.sha256(json.dumps(item["prompt_ids"]).encode()).hexdigest(),
            "usage": usage, "sgl_ext": extensions, "ttft_s": first - start, "e2e_s": end - start,
            "decode_tps": (count - 1) / (last - first), "text_chunks": chunks,
            "output_sha256": hashlib.sha256("".join(output).encode()).hexdigest(),
            "output": "".join(output)}


def prepare(root):
    from transformers import AutoTokenizer
    model = Path(__file__).with_name("model-path.txt").read_text().strip()
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    tasks = [
        ("python_cache", "Implement a production quality Python LRU cache with a capacity limit, get and put methods, an expiry time, and thread safety. Explain the design and write detailed unit tests. Include the complete implementation."),
        ("typescript_agent", "Write a complete TypeScript task runner for an agent. Tasks have dependencies, timeouts, retries with exponential backoff and cancellation. Explain the architecture, include types, implementation and usage examples."),
        ("rust_parser", "Implement a JSON streaming parser in Rust. It should handle escaped strings and nested arrays and objects, report precise errors, and accept chunks. Explain and include a complete implementation and tests."),
        ("sql_analysis", "Design a PostgreSQL schema for an ecommerce marketplace. Include users, listings, orders, payments, refunds and seller payouts. Explain transaction handling and write the DDL and queries for monthly revenue and cohort retention."),
        ("systems_explanation", "Explain how a modern database executes a join between two large tables. Discuss hash joins, merge joins, nested loops, indexes, memory limits, spilling, statistics, query planning, and several worked examples."),
        ("agent_plan", "You are a coding agent working on a Python web service whose latency increased after a deployment. Write a detailed investigation plan covering traces, CPU and memory profiles, database queries, lock contention and deployment rollback. Show shell commands, Python instrumentation and how to interpret results."),
    ]
    cases = []
    for label, task in tasks:
        ids = tok.apply_chat_template([{"role": "user", "content": task}], tokenize=True,
                                      add_generation_prompt=True, enable_thinking=False, return_dict=False)
        cases.append({"label": label, "prompt_ids": ids, "output_tokens": 512})
    (root / "decode-cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    doc = "Repository reference material.\n" + "\n".join(
        f"Module {i}: requests use explicit timeouts, structured logs, bounded retries and database transactions. "
        "The service validates inputs, caches read results, records metrics and propagates cancellation."
        for i in range(3000))
    common = tok.encode(doc, add_special_tokens=False)[:32768]
    cases = []
    for i in range(4):
        suffix = tok.encode(f"\nUser request {i}: explain the timeout handling and propose a useful improvement.\nAssistant:", add_special_tokens=False)
        cases.append({"label": f"shared_prefix_{i}", "prompt_ids": common + suffix, "output_tokens": 128})
    (root / "prefix-cases.json").write_text(json.dumps(cases) + "\n")
    print(json.dumps({"decode_prompt_lengths": [len(x["prompt_ids"]) for x in json.loads((root / 'decode-cases.json').read_text())],
                      "common_prefix_tokens": len(common), "prefix_prompt_lengths": [len(x["prompt_ids"]) for x in cases]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--label")
    parser.add_argument("--kind", choices=["decode", "prefix"], default="decode")
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--reset-route", default="/reset_prefix_cache")
    parser.add_argument("--sglang", action="store_true")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        prepare(args.root)
        return
    dest = args.root / args.label
    dest.mkdir(exist_ok=True)
    cases = json.loads((args.root / f"{args.kind}-cases.json").read_text())
    warmup = dict(cases[0], output_tokens=32)
    request(args.base, warmup, args.sglang)
    api(args.base, args.reset_route, {})
    time.sleep(12)
    raw, before = metrics(args.base)
    (dest / "metrics-before.txt").write_text(raw)
    results = []
    if args.kind == "decode":
        work = [(f"round_{r}", item, True) for r in range(args.rounds) for item in cases]
    else:
        work = [("cold", item, True) for item in cases[1:]]
        work += [("prime", cases[0], True)]
        work += [("warm", item, False) for item in cases[1:]]
    for phase, item, reset in work:
        if reset:
            api(args.base, args.reset_route, {})
        _, request_before = metrics(args.base)
        result = request(args.base, item, args.sglang)
        _, request_after = metrics(args.base)
        result["counter_deltas"] = {key: request_after[key] - request_before.get(key, 0)
                                    for key in request_after
                                    if ("prefix_cache" in key or "spec_decode" in key
                                        or key in ("sglang:cached_tokens_total", "sglang:spec_verify_calls_total"))
                                    and key.endswith("_total")}
        result["phase"] = phase
        results.append(result)
        (dest / "requests.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "output"}), flush=True)
    time.sleep(12)
    raw, after = metrics(args.base)
    (dest / "metrics-after.txt").write_text(raw)
    delta = {key: after[key] - before.get(key, 0) for key in after if key.endswith("_total") or "spec_decode" in key}
    summary = {"label": args.label, "kind": args.kind, "base": args.base, "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
               "request_count": len(results), "mean_decode_tps": statistics.mean(x["decode_tps"] for x in results),
               "mean_ttft_s": statistics.mean(x["ttft_s"] for x in results), "counter_deltas": delta}
    if args.kind == "prefix":
        summary["phases"] = {phase: {"mean_ttft_s": statistics.mean(x["ttft_s"] for x in results if x["phase"] == phase),
                                      "cached_tokens": [x["usage"].get("prompt_tokens_details") for x in results if x["phase"] == phase]}
                              for phase in ["cold", "prime", "warm"]}
    (dest / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
