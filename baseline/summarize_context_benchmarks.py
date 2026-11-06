"""Summarize the completed SGLang client context sweep."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


if __name__ == "__main__":
    folder = Path((ROOT / "context-benchmark-path.txt").read_text().strip())
    rows = []
    for length in (512, 4096, 16384, 32768, 65536):
        source = folder / f"context_{length}.jsonl"
        data = json.loads(source.read_text().splitlines()[-1])
        if data["completed"] != 5 or data["total_output_tokens"] != 5120:
            raise RuntimeError(f"Incomplete benchmark: {source}")
        tpot = data["mean_tpot_ms"]
        rows.append({
            "input_tokens_requested": length,
            "completed_requests": data["completed"],
            "total_input_tokens_reported": data["total_input_tokens"],
            "total_output_tokens": data["total_output_tokens"],
            "mean_first_token_seconds": round(data["mean_ttft_ms"] / 1000, 6),
            "mean_time_per_output_token_ms": round(tpot, 6),
            "decode_tokens_per_second_from_mean_tpot": round(1000 / tpot, 3) if tpot > 0 else None,
            "overall_output_tokens_per_second": round(data["output_throughput"], 3),
            "mean_request_seconds": round(data["mean_e2e_latency_ms"] / 1000, 6),
            "run_duration_seconds": round(data["duration"], 6),
        })
    (folder / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (folder / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Context benchmark results",
        "",
        "SGLang 0.5.22 HTTP benchmark client against vLLM 0.31.0 on port 8000. Five requests per length, one concurrent request, 1,024 output tokens each, range ratio 1.0, default seed 42. Prefix cache reset after the default single warmup request before each length. Server context: 131,072 tokens; RAM n-gram offload and decode CUDA graphs enabled; Marlin backend.",
        "",
        "| Input tokens requested | Completed | Mean first token (s) | Mean TPOT (ms) | Decode tokens/s | Overall output tokens/s | Mean request (s) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['input_tokens_requested']:,} | {row['completed_requests']}/5 | "
            f"{row['mean_first_token_seconds']:.3f} | {row['mean_time_per_output_token_ms']:.3f} | "
            f"{row['decode_tokens_per_second_from_mean_tpot']:.2f} | "
            f"{row['overall_output_tokens_per_second']:.2f} | {row['mean_request_seconds']:.3f} |"
        )
    lines += [
        "",
        "Decode tokens/s is 1,000 divided by mean time per output token. Overall output tokens/s includes prefill and request overhead. Input lengths are the benchmark's requested/reported token lengths; the client sends decoded text, so server retokenization can differ slightly. These are raw completion requests, without a chat template. EOS is ignored by the client defaults. Five samples per length provide a small performance sample, not a latency guarantee.",
        "",
        "Raw aggregate metrics and per-request timing arrays are in the context_<INPUT>.jsonl files; matching .log files contain the full benchmark output. Client compatibility details are documented in ../../context-benchmark-notes.md.",
        "",
    ]
    (folder / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)
