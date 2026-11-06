"""Summarize measured Swift experiments without mixing warmup requests in."""
import argparse
import json
import re
import statistics
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    decode, prefix = [], []
    for path in sorted(root.glob("*/summary.json")):
        info = load(path)
        requests = load(path.with_name("requests.json"))
        if info["kind"] == "prefix":
            phases = {}
            for phase in ("cold", "warm"):
                rows = [x for x in requests if x["phase"] == phase]
                phases[phase] = {"mean_ttft_s": statistics.mean(x["ttft_s"] for x in rows),
                                 "mean_e2e_s": statistics.mean(x["e2e_s"] for x in rows),
                                 "mean_decode_tps": statistics.mean(x["decode_tps"] for x in rows),
                                 "requests": len(rows), "cached_tokens": []}
                for row in rows:
                    hits = row.get("counter_deltas", {}).get("vllm:prefix_cache_hits_total")
                    if hits is None:
                        hits = (row.get("usage", {}).get("prompt_tokens_details") or {}).get("cached_tokens")
                    if hits is None:
                        details = row.get("sgl_ext", {}).get("cached_tokens_details")
                        if details is not None:
                            hits = sum(details.get(key, 0) or 0 for key in ("device", "host", "storage"))
                    if hits is None:
                        hits = row.get("counter_deltas", {}).get("sglang:cached_tokens_total")
                    phases[phase]["cached_tokens"].append(hits)
            prefix.append({"label": info["label"], **phases,
                           "ttft_speedup": phases["cold"]["mean_ttft_s"] / phases["warm"]["mean_ttft_s"]})
            continue
        row = {"label": info["label"], "requests": len(requests),
               "output_tokens": sum(x["usage"]["completion_tokens"] for x in requests),
               "mean_decode_tps": statistics.mean(x["decode_tps"] for x in requests),
               "min_decode_tps": min(x["decode_tps"] for x in requests),
               "max_decode_tps": max(x["decode_tps"] for x in requests),
               "mean_ttft_s": statistics.mean(x["ttft_s"] for x in requests),
               "mean_e2e_s": statistics.mean(x["e2e_s"] for x in requests)}
        counts = info.get("counter_deltas", {})
        drafted = counts.get("vllm:spec_decode_num_draft_tokens_total", 0)
        if drafted:
            accepted = counts["vllm:spec_decode_num_accepted_tokens_total"]
            steps = counts["vllm:spec_decode_num_drafts_total"]
            row["speculation"] = {"drafted_tokens": drafted, "accepted_draft_tokens": accepted,
                                  "verification_steps": steps, "acceptance_rate": accepted / drafted,
                                  "mean_acceptance_length_including_bonus": 1 + accepted / steps}
            positions = []
            for file in ("metrics-before.txt", "metrics-after.txt"):
                text = path.with_name(file).read_text()
                values = {}
                for line in text.splitlines():
                    if line.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total{"):
                        position = int(re.search(r'position="(\d+)"', line).group(1))
                        values[position] = values.get(position, 0) + float(line.rsplit(" ", 1)[1])
                positions.append(values)
            row["speculation"]["per_position_acceptance"] = [
                (positions[1][pos] - positions[0].get(pos, 0)) / steps for pos in sorted(positions[1])]
        sg = [x.get("sgl_ext", {}).get("spec_tokens_details") for x in requests]
        sg = [x for x in sg if x and x.get("spec_num_proposed_drafts", 0)]
        if sg:
            drafted = sum(x["spec_num_proposed_drafts"] for x in sg)
            accepted = sum(x["spec_num_correct_drafts"] for x in sg)
            steps = sum(x["spec_verify_ct"] for x in sg)
            row["speculation"] = {"drafted_tokens": drafted, "accepted_draft_tokens": accepted,
                                  "verification_steps": steps, "acceptance_rate": accepted / drafted,
                                  "mean_acceptance_length_including_bonus": 1 + accepted / steps}
        row["by_case"] = {case: statistics.mean(x["decode_tps"] for x in requests if x["case"] == case)
                          for case in dict.fromkeys(x["case"] for x in requests)}
        decode.append(row)
    baseline = next((x for x in decode if x["label"] == "vllm-mtp-off"), None)
    if baseline:
        for row in decode:
            row["speedup_vs_vllm_mtp_off"] = row["mean_decode_tps"] / baseline["mean_decode_tps"]
    report = {"decode": decode, "prefix": prefix}
    (root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    md = ["# Swift inference experiments", "", "## Decode throughput", "",
          "Six fixed coding, SQL, systems and agent prompts, two rounds each, concurrency one, 512 output tokens per request. "
          "Identical saved prompt token IDs, greedy sampling, seed 42 and ignored EOS. Prefix cache reset before each decode request. "
          "Warmup excluded. These measure performance; they do not evaluate answer quality.", "",
          "| Configuration | Decode tokens/s | Range | TTFT (s) | 512-token response (s) | Speedup | Draft acceptance | Tokens per verify incl. bonus |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in decode:
        spec = row.get("speculation")
        acceptance = f"{spec['acceptance_rate']:.1%}" if spec else "Off"
        length = f"{spec['mean_acceptance_length_including_bonus']:.3f}" if spec else "1"
        md.append(f"| {row['label']} | {row['mean_decode_tps']:.2f} | {row['min_decode_tps']:.2f}–{row['max_decode_tps']:.2f} | "
                  f"{row['mean_ttft_s']:.3f} | {row['mean_e2e_s']:.3f} | {row.get('speedup_vs_vllm_mtp_off',1):.2f}× | {acceptance} | {length} |")
    md += ["", "Draft acceptance counts only proposed draft tokens. Tokens per verification adds the target model's bonus token. "
           "Measured output speed, including draft/verify overhead, determines the actual benefit.", "", "## Shared prefix reuse", "",
           "32,768 identical prefix tokens plus different 19-token follow-up questions; 128 output tokens; three cold requests "
           "with a reset before each, a separate priming request, then three warm requests without resets.", "",
           "| Configuration | Cold TTFT (s) | Warm TTFT (s) | Warm 128-token response (s) | TTFT speedup | Cached tokens per warm request |",
           "|---|---:|---:|---:|---:|---|"]
    for row in prefix:
        md.append(f"| {row['label']} | {row['cold']['mean_ttft_s']:.3f} | {row['warm']['mean_ttft_s']:.3f} | {row['warm']['mean_e2e_s']:.3f} | "
                  f"{row['ttft_speedup']:.2f}× | {row['warm']['cached_tokens']} |")
    md += ["", "Raw responses, timing data and before/after Prometheus snapshots are in each configuration directory. "
           "The counter deltas exclude warmup. Launch logs retain failures as well as successful runs.", ""]
    (root / "summary.md").write_text("\n".join(md))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
