"""Sample CPU and GPU activity during one streamed decode request."""
import importlib.util
import json
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("latency", ROOT / "measure_latency.py")
latency = importlib.util.module_from_spec(spec)
spec.loader.exec_module(latency)


if __name__ == "__main__":
    server_pid = int((ROOT / "server.pid").read_text())
    children = subprocess.check_output(
        ["ps", "--ppid", str(server_pid), "-o", "pid=,args="], text=True
    ).splitlines()
    engine_pid = next(int(line.split()[0]) for line in children if "EngineCore" in line)
    hz = os.sysconf(os.sysconf_names["SC_CLK_TCK"])

    def cpu_seconds():
        fields = Path(f"/proc/{engine_pid}/stat").read_text().split()
        return (int(fields[13]) + int(fields[14])) / hz

    result = {"server_pid": server_pid, "engine_pid": engine_pid}

    def generate():
        try:
            result["request"] = latency.request(
                "http://127.0.0.1:8000/v1",
                "graph_active_utilization",
                "Explain how an LRU cache works, give a small Python example, and describe its time complexity.",
                256,
            )
        except Exception as error:
            result["error"] = str(error)

    fields = [
        "gpu_busy_percent", "memory_busy_percent", "power_watts",
        "sm_clock_mhz", "memory_clock_mhz", "pcie_generation", "pcie_lanes",
    ]
    query = "utilization.gpu,utilization.memory,power.draw,clocks.sm,clocks.mem,pcie.link.gen.current,pcie.link.width.current"
    samples = []
    previous_cpu, previous_time = cpu_seconds(), time.monotonic()
    worker = threading.Thread(target=generate)
    worker.start()
    while worker.is_alive():
        output = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        now, cpu = time.monotonic(), cpu_seconds()
        sample = dict(zip(fields, map(float, output.split(","))))
        sample["engine_cpu_percent_one_core"] = 100 * (cpu - previous_cpu) / (now - previous_time)
        samples.append(sample)
        previous_cpu, previous_time = cpu, now
        worker.join(timeout=0.5)
    worker.join()
    result["samples"] = samples
    active = samples[2:-2] if len(samples) > 5 else samples
    if active:
        result["median_activity"] = {
            key: round(statistics.median(sample[key] for sample in active), 2)
            for key in active[0]
        }
    (ROOT / "active-utilization-graphs.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}), flush=True)
