# Swift local inference

Model: `ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4`

Pinned model revision: `3ff0520224f264a2d0ac4ab56ece8f2f13aadb38`

This setup keeps the BF16 n-gram embedding table in pinned host RAM and runs the language model on the GPU. It uses Marlin on this RTX PRO 6000 (SM120), a 131,072-token context, at most four simultaneous sequences, and CUDA graphs for decoding. Image/video processing and speculative decoding are disabled for this text deployment.

## This machine

- NVIDIA RTX PRO 6000 Blackwell Max-Q, 97,887 MiB reported GPU memory.
- 188 GiB system RAM; approximately 183 GiB available before launch.
- Existing Python environment: `/u701/j692wu/.conda/envs/inference`.
- Existing runtime: vLLM 0.31.0, PyTorch 2.13.0+cu130, Transformers 5.17.0.
- The checkpoint is approximately 186.4 GB on disk. Its BF16 n-gram table accounts for approximately 102.4 GB of host memory. Actual runtime memory is recorded after a successful launch.

## Initial 32K measurements

Successfully launched on 2026-10-21. The server log confirms `weight_dtype=torch.bfloat16, weight_device=cpu, pinned=True` for the PLE n-gram embedding. vLLM reports 73.62 GiB for model weights and other consumed device memory, plus approximately 16.2 GiB for caches. Graph capture used approximately 0.06 GiB. NVIDIA-SMI reports approximately 91 GiB total GPU usage. System memory usage is approximately 137 GiB, with approximately 51 GiB available. The checkpoint's 102.4 GB table size is smaller than total host allocations; these measured system totals include runtime and allocator overhead.

The initial server was bound to `127.0.0.1:8000`, with model name `swift`, 32,768-token context, and at most four active sequences. The context limit has since been increased to 131,072 for the context sweep below. This deployment serves text. It uses the Marlin FP4 weight-only backend with BF16 activations and `FULL_DECODE_ONLY` CUDA graphs captured for batch sizes 1, 2, and 4. vLLM automatically enables breakable CUDA graphs for this architecture so host embedding prefetch can run outside captured segments. Torch compilation is disabled (`mode=0`); normal kernel warmup is enabled. Native FP4 kernels, image/video input, speculative decoding, and concurrency tuning were not evaluated.

Single-client measurements after the first warmup request, with thinking disabled:

| Prompt tokens | Output tokens | First token | Decode throughput | Total time |
| --- | --- | --- | --- | --- |
| 33 | 256 | 0.5435 s | 103.84 tokens/s | 3.0006 s |
| 2,209 | 256 | 0.2844 s | 102.88 tokens/s | 2.7642 s |

The first warmup request returned `Paris` correctly and took 0.85 seconds. Raw measurements and generated outputs are in `latency-results.json` and `latency-results-graphs.json`. These timings are samples, not latency guarantees or a quality evaluation. Reasoning workloads can generate many additional thinking tokens. Initial weight loading from NFS took approximately 459 seconds, followed by weight preparation, cache profiling, graph capture, and warmup.

## Eager execution comparison

The original launch forced `--enforce-eager`, which disabled CUDA graphs and normal JIT kernel warmup. Replacing that flag with `--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4]}'` raised decode throughput by approximately 8.3 times, with the same model, GPU, Marlin backend, host embedding offload, prompt lengths, and output limits.

| Metric | Initial eager execution | Current CUDA graphs |
| --- | --- | --- |
| Short-prompt decode | 12.54 tokens/s | 103.84 tokens/s |
| Longer-prompt decode | 12.48 tokens/s | 102.88 tokens/s |
| 256-token response, short prompt | 21.4640 s | 3.0006 s |
| Sampled GPU activity during decode | ~14% | ~100% |
| Sampled GPU power during decode | ~95 W | ~277 W |

These results show that the initial 12.5 tokens/s was dominated by the execution configuration; this machine can exceed 100 tokens/s while the n-gram table remains offloaded. The comparison does not isolate the residual cost of offload, since both runs keep the table in host RAM. GPU activity samples are coarse NVIDIA-SMI observations, not a kernel trace.

Baseline artifacts are preserved as `serve-eager.sh`, `server-eager.log`, `latency-results-eager.json`, and `active-utilization.json`. The current activity samples are in `active-utilization-graphs.json`; the summary is in `performance-comparison.json`.

The active Conda environment's PATH currently resolves an older CUDA 12.8 compiler, while PyTorch uses CUDA 13.0. FlashInfer's sampling JIT therefore falls back on SM120. Greedy sampling was used for the measurements above. A native FP4 backend evaluation should first select the available CUDA 13 toolkit explicitly. This compiler selection was unchanged between the two measured configurations.

## Download and start

The current server has a 131,072-token context. Override `SWIFT_CONTEXT` to choose another limit. `VLLM_SERVER_DEV_MODE=1` is enabled for the loopback server so the context benchmark can reset the prefix cache.

```bash
/u701/j692wu/.conda/envs/inference/bin/python ~/swift-local/download.py
bash ~/swift-local/serve.sh
```

Downloads use the existing Hugging Face cache without a second copy of the weights. `model-path.txt` records the resolved local snapshot path.

API endpoint: `http://127.0.0.1:8000/v1`. Model name: `swift`.

The initial background launch writes logs to `server.log` and its process ID to `server.pid`. Loading these weights from this machine's NFS cache takes several minutes; it does not represent request latency after startup.

The current 128K background server writes to `server-context-128k.log`.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"swift","messages":[{"role":"user","content":"Hello!"}],"max_tokens":128,"chat_template_kwargs":{"enable_thinking":false}}'
```

For reasoning tasks, omit `chat_template_kwargs` to use the model's default thinking behavior and allow enough output tokens for both reasoning and the answer.

## Use from another machine

Keep this server running and open an SSH tunnel on the client machine:

```bash
ssh -N -L 8001:127.0.0.1:8000 j692wu@gpu-8-vm
```

Replace the SSH destination with this machine's reachable hostname if needed. Point the client at `http://127.0.0.1:8001/v1`, model `swift`. This keeps the server bound to loopback.

To move inference to another Linux GPU machine with comparable memory and a compatible NVIDIA driver, copy this directory, remove the copied `model-path.txt`, and run:

```bash
bash ~/swift-local/setup.sh
bash ~/swift-local/serve.sh
```

The setup script creates an isolated `.venv` using Python 3 and installs the runtime versions used here. Python 3.12 is recommended. The new machine downloads its own checkpoint; copying the Hugging Face cache is optional. Set `SWIFT_PYTHON` to reuse another suitable environment instead. Review GPU and host memory capacity before launch. Override `SWIFT_CONTEXT` or `SWIFT_PORT` as needed.

## Measure latency

```bash
/u701/j692wu/.conda/envs/inference/bin/python ~/swift-local/measure_latency.py
```

This sends a short correctness/warmup request and two streamed requests with thinking disabled. It records prompt and completion token counts, first-token latency, decode throughput, total time, and generated text in `latency-results.json`. These are single-client measurements, not concurrency benchmarks or predictions for thinking workloads.

## Context sweep

All five requested input lengths completed, each with five measured requests, 1,024 output tokens per request, concurrency one, and a prefix cache reset after warmup. SGLang's benchmark client used the current vLLM backend on port 8000. Decode throughput was 101.47–103.70 tokens/s; mean first-token latency ranged from 0.138 seconds at 512 input tokens to 7.387 seconds at 65,536 input tokens.

- [Results table](context-benchmarks/20261021T204924Z/summary.md)
- [CSV summary](context-benchmarks/20261021T204924Z/summary.csv)
- [Benchmark settings and compatibility notes](context-benchmark-notes.md)
- Run again with `bash ~/swift-local/run-context-benchmarks.sh`.

## References

- https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4
- https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next
