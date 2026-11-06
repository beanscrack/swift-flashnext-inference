# Context length sweep

Requested workload: random input lengths 512, 4,096, 16,384, 32,768, and 65,536; output length 1,024; five measured prompts per length; maximum concurrency one; range ratio 1.0; cache reset after warmup before each measured run.

The current model server is vLLM, rather than SGLang. The command uses the SGLang 0.5.22 benchmark client with `--backend vllm --port 8000`, the local model tokenizer, and served model name `swift`. The remaining workload settings match the requested sweep. The server context limit was increased from 32,768 to 131,072 because 65,536 input tokens plus 1,024 output tokens exceed a 64K total context. CPU n-gram offload, Marlin, and decode CUDA graphs are retained.

`VLLM_SERVER_DEV_MODE=1` enables vLLM's `/reset_prefix_cache` endpoint, used by the client's `--flush-cache`. The server remains bound to loopback. The client performs its default warmup, flushes the prefix cache, then measures five prompts for each length. Cache flushing is per benchmark run, as in the requested command, rather than between every prompt.

SGLang's random dataset uses repeated/truncated ShareGPT text to reach the requested token lengths. The cached corpus is recorded in `benchmark-dataset-path.txt`. The client defaults to greedy sampling and ignores EOS to produce the requested output length. This uses the completion API without adding a chat template.

The client is installed in `.bench-venv` with access to the existing inference Python packages. The SGLang package was installed without GPU server dependencies. One client-only compatibility edit replaces the import of the unused disaggregated-serving constant `FAKE_BOOTSTRAP_HOST` with its upstream literal `2.2.2.2`, avoiding imports of optional SGLang GPU server kernels. Dataset generation, HTTP requests, cache flushing, and metric calculation are unchanged. `orjson` and IPython were installed locally for client imports.

Run `bash ~/swift-local/run-context-benchmarks.sh` against the ready server. Each invocation creates a new directory under `context-benchmarks/`; the path is recorded in `context-benchmark-path.txt`. Raw aggregate and per-request data are saved as `context_<INPUT>.jsonl`, with matching `.log` files. `--output-details` adds request details without changing the workload.
