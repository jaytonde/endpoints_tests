# GPT-OSS 120B Results

## Docker command

```bash
sudo docker run --rm \
  --name gpt-oss-120b-vllm \
  --gpus all \
  --ipc=host \
  --cap-add IPC_LOCK \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 127.0.0.1:30000:30000 \
  -v /home/ubuntu/models/gpt-oss-120b/hub:/model_weights/hub:ro \
  -v /home/ubuntu/gpt-oss-120b-runtime-cache/vllm-v0.28.0:/runtime_cache \
  -e HF_HUB_OFFLINE=1 \
  -e HF_HOME=/runtime_cache \
  -e HF_HUB_CACHE=/model_weights/hub \
  -e XDG_CACHE_HOME=/runtime_cache/xdg \
  -e VLLM_CACHE_ROOT=/runtime_cache/vllm_cache \
  -e TORCH_EXTENSIONS_DIR=/runtime_cache/torch_extensions \
  -e TORCHINDUCTOR_CACHE_DIR=/runtime_cache/inductor_cache \
  -e TRITON_CACHE_DIR=/runtime_cache/triton_cache \
  -e CUDA_CACHE_PATH=/runtime_cache/cuda_cache \
  -e FLASHINFER_WORKSPACE_BASE=/runtime_cache/flashinfer_workspace \
  -e VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/runtime_cache/flashinfer_autotune \
  -e TVM_FFI_CACHE_DIR=/runtime_cache/tvm_ffi \
  -e DG_JIT_CACHE_DIR=/runtime_cache/vllm_cache/deep_gemm \
  -e TILELANG_CACHE_DIR=/runtime_cache/tilelang_cache \
  -e CUTE_DSL_CACHE_DIR=/runtime_cache/cute_dsl_cache \
  -e VLLM_ENABLE_STARTUP_PLAN=1 \
  serverless_vllm:v0.28.0 \
  --model openai/gpt-oss-120b \
  --revision b5c939de8f754692c1647ca79fbf85e8c1e70f8a \
  --served-model-name gpt-oss-120b \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --load-format instanttensor \
  --max-model-len 131072 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --async-scheduling \
  --enable-auto-tool-choice \
  --tool-call-parser openai \
  --host 0.0.0.0 \
  --port 30000
```

## Acceptance test results

Command:

```bash
python acceptance_tests.py
```

```text
PASS basic chat
PASS tool calling
PASS no unnecessary tool call
PASS reasoning field
PASS final content
FAIL image understanding
```

Five of the six acceptance checks passed. The server reported vLLM version
`0.28.0-3b7074f4`. The basic chat response returned `4`, tool calling selected
`get_weather` with `{"city":"London"}`, and the reasoning test correctly
identified `9.8` as larger than `9.11`.

Image understanding failed because GPT-OSS 120B is being served as a text-only
model. The response stated that it could not see the image, and the usage data
reported no multimodal tokens.

## Concurrency benchmarks with SLA TTFT < 5000 ms

Each run used 600 random prompts with a 10,000-token target input and a
100-token output. All six runs completed 600 requests with zero failures.

| Concurrency | Request throughput (req/s) | TTFT P99 (ms) | SLA PASS/FAIL |
|---:|---:|---:|:---:|
| 16 | 2.694372 | 2922.99 | PASS |
| 32 | 2.991529 | 6964.80 | FAIL |
| 64 | 3.046455 | 16286.11 | FAIL |
| 128 | 3.409458 | 35377.80 | FAIL |
| 256 | 3.376223 | 72955.14 | FAIL |
| 512 | 3.300617 | 152811.08 | FAIL |

The highest concurrency satisfying the TTFT P99 SLA was **16**.

## Phase-wise loading time

| Loading phase | Time |
|---|---:|
| API initialization, configuration and engine-process startup | ~19.00 s |
| Engine/distributed initialization | ~1.00 s |
| Model loading | 12.87 s |
| └ Safetensors weight loading | 10.26 s |
| Startup-plan and cached-compilation preparation | ~1.00 s |
| └ Cached Torch compilation loading | 0.37 s |
| └ Initial profiling/warmup | 0.22 s |
| KV-cache allocation | ~1.00 s |
| Kernel warmup and FlashInfer autotuning | <1.00 s |
| CUDA graph capture | 21.00 s |
| Remaining engine initialization | ~2.00 s |
| API setup and application startup | ~4.00 s |
| **Total: container log start → API ready** | **~63.00 s** |

The persisted startup plan and AOT compilation cache were reused. Approximate
phase values are derived from log timestamps; explicit vLLM timings are shown
without an approximation marker.

## Opencode test results

The following lifecycle test was executed to verify file creation, execution, and deletion:

- **File creation** – `opencode_test.py` successfully created a temporary Python script.
- **Execution** – The script ran with `python3` and produced the expected output `Hello, World!`.
- **Deletion** – The temporary script file was removed and its absence confirmed.

All steps passed, confirming the basic file‑handling workflow works as intended.
