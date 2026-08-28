# Gemma 4 E2B IT Results

## Docker command

```bash
sudo docker run --rm \
  --name gemma4-e2b-vllm \
  --gpus all \
  --ipc=host \
  --cap-add IPC_LOCK \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 127.0.0.1:30000:30000 \
  -v /home/ubuntu/models/gemma-4-E2B-it:/model_weights:ro \
  -v /home/ubuntu/gemma4-e2b-runtime-cache/vllm-v0.28.0:/runtime_cache \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HOME=/runtime_cache \
  -e XDG_CACHE_HOME=/runtime_cache/xdg \
  -e VLLM_CACHE_ROOT=/runtime_cache/vllm \
  -e TORCH_EXTENSIONS_DIR=/runtime_cache/torch_extensions \
  -e TORCHINDUCTOR_CACHE_DIR=/runtime_cache/inductor_cache \
  -e TRITON_CACHE_DIR=/runtime_cache/triton_cache \
  -e CUDA_CACHE_PATH=/runtime_cache/cuda_cache \
  -e FLASHINFER_WORKSPACE_BASE=/runtime_cache/flashinfer_workspace \
  -e VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/runtime_cache/flashinfer_autotune \
  -e TVM_FFI_CACHE_DIR=/runtime_cache/tvm_ffi \
  -e DG_JIT_CACHE_DIR=/runtime_cache/deep_gemm \
  -e TILELANG_CACHE_DIR=/runtime_cache/tilelang \
  serverless_vllm:v0.28.0 \
  /model_weights \
  --served-model-name gemma-4-E2B-it \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 131072 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 16384 \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --async-scheduling \
  --limit-mm-per-prompt '{"image":4,"audio":1,"video":1}' \
  --mm-processor-kwargs '{"max_soft_tokens":1120}' \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4 \
  --chat-template examples/tool_chat_template_gemma4.jinja \
  --default-chat-template-kwargs '{"enable_thinking":true}' \
  --host 0.0.0.0 \
  --port 30000
```

## Acceptance tests results

```text
PASS basic chat
PASS tool calling
PASS no unnecessary tool call
PASS reasoning field
PASS final content
PASS image understanding
```

## Concurrency benchmarks with SLA TTFT < 5000 ms

Each run used 600 random prompts with a 10,000-token target input and a
100-token output. All six runs completed 600 requests with zero failures.

| Concurrency | TTFT P99 (ms) | SLA PASS/FAIL |
|---:|---:|:---:|
| 16 | 1506.61 | PASS |
| 32 | 2482.72 | PASS |
| 64 | 5465.46 | FAIL |
| 128 | 11856.37 | FAIL |
| 256 | 24756.11 | FAIL |
| 512 | 50809.67 | FAIL |

The highest concurrency satisfying the TTFT SLA was **32**.

## Phase-wise loading time

| Loading phase | Time |
|---|---:|
| API initialization, configuration, and engine-process startup | ~38 s |
| Engine/distributed initialization | ~5 s |
| Model loading | 3.20 s |
| └ Safetensors weight loading | 1.78 s |
| Engine profiling, KV-cache creation, and model warmup | 42.50 s |
| ├ Encoder-cache profiling and setup | ~19 s |
| ├ Cached Torch AOT compilation loading | 0.67 s |
| ├ Initial profiling/warmup | 0.70 s |
| ├ KV-cache profiling and allocation | ~1 s |
| ├ Kernel warmup and FlashInfer autotuning | ~7 s |
| ├ CUDA graph capture | 11 s |
| └ Remaining engine initialization | ~3 s |
| API post-engine startup | ~25 s |
| ├ Multi-modal warmup | 24.171 s |
| ├ Read-only multi-modal warmup | 0.122 s |
| └ Route setup and application startup | <1 s |
| **Total: first API log → application ready** | **~114 s** |

The total is measured from the first API-server log at `16:53:59` to
application readiness at approximately `16:55:53`. Values prefixed with `~`
are derived from second-resolution log timestamps; the other durations are
reported directly by vLLM. The persisted Torch AOT cache was reused during
startup.
