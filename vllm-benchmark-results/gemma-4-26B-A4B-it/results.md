# Gemma 4 26B A4B IT Results

## Docker command

```bash
sudo docker run --rm \
  --name gemma4-26b-a4b-vllm \
  --gpus all \
  --ipc=host \
  --cap-add IPC_LOCK \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 127.0.0.1:30000:30000 \
  -v /home/ubuntu/models/google-gemma-4-26B-A4B-it:/model_weights:ro \
  -v /home/ubuntu/gemma4-26b-a4b-runtime-cache/vllm-v0.28.0:/runtime_cache \
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
  --served-model-name gemma-4-26B-A4B-it \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 262144 \
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

| Concurrency | TTFT P99 (ms) | SLA PASS/FAIL |
|---:|---:|:---:|
| 16 | 2319.64 | PASS |
| 32 | 4858.62 | PASS |
| 64 | 10544.97 | FAIL |
| 128 | 22572.25 | FAIL |
| 256 | 47820.06 | FAIL |
| 512 | 97915.91 | FAIL |

## Phase-wise loading time

| Loading phase | Time |
|---|---:|
| API initialization, configuration and engine-process startup | ~21.00 s |
| Engine/distributed initialization | ~6.00 s |
| Model loading | 12.62 s |
| └ Safetensors weight loading | 12.00 s |
| Post-load encoder/cache preparation | ~25.00 s |
| └ Cached Torch compilation loading | 0.59 s |
| └ Initial profiling/warmup | 0.62 s |
| KV-cache profiling and allocation | ~10.00 s |
| Kernel warmup and FlashInfer autotuning | ~2.00 s |
| CUDA graph capture | 17.00 s |
| Remaining engine initialization | ~2.00 s |
| API setup and application startup | ~60.00 s |
| └ Multi-modal warmup | 56.80 s |
| └ Read-only multi-modal warmup | 0.11 s |
| **Total: container log start → API ready** | **~156.00 s** |

The persisted AOT compilation cache was reused during startup. Approximate phase
values are derived from log timestamps; explicit vLLM timings are shown without
an approximation marker.
