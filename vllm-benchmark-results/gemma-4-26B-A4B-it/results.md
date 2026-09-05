# Gemma 4 26B A4B IT Results

## Docker command

```bash
sudo docker run --rm \
  --name gemma-4-26B-A4B-it-vllm \
  --gpus all \
  --ipc=host \
  --cap-add IPC_LOCK \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 127.0.0.1:30000:30000 \
  -v /home/ubuntu/models/gemma-4-26B-A4B-it/hub:/model_weights/hub:ro \
  -v /home/ubuntu/gemma-4-26B-A4B-it-runtime-cache/vllm-v0.28.0:/runtime_cache \
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
  --model google/gemma-4-26B-A4B-it \
  --revision 4d7ae4984b7db7de8f8457170b3f1a419ee76d52 \
  --served-model-name gemma-4-26B-A4B-it \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --load-format instanttensor \
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
| API initialization, configuration and engine-process startup | ~22.00 s |
| Engine/distributed initialization | ~5.00 s |
| Model loading | 9.23 s |
| └ Safetensors weight loading | 8.59 s |
| Post-load encoder/cache preparation | ~24.00 s |
| └ Cached Torch compilation loading | 0.60 s |
| └ Initial profiling/warmup | 0.62 s |
| KV-cache profiling and allocation | ~1.00 s |
| Kernel warmup and FlashInfer autotuning | ~2.00 s |
| CUDA graph capture | 24.00 s |
| Remaining engine initialization | ~2.00 s |
| API setup and application startup | ~58.00 s |
| └ Multi-modal warmup | 54.78 s |
| └ Read-only multi-modal warmup | 0.13 s |
| **Total: container log start → API ready** | **~150.00 s** |

The persisted AOT compilation cache was reused during startup. Approximate phase
values are derived from log timestamps; explicit vLLM timings are shown without
an approximation marker.
