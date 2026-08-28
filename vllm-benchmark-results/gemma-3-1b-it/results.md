# Gemma 3 1B IT Results

## Docker command

```bash
sudo docker run --rm \
  --name gemma3-1b-vllm \
  --gpus all \
  --ipc=host \
  --cap-add IPC_LOCK \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 127.0.0.1:30000:30000 \
  -v /home/ubuntu/models/gemma-3-1b-it:/model_weights:ro \
  -v /home/ubuntu/gemma3-1b-runtime-cache/vllm-v0.28.0:/runtime_cache \
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
  --served-model-name gemma-3-1b-it \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --host 0.0.0.0 \
  --port 30000
```

## Acceptance tests results

```text
PASS basic chat
SKIP tool calling
SKIP reasoning field
SKIP image understanding
```

## Concurrency benchmarks with SLA TTFT < 5000 ms

| Concurrency | TTFT P99 (ms) | SLA PASS/FAIL |
|---:|---:|:---:|
| 16 | 608.23 | PASS |
| 32 | 1125.04 | PASS |
| 64 | 2238.97 | PASS |
| 128 | 4275.66 | PASS |
| 256 | 8863.67 | FAIL |
| 512 | 18100.31 | FAIL |

## Phase-wise loading time

| Loading phase | Time |
|---|---:|
| API initialization, configuration and engine-process startup | 12.98 s |
| Engine/distributed initialization | 5.27 s |
| Model loading | 1.51 s |
| └ Safetensors weight loading | 0.37 s |
| Cached Torch compilation loading | 0.37 s |
| Initial profiling/warmup | 0.20 s |
| KV-cache profiling and allocation | ~0.70 s |
| Kernel warmup and FlashInfer autotuning | ~2.82 s |
| CUDA graph capture | ~3.00 s |
| Remaining engine initialization | ~0.28 s |
| API route setup and application startup | 2.66 s |
| **Total: container log start → API ready** | **29.53 s** |

The persisted AOT compilation cache was reused during startup. Some phase values are approximate because vLLM uses multiple processes and buffers progress-bar timestamps.
