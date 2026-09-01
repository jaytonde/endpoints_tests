# Qwen 3.6 35B A3B FP8 Results

## Docker command

```bash
sudo docker run --rm \
  --name qwen3.6-35b-a3b-fp8-vllm \
  --gpus all \
  --ipc=host \
  --cap-add IPC_LOCK \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 127.0.0.1:30000:30000 \
  -v /home/ubuntu/models/Qwen3.6-35B-A3B-FP8:/model_weights:ro \
  -v /home/ubuntu/qwen3.6-35B-A3B-fp8-runtime-cache/vllm-v0.28.0:/runtime_cache \
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
  --served-model-name Qwen3.6-35B-A3B-FP8 \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 262144 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 16384 \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --async-scheduling \
  --limit-mm-per-prompt '{"image":4,"video":1}' \
  --mm-processor-kwargs '{"max_soft_tokens":1120}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
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
PASS image understanding
```

All six acceptance checks passed. The response fingerprint reported vLLM
version `0.28.0-63418ac2`. The basic chat response returned `2 + 2 = 4`, tool
calling selected `get_weather` with `{"city": "London"}`, reasoning and final
content were returned in their respective fields, and image understanding
returned `5`.

## Phase-wise loading time

| Loading phase | Time |
|---|---:|
| API initialization, configuration and engine-process startup | ~26.00 s |
| Engine/distributed initialization | ~8.00 s |
| Model loading | 22.93 s |
| └ Safetensors weight loading | 20.10 s |
| Post-load encoder/cache preparation | ~10.00 s |
| └ Cached Torch compilation loading | 0.24 s |
| └ Initial profiling/warmup | 7.40 s |
| KV-cache profiling and allocation | ~2.00 s |
| Kernel warmup and FlashInfer autotuning | ~1.00 s |
| CUDA graph capture | 26.00 s |
| Remaining engine initialization | ~2.00 s |
| API setup and application startup | ~21.00 s |
| └ Multi-modal warmup | 17.66 s |
| └ Read-only multi-modal warmup | 1.84 s |
| **Total: container log start → API ready** | **~120.00 s** |

The persisted AOT compilation cache was reused during startup. Approximate phase
values are derived from log timestamps; explicit vLLM timings are shown without
an approximation marker.
