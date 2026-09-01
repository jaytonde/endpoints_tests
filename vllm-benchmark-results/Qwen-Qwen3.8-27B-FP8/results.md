# Qwen 3.8 27B FP8 Results

## Docker command

```bash
sudo docker run --rm \
  --name qwen3.8-27b-fp8-vllm \
  --gpus all \
  --ipc=host \
  --cap-add IPC_LOCK \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 127.0.0.1:30000:30000 \
  -v /home/ubuntu/models/Qwen-Qwen3.8-27B-FP8:/model_weights:ro \
  -v /home/ubuntu/qwen3.8-27b-fp8-runtime-cache/vllm-v0.28.0:/runtime_cache \
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
  --served-model-name Qwen-Qwen3.8-27B-FP8 \
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

All six acceptance checks passed. The server reported vLLM version
`0.28.0-a096cf33`. The basic chat response returned `4`, tool calling selected
`get_weather` with `{"city":"London"}`, and image understanding returned `5`.

## Concurrency benchmarks with SLA TTFT < 5000 ms

Each run used 600 random prompts with a 10,000-token target input and a
100-token output. All six runs completed 600 requests with zero failures.

| Concurrency | Request throughput (req/s) | TTFT P99 (ms) | SLA PASS/FAIL |
|---:|---:|---:|:---:|
| 16 | 1.508055 | 7070.74 | FAIL |
| 32 | 1.593661 | 16316.97 | FAIL |
| 64 | 1.639218 | 34550.67 | FAIL |
| 128 | 1.648560 | 74309.95 | FAIL |
| 256 | 1.642803 | 151511.38 | FAIL |
| 512 | 1.641532 | 306550.23 | FAIL |

No concurrency level met the TTFT P99 SLA of less than 5000 ms.

## Phase-wise loading time

| Loading phase | Time |
|---|---:|
| API initialization, configuration and engine-process startup | ~25.00 s |
| Engine/distributed initialization | ~8.00 s |
| Model loading | 10.48 s |
| └ Safetensors weight loading | 7.03 s |
| Post-load encoder/cache preparation | ~9.00 s |
| └ Cached Torch compilation loading | 1.27 s |
| └ Initial profiling/warmup | 7.56 s |
| KV-cache profiling and allocation | ~2.00 s |
| Kernel warmup and FlashInfer autotuning | ~2.00 s |
| CUDA graph capture | 22.00 s |
| Remaining engine initialization | ~2.00 s |
| API setup and application startup | ~21.00 s |
| └ Multi-modal warmup | 18.02 s |
| └ Read-only multi-modal warmup | 1.87 s |
| **Total: container log start → API ready** | **~103.00 s** |

The persisted AOT compilation cache was reused during startup. Approximate phase
values are derived from log timestamps; explicit vLLM timings are shown without
an approximation marker.
