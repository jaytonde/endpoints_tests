# Muse Glimmer 30B Results

## Docker command

```bash
sudo docker run --rm \
  --name Muse-Glimmer-30B-vllm \
  --gpus all \
  --ipc=host \
  --cap-add IPC_LOCK \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 127.0.0.1:30000:30000 \
  -v /home/ubuntu/models/Muse-Glimmer-30B/hub:/model_weights/hub:ro \
  -v /home/ubuntu/Muse-Glimmer-30B-runtime-cache/vllm-v0.28.0:/runtime_cache \
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
  --model meta-models/Muse-Glimmer-30B \
  --revision a4e59da52a7bc87ae7251dd5545c0dd437c44b68 \
  --served-model-name Muse-Glimmer-30B \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --load-format instanttensor \
  --max-model-len 131072 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 16384 \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --async-scheduling \
  --limit-mm-per-prompt '{"image":4}' \
  --enable-auto-tool-choice \
  --tool-call-parser muse_glimmer \
  --reasoning-parser muse_glimmer \
  --host 0.0.0.0 \
  --port 30000
```

## Acceptance test results

Command:

```bash
python acceptance_tests.py
```

- PASS basic chat
- PASS tool calling
- PASS no unnecessary tool call
- PASS reasoning field
- PASS final content
- PASS image understanding

All six acceptance checks passed. The server reported vLLM version
`0.28.0-60455b03`. The basic chat response returned `4`, tool calling selected
`get_weather` with `{"city":"London"}`, the reasoning test correctly identified
`9.8` as larger than `9.11`, and image understanding identified the beach scene.

## Concurrency benchmarks with SLA TTFT < 5000 ms

Each run used 600 random prompts with a 10,000-token target input and a
100-token output. All six runs completed 600 requests with zero failures.

| Concurrency | TTFT P99 (ms) | SLA PASS/FAIL |
|---:|---:|:---:|
| 16 | 8733.54 | FAIL |
| 32 | 22855.15 | FAIL |
| 64 | 49580.60 | FAIL |
| 128 | 105025.43 | FAIL |
| 256 | 216316.54 | FAIL |
| 512 | 439548.35 | FAIL |

No concurrency level met the TTFT P99 SLA of less than 5000 ms.

## Phase-wise loading time

| Loading phase | Time |
|---|---:|
| API initialization, configuration and engine-process startup | ~13.00 s |
| Engine/distributed initialization | ~8.00 s |
| Model loading | 10.37 s |
| └ Safetensors weight loading | 9.11 s |
| Post-load encoder/cache preparation | ~3.00 s |
| └ Cached Torch compilation loading | 0.52 s |
| └ Initial profiling/warmup | 0.02 s |
| KV-cache allocation | ~1.00 s |
| Kernel warmup and FlashInfer autotuning | ~1.00 s |
| CUDA graph capture | 8.00 s |
| Remaining engine initialization | ~2.00 s |
| API setup and application startup | ~3.00 s |
| └ Multi-modal warmup | 0.331 s |
| └ Read-only multi-modal warmup | 0.254 s |
| **Total: container log start → API ready** | **~50.00 s** |

The persisted startup plan and AOT compilation cache were reused. Approximate
phase values are derived from log timestamps; explicit vLLM timings are shown
without an approximation marker.

## Opencode tests

Manual interaction tests performed via opencode CLI:

- Essay generation
  - Generated ~900-word essay on "Why Humans Exist" covering evolutionary, ecological, cosmic, and philosophical perspectives.
- File operations
  - Create: Created `/home/ubuntu/temp_test.py` with content `print("Hello, humans exist")`
  - Run: Executed with `python3 /home/ubuntu/temp_test.py`, output `Hello, humans exist`
  - Delete: Removed file with `rm /home/ubuntu/temp_test.py`

