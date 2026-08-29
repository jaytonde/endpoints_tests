# Gemma 4 31B IT Results

## Docker command

```bash
sudo docker run --rm \
  --name gemma4-31b-vllm \
  --gpus all \
  --ipc=host \
  --cap-add IPC_LOCK \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 127.0.0.1:30000:30000 \
  -v /home/ubuntu/models/google-gemma-4-31B-it:/model_weights:ro \
  -v /home/ubuntu/gemma4-31B-runtime-cache/vllm-v0.28.0:/runtime_cache \
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
  --served-model-name gemma-4-31B-it \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 262144 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 16384 \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --async-scheduling \
  --limit-mm-per-prompt '{"image":4}' \
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
| 16 | 11162.46 | FAIL |
| 32 | 27413.51 | FAIL |
| 64 | 66537.61 | FAIL |
| 128 | 142049.79 | FAIL |
| 256 | 294211.10 | FAIL |
| 512 | 596274.78 | FAIL |



## OpenCode Harness Tests

```text
PASS creating python file
PASS running python file
PASS updating python file
PASS deleting python file
```
