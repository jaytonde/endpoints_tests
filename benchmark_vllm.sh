#!/usr/bin/env bash
set -uo pipefail

: "${VLLM_MODEL:?Set VLLM_MODEL to the served model name}"
: "${RULER_DATASET_PATH:?Set RULER_DATASET_PATH to a RULER bucket JSONL file}"

SERVER_URL="${VLLM_SERVER_URL:-http://localhost:8000}"
TOKENIZER="${VLLM_TOKENIZER:-$VLLM_MODEL}"
API_KEY="${VLLM_API_KEY:-}"
INPUT_LEN="${INPUT_LEN:-30000}"
INPUT_LEN_TOLERANCE="${INPUT_LEN_TOLERANCE:-512}"
OUTPUT_LEN="${OUTPUT_LEN:-512}"
TTFT_SLA_MS="${TTFT_SLA_MS:-5000}"
NUM_PROMPTS="${NUM_PROMPTS:-600}"
NUM_WARMUPS="${NUM_WARMUPS:-4}"
CONCURRENCY_LEVELS="${CONCURRENCY_LEVELS:-16 32 64 128 512}"
RESULT_DIR="${RESULT_DIR:-./vllm-benchmark-results}"
read -r -a concurrencies <<< "$CONCURRENCY_LEVELS"

mkdir -p "$RESULT_DIR"
auth_args=()
if [[ -n "$API_KEY" ]]; then
    auth_args=(--header "Authorization=Bearer $API_KEY")
fi

overall_status=0
max_passing_concurrency=0

for concurrency in "${concurrencies[@]}"; do
    result_file="concurrency-${concurrency}.json"

    echo
    echo "Testing concurrency=$concurrency with $NUM_PROMPTS RULER prompts"

    if ! vllm bench serve \
        --backend openai-chat \
        --base-url "$SERVER_URL" \
        --endpoint /v1/chat/completions \
        --model "$VLLM_MODEL" \
        --tokenizer "$TOKENIZER" \
        --dataset-name custom \
        --dataset-path "$RULER_DATASET_PATH" \
        --custom-output-len "$OUTPUT_LEN" \
        --no-oversample \
        --ignore-eos \
        --request-rate inf \
        --max-concurrency "$concurrency" \
        --num-prompts "$NUM_PROMPTS" \
        --num-warmups "$NUM_WARMUPS" \
        --percentile-metrics ttft \
        --metric-percentiles 99 \
        --goodput "ttft:$TTFT_SLA_MS" \
        --save-result \
        --save-detailed \
        --result-dir "$RESULT_DIR" \
        --result-filename "$result_file" \
        "${auth_args[@]}"; then
        echo "FAIL concurrency=$concurrency: benchmark command failed"
        overall_status=1
        continue
    fi

    if python3 - \
        "$RESULT_DIR/$result_file" \
        "$concurrency" \
        "$NUM_PROMPTS" \
        "$TTFT_SLA_MS" \
        "$INPUT_LEN" \
        "$INPUT_LEN_TOLERANCE" \
        "$OUTPUT_LEN" <<'PY'
import json
import sys

path = sys.argv[1]
concurrency = int(sys.argv[2])
requested = int(sys.argv[3])
sla_ms = float(sys.argv[4])
target_input = int(sys.argv[5])
tolerance = int(sys.argv[6])
required_output = int(sys.argv[7])

with open(path, encoding="utf-8") as file:
    result = json.load(file)

p99 = result.get("p99_ttft_ms", float("inf"))
completed = result.get("completed", 0)
input_lens = result.get("input_lens") or []
output_lens = result.get("output_lens") or []
minimum = target_input - tolerance
maximum = target_input + tolerance

passed = (
    completed == requested
    and len(input_lens) == requested
    and all(minimum <= length <= maximum for length in input_lens)
    and len(output_lens) == requested
    and all(length >= required_output for length in output_lens)
    and p99 < sla_ms
)

status = "PASS" if passed else "FAIL"
input_range = f"{min(input_lens)}-{max(input_lens)}" if input_lens else "missing"
output_range = f"{min(output_lens)}-{max(output_lens)}" if output_lens else "missing"
print(
    f"{status} concurrency={concurrency}: p99 TTFT={p99:.2f} ms "
    f"(SLA <{sla_ms:.0f}), completed={completed}/{requested}, "
    f"input={input_range} tokens (required {minimum}-{maximum}), "
    f"output={output_range} tokens (required >= {required_output})"
)
raise SystemExit(0 if passed else 1)
PY
    then
        if ((concurrency > max_passing_concurrency)); then
            max_passing_concurrency=$concurrency
        fi
    else
        overall_status=1
    fi
done

echo
if ((max_passing_concurrency == 0)); then
    echo "MAX CONCURRENCY: none of the tested levels passed."
elif ((overall_status == 0)); then
    echo "MAX CONCURRENCY: at least $max_passing_concurrency (all tested levels passed)."
else
    echo "MAX CONCURRENCY: $max_passing_concurrency (highest tested passing level)."
fi

exit "$overall_status"
