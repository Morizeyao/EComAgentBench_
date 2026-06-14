#!/bin/bash
# 评测脚本
set -e

CONFIG=${CONFIG:-"configs/settings.yaml"}
PREDICTIONS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --predictions)
            if [[ $# -lt 2 ]]; then
                echo "Error: --predictions requires a file path" >&2
                exit 1
            fi
            PREDICTIONS="$2"
            shift 2
            ;;
        *)
            echo "Usage: $0 [--predictions <predictions.jsonl>]" >&2
            exit 1
            ;;
    esac
done

echo "Running evaluation..."
echo "Config: ${CONFIG}"
if [[ -n "${PREDICTIONS}" ]]; then
    echo "Predictions: ${PREDICTIONS}"
fi

CMD=(uv run python -m src.evaluation.run_eval --config "${CONFIG}")
if [[ -n "${PREDICTIONS}" ]]; then
    CMD+=(--predictions "${PREDICTIONS}")
fi

"${CMD[@]}"
