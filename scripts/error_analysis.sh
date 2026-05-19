#!/usr/bin/env bash
# run_error_analysis.sh
# Run error-run for the three selected models, sequentially.

set -euo pipefail

MODELS=(
    "models/m2m100_finetuned_20260427_multiun+opus"
    # "models/nllb600m_finetuned"
    # "models/m2m100_finetuned_20260426_multiun"
    # "models/mbart_finetuned"
)

TOTAL=${#MODELS[@]}
FAILED=()

echo "======================================================"
echo " Error analysis — ${TOTAL} models"
echo "======================================================"


for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    echo ""
    echo "[$(( i + 1 ))/${TOTAL}] ${MODEL}"
    echo "------------------------------------------------------"

    if [ ! -d "${MODEL}" ]; then
        echo "  ⚠  Directory not found — skipping."
        FAILED+=("${MODEL} (not found)")
        continue
    fi

    if make error-run ERROR_MODEL="${MODEL}"; then
        echo "  ✓  Done."
    else
        echo "  ✗  Failed."
        FAILED+=("${MODEL} (make error)")
    fi
done

echo ""
echo "======================================================"
echo " Summary"
echo "======================================================"
echo "  Ran    : ${TOTAL} models"
echo "  Failed : ${#FAILED[@]}"
for f in "${FAILED[@]}"; do
    echo "    - ${f}"
done
echo "======================================================"