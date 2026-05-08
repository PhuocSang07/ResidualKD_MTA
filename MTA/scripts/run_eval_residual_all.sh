#! /bin/bash
# Chạy toàn bộ eval cho các residual model.
# Mặc định chạy tuần tự. Dùng --parallel để chạy song song (cẩn thận VRAM).
#
# Cách dùng (từ thư mục MTA/):
#   bash scripts/run_eval_residual_all.sh             # tuần tự
#   bash scripts/run_eval_residual_all.sh --parallel  # song song
#   bash scripts/run_eval_residual_all.sh gpt2        # chỉ GPT2
#   bash scripts/run_eval_residual_all.sh llama opt   # LLaMA + OPT

set -e
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ==== Danh sách scripts theo nhóm ====
GPT2_SCRIPTS=(
    # "eval_gpt2_0.1B_residual_mta.sh"
    # "eval_gpt2_0.1B_residual_mta_entropy.sh"
    # "eval_gpt2_0.35B_residual_mta_entropy.sh"
    "eval_gpt2_1.5B_residual_paper.sh"
    "eval_gpt2_1.5B_residual_mta_entropy.sh"
)

LLAMA_SCRIPTS=(
    "eval_llama_1.1B_residual_paper.sh"
    "eval_llama_1.1B_residual_mta_entropy.sh"
)

OPT_SCRIPTS=(
    "eval_opt_2.7B_residual_paper.sh"
    "eval_opt_2.7B_residual_mta_entropy.sh"
)

# ==== Parse arguments ====
PARALLEL=false
RUN_GROUPS=()

for arg in "$@"; do
    case "$arg" in
        --parallel) PARALLEL=true ;;
        gpt2)  RUN_GROUPS+=("gpt2") ;;
        llama) RUN_GROUPS+=("llama") ;;
        opt)   RUN_GROUPS+=("opt") ;;
        *)     echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# Mặc định: chạy tất cả các nhóm
if [ ${#RUN_GROUPS[@]} -eq 0 ]; then
    RUN_GROUPS=("gpt2" "llama" "opt")
fi

# Gom scripts theo nhóm được chọn
SELECTED=()
for g in "${RUN_GROUPS[@]}"; do
    case "$g" in
        gpt2)  SELECTED+=("${GPT2_SCRIPTS[@]}") ;;
        llama) SELECTED+=("${LLAMA_SCRIPTS[@]}") ;;
        opt)   SELECTED+=("${OPT_SCRIPTS[@]}") ;;
    esac
done

# ==== Chạy ====
echo "=================================================="
echo " ResidualKD — Run All Eval"
echo " Groups  : ${RUN_GROUPS[*]}"
echo " Mode    : $([ "$PARALLEL" = true ] && echo 'parallel' || echo 'sequential')"
echo " Scripts : ${#SELECTED[@]}"
echo "=================================================="
echo ""

PIDS=()
FAILED=()
START_ALL=$(date +%s)

run_script() {
    local script="$1"
    local path="${SCRIPTS_DIR}/${script}"

    if [ ! -f "$path" ]; then
        echo "[SKIP] $script — file not found"
        return 1
    fi

    echo "[START] $script"
    local t0=$(date +%s)
    bash "$path"
    local status=$?
    local elapsed=$(( $(date +%s) - t0 ))

    if [ $status -eq 0 ]; then
        printf "[DONE]  %s  (%.0fm%.0fs)\n" "$script" "$((elapsed/60))" "$((elapsed%60))"
    else
        printf "[FAIL]  %s  (exit %d)\n" "$script" "$status"
        return 1
    fi
}

if [ "$PARALLEL" = true ]; then
    for script in "${SELECTED[@]}"; do
        run_script "$script" &
        PIDS+=($!)
    done

    for i in "${!PIDS[@]}"; do
        if ! wait "${PIDS[$i]}"; then
            FAILED+=("${SELECTED[$i]}")
        fi
    done
else
    for script in "${SELECTED[@]}"; do
        if ! run_script "$script"; then
            FAILED+=("$script")
        fi
        echo ""
    done
fi

# ==== Tổng kết ====
TOTAL_ELAPSED=$(( $(date +%s) - START_ALL ))
echo "=================================================="
printf " Tổng thời gian: %dm%ds\n" "$((TOTAL_ELAPSED/60))" "$((TOTAL_ELAPSED%60))"
echo " Hoàn thành    : $(( ${#SELECTED[@]} - ${#FAILED[@]} )) / ${#SELECTED[@]}"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo " Thất bại      :"
    for f in "${FAILED[@]}"; do echo "   - $f"; done
    echo "=================================================="
    exit 1
fi
echo "=================================================="
