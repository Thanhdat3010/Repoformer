#!/usr/bin/env bash
# =============================================================================
# Master script: Run Repoformer baseline with Qwen2.5-Coder-7B
# on all 3 benchmarks (RepoEval, CrossCodeEval, ReccEval)
#
# Usage:
#   bash setup_server.sh        # cài deps + extract data
#   bash run_all_qwen7b.sh      # chạy eval
# =============================================================================

set -e

MODEL="qwen2.5-coder-7b"
EXP="rcfcl_rg1"          # left + right context + retrieved cross-file context

echo "========================================"
echo " Running Repoformer baseline with ${MODEL}"
echo " Experiment setting: ${EXP}"
echo "========================================"

# --- 1. RepoEval (Python: line, API, function) ---
echo ""
echo ">>> [1/3] RepoEval (line, API, function completion)"
cd repo_eval
bash run_fim_hf.sh ${MODEL} ${EXP} sparse
cd ..

# --- 2. CrossCodeEval (Python, Java, TypeScript, C#) ---
echo ""
echo ">>> [2/3] CrossCodeEval (Python, Java, TypeScript, C#)"
cd cceval
bash run_fim_hf.sh ${MODEL} ${EXP} bm25
cd ..

# --- 3. ReccEval (Python only, from DraCo) ---
echo ""
echo ">>> [3/3] ReccEval (Python)"
cd cceval
# Override data_root to point to recceval_processed_data
data_root=$(realpath ./recceval_processed_data)
HOME_DIR=$(realpath ..)
output_root=${HOME_DIR}/results/recceval

# Source the model config from run_fim_hf.sh
source <(grep -A1 'py_model_zoo\["'${MODEL}'"\]' run_fim_hf.sh | head -1)
model_name=${py_model_zoo["$MODEL"]}
if [ -z "$model_name" ]; then
    model_name="Qwen/Qwen2.5-Coder-7B"
fi

prompt_file="${data_root}/python/line_completion_rg1_bm25.jsonl"
if [ -f "$prompt_file" ]; then
    out_dirname=$(echo $model_name | tr '[:upper:]' '[:lower:]' | tr '/-' '_')
    output_dir=$output_root/python/${EXP}/bm25/line_completion/$out_dirname
    mkdir -p $output_dir

    accelerate launch --main_process_port 29571 ${HOME_DIR}/repo_eval/eval_hf.py \
        --task line_completion \
        --compute_cceval_metric \
        --model_type codelm_right_cfc_left \
        --model_name_or_path $model_name \
        --use_fim_prompt \
        --preprocessing_num_workers 1 \
        --cfc_seq_length 512 \
        --min_cfc_score 0.0 \
        --prompt_file $prompt_file \
        --gen_length 50 \
        --max_seq_length 8192 \
        --batch_size 1 \
        --output_dir $output_dir \
        --dtype bf16 \
        --ts_lib ${HOME_DIR}/build/python-lang-parser.so \
        --language python 2>&1 | tee $output_dir/log.txt
else
    echo "WARNING: ReccEval data not found at ${prompt_file}"
    echo "  Run: bash download_all_data.sh   to prepare it"
fi
cd ..

echo ""
echo "========================================"
echo " All experiments complete!"
echo " Collecting results..."
echo "========================================"

python collect_results.py --results_dir results

echo ""
echo " Results saved under: results/"
echo " Summary: results/summary.json"
echo "========================================"
