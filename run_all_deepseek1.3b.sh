#!/usr/bin/env bash
# Master script: DeepSeek-Coder-1.3B
set -e
MODEL="deepseek-coder-1.3b"
EXP="rcfcl_rg1"
echo "Running ${MODEL}..."
cd repo_eval && bash run_fim_hf.sh ${MODEL} ${EXP} sparse && cd ..
cd cceval && bash run_fim_hf.sh ${MODEL} ${EXP} bm25 && cd ..

# --- ReccEval (Python) ---
# Note: ReccEval data doesn't have right_context, so use codelm_cfc
cd cceval
data_root=$(realpath ./recceval_processed_data)
HOME_DIR=$(realpath ..)
output_root=${HOME_DIR}/results/recceval

model_name="deepseek-ai/deepseek-coder-1.3b-base"
prompt_file="${data_root}/python/line_completion_rg1_bm25.jsonl"

if [ -f "$prompt_file" ]; then
    out_dirname=$(echo $model_name | tr '[:upper:]' '[:lower:]' | tr '/-' '_')
    output_dir=$output_root/python/${EXP}/bm25/line_completion/$out_dirname
    mkdir -p $output_dir

    accelerate launch --main_process_port 29572 ${HOME_DIR}/repo_eval/eval_hf.py \
        --task line_completion \
        --model_type codelm_cfc \
        --model_name_or_path $model_name \
        --use_fim_prompt \
        --preprocessing_num_workers 1 \
        --cfc_seq_length 512 \
        --min_cfc_score 0.0 \
        --prompt_file $prompt_file \
        --gen_length 50 \
        --max_seq_length 4096 \
        --batch_size 8 \
        --output_dir $output_dir \
        --dtype bf16 \
        --ts_lib ${HOME_DIR}/build/python-lang-parser.so \
        --language python 2>&1 | tee $output_dir/log.txt
fi
cd ..
python collect_results.py --results_dir results
