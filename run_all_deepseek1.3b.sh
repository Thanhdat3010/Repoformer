#!/usr/bin/env bash
# Master script: DeepSeek-Coder-1.3B
set -e
MODEL="deepseek-coder-1.3b"
EXP="rcfcl_rg1"
echo "Running ${MODEL}..."
cd repo_eval && bash run_fim_hf.sh ${MODEL} ${EXP} sparse && cd ..
cd cceval && bash run_fim_hf.sh ${MODEL} ${EXP} bm25 && cd ..
# ReccEval (Manual run for safety)
cd cceval
data_root=$(realpath ./recceval_processed_data)
accelerate launch --main_process_port 29572 ../repo_eval/eval_hf.py \
    --task line_completion --model_name_or_path deepseek-ai/deepseek-coder-1.3b-base \
    --use_fim_prompt --model_type codelm_right_cfc_left --prompt_file ${data_root}/python/line_completion_rg1_bm25.jsonl \
    --gen_length 50 --max_seq_length 4096 --batch_size 8 --dtype bf16 --language python
cd ..
python collect_results.py --results_dir results
