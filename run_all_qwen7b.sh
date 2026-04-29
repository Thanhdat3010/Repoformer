#!/usr/bin/env bash
# =============================================================================
# Master script: Run Repoformer baseline with Qwen2.5-Coder-7B
# on all 3 benchmarks (RepoEval, CrossCodeEval, ReccEval)
#
# Usage on server:
#   1. git clone / rsync the repo to the server
#   2. pip install -r requirements.txt
#   3. bash scripts/build_treesitter.sh
#   4. Prepare data (see sections below)
#   5. bash run_all_qwen7b.sh
# =============================================================================

set -e

MODEL="qwen2.5-coder-7b"
EXP="rcfcl_rg1"          # left + right context + retrieved cross-file context

echo "========================================"
echo " Running Repoformer baseline with ${MODEL}"
echo " Experiment setting: ${EXP}"
echo "========================================"

# --- 1. RepoEval ---
# Prerequisites:
#   cd repo_eval/data && bash download.sh && bash prepare.sh
#   cd repo_eval/cfc_retrieval && bash run.sh
echo ""
echo ">>> [1/3] RepoEval (line, API, function completion)"
cd repo_eval
bash run_fim_hf.sh ${MODEL} ${EXP} sparse
cd ..

# --- 2. CrossCodeEval ---
# Prerequisites:
#   cd cceval && bash prepare_data.sh
echo ""
echo ">>> [2/3] CrossCodeEval (Python, Java, TypeScript, C#)"
cd cceval
bash run_fim_hf.sh ${MODEL} ${EXP} bm25
cd ..

# --- 3. ReccEval ---
# Prerequisites:
#   Download ReccEval data from DraCo repo:
#     https://github.com/peterchenyipu/draco
#   Place processed data in cceval/processed_data/ following the same format
echo ""
echo ">>> [3/3] ReccEval (Python)"
echo "NOTE: ReccEval uses the same pipeline as CrossCodeEval."
echo "      Make sure ReccEval data is placed in cceval/processed_data/python/"
echo "      with filenames matching the expected format."
# Uncomment below once ReccEval data is in place:
# cd cceval
# bash run_fim_hf.sh ${MODEL} ${EXP} bm25
# cd ..

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
