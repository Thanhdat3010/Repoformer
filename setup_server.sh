#!/usr/bin/env bash
# ============================================================
# Cài dependencies + prepare data + chạy eval
# Yêu cầu: đã clone repo và activate conda env
# Usage: conda activate repoformer && bash setup_server.sh
# ============================================================
set -e

# ---- 1. Cài PyTorch (CUDA 12.1) ----
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

# ---- 2. Cài transformers + accelerate ----
pip install "transformers>=4.45.0" accelerate datasets

# ---- 3. Cài dependencies còn lại ----
pip install tree-sitter==0.21.3 timeout-decorator editdistance
pip install bitsandbytes scikit-learn rank-bm25
pip install fuzzywuzzy python-Levenshtein nltk sacrebleu sentencepiece
pip install tensorboard gputil jsonlines codebleu numpy

# ---- 4. Build tree-sitter parsers ----
bash ts_package/build_treesitter.sh

# ---- 5. Prepare RepoEval data ----
cd repo_eval/data
bash download.sh
bash prepare.sh
cd ../cfc_retrieval
bash run.sh
cd ../..

# ---- 6. Prepare CrossCodeEval data ----
cd cceval
bash prepare_data.sh
cd ..

echo ""
echo "Setup done! Chạy eval:"
echo "  bash run_all_qwen7b.sh"
