#!/usr/bin/env bash
# ============================================================
# Setup server: cài dependencies + giải nén data
# Yêu cầu: đã clone repo và activate conda env
#
# 2 cách dùng:
#   Cách 1 (có data zip từ Drive):
#     # Tải repoformer_all_data.tar.gz vào thư mục Repoformer
#     conda activate repoformer && bash setup_server.sh
#
#   Cách 2 (tải data từ đầu):
#     conda activate repoformer && bash setup_server.sh --download-data
# ============================================================
set -e

DOWNLOAD_DATA=false
for arg in "$@"; do
    if [ "$arg" == "--download-data" ]; then
        DOWNLOAD_DATA=true
    fi
done

# ---- 1. Cài PyTorch (CUDA 12.1) ----
echo ">>> [1/4] Installing PyTorch..."
pip install "torch==2.4.1" "torchvision==0.19.1" --index-url https://download.pytorch.org/whl/cu121
pip install "transformers==4.47.1" "accelerate>=0.34.0" "datasets>=3.0.0" "tree-sitter>=0.23.0"
pip install timeout-decorator editdistance
pip install tree-sitter-python tree-sitter-java tree-sitter-c-sharp tree-sitter-typescript
pip install bitsandbytes scikit-learn rank-bm25
pip install fuzzywuzzy python-Levenshtein nltk sacrebleu sentencepiece
pip install tensorboard gputil jsonlines codebleu numpy

# ---- 3. Build tree-sitter parsers ----
echo ">>> [3/4] Building tree-sitter..."
python ts_package/build_ts_lib.py

# ---- 4. Data ----
if [ "$DOWNLOAD_DATA" = true ]; then
    echo ">>> [4/4] Downloading data from scratch..."
    bash download_all_data.sh
elif [ -f "repoformer_all_data.tar.gz" ]; then
    echo ">>> [4/4] Extracting data from repoformer_all_data.tar.gz..."
    tar -xzf repoformer_all_data.tar.gz
    echo "Data extracted successfully!"
else
    echo ">>> [4/4] No data archive found."
    echo "    Option A: Place repoformer_all_data.tar.gz here and re-run"
    echo "    Option B: Run with --download-data flag"
fi

echo ""
echo "========================================"
echo " Setup done! Chạy eval:"
echo "   bash run_all_qwen7b.sh"
echo "========================================"
