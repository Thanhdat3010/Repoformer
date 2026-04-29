#!/usr/bin/env bash
# ============================================================
# download_all_data.sh
# Tải + chuẩn bị tất cả 3 bộ data và zip thành 1 file.
#
# Usage:  cd Repoformer && bash download_all_data.sh
# Output: repoformer_all_data.tar.gz
# ============================================================
set -e

BASEDIR=$(cd "$(dirname "$0")" && pwd)
cd "$BASEDIR"

echo "========================================"
echo " Step 1/6: Download RepoEval data"
echo "========================================"
cd repo_eval/data
bash download.sh
cd "$BASEDIR"

echo ""
echo "========================================"
echo " Step 2/6: Prepare RepoEval data"
echo "========================================"
cd repo_eval/data
bash prepare.sh
cd "$BASEDIR"

echo ""
echo "========================================"
echo " Step 3/6: Run CFC retrieval (RepoEval)"
echo "========================================"
cd repo_eval/cfc_retrieval
bash run.sh
cd "$BASEDIR"

echo ""
echo "========================================"
echo " Step 4/6: Download CrossCodeEval data"
echo "========================================"
cd cceval
bash prepare_data.sh
cd "$BASEDIR"

echo ""
echo "========================================"
echo " Step 5/6: Download & convert ReccEval"
echo "========================================"
DRACO_DIR="/tmp/draco_data"
if [ ! -d "$DRACO_DIR" ]; then
    git clone https://github.com/nju-websoft/DraCo.git "$DRACO_DIR"
fi

# Extract source code
cd "$DRACO_DIR/ReccEval"
if [ ! -d "Source_Code" ]; then
    tar -zvxf Source_Code.tar.gz
fi
cd "$BASEDIR"

# Convert to cceval format
# ReccEval is Python-only, output to cceval/processed_data/python/
# (separate dir to avoid overwriting CrossCodeEval python data)
mkdir -p cceval/recceval_processed_data/python
python convert_recceval.py \
    --input "$DRACO_DIR/ReccEval/metadata.jsonl" \
    --output_dir cceval/recceval_processed_data/python \
    --source_code_dir "$DRACO_DIR/ReccEval/Source_Code"

echo ""
echo "========================================"
echo " Step 6/6: Zipping all data..."
echo "========================================"

tar -czf repoformer_all_data.tar.gz \
    repo_eval/processed_data/ \
    cceval/processed_data/ \
    cceval/recceval_processed_data/ \
    repo_eval/data/datasets/ \
    repo_eval/data/repositories/

SIZE=$(du -h repoformer_all_data.tar.gz | cut -f1)

echo ""
echo "========================================"
echo " DONE!"
echo " Output: repoformer_all_data.tar.gz ($SIZE)"
echo ""
echo " Upload to Google Drive, then on new server:"
echo "   cd Repoformer"
echo "   tar -xzf repoformer_all_data.tar.gz"
echo "   bash run_all_qwen7b.sh"
echo "========================================"
