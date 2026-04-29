#!/usr/bin/env python3
"""
Convert ReccEval (DraCo format) → Repoformer cceval format.

DraCo ReccEval format (metadata.jsonl):
  {"pkg": "...", "fpath": "...", "input": "...", "gt": "..."}

Repoformer cceval expects:
  processed_data/{language}/line_completion.jsonl           (baseline)
  processed_data/{language}/line_completion_rg1_bm25.jsonl  (with retrieval)

ReccEval has no cross-file retrieval built-in, so we create both files
with the same content (empty crossfile_context). The eval pipeline will
use whichever matches the experiment setting.
"""
import json
import argparse
import os


def convert_recceval(input_file, output_dir, source_code_dir=None):
    """Convert metadata.jsonl → cceval format .jsonl files."""
    os.makedirs(output_dir, exist_ok=True)
    
    samples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            
            # Build crossfile_context from source code if available
            crossfile_context = ""
            if source_code_dir:
                pkg_dir = os.path.join(source_code_dir, item["pkg"])
                if os.path.isdir(pkg_dir):
                    # Find .py files in the package (excluding the target file)
                    for root, dirs, files in os.walk(pkg_dir):
                        for fname in files[:5]:  # limit to 5 files for context
                            if fname.endswith('.py'):
                                fpath_full = os.path.join(root, fname)
                                rel = os.path.relpath(fpath_full, pkg_dir)
                                if rel != item.get("fpath", ""):
                                    try:
                                        with open(fpath_full, 'r', errors='ignore') as cf:
                                            content = cf.read()[:2000]
                                            crossfile_context += f"# {rel}\n{content}\n\n"
                                    except:
                                        pass
            
            sample = {
                "prompt": item["input"],
                "groundtruth": item["gt"],
                "crossfile_context": crossfile_context,
                "metadata": {
                    "task_id": f"{item['pkg']}/{item['fpath']}",
                    "pkg": item["pkg"],
                    "fpath": item["fpath"],
                }
            }
            samples.append(sample)
    
    # Write files for different experiment settings
    filenames = [
        "line_completion.jsonl",              # baseline
        "line_completion_rg1_bm25.jsonl",     # rcfcl_rg1 + bm25
        "line_completion_oracle_bm25.jsonl",  # rcfcl_oracle + bm25
    ]
    
    for fname in filenames:
        output_file = os.path.join(output_dir, fname)
        with open(output_file, 'w', encoding='utf-8') as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')
        print(f"  Written {len(samples)} samples → {output_file}")
    
    return len(samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert ReccEval (DraCo) to Repoformer cceval format"
    )
    parser.add_argument("--input", required=True,
                        help="Path to ReccEval metadata.jsonl")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory (e.g., cceval/processed_data/python)")
    parser.add_argument("--source_code_dir", default=None,
                        help="Path to extracted Source_Code/ for cross-file context")
    args = parser.parse_args()
    
    n = convert_recceval(args.input, args.output_dir, args.source_code_dir)
    print(f"\nDone! Converted {n} ReccEval samples.")
