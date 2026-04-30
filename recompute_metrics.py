import os
import subprocess

def main():
    # Ánh xạ: prediction_file -> (prompt_file, use_cceval_metric)
    mapping = {
        # --- RepoEval (prompt pattern: repo_eval/processed_data/python_{task}_sparse_rg1.jsonl) ---
        "results/repoeval/rcfcl_rg1/sparse/api_completion/deepseek_ai_deepseek_coder_1.3b_base/prediction.jsonl":
            ("repo_eval/processed_data/python_api_completion_sparse_rg1.jsonl", False),

        "results/repoeval/rcfcl_rg1/sparse/function_completion/deepseek_ai_deepseek_coder_1.3b_base/prediction.jsonl":
            ("repo_eval/processed_data/python_function_completion_sparse_rg1.jsonl", False),

        "results/repoeval/rcfcl_rg1/sparse/line_completion/deepseek_ai_deepseek_coder_1.3b_base/prediction.jsonl":
            ("repo_eval/processed_data/python_line_completion_sparse_rg1.jsonl", False),

        # --- ReccEval ---
        "results/recceval/python/rcfcl_rg1/bm25/line_completion/deepseek_ai_deepseek_coder_1.3b_base/prediction.jsonl":
            ("cceval/recceval_processed_data/python/line_completion_rg1_bm25.jsonl", True),

        # --- CrossCodeEval ---
        "results/cceval/python/rcfcl_rg1/bm25/line_completion/deepseek_ai_deepseek_coder_1.3b_base/prediction.jsonl":
            ("cceval/processed_data/python/line_completion_rg1_bm25.jsonl", True),

        "results/cceval/typescript/rcfcl_rg1/bm25/line_completion/deepseek_ai_deepseek_coder_1.3b_base/prediction.jsonl":
            ("cceval/processed_data/typescript/line_completion_rg1_bm25.jsonl", True),

        "results/cceval/csharp/rcfcl_rg1/bm25/line_completion/deepseek_ai_deepseek_coder_1.3b_base/prediction.jsonl":
            ("cceval/processed_data/csharp/line_completion_rg1_bm25.jsonl", True),

        "results/cceval/java/rcfcl_rg1/bm25/line_completion/deepseek_ai_deepseek_coder_1.3b_base/prediction.jsonl":
            ("cceval/processed_data/java/line_completion_rg1_bm25.jsonl", True),
    }

    for pred_path, (prompt_path, use_cceval) in mapping.items():
        if not os.path.exists(pred_path):
            print(f"Skipping {pred_path} (prediction not found)")
            continue

        if not os.path.exists(prompt_path):
            print(f"ERROR: Prompt file {prompt_path} not found! Skipping.")
            continue

        print(f"\n>>> Computing metrics for: {pred_path}")
        print(f"    Using prompt file: {prompt_path}")

        # Determine task and language from path
        task = "line_completion"
        if "api_completion" in pred_path: task = "api_completion"
        if "function_completion" in pred_path: task = "function_completion"

        lang = "python"
        if "/java/" in pred_path: lang = "java"
        if "/typescript/" in pred_path: lang = "typescript"
        if "/csharp/" in pred_path: lang = "csharp"

        output_dir = os.path.dirname(pred_path)

        cmd = [
            "python", "repo_eval/eval_hf.py",
            "--only_compute_metric",
            "--task", task,
            "--language", lang,
            "--prompt_file", prompt_path,
            "--output_dir", output_dir,
            "--ts_lib", f"build/{lang}-lang-parser.so",
            "--model_name_or_path", "deepseek-ai/deepseek-coder-1.3b-base",
        ]

        if use_cceval:
            cmd.append("--compute_cceval_metric")

        try:
            subprocess.run(cmd, check=True)
            print(f"SUCCESS: Metrics saved in {output_dir}")
        except Exception as e:
            print(f"FAILED: {e}")

    print("\n" + "="*60)
    print("All metrics computed! Now run:")
    print("  python collect_results.py --results_dir results")
    print("="*60)

if __name__ == "__main__":
    main()
