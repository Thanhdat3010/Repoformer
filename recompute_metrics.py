import os
import subprocess

def main():
    # Ánh xạ: prediction_file -> prompt_file
    mapping = {
        "results/repoeval/rcfcl_rg1/sparse/api_completion/deepseek_ai_deepseek_coder_1.3b_base/prediction.jsonl": 
            "data/repoeval/python/api_completion_rg1_sparse.jsonl",
            
        "results/repoeval/rcfcl_rg1/sparse/function_completion/deepseek_ai_deepseek_coder_1.3b_base/prediction.jsonl": 
            "data/repoeval/python/function_completion_rg1_sparse.jsonl",
            
        "results/repoeval/rcfcl_rg1/sparse/line_completion/deepseek_ai_deepseek_coder_1.3b_base/prediction.jsonl": 
            "data/repoeval/python/line_completion_rg1_sparse.jsonl",
            
        "results/recceval/python/rcfcl_rg1/bm25/line_completion/deepseek_ai_deepseek_coder_1.3b_base/prediction.jsonl": 
            "cceval/recceval_processed_data/python/line_completion_rg1_bm25.jsonl",
            
        "results/cceval/python/rcfcl_rg1/bm25/line_completion/deepseek_ai_deepseek_coder_1.3b_base/prediction.jsonl": 
            "cceval/processed_data/python/line_completion_rg1_bm25.jsonl",
            
        "results/cceval/typescript/rcfcl_rg1/bm25/line_completion/deepseek_ai_deepseek_coder_1.3b_base/prediction.jsonl": 
            "cceval/processed_data/typescript/line_completion_rg1_bm25.jsonl",
            
        "results/cceval/csharp/rcfcl_rg1/bm25/line_completion/deepseek_ai_deepseek_coder_1.3b_base/prediction.jsonl": 
            "cceval/processed_data/csharp/line_completion_rg1_bm25.jsonl",
            
        "results/cceval/java/rcfcl_rg1/bm25/line_completion/deepseek_ai_deepseek_coder_1.3b_base/prediction.jsonl": 
            "cceval/processed_data/java/line_completion_rg1_bm25.jsonl"
    }

    for pred_path, prompt_path in mapping.items():
        if not os.path.exists(pred_path):
            print(f"Skipping {pred_path} (not found)")
            continue
        
        # Check if prompt file exists
        if not os.path.exists(prompt_path):
            print(f"ERROR: Prompt file {prompt_path} not found!")
            continue
            
        print(f"\n>>> Computing metrics for: {pred_path}")
        print(f"    Using prompt file: {prompt_path}")
        
        # Xác định task và language từ đường dẫn
        task = "line_completion"
        if "api_completion" in pred_path: task = "api_completion"
        if "function_completion" in pred_path: task = "function_completion"
        
        lang = "python"
        if "java" in pred_path: lang = "java"
        if "typescript" in pred_path: lang = "typescript"
        if "csharp" in pred_path: lang = "csharp"
        
        output_dir = os.path.dirname(pred_path)
        
        # Build command
        cmd = [
            "python", "repo_eval/eval_hf.py",
            "--only_compute_metric",
            "--task", task,
            "--language", lang,
            "--prompt_file", prompt_path,
            "--output_dir", output_dir,
            "--ts_lib", f"build/{lang}-lang-parser.so",
            "--model_name_or_path", "deepseek-coder-1.3b"
        ]
        
        if "cceval" in pred_path or "recceval" in pred_path:
            cmd.append("--compute_cceval_metric")
            
        try:
            subprocess.run(cmd, check=True)
            print(f"SUCCESS: Metrics saved in {output_dir}")
        except Exception as e:
            print(f"FAILED: Error computing metrics for {output_dir}: {e}")

    print("\nAll metrics computed! Now run: python collect_results.py --results_dir results")

if __name__ == "__main__":
    main()
