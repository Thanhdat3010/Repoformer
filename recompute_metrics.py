import os
import subprocess
import json

def main():
    # Danh sách các file prediction bạn vừa tìm thấy
    predictions = [
        "results/repoeval/rcfcl_rg1/sparse/api_completion/deepseek_ai_deepseek_coder_1.3b_base/prediction.jsonl",
        "results/repoeval/rcfcl_rg1/sparse/function_completion/deepseek_ai_deepseek_coder_1.3b_base/prediction.jsonl",
        "results/repoeval/rcfcl_rg1/sparse/line_completion/deepseek_ai_deepseek_coder_1.3b_base/prediction.jsonl",
        "results/recceval/python/rcfcl_rg1/bm25/line_completion/deepseek_ai_deepseek_coder_1.3b_base/prediction.jsonl",
        "results/cceval/python/rcfcl_rg1/bm25/line_completion/deepseek_ai_deepseek_coder_1.3b_base/prediction.jsonl",
        "results/cceval/typescript/rcfcl_rg1/bm25/line_completion/deepseek_ai_deepseek_coder_1.3b_base/prediction.jsonl",
        "results/cceval/csharp/rcfcl_rg1/bm25/line_completion/deepseek_ai_deepseek_coder_1.3b_base/prediction.jsonl",
        "results/cceval/java/rcfcl_rg1/bm25/line_completion/deepseek_ai_deepseek_coder_1.3b_base/prediction.jsonl"
    ]

    for pred_path in predictions:
        if not os.path.exists(pred_path):
            print(f"Skipping {pred_path} (not found)")
            continue
            
        print(f"\n>>> Computing metrics for: {pred_path}")
        
        # Xác định task và language từ đường dẫn
        task = "line_completion"
        if "api_completion" in pred_path: task = "api_completion"
        if "function_completion" in pred_path: task = "function_completion"
        
        lang = "python"
        if "java" in pred_path: lang = "java"
        if "typescript" in pred_path: lang = "typescript"
        if "csharp" in pred_path: lang = "csharp"
        
        output_dir = os.path.dirname(pred_path)
        
        # Build command để chạy tính điểm (không dùng GPU)
        cmd = [
            "python", "repo_eval/eval_hf.py",
            "--only_compute_metric",
            "--task", task,
            "--language", lang,
            "--output_dir", output_dir,
            "--ts_lib", f"build/{lang}-lang-parser.so"
        ]
        
        if "cceval" in pred_path or "recceval" in pred_path:
            cmd.append("--compute_cceval_metric")
            
        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully computed metrics for {output_dir}")
        except Exception as e:
            print(f"Error computing metrics for {output_dir}: {e}")

    print("\nAll done! Now you can run: python collect_results.py --results_dir results")

if __name__ == "__main__":
    main()
