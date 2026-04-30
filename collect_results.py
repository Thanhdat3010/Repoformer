import os
import json
import argparse
from tabulate import tabulate

def collect_metrics(results_dir):
    all_data = []
    
    # Walk through results directory
    for root, dirs, files in os.walk(results_dir):
        if 'results.json' in files:
            file_path = os.path.join(root, 'results.json')
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Extract info from path
                # Example path: results/repoeval/rcfcl_rg1/sparse/line_completion/qwen_7b
                parts = root.split(os.sep)
                
                benchmark = parts[1] if len(parts) > 1 else "Unknown"
                setting = parts[2] if len(parts) > 2 else "Default"
                task = parts[4] if len(parts) > 4 else "All"
                model = parts[-1] if len(parts) > 0 else "Model"
                
                all_data.append({
                    "Model": model,
                    "Benchmark": benchmark,
                    "Setting": setting,
                    "Task": task,
                    "EM": data.get("em", data.get("Exact Match", "N/A")),
                    "ES": data.get("es", data.get("Edit Similarity", "N/A")),
                    "Total": data.get("total", "N/A")
                })
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    return all_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()
    
    metrics = collect_metrics(args.results_dir)
    
    if not metrics:
        print(f"No results found in '{args.results_dir}'. Make sure evaluation has finished.")
        return

    # Print Table
    print("\n" + "="*80)
    print(" REPOFORMER EVALUATION SUMMARY (IJCAI STYLE)")
    print("="*80)
    
    headers = ["Model", "Benchmark", "Task", "Setting", "EM (%)", "ES (%)"]
    table_data = []
    for m in metrics:
        table_data.append([
            m["Model"], m["Benchmark"], m["Task"], m["Setting"], m["EM"], m["ES"]
        ])
    
    # Sort by model then benchmark
    table_data.sort(key=lambda x: (x[0], x[1]))
    
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Save to summary.json
    with open(os.path.join(args.results_dir, "summary.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved detailed summary to {args.results_dir}/summary.json")

if __name__ == "__main__":
    try:
        main()
    except ImportError:
        # Fallback if tabulate is not installed
        print("Install 'tabulate' for better formatting: pip install tabulate")
        print("Printing raw summary data:")
        # Simple print logic here if needed
