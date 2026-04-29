#!/usr/bin/env python3
"""
Collect and summarize all evaluation results for the ACAR paper.

Scans the results/ directory and aggregates:
  - EM (Exact Match) and ES (Edit Similarity) per benchmark/task/model
  - Prompt generation time per benchmark/model
  - Per-language breakdown for CrossCodeEval

Usage:
  python collect_results.py [--results_dir results]

Output:
  - Prints formatted tables to stdout
  - Saves results/summary.json
"""

import os
import json
import argparse
from collections import defaultdict


def find_result_files(results_dir):
    """Walk results directory and find all results.json and prompt_gen_time.json files."""
    entries = []
    for root, dirs, files in os.walk(results_dir):
        for fname in files:
            if fname in ("results.json", "prompt_gen_time.json"):
                fpath = os.path.join(root, fname)
                # Parse path to extract benchmark/task/model info
                rel = os.path.relpath(fpath, results_dir)
                entries.append((rel, fpath, fname))
    return entries


def load_json(fpath):
    with open(fpath, 'r') as f:
        return json.load(f)


def collect_all(results_dir):
    entries = find_result_files(results_dir)
    
    results = defaultdict(dict)
    prompt_times = defaultdict(dict)
    
    for rel, fpath, fname in entries:
        parts = rel.replace("\\", "/").split("/")
        
        if fname == "results.json":
            data = load_json(fpath)
            # Key: full relative path minus the filename
            key = "/".join(parts[:-1])
            results[key] = data
        
        elif fname == "prompt_gen_time.json":
            data = load_json(fpath)
            key = "/".join(parts[:-1])
            prompt_times[key] = data

    return results, prompt_times


def print_table(title, rows, headers):
    """Print a formatted ASCII table."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    
    col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
    
    header_line = " | ".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-+-".join("-" * w for w in col_widths))
    
    for row in rows:
        print(" | ".join(str(v).ljust(w) for v, w in zip(row, col_widths)))


def main():
    parser = argparse.ArgumentParser(description="Collect ACAR paper results")
    parser.add_argument("--results_dir", default="results", help="Root results directory")
    args = parser.parse_args()
    
    results, prompt_times = collect_all(args.results_dir)
    
    if not results and not prompt_times:
        print(f"No results found in '{args.results_dir}/'")
        print("Run the evaluation scripts first.")
        return
    
    # --- Print EM/ES Results ---
    print("\n" + "=" * 80)
    print("  EVALUATION RESULTS SUMMARY")
    print("=" * 80)
    
    headers = ["Benchmark/Task/Model", "EM (%)", "ES (%)", "Samples"]
    rows = []
    for key in sorted(results.keys()):
        r = results[key]
        rows.append([
            key,
            r.get("em", "N/A"),
            r.get("es_repoeval", r.get("es", "N/A")),
            r.get("total", "N/A")
        ])
    
    if rows:
        print_table("EM & ES Scores", rows, headers)
    
    # --- Print Prompt Gen Times ---
    if prompt_times:
        headers_pt = ["Benchmark/Task/Model", "Avg Prompt Gen (ms)", "Total (ms)", "Samples"]
        rows_pt = []
        for key in sorted(prompt_times.keys()):
            pt = prompt_times[key]
            rows_pt.append([
                key,
                pt.get("avg_prompt_gen_time_ms", "N/A"),
                pt.get("total_prompt_gen_time_ms", "N/A"),
                pt.get("num_samples", "N/A")
            ])
        print_table("Prompt Generation Time", rows_pt, headers_pt)
    
    # --- Aggregate per benchmark ---
    print("\n" + "=" * 80)
    print("  AGGREGATED BY BENCHMARK (for paper tables)")
    print("=" * 80)
    
    benchmarks = defaultdict(list)
    for key, r in results.items():
        parts = key.split("/")
        # Try to identify benchmark
        if "repoeval" in key.lower():
            bmark = "RepoEval"
        elif "cceval" in key.lower():
            bmark = "CrossCodeEval"
        elif "recceval" in key.lower():
            bmark = "ReccEval"
        else:
            bmark = parts[0] if parts else "Unknown"
        benchmarks[bmark].append((key, r))
    
    for bmark, entries in sorted(benchmarks.items()):
        print(f"\n--- {bmark} ---")
        for key, r in entries:
            em = r.get("em", "N/A")
            es = r.get("es_repoeval", r.get("es", "N/A"))
            print(f"  {key}: EM={em}%, ES={es}%")
    
    # --- Save summary ---
    summary = {
        "results": dict(results),
        "prompt_gen_times": dict(prompt_times)
    }
    summary_path = os.path.join(args.results_dir, "summary.json")
    os.makedirs(args.results_dir, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n\nFull summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
