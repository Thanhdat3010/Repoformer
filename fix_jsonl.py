import json
import os

def fix_jsonl(file_path):
    print(f"Checking {file_path}...")
    valid_lines = []
    error_count = 0
    
    if not os.path.exists(file_path):
        print(f"File {file_path} not found!")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                json.loads(line)
                valid_lines.append(line)
            except Exception as e:
                print(f"Error at line {i+1}: {e}")
                error_count += 1
    
    if error_count > 0:
        print(f"Found {error_count} bad lines. Fixing...")
        with open(file_path, 'w', encoding='utf-8') as f:
            for line in valid_lines:
                f.write(line)
        print("Fixed!")
    else:
        print("File is clean.")

if __name__ == "__main__":
    ts_file = "cceval/processed_data/typescript/line_completion_rg1_bm25.jsonl"
    fix_jsonl(ts_file)
