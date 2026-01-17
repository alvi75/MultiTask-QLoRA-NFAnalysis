import os
import subprocess
import json
import pandas as pd
from collections import defaultdict

def run_lizard_analysis(input_folder, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    code_files = []
    file_extension = None
    for ext in [".java", ".cs", ".py"]:
        files = [f for f in os.listdir(input_folder) if f.endswith(ext)]
        if files:
            code_files = files
            file_extension = ext
            break
    
    print(f"Found {len(code_files)} {file_extension} files to analyze with Lizard")
    
    if not code_files:
        print(f"No code files found in {input_folder}")
        return
    
    results = []
    all_metrics = {
        "cyclomatic_complexity": [],
        "lines_of_code": [],
        "token_count": [],
        "parameter_count": []
    }
    
    failed_files = [] 
    files_with_functions = 0
    total_files = len(code_files)
    
    for idx, filename in enumerate(code_files, 1):
        file_path = os.path.join(input_folder, filename)
        task_id = filename.replace(file_extension, "")
        
        if idx % 100 == 0:
            print(f"Processing {idx}/{total_files}... (Functions found: {len(all_metrics['cyclomatic_complexity'])}, Failed: {len(failed_files)})")
        
        result = {"Task ID": task_id}
        
        # Read the code
        code_content = ""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code_content = f.read()
                result["Generated Code"] = code_content
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
        
        output_file = os.path.join(output_dir, f"{task_id}_lizard.txt")
        process = subprocess.run(["lizard", file_path], capture_output=True, text=True)
        
        with open(output_file, 'w') as f:
            f.write(process.stdout)
        
        functions_found = 0
        for line in process.stdout.split('\n'):
            line = line.strip()
            if '@' in line and line and line[0].isdigit():
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        nloc = int(parts[0])
                        ccn = int(parts[1])
                        token = int(parts[2])
                        param = int(parts[3])
                        
                        all_metrics["lines_of_code"].append(nloc)
                        all_metrics["cyclomatic_complexity"].append(ccn)
                        all_metrics["token_count"].append(token)
                        all_metrics["parameter_count"].append(param)
                        
                        functions_found += 1
                    except ValueError:
                        continue
        
        if functions_found > 0:
            files_with_functions += 1
        else:
            failed_files.append({
                "filename": filename,
                "task_id": task_id,
                "code": code_content
            })
        
        result["functions_found"] = functions_found
        results.append(result)
    
    print(f"\n=== ANALYSIS COMPLETE ===")
    print(f"Files analyzed: {total_files}")
    print(f"Files with functions: {files_with_functions}")
    print(f"Files without functions: {len(failed_files)}")
    print(f"Detection rate: {(files_with_functions/total_files*100):.1f}%")
    print(f"Total functions found: {len(all_metrics['cyclomatic_complexity'])}")
    
    if failed_files:
        failed_df = pd.DataFrame(failed_files)
        failed_csv_path = os.path.join(output_dir, "failed_translations.csv")
        failed_df.to_csv(failed_csv_path, index=False)
        print(f"\n⚠ Failed files CSV saved: {failed_csv_path}")
    
    if results:
        pd.DataFrame(results).to_excel(os.path.join(output_dir, 'lizard_analysis.xlsx'), index=False)
        generate_lizard_summary_json(output_dir, len(code_files), all_metrics, len(failed_files))
        print(f"✓ Results saved to: {output_dir}")

def generate_lizard_summary_json(output_dir, total_files, all_metrics, failed_count):
    summary = {
        "Total files": total_files,
        "Failed files": failed_count,
        "Detection rate": f"{((total_files-failed_count)/total_files*100):.1f}%",
        "Total functions": len(all_metrics["cyclomatic_complexity"]),
        "Sum Cyclomatic Complexity": sum(all_metrics["cyclomatic_complexity"]),
        "Sum Lines of Code": sum(all_metrics["lines_of_code"]),
        "Sum Token Count": sum(all_metrics["token_count"]),
        "Sum Parameter Count": sum(all_metrics["parameter_count"])
    }
    
    with open(os.path.join(output_dir, "lizard_summary.json"), 'w') as json_file:
        json.dump(summary, json_file, indent=4)
    
    print("\nSummary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    input_folder = "path/to/code_files"
    output_dir = "path/to/output/lizard"
    
    run_lizard_analysis(input_folder, output_dir)
