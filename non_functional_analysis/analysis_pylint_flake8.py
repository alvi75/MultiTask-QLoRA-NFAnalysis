import os
import subprocess
import json
import pandas as pd
import re
from collections import defaultdict

def extract_pylint_data(report_path):
    """Extract categorized errors and full output from pylint report."""
    categories = {"Error": [], "Warning": [], "Convention": [], "Refactor": [], "Information": []}
    output_lines = []
    
    with open(report_path, 'r') as file:
        for line in file:
            output_lines.append(line.strip())
            parts = line.split(":")
            if len(parts) > 4:
                error_code = parts[3].strip()
                message = parts[4].strip()
                if error_code.startswith("E"):
                    categories["Error"].append(f"{error_code}: {message}")
                elif error_code.startswith("W"):
                    categories["Warning"].append(f"{error_code}: {message}")
                elif error_code.startswith("C"):
                    categories["Convention"].append(f"{error_code}: {message}")
                elif error_code.startswith("R"):
                    categories["Refactor"].append(f"{error_code}: {message}")
                elif error_code.startswith("I"):
                    categories["Information"].append(f"{error_code}: {message}")
    
    return {category: "\n".join(messages) for category, messages in categories.items()}, "\n".join(output_lines)

def extract_flake8_data(report_path):
    """Extract categorized errors and full output from flake8 report."""
    categories = {"ERROR": [], "WARNING": [], "PyFlakes": [], "Complexity": [], "Naming Convention": []}
    output_lines = []
    
    with open(report_path, 'r') as file:
        for line in file:
            output_lines.append(line.strip())
            parts = line.split(":")
            if len(parts) > 3:
                error_code = parts[3].strip()
                message = parts[3].strip()
                if error_code.startswith("E"):
                    categories["ERROR"].append(f"{error_code}: {message}")
                elif error_code.startswith("W"):
                    categories["WARNING"].append(f"{error_code}: {message}")
                elif error_code.startswith("F"):
                    categories["PyFlakes"].append(f"{error_code}: {message}")
                elif error_code.startswith("C"):
                    categories["Complexity"].append(f"{error_code}: {message}")
                elif error_code.startswith("N"):
                    categories["Naming Convention"].append(f"{error_code}: {message}")
    
    return {category: "\n".join(messages) for category, messages in categories.items()}, "\n".join(output_lines)

def run_pylint_analysis(input_folder, pylint_output_dir):
    os.makedirs(pylint_output_dir, exist_ok=True)
    pylint_data = []
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.py'):
            file_path = os.path.join(input_folder, filename)
            task_id = filename.replace('.py', '')

            # Run pylint
            pylint_output_file = os.path.join(pylint_output_dir, f'{filename}_pylint.txt')
            with open(pylint_output_file, 'w') as output:
                subprocess.run(['pylint', file_path], stdout=output, stderr=output)
            categorized_pylint_errors, pylint_output = extract_pylint_data(pylint_output_file)
            
            # Read generated code
            with open(file_path, 'r') as code_file:
                generated_code = code_file.read()

            pylint_data.append({"Task ID": task_id, "Generated Code": generated_code, "Pylint Output": pylint_output, **categorized_pylint_errors})

    # Save to Excel
    pd.DataFrame(pylint_data).to_excel(os.path.join(pylint_output_dir, 'pylint_analysis.xlsx'), index=False)
    generate_pylint_summary_json(pylint_output_dir, os.path.join(pylint_output_dir, "pylint_summary.json"))

def run_flake8_analysis(input_folder, flake8_output_dir):
    os.makedirs(flake8_output_dir, exist_ok=True)
    flake8_data = []
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.py'):
            file_path = os.path.join(input_folder, filename)
            task_id = filename.replace('.py', '')

            # Run flake8
            flake8_output_file = os.path.join(flake8_output_dir, f'{filename}_flake8.txt')
            with open(flake8_output_file, 'w') as output:
                subprocess.run(['flake8', file_path], stdout=output, stderr=output)
            categorized_flake8_errors, flake8_output = extract_flake8_data(flake8_output_file)
            
            # Read generated code
            with open(file_path, 'r') as code_file:
                generated_code = code_file.read()

            flake8_data.append({"Task ID": task_id, "Generated Code": generated_code, "Flake8 Output": flake8_output, **categorized_flake8_errors})

    # Save to Excel
    pd.DataFrame(flake8_data).to_excel(os.path.join(flake8_output_dir, 'flake8_analysis.xlsx'), index=False)
    generate_flake8_summary_json(flake8_output_dir, os.path.join(flake8_output_dir, "flake8_summary.json"))

def generate_pylint_summary_json(output_dir, json_path):
    """Generate a JSON summary report for pylint rule violations."""
    parse_and_generate_summary(output_dir, json_path, extract_pylint_data)

def generate_flake8_summary_json(output_dir, json_path):
    """Generate a JSON summary report for flake8 rule violations."""
    parse_and_generate_summary(output_dir, json_path, extract_flake8_data)

def parse_and_generate_summary(output_dir, json_path, parser_function):
    """Generate a JSON summary report for rule violations."""
    total_code_count = 0
    total_violations = defaultdict(int)
    error_code_counts = defaultdict(lambda: defaultdict(int))

    for filename in os.listdir(output_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(output_dir, filename)
            total_code_count += 1
            parsed_data, _ = parser_function(file_path)

            for category, messages in parsed_data.items():
                message_list = messages.split('\n') if messages else []
                total_violations[category] += len(message_list)
                for msg in message_list:
                    error_code = msg.split(":")[0]
                    error_code_counts[category][error_code] += 1
    
    summary = {
        "Total code count": {"files": total_code_count},
        "Total Rule Violations": dict(total_violations),
        "Error Code Counts": {category: dict(codes) for category, codes in error_code_counts.items()}
    }
    
    with open(json_path, 'w') as json_file:
        json.dump(summary, json_file, indent=4)
    print(f"JSON summary saved to {json_path}")



if __name__ == "__main__":
    input_folder = "path/to/python_files"
    pylint_output_dir = "path/to/output/pylint"
    flake8_output_dir = "path/to/output/flake8"
    
    run_pylint_analysis(input_folder, pylint_output_dir)
    run_flake8_analysis(input_folder, flake8_output_dir)
