import json
import os
from pathlib import Path


def extract_code_to_files(jsonl_path, output_dir, language='java'):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    file_extension = '.java' if language == 'java' else '.py'
    
    successful = 0
    failed = 0
    
    with open(jsonl_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
                
            try:
                data = json.loads(line)
                problem_id = data.get('_id')
                
                if 'generate_results' in data and data['generate_results']:
                    generated_code = data['generate_results'][0]
                else:
                    print(f"Line {line_num}: No generated code for {problem_id}")
                    failed += 1
                    continue
                
                file_name = f"{problem_id}{file_extension}"
                file_path = output_path / file_name
                
                with open(file_path, 'w', encoding='utf-8') as code_file:
                    code_file.write(generated_code)
                
                successful += 1
                
            except json.JSONDecodeError as e:
                print(f"Line {line_num}: JSON decode error - {e}")
                failed += 1
            except Exception as e:
                print(f"Line {line_num}: Error processing - {e}")
                failed += 1
    
    print(f"Created {successful} {language} files in {output_dir}")
    if failed > 0:
        print(f"Failed to process {failed} entries")
    
    return successful, failed


def process_all_jsonl_files(base_path):
    base_path = Path(base_path)
    
    total_java_files = 0
    total_python_files = 0
    
    for jsonl_file in base_path.rglob('*.jsonl'):
        print(f"\nProcessing: {jsonl_file.relative_to(base_path)}")
        
        if '/java/' in str(jsonl_file):
            language = 'java'
        elif '/py/' in str(jsonl_file) or '/python/' in str(jsonl_file):
            language = 'python'
        else:
            print(f"  Skipping - cannot determine language")
            continue
        
        output_dir = jsonl_file.parent / f"{jsonl_file.stem}_{language}_files"
        
        successful, failed = extract_code_to_files(
            jsonl_path=str(jsonl_file),
            output_dir=str(output_dir),
            language=language
        )
        
        if language == 'java':
            total_java_files += successful
        else:
            total_python_files += successful


if __name__ == "__main__":
    BASE_PATH = "code-generation/generation"
    
    print("="*60)
    print("JSONL TO CODE FILES CONVERTER")
    print("="*60)
    
    process_all_jsonl_files(BASE_PATH)
    
    print("\n" + "="*60)
    print("CONVERSION COMPLETE!")
