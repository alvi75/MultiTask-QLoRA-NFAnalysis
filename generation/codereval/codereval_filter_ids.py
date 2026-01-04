import json
import os
from pathlib import Path

with open('ids_to_discard.json', 'r') as f:
    ids_data = json.load(f)
    java_discard = set(ids_data["CoderEval Java Ids with Unreliable Tests"])
    python_discard = set(ids_data["CoderEval Python Ids with Unreliable Tests"])

base_path = "code-generation/generation"

for root, dirs, files in os.walk(base_path):
    for file in files:
        if file.endswith('.jsonl'):
            file_path = os.path.join(root, file)
            
            if '/java/' in file_path:
                discard_ids = java_discard
                lang = "Java"
            elif '/py/' in file_path:
                discard_ids = python_discard
                lang = "Python"
            else:
                continue
            
            filtered_data = []
            total = 0
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        total += 1
                        if data['_id'] not in discard_ids:
                            filtered_data.append(data)
            
            with open(file_path, 'w') as f:
                for item in filtered_data:
                    f.write(json.dumps(item) + '\n')
            
            print(f"Filtered {file_path}")
            print(f"  {lang}: {total} -> {len(filtered_data)} tasks")

print("\nFiltering complete!")
