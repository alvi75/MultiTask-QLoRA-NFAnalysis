import json
import os
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import time
from collections import Counter
import numpy as np

INPUT_FILE = "path/to/predictions.jsonl"
OUTPUT_FOLDER = "path/to/output_folder"
LANGUAGE = "java"  # or "python"
SUMMARY_FIELD = "generated_summary"
MODEL_NAME = "model_name"

NUM_RUNS = 5

DELAY_BETWEEN_RUNS = 30


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_jsonl(file_path):
    """Load JSONL file"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def create_prompt(code, summary, language='java'):
    """Create the exact prompt as specified"""
    lang = language.capitalize()
    
    prompt = f"""You will be provided with a {lang} function ("Function") and a textual summary of it ("Comment"). The goal of the Comment is to document the functionality implemented in the Function. Your role is to evaluate the Comment across three criteria, providing as output for each of them a rating and a rationale as described in the following.

# Evaluation Criteria
* Content adequacy: the extent to which the comment summarizes all information that can be inferred from the source code.

* Conciseness: the extent to which the comment contains unnecessary information.

* Fluency: the extent to which the comment is easy to understand.

For each criterion, provide a score on a scale from 1 to 5:

1. Very poor
2. Poor
3. Fair
4. Good
5. Very good

# Function
{code}

# Comment
{summary}

Please provide your ratings in this exact format:
Content adequacy: [score]
Conciseness: [score]
Fluency: [score]"""
    
    return prompt

def call_gpt5(prompt, max_retries=3):
    """Call GPT-5-mini using the responses API with retries"""
    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model="gpt-5-mini",
                input=prompt,
                text={"verbosity": "low"},
                max_output_tokens=1024
            )
            
            output_text = ""
            
            if hasattr(response, 'output') and response.output:
                for item in response.output:
                    if hasattr(item, 'type') and item.type == 'reasoning':
                        continue

                    if hasattr(item, 'type') and item.type == 'message':
                        if hasattr(item, 'content') and item.content:
                            for content_item in item.content:
                                if hasattr(content_item, 'text') and content_item.text:
                                    output_text += content_item.text
            
            if output_text:
                return output_text
            
            if attempt < max_retries - 1:
                print(f"  Empty response, retrying... (attempt {attempt + 2}/{max_retries})")
                time.sleep(1)
                            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Error: {e}, retrying... (attempt {attempt + 2}/{max_retries})")
                time.sleep(1)
            else:
                print(f"  Error after {max_retries} attempts: {e}")
    
    return ""

def extract_scores(model_output):
    if not model_output:
        return None, None, None, ""
    
    scores = {'CA': None, 'Conciseness': None, 'Fluency': None}
    formatted_output = []
    
    lines = model_output.split('\n')
    
    for line in lines:
        line_lower = line.lower()
        
        if 'content adequacy' in line_lower:
            for char in line:
                if char.isdigit() and 1 <= int(char) <= 5:
                    scores['CA'] = int(char)
                    formatted_output.append(f"Content adequacy: {char}")
                    break
        elif 'conciseness' in line_lower:
            for char in line:
                if char.isdigit() and 1 <= int(char) <= 5:
                    scores['Conciseness'] = int(char)
                    formatted_output.append(f"Conciseness: {char}")
                    break
        elif 'fluency' in line_lower:
            for char in line:
                if char.isdigit() and 1 <= int(char) <= 5:
                    scores['Fluency'] = int(char)
                    formatted_output.append(f"Fluency: {char}")
                    break
    
    formatted_string = '\n'.join(formatted_output) if formatted_output else model_output
    
    return scores['CA'], scores['Conciseness'], scores['Fluency'], formatted_string

def evaluate_with_retry(prompt, target_id, max_total_attempts=5):
    for attempt in range(max_total_attempts):
        model_output = call_gpt5(prompt)
        ca, conciseness, fluency, formatted_output = extract_scores(model_output)
        
        if ca is not None and conciseness is not None and fluency is not None:
            return ca, conciseness, fluency, formatted_output
        
        if attempt < max_total_attempts - 1:
            missing = []
            if ca is None: missing.append("CA")
            if conciseness is None: missing.append("Conciseness")
            if fluency is None: missing.append("Fluency")
            print(f"  {target_id}: Missing {', '.join(missing)}. Retry {attempt + 1}/{max_total_attempts - 1}...")
            time.sleep(1)
    

    return ca, conciseness, fluency, formatted_output

def run_single_evaluation(run_number, data):
    
    print(f"\n{'='*60}")
    print(f"Starting Evaluation Run {run_number} of {NUM_RUNS}")
    print(f"{'='*60}")
    
    results = []
    skipped_count = 0
    blank_count = 0
    
    for item in tqdm(data, desc=f"Run {run_number} Progress"):
        target_id = item.get('id', '')
        code = item.get('code', '')
        reference_summary = item.get('reference_summary', '')
        generated_summary = item.get(SUMMARY_FIELD, '')
        
        if not generated_summary or len(generated_summary.strip()) < 3:
            skipped_count += 1
            continue
        
        prompt = create_prompt(code, generated_summary, LANGUAGE)
        ca, conciseness, fluency, formatted_output = evaluate_with_retry(prompt, target_id)
        
        if ca is None or conciseness is None or fluency is None:
            blank_count += 1

        if len(results) < 1 and run_number == 1:
            print(f"\nDebug - Extracted: CA={ca}, Conciseness={conciseness}, Fluency={fluency}")
        
        result = {
            'target_id': target_id,
            'target': reference_summary,
            'generated_by': MODEL_NAME,
            'summary': generated_summary,
            'prompt': prompt,
            'model_output': formatted_output,
            f'gpt-5-mini_CA_{run_number}': ca,
            f'gpt-5-mini_Conciseness_{run_number}': conciseness,
            f'gpt-5-mini_Fluency_{run_number}': fluency
        }
        
        results.append(result)
        
        time.sleep(0.5)
    
    print(f"\n✓ Run {run_number} completed:")
    print(f"  Evaluated: {len(results)} items")
    print(f"  Skipped: {skipped_count} items")
    print(f"  Items with blanks: {blank_count}")
    
    return pd.DataFrame(results)

def merge_all_runs(all_dfs):
    """Merge all runs and calculate final scores"""
    
    print(f"\n{'='*60}")
    print(f"Merging All Runs and Calculating Final Scores")
    print(f"{'='*60}")
    
    merged_df = all_dfs[0][['target_id', 'target', 'generated_by', 'summary', 'prompt']].copy()
    
    for run_num in range(1, NUM_RUNS + 1):
        for metric in ['CA', 'Conciseness', 'Fluency']:
            col_name = f'gpt-5-mini_{metric}_{run_num}'
            if col_name in all_dfs[run_num - 1].columns:
                merged_df[col_name] = all_dfs[run_num - 1][col_name]

    for metric in ['CA', 'Conciseness', 'Fluency']:
        metric_cols = [f'gpt-5-mini_{metric}_{i}' for i in range(1, NUM_RUNS + 1)]
        
        final_scores = []
        for _, row in merged_df.iterrows():
            scores = [row[col] for col in metric_cols if pd.notna(row[col])]
            if scores:
                score_counts = Counter(scores)
                max_count = max(score_counts.values())
                modes = [score for score, count in score_counts.items() if count == max_count]
                
                if len(modes) == 1:
                    final_scores.append(modes[0])
                else:
                    final_scores.append(np.median(scores))
            else:
                final_scores.append(None)
        
        merged_df[f'gpt-5-mini_{metric}_final'] = final_scores
    
    return merged_df

def main():
    print("\n" + "="*60)
    print("GPT-5-MINI 5-RUN EVALUATION")
    print("="*60)
    print(f"Input: {INPUT_FILE}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"Language: {LANGUAGE}")
    print(f"Runs: {NUM_RUNS}")
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    print("\nLoading data...")
    data = load_jsonl(INPUT_FILE)
    print(f"Found {len(data)} entries")
    
    all_dfs = []
    total_start = time.time()
    
    for run_num in range(1, NUM_RUNS + 1):
        run_start = time.time()
        
        df = run_single_evaluation(run_num, data)
        all_dfs.append(df)
        
        base_name = os.path.basename(OUTPUT_FOLDER)
        output_file = os.path.join(OUTPUT_FOLDER, f"{base_name}_{run_num}.csv")
        df.to_csv(output_file, index=False)
        print(f"  Saved to: {output_file}")
        
        run_time = time.time() - run_start
        print(f"  Time: {run_time:.1f} seconds")
        
        if run_num < NUM_RUNS:
            print(f"\n⏳ Waiting {DELAY_BETWEEN_RUNS} seconds before next run...")
            time.sleep(DELAY_BETWEEN_RUNS)
    
    merged_df = merge_all_runs(all_dfs)

    base_name = os.path.basename(OUTPUT_FOLDER)
    merged_file = os.path.join(OUTPUT_FOLDER, f"{base_name}_FINAL_MERGED.csv")
    merged_df.to_csv(merged_file, index=False)
    
    print(f"\n{'='*40}")
    print("FINAL SCORE STATISTICS")
    print("="*40)
    for metric in ['CA', 'Conciseness', 'Fluency']:
        col_name = f'gpt-5-mini_{metric}_final'
        valid = merged_df[col_name].dropna()
        if len(valid) > 0:
            print(f"{metric}: Mean={valid.mean():.2f}, Std={valid.std():.2f}")
    
    total_time = time.time() - total_start
    print(f"\n Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f" Files saved in: {OUTPUT_FOLDER}")
    print(f" Merged file: {merged_file}")

if __name__ == "__main__":
    main()