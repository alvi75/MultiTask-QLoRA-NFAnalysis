import pandas as pd
import numpy as np
import os
from glob import glob


INPUT_FOLDER = "path/to/llm_judge_results"
OUTPUT_FILE_NAME = "FINAL_MERGED_MEAN.csv"

NUM_RUNS = 5


def load_all_runs(input_folder, num_runs):
    """Load all run CSV files"""
    all_dfs = []

    base_name = os.path.basename(input_folder)
    
    for run_num in range(1, num_runs + 1):
        pattern = os.path.join(input_folder, f"{base_name}_{run_num}.csv")
        matching_files = glob(pattern)
        
        if not matching_files:
            pattern = os.path.join(input_folder, f"*_{run_num}.csv")
            matching_files = glob(pattern)
        
        if matching_files:
            print(f"Loading Run {run_num}: {matching_files[0]}")
            df = pd.read_csv(matching_files[0])
            all_dfs.append(df)
            print(f"  Loaded {len(df)} rows")
        else:
            print(f"Warning: Could not find file for Run {run_num}")
    
    return all_dfs

def merge_runs_with_mean(all_dfs):
    
    if not all_dfs:
        raise ValueError("No dataframes to merge!")
    
    print(f"\nMerging {len(all_dfs)} runs and calculating mean scores...")
    
    base_cols = ['target_id', 'target', 'generated_by', 'summary', 'prompt']
    available_base_cols = [col for col in base_cols if col in all_dfs[0].columns]
    merged_df = all_dfs[0][available_base_cols].copy()
    
    for i, df in enumerate(all_dfs, 1):
        for metric in ['CA', 'Conciseness', 'Fluency']:
            col_name = f'gpt-5-mini_{metric}_{i}'
            if col_name in df.columns:
                # Ensure we're aligning by index
                merged_df[col_name] = df[col_name].values
            else:
                print(f"Warning: Column {col_name} not found in run {i}")

    for metric in ['CA', 'Conciseness', 'Fluency']:
        metric_cols = [f'gpt-5-mini_{metric}_{i}' for i in range(1, len(all_dfs) + 1)]
        existing_cols = [col for col in metric_cols if col in merged_df.columns]
        
        if existing_cols:
            # Calculate mean, ignoring NaN values
            merged_df[f'gpt-5-mini_{metric}_final'] = merged_df[existing_cols].mean(axis=1)
            
            merged_df[f'gpt-5-mini_{metric}_std'] = merged_df[existing_cols].std(axis=1)
  
            merged_df[f'gpt-5-mini_{metric}_count'] = merged_df[existing_cols].count(axis=1)
            
            print(f"\n{metric} Statistics:")
            print(f"  Mean of means: {merged_df[f'gpt-5-mini_{metric}_final'].mean():.3f}")
            print(f"  Mean std dev: {merged_df[f'gpt-5-mini_{metric}_std'].mean():.3f}")
            print(f"  Samples with all {len(all_dfs)} scores: {(merged_df[f'gpt-5-mini_{metric}_count'] == len(all_dfs)).sum()}")
    
    return merged_df

def print_comparison_stats(merged_df):
    """Print detailed statistics about the final scores"""
    
    print("\n" + "="*60)
    print("FINAL SCORE STATISTICS (Using Mean)")
    print("="*60)
    
    for metric in ['CA', 'Conciseness', 'Fluency']:
        final_col = f'gpt-5-mini_{metric}_final'
        std_col = f'gpt-5-mini_{metric}_std'
        count_col = f'gpt-5-mini_{metric}_count'
        
        if final_col in merged_df.columns:
            valid_scores = merged_df[final_col].dropna()
            
            if len(valid_scores) > 0:
                print(f"\n{metric}:")
                print(f"  Total samples: {len(merged_df)}")
                print(f"  Valid samples: {len(valid_scores)}")
                print(f"  Mean score: {valid_scores.mean():.3f}")
                print(f"  Std of mean scores: {valid_scores.std():.3f}")
                
                if std_col in merged_df.columns:
                    avg_std = merged_df[std_col].dropna().mean()
                    print(f"  Average std across runs: {avg_std:.3f}")
                
                if count_col in merged_df.columns:
                    full_data = (merged_df[count_col] == NUM_RUNS).sum()
                    print(f"  Samples with all {NUM_RUNS} runs: {full_data}")
                
                print(f"  Score distribution:")
                for score in range(1, 6):
                    count = ((valid_scores >= score - 0.5) & (valid_scores < score + 0.5)).sum()
                    pct = (count / len(valid_scores)) * 100
                    print(f"    ~{score}: {count} ({pct:.1f}%)")

def main():
    print("="*60)
    print("RECALCULATING FINAL SCORES USING MEAN")
    print("="*60)
    print(f"Input folder: {INPUT_FOLDER}")
    print(f"Expected runs: {NUM_RUNS}")
    
    all_dfs = load_all_runs(INPUT_FOLDER, NUM_RUNS)
    
    if len(all_dfs) == 0:
        print("Error: No CSV files found!")
        return
    
    print(f"\nSuccessfully loaded {len(all_dfs)} run files")
    
    merged_df = merge_runs_with_mean(all_dfs)

    output_path = os.path.join(INPUT_FOLDER, OUTPUT_FILE_NAME)
    merged_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved merged file with mean scores to:\n  {output_path}")
    
    print_comparison_stats(merged_df)
    
    old_merged_pattern = os.path.join(INPUT_FOLDER, "*FINAL_MERGED.csv")
    old_files = glob(old_merged_pattern)
    
    if old_files and OUTPUT_FILE_NAME not in old_files[0]:
        print("\n" + "="*60)
        print("COMPARISON WITH VOTING METHOD")
        print("="*60)
        
        old_df = pd.read_csv(old_files[0])
        
        for metric in ['CA', 'Conciseness', 'Fluency']:
            old_col = f'gpt-5-mini_{metric}_final'
            new_col = f'gpt-5-mini_{metric}_final'
            
            if old_col in old_df.columns and new_col in merged_df.columns:
                if 'target_id' in old_df.columns and 'target_id' in merged_df.columns:
                    comparison = pd.merge(
                        old_df[['target_id', old_col]],
                        merged_df[['target_id', new_col]],
                        on='target_id',
                        suffixes=('_voting', '_mean')
                    )
                    
                    diff = comparison[f'{old_col}_mean'] - comparison[f'{old_col}_voting']
                    
                    print(f"\n{metric} Differences (Mean - Voting):")
                    print(f"  Mean difference: {diff.mean():.3f}")
                    print(f"  Std of differences: {diff.std():.3f}")
                    print(f"  Max positive diff: {diff.max():.3f}")
                    print(f"  Max negative diff: {diff.min():.3f}")
                    print(f"  Samples with difference: {(diff.abs() > 0.01).sum()} / {len(diff)}")

if __name__ == "__main__":
    main()
