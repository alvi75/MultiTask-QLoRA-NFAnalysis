import os
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    return parser.parse_args()

def wrap_java_code(code, class_name):
    """Wrap Java method in a class."""
    wrapped = f"""import java.util.*;

public class {class_name} {{
    {code}
}}"""
    return wrapped

def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    java_files = [f for f in os.listdir(args.input_dir) if f.endswith('.java')]
    
    print(f"Processing {len(java_files)} Java files...")
    
    for filename in tqdm(java_files):
        with open(os.path.join(args.input_dir, filename), 'r') as f:
            code = f.read()
        
        class_name = 'Generated_' + filename.replace('.java', '')
        
        wrapped = wrap_java_code(code, class_name)
        
        with open(os.path.join(args.output_dir, filename), 'w') as f:
            f.write(wrapped)
    
    print(f"✓ Saved {len(java_files)} wrapped files to: {args.output_dir}")

if __name__ == "__main__":
    main()

# python add_java_wrappers_cg.py --input_dir code-generation/generation/qlora/java/multitask/qwen0_5_java_files --output_dir code-generation/generation/qlora/java/multitask/qwen0_5_java_files_wrapped
