import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse
from pathlib import Path
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on JSONL dataset with fine-tuned model")
    parser.add_argument('--model_path', type=str, required=True, 
                       help="Path to fine-tuned model (e.g., /scratch/mhaque/results/Qwen2.5-Coder-0.5B-Instruct_summarization_fft_qwen3)")
    parser.add_argument('--input_jsonl', type=str, required=True,
                       help="Path to input JSONL file with code to summarize")
    parser.add_argument('--output_file', type=str, default=None,
                       help="Output JSONL file path (default: auto-generated based on model name)")
    parser.add_argument('--language', type=str, default='java', choices=['java', 'python'],
                       help="Programming language of the code")
    parser.add_argument('--batch_size', type=int, default=8,
                       help="Batch size for inference")
    parser.add_argument('--max_new_tokens', type=int, default=128,
                       help="Maximum tokens to generate")
    parser.add_argument('--device', type=str, default='cuda',
                       help="Device to use (cuda/cpu)")
    return parser.parse_args()

def load_jsonl_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def prepare_prompt(code, language, tokenizer):
    if language == 'java':
        system_prompt = "You are an expert Java developer. Provide clear and concise summarization for Java code segments."
    else:
        system_prompt = "You are an expert Python developer. Provide clear and concise summarization for Python code segments."
    
    chat = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Summarize this {language.capitalize()} code:\n{code}"},
    ]
    
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    return prompt

def extract_summary(generated_text):
    if "Assistant: " in generated_text:
        summary = generated_text.split("Assistant: ")[-1].strip()
    else:
        summary = generated_text.strip()

    if summary.startswith("SUMMARY:"):
        summary = summary.replace("SUMMARY:", "", 1).strip()
    
    if "DONE" in summary:
        summary = summary.split("DONE")[0].strip()
    
    return summary

def generate_summaries_batch(model, tokenizer, codes, language, device, max_new_tokens=128):
    prompts = [prepare_prompt(code, language, tokenizer) for code in codes]
    
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_length = inputs['input_ids'].shape[1]
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated_ids = outputs[:, input_length:]
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    summaries = []
    for text in generated_texts:
        summary = text.strip()
        
        if "SUMMARY:" in summary:
            summary = summary.split("SUMMARY:")[-1].strip()

        if "DONE" in summary:
            summary = summary.split("DONE")[0].strip()
        
        summaries.append(summary)
    
    return summaries

def main():
    args = parse_args()
    
    print("="*80)
    print("Code Summarization Inference")
    print(f"Model: {args.model_path}")
    print(f"Input: {args.input_jsonl}")
    print(f"Language: {args.language}")
    print("="*80)
    
    model_name = Path(args.model_path).name
    if "Qwen2.5-Coder-0.5B" in model_name or "0.5B" in args.model_path:
        summary_column = "summary_qwen0_5b"
    elif "Qwen2.5-Coder-1.5B" in model_name or "1.5B" in args.model_path:
        summary_column = "summary_qwen1_5b"
    elif "Qwen2.5-Coder-3B" in model_name or "3B" in args.model_path:
        summary_column = "summary_qwen3b"
    else:
        summary_column = "summary_" + model_name.lower().replace("-", "_").replace(".", "_")
    
    print(f"Summary column name: {summary_column}")
    
    print("\nLoading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto" if args.device == "cuda" else None,
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        padding_side="left"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if args.device == "cuda" and not hasattr(model, 'hf_device_map'):
        model = model.to(args.device)
    
    model.eval()
    
    print(f"\nLoading data from {args.input_jsonl}...")
    data = load_jsonl_data(args.input_jsonl)
    print(f"Found {len(data)} entries to process")
    
    all_results = []
    batch_size = args.batch_size
    
    for i in tqdm(range(0, len(data), batch_size), desc="Generating summaries"):
        batch = data[i:i + batch_size]
        codes = [entry['code'] for entry in batch]
        
        summaries = generate_summaries_batch(
            model, tokenizer, codes, args.language, 
            args.device, args.max_new_tokens
        )
        
        for entry, summary in zip(batch, summaries):
            result = {
                "id": entry.get('id', ''),
                "code": entry['code'],
                "reference_summary": entry.get('reference_summary', ''),
                summary_column: summary
            }
            all_results.append(result)
    
    if args.output_file:
        output_path = args.output_file
    else:
        input_name = Path(args.input_jsonl).stem
        output_path = f"inference_results_{input_name}_{model_name}.csv"
    
    df = pd.DataFrame(all_results)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to: {output_path}")
    
    jsonl_path = output_path.replace('.csv', '.jsonl')
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    print(f"✓ JSONL results saved to: {jsonl_path}")
    
    print("\n" + "="*80)
    print("Sample Results (first 3):")
    print("="*80)
    for i, result in enumerate(all_results[:3], 1):
        print(f"\n--- Sample {i} ---")
        print(f"ID: {result['id']}")
        print(f"Reference: {result['reference_summary'][:100]}...")
        print(f"Generated: {result[summary_column][:100]}...")
    
    print("\n" + "="*80)
    print("Inference completed successfully!")
    print("="*80)

if __name__ == "__main__":
    main()

# Usage:
#   python infer_summarization_fft.py \
#       --model_path path/to/model \
#       --input_jsonl path/to/input.jsonl \
#       --output_file path/to/output.csv \
#       --language java \
#       --batch_size 8 \
#       --max_new_tokens 128
