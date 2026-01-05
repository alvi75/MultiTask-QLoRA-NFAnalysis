import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
import argparse
from pathlib import Path
from peft import PeftModel

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with QLoRA fine-tuned model")
    parser.add_argument('--model_path', type=str, required=True, 
                       help="Path to QLoRA adapter (e.g., results/Qwen2.5-Coder-0.5B-Instruct_summarization_qlora_qwen0_5)")
    parser.add_argument('--input_jsonl', type=str, required=True,
                       help="Path to input JSONL file with code to summarize")
    parser.add_argument('--output_file', type=str, default=None,
                       help="Output JSONL file path (default: auto-generated based on model name)")
    parser.add_argument('--language', type=str, default='java', choices=['java', 'python'],
                       help="Programming language of the code")
    parser.add_argument('--batch_size', type=int, default=4,
                       help="Batch size for inference")
    parser.add_argument('--max_new_tokens', type=int, default=128,
                       help="Maximum tokens to generate")
    return parser.parse_args()

def load_jsonl_data(file_path):
    """Load data from JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def prepare_prompt(code, language, tokenizer):
    """Prepare the prompt for the model using the same template as training"""
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

def generate_summaries_batch(model, tokenizer, codes, language, max_new_tokens=128):
    prompts = [prepare_prompt(code, language, tokenizer) for code in codes]
    
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_length = inputs['input_ids'].shape[1]
    
    device = next(model.parameters()).device
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
    print("QLoRA Code Summarization Inference")
    print(f"Model Path: {args.model_path}")
    print(f"Input: {args.input_jsonl}")
    print(f"Language: {args.language}")
    print("="*80)
    
    model_path_lower = args.model_path.lower()
    if "0.5b" in model_path_lower:
        base_model = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
        summary_column = "summary_qwen0_5b_qlora"
    elif "1.5b" in model_path_lower:
        base_model = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
        summary_column = "summary_qwen1_5b_qlora"
    elif "3b" in model_path_lower:
        base_model = "Qwen/Qwen2.5-Coder-3B-Instruct"
        summary_column = "summary_qwen3b_qlora"
    elif "7b" in model_path_lower:
        base_model = "Qwen/Qwen2.5-Coder-7B-Instruct"
        summary_column = "summary_qwen7b_qlora"
    else:
        raise ValueError(f"Cannot auto-detect base model from path: {args.model_path}. Please ensure path contains model size (0.5B, 1.5B, 3B, or 7B)")
    
    print(f"Auto-detected base model: {base_model}")
    print(f"Summary column name: {summary_column}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    print("\nLoading base model with 4-bit quantization...")
    base_model_loaded = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            padding_side="left"
        )
    except:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True,
            padding_side="left"
        )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model_loaded, args.model_path)
    
    model.eval()
    print("Model loaded successfully!")

    print(f"\nLoading data from {args.input_jsonl}...")
    data = load_jsonl_data(args.input_jsonl)
    print(f"Found {len(data)} entries to process")
    
    all_results = []
    batch_size = args.batch_size
    
    for i in tqdm(range(0, len(data), batch_size), desc="Generating summaries"):
        batch = data[i:i + batch_size]
        codes = [entry['code'] for entry in batch]
        
        summaries = generate_summaries_batch(
            model, tokenizer, codes, args.language, args.max_new_tokens
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
        adapter_name = Path(args.model_path).name
        output_path = f"inference_results_{input_name}_{adapter_name}.jsonl"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    print(f"\n✓ Results saved to: {output_path}")
    
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
#   python infer_summarization_qlora.py \
#       --model_path path/to/qlora_adapter \
#       --input_jsonl path/to/input.jsonl \
#       --output_file path/to/output \
#       --language java \
#       --batch_size 8 \
#       --max_new_tokens 128
