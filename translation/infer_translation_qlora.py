import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from datasets import load_dataset
import json
import os
from tqdm import tqdm
from codebleu import calc_codebleu
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run code translation and evaluation for QLoRA models")
    parser.add_argument('--model_path', type=str, required=True, help="Path to fine-tuned QLoRA model")
    parser.add_argument('--direction', type=str, choices=['cs_to_java', 'java_to_cs'], required=True, help="Translation direction")
    parser.add_argument('--output_dir', type=str, default="./translation_results", help="Output directory")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for inference")
    parser.add_argument('--max_samples', type=int, default=None, help="Limit number of samples to process")
    args = parser.parse_args()
    return args

def format_code(code, language):
    """Format code with proper indentation."""
    # Remove existing formatting
    code = ' '.join(code.split())
    
    # Add line breaks
    code = code.replace(';', ';\n')
    code = code.replace('{', '{\n')
    code = code.replace('}', '}\n')
    
    # Clean up multiple newlines
    lines = [line.strip() for line in code.split('\n') if line.strip()]
    
    # Add proper indentation
    formatted_lines = []
    indent_level = 0
    
    for line in lines:
        if line.startswith('}'):
            indent_level = max(0, indent_level - 1)
        formatted_lines.append('    ' * indent_level + line)
        if line.endswith('{'):
            indent_level += 1
    
    return '\n'.join(formatted_lines)

def clean_generated_code(code):
    if "Assistant:" in code:
        code = code.split("Assistant:")[-1].strip()
    
    if '```java' in code:
        code = code.split('```java')[1].split('```')[0].strip()
    elif '```csharp' in code or '```cs' in code:
        code = code.split('```')[1].split('```')[0].strip()
    elif '```' in code:
        code = code.split('```')[1].split('```')[0].strip()
    
    return code.strip()

def main():
    args = parse_args()
    
    print("="*80)
    print("Code Translation and Evaluation")
    print(f"Model: {args.model_path}")
    print(f"Direction: {args.direction}")
    print("="*80)
    
    print("\nLoading model and tokenizer...")
    
    peft_config = PeftConfig.from_pretrained(args.model_path)
    base_model_name = peft_config.base_model_name_or_path
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN")
    )
    
    model = PeftModel.from_pretrained(model, args.model_path)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        padding_side="left",
        trust_remote_code=True
    )
    
    if tokenizer.vocab_size == 0:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            padding_side="left",
            trust_remote_code=True,
            token=os.environ.get("HF_TOKEN")
        )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set chat template
    tokenizer.chat_template = "{%- for message in messages -%}\n{% if message['role'] == 'system' %}{% if loop.first %}{{ message['content'] + '\n' }}{% endif %}{% elif message['role'] == 'user' %}{{ '\nHuman: ' + message['content'] + '\n\n' }}{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + '\n\n' }}{% endif %}\n{%- endfor -%}\n{% if add_generation_prompt %}Assistant: {% endif %}"
    
    # Load dataset
    print("Loading test dataset...")
    ds = load_dataset("google/code_x_glue_cc_code_to_code_trans")
    test_data = ds["test"]
    
    if args.max_samples:
        test_data = test_data.select(range(min(args.max_samples, len(test_data))))
    
    print(f"Processing {len(test_data)} samples...")
    
    # Setup directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine languages
    if args.direction == "cs_to_java":
        source_lang = "cs"
        target_lang = "java"
        source_lang_name = "C#"
        target_lang_name = "Java"
        file_ext = ".java"
        codebleu_lang = "java"
    else:  # java_to_cs
        source_lang = "java"
        target_lang = "cs"
        source_lang_name = "Java"
        target_lang_name = "C#"
        file_ext = ".cs"
        codebleu_lang = "c_sharp"
    
    code_dir = os.path.join(args.output_dir, f"{target_lang}_files")
    os.makedirs(code_dir, exist_ok=True)
    
    all_predictions = []
    all_references = []
    
    print(f"\nTranslating {source_lang_name} to {target_lang_name}...")
    
    for i in tqdm(range(0, len(test_data), args.batch_size)):
        batch_end = min(i + args.batch_size, len(test_data))
        batch_items = [test_data[j] for j in range(i, batch_end)]
      
        prompts = []
        for item in batch_items:
            source_code = item[source_lang]
            
            system = "You are an expert code translator. Translate code from the source language to the target language. Preserve semantics, be idiomatic in the target language, and output only the translated code—no explanations."
            
            if args.direction == "cs_to_java":
                user = f"Translate from C# to Java:\n{source_code}"
            else:  # java_to_cs
                user = f"Translate from Java to C#:\n{source_code}"
            
            chat = [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ]
            
            prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)
            
            all_references.append(item[target_lang])
        
        inputs = tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=False, 
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode outputs
        for j, output in enumerate(outputs):
            decoded = tokenizer.decode(output, skip_special_tokens=True)
            
            if "Assistant:" in decoded:
                generated = decoded.split("Assistant:")[-1].strip()
            else:
                generated = decoded[len(prompts[j]):].strip()
            
            cleaned_code = clean_generated_code(generated)
            all_predictions.append(cleaned_code)
            
            # Save to file
            file_name = f"translation_{i+j}{file_ext}"
            file_path = os.path.join(code_dir, file_name)
            
            try:
                formatted_code = format_code(cleaned_code, target_lang)
            except:
                formatted_code = cleaned_code  # Use unformatted if formatting fails
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(formatted_code)
    
    print(f"\n Saved {len(all_predictions)} {target_lang_name} files to: {code_dir}")
    
    print("\n" + "="*80)
    print("Calculating CodeBLEU scores...")
    print("="*80)
    
    result = calc_codebleu(
        references=all_references,
        predictions=all_predictions,
        lang=codebleu_lang
    )
    
    codebleu_score = result['codebleu'] * 100
    ngram_score = result['ngram_match_score'] * 100
    weighted_ngram_score = result['weighted_ngram_match_score'] * 100
    syntax_score = result['syntax_match_score'] * 100
    dataflow_score = result['dataflow_match_score'] * 100
    
    print(f"\nTranslation: {source_lang_name} → {target_lang_name}")
    print(f"Samples: {len(all_predictions)}")
    print("-"*50)
    print(f"CodeBLEU:        {codebleu_score:.2f}%")
    print(f"  N-gram:        {ngram_score:.2f}%")
    print(f"  Weighted:      {weighted_ngram_score:.2f}%")
    print(f"  Syntax:        {syntax_score:.2f}%")
    print(f"  Dataflow:      {dataflow_score:.2f}%")
    print("="*80)
    
    results = {
        "model": os.path.basename(args.model_path),
        "direction": args.direction,
        "source_language": source_lang_name,
        "target_language": target_lang_name,
        "num_samples": len(all_predictions),
        "scores": {
            "codebleu": codebleu_score,
            "ngram_match": ngram_score,
            "weighted_ngram": weighted_ngram_score,
            "syntax_match": syntax_score,
            "dataflow_match": dataflow_score
        }
    }
    
    results_file = os.path.join(args.output_dir, f"results_{args.direction}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    
    print("\n" + "="*80)
    print("Sample Translation (First Example)")
    print("="*80)
    print(f"\nOriginal {source_lang_name}:")
    print("-"*40)
    print(test_data[0][source_lang][:300] + "..." if len(test_data[0][source_lang]) > 300 else test_data[0][source_lang])
    print(f"\nGenerated {target_lang_name}:")
    print("-"*40)
    print(all_predictions[0][:300] + "..." if len(all_predictions[0]) > 300 else all_predictions[0])
    print(f"\nReference {target_lang_name}:")
    print("-"*40)
    print(all_references[0][:300] + "..." if len(all_references[0]) > 300 else all_references[0])
    print("="*80)

if __name__ == "__main__":
    main()

# Example usage for C# to Java:
# python translate_and_eval_qlora.py --model_path /path/to/model --direction cs_to_java --output_dir ./results --batch_size 8

# Example usage for Java to C#:
# python translate_and_eval_qlora.py --model_path /path/to/model --direction java_to_cs --output_dir ./results --batch_size 8
