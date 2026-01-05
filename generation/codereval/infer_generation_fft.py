import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Generate code for CoderEval benchmark")
    parser.add_argument('--model_path', type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument('--input_file', type=str, required=True, help="CEPythonHumanLabel.jsonl or CEJavaHumanLabel.jsonl")
    parser.add_argument('--output_file', type=str, required=True, help="Output JSONL file for evaluation")
    parser.add_argument('--language', type=str, required=True, choices=['java', 'python'], help="Programming language")
    parser.add_argument('--num_samples', type=int, default=1, help="Number of generations per task")
    parser.add_argument('--temperature', type=float, default=0.0, help="Sampling temperature")
    parser.add_argument('--max_new_tokens', type=int, default=512, help="Maximum tokens to generate")
    parser.add_argument('--debug', action='store_true', help="Print debug information")
    args = parser.parse_args()
    return args

def apply_chat_template(codereval_input, language, tokenizer):
    """Apply chat template using CoderEval input directly"""
    
    if language == 'java':
        system_content = "You are an expert Java developer. Generate complete and efficient Java code based on the given specifications."
    else:
        system_content = "You are an expert Python developer. Generate complete and efficient Python code based on the given specifications."
    
    chat = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": codereval_input},
    ]
    
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{%- for message in messages -%}\n{% if message['role'] == 'system' %}{% if loop.first %}{{ message['content'] + '\n' }}{% endif %}{% elif message['role'] == 'user' %}{{ '\nHuman: ' + message['content'] + '\n\n' }}{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + '\n\n' }}{% endif %}\n{%- endfor -%}\n{% if add_generation_prompt %}Assistant: {% endif %}"
    
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

def generate_code(model, tokenizer, prompt, num_samples=1, temperature=0.0, max_new_tokens=512, debug=False):
    """Generate code samples for a single task"""
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
    input_length = inputs['input_ids'].shape[1]
    
    generations = []
    for i in range(num_samples):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=(temperature > 0.0),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        generated_tokens = outputs[0][input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        if debug and i == 0:
            print("\n" + "="*50)
            print("DEBUG - Prompt (last 200 chars):")
            print(prompt[-200:])
            print("\nDEBUG - Generated text (first 300 chars):")
            print(generated_text[:300])
            print("="*50 + "\n")
        
        code = generated_text.strip()
        
        if "Human:" in code:
            code = code.split("Human:")[0].strip()

        if code.startswith("Assistant:"):
            code = code[len("Assistant:"):].strip()
        
        generations.append(code)
    
    return generations

def main():
    args = parse_args()
    
    print(f"Loading model from: {args.model_path}")
    print(f"Input file: {args.input_file}")
    print(f"Language: {args.language}")
    print(f"Temperature: {args.temperature}")
    print(f"Generating {args.num_samples} samples per task")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    tasks = []
    with open(args.input_file, 'r') as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    
    print(f"Loaded {len(tasks)} tasks")
    
    results = []
    for idx, task in enumerate(tqdm(tasks, desc="Generating code")):
        codereval_input = task['input']
        
        # Create prompt
        prompt = apply_chat_template(codereval_input, args.language, tokenizer)
        
        generations = generate_code(
            model, tokenizer, prompt,
            num_samples=args.num_samples,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            debug=(args.debug and idx == 0)
        )
        
        result = {
            "_id": task["question_id"],
            "generate_results": generations
        }
        results.append(result)
        
        if len(results) % 10 == 0:
            torch.cuda.empty_cache()
    
    # Save results
    with open(args.output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"\nResults saved to: {args.output_file}")
    print(f"Ready for evaluation with PythonExec.py or JavaExec.py")
    
    # Print sample output
    print("\nSample outputs (first 3 tasks):")
    for i in range(min(3, len(results))):
        print(f"\nTask {i+1} ID: {results[i]['_id']}")
        code = results[i]['generate_results'][0]
        print(f"Generated code (first 200 chars):\n{code[:200]}...")

if __name__ == "__main__":
    main()

# Usage:
#   python infer_generation_fft.py \
#       --model_path path/to/model \
#       --input_file path/to/input.jsonl \
#       --output_file path/to/output.jsonl \
#       --language java \
#       --num_samples 1 \
#       --temperature 0.0
