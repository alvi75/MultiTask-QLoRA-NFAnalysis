import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from datasets import load_dataset, concatenate_datasets, DatasetDict
import argparse
from trl import SFTTrainer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from transformers import TrainingArguments, BitsAndBytesConfig, EarlyStoppingCallback
import json
import sacrebleu
from codebleu import calc_codebleu

def parse_args():
    parser = argparse.ArgumentParser(description="Run multilingual multitask fine-tuning for code tasks")
    parser.add_argument('--base_model_name', type=str, default='Qwen/Qwen2.5-Coder-1.5B-Instruct', help="Base model name")
    parser.add_argument('--device_batch_size', type=int, default=2, help="Per device train batch size")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument('--save_processed_data', type=bool, default=True, help="Save processed datasets")
    parser.add_argument('--sample_size', type=int, default=-1, help="Sample size for training data per task (-1 for full dataset)")
    parser.add_argument('--val_sample_size', type=int, default=-1, help="Sample size for validation data per task (-1 for full dataset)")
    parser.add_argument('--eval_samples', type=int, default=400, help="Number of samples for evaluation during training")
    parser.add_argument('--early_stopping_patience', type=int, default=3, help="Early stopping patience")
    parser.add_argument('--early_stopping_threshold', type=float, default=0.001, help="Early stopping threshold")
    parser.add_argument('--num_train_epochs', type=int, default=10, help="Number of training epochs")
    args = parser.parse_args()
    return args

def apply_chat_template(example, tokenizer, language=None, task="summarization", src_lang=None, tgt_lang=None):
    """Chat template for all tasks"""
    if task == "summarization":
        code = ' '.join(example['code_tokens'])
        if language == 'java':
            chat = [
                {"role": "system", "content": "You are an expert Java developer. Provide clear and concise summarization for Java code segments."},
                {"role": "user", "content": f"Summarize this Java code:\n{code}"},
                {"role": "assistant", "content": f"SUMMARY: {' '.join(example['docstring_tokens'])} DONE"},
            ]
        elif language == 'python':
            chat = [
                {"role": "system", "content": "You are an expert Python developer. Provide clear and concise summarization for Python code segments."},
                {"role": "user", "content": f"Summarize this Python code:\n{code}"},
                {"role": "assistant", "content": f"SUMMARY: {' '.join(example['docstring_tokens'])} DONE"},
            ]
            
    elif task == "generation":
        input_text = example['input']
        output_text = example['output']
        
        summary_part = input_text.split("Summary:")[1].split("Signature:")[0].strip()
        signature_part = input_text.split("Signature:")[1].strip()
        
        if language == 'java':
            signature_part = signature_part.rstrip('{').strip()
        elif language == 'python':
            signature_part = signature_part.rstrip(':').strip()
            
        user_prompt = f"Generate {language.capitalize()} code with the following specification:\nSignature: {signature_part}\nDescription: {summary_part}"
        
        if language == 'java':
            chat = [
                {"role": "system", "content": "You are an expert Java developer. Generate complete and efficient Java code based on the given specifications."},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": output_text},
            ]
        elif language == 'python':
            chat = [
                {"role": "system", "content": "You are an expert Python developer. Generate complete and efficient Python code based on the given specifications."},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": output_text},
            ]
            
    elif task == "translation":
        chat = [
            {"role": "system", "content": "You are an expert code translator. Translate code from the source language to the target language. Preserve semantics, be idiomatic in the target language, and output only the translated code—no explanations."},
            {"role": "user", "content": f"Translate from {src_lang} to {tgt_lang}:\n{example['java'] if src_lang == 'Java' else example['cs']}"},
            {"role": "assistant", "content": example['cs'] if tgt_lang == 'C#' else example['java']},
        ]
    
    # Set default chat template if not available
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{%- for message in messages -%}\n{% if message['role'] == 'system' %}{% if loop.first %}{{ message['content'] + '\n' }}{% endif %}{% elif message['role'] == 'user' %}{{ '\nHuman: ' + message['content'] + '\n\n' }}{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + '\n\n' }}{% endif %}\n{%- endfor -%}\n{% if add_generation_prompt %}Human: {{ '' }}{% endif %}"
    
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)

def process_code_summarization_datasets(tokenizer, sample_size=2000, val_sample_size=100):
    print("Loading code summarization datasets...")
    
    # Load Java dataset
    ds_code_java = load_dataset("google/code_x_glue_ct_code_to_text", "java", cache_dir='datasets')
    
    # Handle full dataset or sample
    if sample_size == -1:
        train_java_sum = ds_code_java["train"].shuffle(seed=42)
    else:
        train_java_sum = ds_code_java["train"].shuffle(seed=42).select(range(min(sample_size, len(ds_code_java["train"]))))
    
    if val_sample_size == -1:
        test_java_sum = ds_code_java["validation"].shuffle(seed=42)
    else:
        test_java_sum = ds_code_java["validation"].shuffle(seed=42).select(range(min(val_sample_size, len(ds_code_java["validation"]))))
    
    # Load Python dataset
    ds_code_python = load_dataset("google/code_x_glue_ct_code_to_text", "python", cache_dir='datasets')
    
    if sample_size == -1:
        train_python_sum = ds_code_python["train"].shuffle(seed=42)
    else:
        train_python_sum = ds_code_python["train"].shuffle(seed=42).select(range(min(sample_size, len(ds_code_python["train"]))))
    
    if val_sample_size == -1:
        test_python_sum = ds_code_python["validation"].shuffle(seed=42)
    else:
        test_python_sum = ds_code_python["validation"].shuffle(seed=42).select(range(min(val_sample_size, len(ds_code_python["validation"]))))
    
    # Process datasets
    def process_summarization(example, lang):
        text = apply_chat_template(example, tokenizer, language=lang, task="summarization")
        return {
            "text": text,
            "task": "summarization",
            "language": lang
        }
    
    # Process Java summarization
    train_java_sum_processed = train_java_sum.map(
        lambda x: process_summarization(x, "java"), 
        remove_columns=train_java_sum.column_names, 
        desc="Processing Java summarization train"
    )
    test_java_sum_processed = test_java_sum.map(
        lambda x: process_summarization(x, "java"), 
        remove_columns=test_java_sum.column_names, 
        desc="Processing Java summarization test"
    )
    
    # Process Python summarization
    train_python_sum_processed = train_python_sum.map(
        lambda x: process_summarization(x, "python"), 
        remove_columns=train_python_sum.column_names, 
        desc="Processing Python summarization train"
    )
    test_python_sum_processed = test_python_sum.map(
        lambda x: process_summarization(x, "python"), 
        remove_columns=test_python_sum.column_names, 
        desc="Processing Python summarization test"
    )
    
    return {
        "train": [train_java_sum_processed, train_python_sum_processed],
        "test": [test_java_sum_processed, test_python_sum_processed]
    }

def process_code_generation_datasets(tokenizer, sample_size=2000, val_sample_size=100):
    print("Loading code generation datasets...")
    
    java_path = '/home/mhaque/QLoRA-Code-Summarization/Multitask/code-generation/codegen_codexglue/java'
    python_path = '/home/mhaque/QLoRA-Code-Summarization/Multitask/code-generation/codegen_codexglue/python'
    
    # Load Java dataset
    ds_java_gen = DatasetDict.load_from_disk(java_path)
    
    if sample_size == -1:
        train_java_gen = ds_java_gen["train"].shuffle(seed=42)
    else:
        train_java_gen = ds_java_gen["train"].shuffle(seed=42).select(range(min(sample_size, len(ds_java_gen["train"]))))
    
    if val_sample_size == -1:
        test_java_gen = ds_java_gen["validation"].shuffle(seed=42)
    else:
        test_java_gen = ds_java_gen["validation"].shuffle(seed=42).select(range(min(val_sample_size, len(ds_java_gen["validation"]))))

    # Load Python dataset
    ds_python_gen = DatasetDict.load_from_disk(python_path)
    
    if sample_size == -1:
        train_python_gen = ds_python_gen["train"].shuffle(seed=42)
    else:
        train_python_gen = ds_python_gen["train"].shuffle(seed=42).select(range(min(sample_size, len(ds_python_gen["train"]))))
    
    if val_sample_size == -1:
        test_python_gen = ds_python_gen["validation"].shuffle(seed=42)
    else:
        test_python_gen = ds_python_gen["validation"].shuffle(seed=42).select(range(min(val_sample_size, len(ds_python_gen["validation"]))))
    
    def process_generation(example, lang):
        text = apply_chat_template(example, tokenizer, language=lang, task="generation")
        return {
            "text": text,
            "task": "generation",
            "language": lang
        }
    
    # Process Java generation
    train_java_gen_processed = train_java_gen.map(
        lambda x: process_generation(x, "java"), 
        remove_columns=train_java_gen.column_names, 
        desc="Processing Java generation train"
    )
    test_java_gen_processed = test_java_gen.map(
        lambda x: process_generation(x, "java"), 
        remove_columns=test_java_gen.column_names, 
        desc="Processing Java generation test"
    )
    
    # Process Python generation
    train_python_gen_processed = train_python_gen.map(
        lambda x: process_generation(x, "python"), 
        remove_columns=train_python_gen.column_names, 
        desc="Processing Python generation train"
    )
    test_python_gen_processed = test_python_gen.map(
        lambda x: process_generation(x, "python"), 
        remove_columns=test_python_gen.column_names, 
        desc="Processing Python generation test"
    )
    
    return {
        "train": [train_java_gen_processed, train_python_gen_processed],
        "test": [test_java_gen_processed, test_python_gen_processed]
    }

def process_code_translation_datasets(tokenizer, sample_size=2000, val_sample_size=100):
    print("Loading code translation datasets...")
    
    # Load translation dataset
    ds_translation = load_dataset("google/code_x_glue_cc_code_to_code_trans", cache_dir='datasets')
    
    if sample_size == -1:
        train_trans = ds_translation["train"].shuffle(seed=42)
    else:
        train_trans = ds_translation["train"].shuffle(seed=42).select(range(min(sample_size, len(ds_translation["train"]))))
    
    if val_sample_size == -1:
        test_trans = ds_translation["validation"].shuffle(seed=42)
    else:
        test_trans = ds_translation["validation"].shuffle(seed=42).select(range(min(val_sample_size, len(ds_translation["validation"]))))
    
    def process_translation(example, src_lang, tgt_lang):
        text = apply_chat_template(example, tokenizer, task="translation", src_lang=src_lang, tgt_lang=tgt_lang)
        return {
            "text": text,
            "task": "translation",
            "src_lang": src_lang,
            "tgt_lang": tgt_lang
        }
    
    # Java to C#
    train_java_cs = train_trans.map(
        lambda x: process_translation(x, "Java", "C#"), 
        remove_columns=train_trans.column_names, 
        desc="Processing Java->C# translation train"
    )
    test_java_cs = test_trans.map(
        lambda x: process_translation(x, "Java", "C#"), 
        remove_columns=test_trans.column_names, 
        desc="Processing Java->C# translation test"
    )
    
    # C# to Java
    train_cs_java = train_trans.map(
        lambda x: process_translation(x, "C#", "Java"), 
        remove_columns=train_trans.column_names, 
        desc="Processing C#->Java translation train"
    )
    test_cs_java = test_trans.map(
        lambda x: process_translation(x, "C#", "Java"), 
        remove_columns=test_trans.column_names, 
        desc="Processing C#->Java translation test"
    )
    
    return {
        "train": [train_java_cs, train_cs_java],
        "test": [test_java_cs, test_cs_java]
    }

def setup_multitask_metrics(tokenizer):
    """Setup metrics computation for multitask learning"""
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        
        # Convert logits to token ids if needed
        if predictions.ndim == 3:
            predictions = np.argmax(predictions, axis=-1)
        
        # Replace -100 with eos token for decoding
        predictions = np.where(predictions != -100, predictions, tokenizer.eos_token_id)
        labels = np.where(labels != -100, labels, tokenizer.eos_token_id)
        
        # Decode
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Store scores
        sacrebleu_scores = []  # For CS
        codebleu_scores = []   # For CG and CT
        
        for pred, label in zip(decoded_preds, decoded_labels):
            # Extract content after "Assistant:"
            pred_content = pred.split("Assistant:")[-1].strip() if "Assistant:" in pred else pred
            label_content = label.split("Assistant:")[-1].strip() if "Assistant:" in label else label
            
            # For summarization, extract between SUMMARY: and DONE
            if "SUMMARY:" in label_content:
                label_content = label_content.split("SUMMARY:")[-1].split("DONE")[0].strip()
            if "SUMMARY:" in pred_content:
                pred_content = pred_content.split("SUMMARY:")[-1].split("DONE")[0].strip()
            
            # Determine task type from content
            if "Summarize this" in label or "SUMMARY:" in label:
                # Code Summarization - use SacreBLEU
                try:
                    score = sacrebleu.sentence_bleu(pred_content, [label_content]).score
                    sacrebleu_scores.append(score)
                except:
                    sacrebleu_scores.append(0.0)
                    
            else:
                # Code Generation or Translation - use CodeBLEU
                try:
                    # Simple language detection
                    if "public" in label_content or "private" in label_content:
                        lang = "java"
                    elif "def " in label_content or "import " in label_content:
                        lang = "python"
                    elif "namespace" in label_content or "using System" in label_content:
                        lang = "c_sharp"
                    else:
                        lang = "java"  # default
                    
                    result = calc_codebleu(
                        references=[label_content],
                        predictions=[pred_content],
                        lang=lang
                    )
                    codebleu_scores.append(result['codebleu'] * 100)  # Convert to percentage
                except:
                    codebleu_scores.append(0.0)
        
        # Calculate metrics
        metrics = {}
        
        if sacrebleu_scores:
            metrics["eval_sacrebleu"] = np.mean(sacrebleu_scores)
        
        if codebleu_scores:
            metrics["eval_codebleu"] = np.mean(codebleu_scores)
        
        # Combined score for early stopping
        all_scores = sacrebleu_scores + codebleu_scores
        metrics["eval_combined_score"] = np.mean(all_scores) if all_scores else 0.0
        
        # Generation length
        gen_lengths = [len(tokenizer.tokenize(p)) for p in decoded_preds]
        metrics["eval_gen_len"] = np.mean(gen_lengths) if gen_lengths else 0.0
        
        return metrics
    
    return compute_metrics

def main():
    args = parse_args()
    
    print("="*80)
    print("Running Multilingual Multitask Training with QLoRA")
    print("Tasks: Code Summarization (SacreBLEU), Generation (CodeBLEU), Translation (CodeBLEU)")
    print("="*80)

    # BitsAndBytes config for 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    hf_token = os.environ.get("HF_TOKEN")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_name, 
        token=hf_token, 
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Setup metrics
    compute_metrics = setup_multitask_metrics(tokenizer)
    
    # Setup callbacks
    callbacks = [EarlyStoppingCallback(
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold
    )]
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
        trust_remote_code=True
    )
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # https://qwen.readthedocs.io/en/v1.5/training/SFT/example.html
        inference_mode=False, 
        r=8, 
        lora_alpha=16, 
        lora_dropout=0.1
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Process all datasets
    summarization_data = process_code_summarization_datasets(tokenizer, args.sample_size, args.val_sample_size)
    generation_data = process_code_generation_datasets(tokenizer, args.sample_size, args.val_sample_size)
    translation_data = process_code_translation_datasets(tokenizer, args.sample_size, args.val_sample_size)
    
    # Combine all training datasets
    all_train_datasets = []
    all_test_datasets = []
    
    all_train_datasets.extend(summarization_data["train"])
    all_test_datasets.extend(summarization_data["test"])
    
    all_train_datasets.extend(generation_data["train"])
    all_test_datasets.extend(generation_data["test"])
    
    all_train_datasets.extend(translation_data["train"])
    all_test_datasets.extend(translation_data["test"])
    
    # Concatenate and shuffle
    print("\nMerging all multitask datasets...")
    multitask_train_dataset = concatenate_datasets(all_train_datasets).shuffle(seed=42)
    multitask_test_dataset = concatenate_datasets(all_test_datasets).shuffle(seed=42)
    
    print(f"Total multitask train size: {len(multitask_train_dataset)}")
    print(f"Total multitask test size: {len(multitask_test_dataset)}")
    
    # Print sample examples
    print("\n" + "="*80)
    print("SAMPLE EXAMPLES FROM MULTITASK DATASET:")
    print("="*80)
    
    for i in range(min(3, len(multitask_train_dataset))):
        sample = multitask_train_dataset[i]
        task = sample.get('task', 'unknown')
        
        if task == "translation":
            lang_info = f"{sample.get('src_lang', 'unknown')} -> {sample.get('tgt_lang', 'unknown')}"
        else:
            lang_info = sample.get('language', 'unknown')
        
        print(f"\nSample {i+1} - Task: {task.upper()}, Language: {lang_info}")
        print("-" * 50)
        print(sample['text'][:300] + "..." if len(sample['text']) > 300 else sample['text'])
    
    print("="*80 + "\n")
    
    # Save processed datasets
    if args.save_processed_data:
        print("Saving processed multitask datasets...")
        save_dir = "./processed_datasets_multitask"
        os.makedirs(save_dir, exist_ok=True)
        
        multitask_train_dataset.save_to_disk(f"{save_dir}/train_multitask")
        multitask_test_dataset.save_to_disk(f"{save_dir}/test_multitask")
        
        # Calculate and save statistics
        task_distribution = {"summarization": 0, "generation": 0, "translation": 0}
        for sample in multitask_train_dataset:
            task = sample.get('task', 'unknown')
            task_distribution[task] = task_distribution.get(task, 0) + 1
        
        stats = {
            "total_train_size": len(multitask_train_dataset),
            "total_test_size": len(multitask_test_dataset),
            "task_distribution": task_distribution,
            "sample_size_arg": args.sample_size,
            "val_sample_size_arg": args.val_sample_size
        }
        
        with open(f"{save_dir}/dataset_statistics.json", "w") as f:
            json.dump(stats, f, indent=2)
        
        print(f"Datasets saved to {save_dir}")
        print(f"Task distribution: {task_distribution}")
    
    # Analyze sequence lengths
    sample_size_for_length = min(1000, len(multitask_test_dataset))
    all_lengths = [len(tokenizer.tokenize(multitask_test_dataset[i]['text'])) for i in range(sample_size_for_length)]
    percentiles = np.percentile(all_lengths, [25, 50, 75, 95])
    print('\n========== Sequence Length Percentiles ==========')
    print(f"25th: {percentiles[0]:.0f}, 50th: {percentiles[1]:.0f}, 75th: {percentiles[2]:.0f}, 95th: {percentiles[3]:.0f}")
    
    # Training configuration
    output_dir = f"/scratch/mhaque/results/{args.base_model_name.split('/')[-1]}_multitask_qlora"
    
    training_args = TrainingArguments(
        per_device_train_batch_size=args.device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=100,
        report_to=[],
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        num_train_epochs=args.num_train_epochs,
        logging_steps=50,
        optim="adamw_bnb_8bit",
        bf16=True,
        output_dir=output_dir,
        logging_strategy="steps",
        dataloader_num_workers=4,
        save_total_limit=3,
        do_eval=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=5_000,
        eval_steps=5_000,
        metric_for_best_model="eval_combined_score",
        greater_is_better=True,
        load_best_model_at_end=True,
        gradient_checkpointing=False,
        remove_unused_columns=False,
    )
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    # GPU information
    print(f"\nGPU Information:")
    print(f"Visible GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    
    # Initialize trainer
    trainer = SFTTrainer(
        model,
        packing=True,
        max_seq_length=300,
        args=training_args,
        train_dataset=multitask_train_dataset,
        eval_dataset=multitask_test_dataset.select(range(min(args.eval_samples, len(multitask_test_dataset)))),
        peft_config=peft_config,
        dataset_text_field='text',
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )
    
    print("\n" + "="*80)
    print("Starting Multitask Training")
    print(f"Output directory: {output_dir}")
    print(f"Model: {args.base_model_name}")
    print(f"Epochs: {args.num_train_epochs}")
    print(f"Early stopping patience: {args.early_stopping_patience}")
    print(f"Batch size: {args.device_batch_size} per device")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps} steps")
    print(f"Effective batch size: {args.device_batch_size * args.gradient_accumulation_steps * torch.cuda.device_count()}")
    print("Metrics: SacreBLEU (CS), CodeBLEU aggregated (CG/CT)")
    print("="*80 + "\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Train the model 
    print("Starting training...")
    trainer.train()
    
    # Save the model 
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print("\n" + "="*80)
    print("Multitask Training Completed!")
    print(f"Model saved to: {output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()