import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from datasets import concatenate_datasets, DatasetDict
import argparse
from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback
import json
from codebleu import calc_codebleu

def parse_args():
    parser = argparse.ArgumentParser(description="Run multilingual code generation full fine-tuning (Java + Python)")
    parser.add_argument('--base_model_name', type=str, default='Qwen/Qwen2.5-Coder-1.5B-Instruct', help="Base model name")
    parser.add_argument('--device_batch_size', type=int, default=2, help="Per device train batch size")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument('--sample_size', type=int, default=-1, help="Sample size for training data per language (-1 for full dataset)")
    parser.add_argument('--val_sample_size', type=int, default=-1, help="Sample size for validation data per language (-1 for full dataset)")
    parser.add_argument('--eval_samples', type=int, default=500, help="Number of samples for evaluation during training")
    parser.add_argument('--save_processed_data', type=bool, default=True, help="Save processed datasets")
    parser.add_argument('--num_train_epochs', type=int, default=5, help="Number of training epochs")
    args = parser.parse_args()
    return args

def apply_chat_template_java(example, tokenizer):
    input_text = example['input']
    output_text = example['output']
    
    summary_part = input_text.split("Summary:")[1].split("Signature:")[0].strip()
    signature_part = input_text.split("Signature:")[1].strip()
    
        
    user_prompt = f"{summary_part}\n\nFunction to implement:\n{signature_part}"

    chat = [
        {"role": "system", "content": "You are an expert Java developer. Generate complete and efficient Java code based on the given specifications."},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": output_text},
    ]
    
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{%- for message in messages -%}\n{% if message['role'] == 'system' %}{% if loop.first %}{{ message['content'] + '\n' }}{% endif %}{% elif message['role'] == 'user' %}{{ '\nHuman: ' + message['content'] + '\n\n' }}{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + '\n\n' }}{% endif %}\n{%- endfor -%}\n{% if add_generation_prompt %}Human: {{ '' }}{% endif %}"
    
    example["text"] = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
    example["language"] = "java"
    return example

def apply_chat_template_python(example, tokenizer):
    input_text = example['input']
    output_text = example['output']
    
    summary_part = input_text.split("Summary:")[1].split("Signature:")[0].strip()
    signature_part = input_text.split("Signature:")[1].strip()
    
        
    user_prompt = f"{summary_part}\n\nFunction to implement:\n{signature_part}"

    chat = [
        {"role": "system", "content": "You are an expert Python developer. Generate complete and efficient Python code based on the given specifications."},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": output_text},
    ]
    
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{%- for message in messages -%}\n{% if message['role'] == 'system' %}{% if loop.first %}{{ message['content'] + '\n' }}{% endif %}{% elif message['role'] == 'user' %}{{ '\nHuman: ' + message['content'] + '\n\n' }}{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + '\n\n' }}{% endif %}\n{%- endfor -%}\n{% if add_generation_prompt %}Human: {{ '' }}{% endif %}"
    
    example["text"] = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
    example["language"] = "python"
    return example

def compute_metrics_codebleu(eval_pred, tokenizer):
    predictions, labels = eval_pred

    if isinstance(predictions, list):
        predictions = np.concatenate(predictions, axis=0)
    if isinstance(labels, list):
        labels = np.concatenate(labels, axis=0)

    if predictions.ndim == 3:
        predictions = np.argmax(predictions, axis=-1)
    
    predictions = np.where(predictions != -100, predictions, tokenizer.eos_token_id)
    labels = np.where(labels != -100, labels, tokenizer.eos_token_id)
    
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decoded_preds = [pred.split("Assistant:")[-1].strip() if "Assistant:" in pred else pred.strip() 
                    for pred in decoded_preds]
    decoded_labels = [label.split("Assistant:")[-1].strip() if "Assistant:" in label else label.strip() 
                     for label in decoded_labels]
    
    codebleu_scores = []
    for pred, label in zip(decoded_preds, decoded_labels):
        try:
            if "public" in label or "private" in label or "class" in label[:50]:
                lang = "java"
            elif "def " in label or "import " in label or "__" in label:
                lang = "python"
            else:
                lang = "java"
            
            result = calc_codebleu(references=[label], predictions=[pred], lang=lang)
            codebleu_scores.append(result['codebleu'] * 100)
        except:
            codebleu_scores.append(0.0)
    
    gen_lengths = [len(tokenizer.tokenize(p)) for p in decoded_preds]
    
    return {
        "eval_codebleu": np.mean(codebleu_scores),
        "eval_gen_len": np.mean(gen_lengths)
    }

def main():
    args = parse_args()
    
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    print("="*80)
    print("Multilingual Code Generation Training")
    print(f"Languages: Java + Python")
    print(f"Model: {args.base_model_name}")
    print("="*80)
    
    hf_token = os.environ.get("HF_TOKEN")
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_name, 
        token=hf_token, 
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=hf_token,
        trust_remote_code=True
    )
    model.config.use_cache = False
    
    print("Loading Java code generation dataset...")
    ds_java = DatasetDict.load_from_disk('dataset/codegen_codexglue/java')
    
    if args.sample_size == -1:
        train_java = ds_java["train"].shuffle(seed=42)
    else:
        train_java = ds_java["train"].shuffle(seed=42).select(range(min(args.sample_size, len(ds_java["train"]))))
    
    if args.val_sample_size == -1:
        test_java = ds_java["validation"].shuffle(seed=42)
    else:
        test_java = ds_java["validation"].shuffle(seed=42).select(range(min(args.val_sample_size, len(ds_java["validation"]))))
    
    column_names = list(train_java.features)
    train_java_processed = train_java.map(
        lambda x: apply_chat_template_java(x, tokenizer),
        remove_columns=column_names,
        desc="Processing Java train"
    )
    test_java_processed = test_java.map(
        lambda x: apply_chat_template_java(x, tokenizer),
        remove_columns=column_names,
        desc="Processing Java test"
    )
    print(f"Java - Train: {len(train_java_processed)}, Test: {len(test_java_processed)}")
    
    print("Loading Python code generation dataset...")
    ds_python = DatasetDict.load_from_disk('dataset/codegen_codexglue/python')
    
    if args.sample_size == -1:
        train_python = ds_python["train"].shuffle(seed=42)
    else:
        train_python = ds_python["train"].shuffle(seed=42).select(range(min(args.sample_size, len(ds_python["train"]))))
    
    if args.val_sample_size == -1:
        test_python = ds_python["validation"].shuffle(seed=42)
    else:
        test_python = ds_python["validation"].shuffle(seed=42).select(range(min(args.val_sample_size, len(ds_python["validation"]))))
    
    column_names = list(train_python.features)
    train_python_processed = train_python.map(
        lambda x: apply_chat_template_python(x, tokenizer),
        remove_columns=column_names,
        desc="Processing Python train"
    )
    test_python_processed = test_python.map(
        lambda x: apply_chat_template_python(x, tokenizer),
        remove_columns=column_names,
        desc="Processing Python test"
    )
    print(f"Python - Train: {len(train_python_processed)}, Test: {len(test_python_processed)}")
    
    print("Merging multilingual datasets...")
    train_dataset = concatenate_datasets([train_java_processed, train_python_processed]).shuffle(seed=42)
    test_dataset = concatenate_datasets([test_java_processed, test_python_processed]).shuffle(seed=42)
    
    print(f"Total train size: {len(train_dataset)}")
    print(f"Total test size: {len(test_dataset)}")
    
    if args.save_processed_data:
        save_dir = f"./processed_datasets_multilingual_codegen"
        os.makedirs(save_dir, exist_ok=True)
        train_dataset.save_to_disk(f"{save_dir}/train")
        test_dataset.save_to_disk(f"{save_dir}/test")
        print(f"Datasets saved to {save_dir}")
    
    output_dir = f"results/{args.base_model_name.split('/')[-1]}_generation_fft_qwen1_5"
    
    training_args = TrainingArguments(
        per_device_train_batch_size=args.device_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_accumulation_steps=1,
        eval_do_concat_batches=False,
        dataloader_pin_memory=False,
        warmup_steps=1000,
        report_to=[],
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        num_train_epochs=args.num_train_epochs,
        logging_steps=50,
        optim="adamw_torch",
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
        metric_for_best_model="eval_codebleu",
        greater_is_better=True,
        load_best_model_at_end=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )
    
    torch.cuda.empty_cache()
    
    print(f"\nGPU Information:")
    print(f"Visible GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    
    # Initialize trainer
    trainer = SFTTrainer(
        model,
        packing=True,
        max_seq_length=512,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset.select(range(min(args.eval_samples, len(test_dataset)))),
        dataset_text_field='text',
        compute_metrics=lambda eval_pred: compute_metrics_codebleu(eval_pred, tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3,early_stopping_threshold=0.001)]
    )
    
    print("\nStarting multilingual training...")
    trainer.train()
    
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"\nMultilingual training completed! Model saved to: {output_dir}")

if __name__ == "__main__":
    main()
