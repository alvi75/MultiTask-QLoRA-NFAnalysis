import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from datasets import load_dataset, concatenate_datasets
import argparse
from trl import SFTTrainer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from transformers import TrainingArguments, EarlyStoppingCallback, BitsAndBytesConfig
import json
import sacrebleu

def parse_args():
    parser = argparse.ArgumentParser(description="Run multilingual code summarization fine-tuning (Java + Python)")
    parser.add_argument('--base_model_name', type=str, default='Qwen/Qwen2.5-Coder-1.5B-Instruct', help="Base model name")
    parser.add_argument('--device_batch_size', type=int, default=2, help="Per device train batch size")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument('--sample_size', type=int, default=-1, help="Sample size for training data per language (-1 for full dataset)")
    parser.add_argument('--val_sample_size', type=int, default=-1, help="Sample size for validation data per language (-1 for full dataset)")
    parser.add_argument('--eval_samples', type=int, default=250, help="Number of samples for evaluation during training")
    parser.add_argument('--save_processed_data', type=bool, default=True, help="Save processed datasets")
    parser.add_argument('--num_train_epochs', type=int, default=3, help="Number of training epochs")
    args = parser.parse_args()
    return args

def apply_chat_template_java(example, tokenizer):
    code = ' '.join(example['code_tokens'])
    chat = [
        {"role": "system", "content": "You are an expert Java developer. Provide clear and concise summarization for Java code segments."},
        {"role": "user", "content": f"Summarize this Java code:\n{code}"},
        {"role": "assistant", "content": f"SUMMARY: {' '.join(example['docstring_tokens'])} DONE"},
    ]
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{%- for message in messages -%}\n{% if message['role'] == 'system' %}{% if loop.first %}{{ message['content'] + '\n' }}{% endif %}{% elif message['role'] == 'user' %}{{ '\nHuman: ' + message['content'] + '\n\n' }}{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + '\n\n' }}{% endif %}\n{%- endfor -%}\n{% if add_generation_prompt %}Human: {{ '' }}{% endif %}"
    example["text"] = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
    example["language"] = "java"
    return example

def apply_chat_template_python(example, tokenizer):
    code = ' '.join(example['code_tokens'])
    chat = [
        {"role": "system", "content": "You are an expert Python developer. Provide clear and concise summarization for Python code segments."},
        {"role": "user", "content": f"Summarize this Python code:\n{code}"},
        {"role": "assistant", "content": f"SUMMARY: {' '.join(example['docstring_tokens'])} DONE"},
    ]
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{%- for message in messages -%}\n{% if message['role'] == 'system' %}{% if loop.first %}{{ message['content'] + '\n' }}{% endif %}{% elif message['role'] == 'user' %}{{ '\nHuman: ' + message['content'] + '\n\n' }}{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + '\n\n' }}{% endif %}\n{%- endfor -%}\n{% if add_generation_prompt %}Human: {{ '' }}{% endif %}"
    example["text"] = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
    example["language"] = "python"
    return example

def compute_metrics_sacrebleu(eval_pred, tokenizer):
    predictions, labels = eval_pred
    
    if predictions.ndim == 3:
        predictions = np.argmax(predictions, axis=-1)
    
    predictions = np.where(predictions != -100, predictions, tokenizer.eos_token_id)
    labels = np.where(labels != -100, labels, tokenizer.eos_token_id)
    
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Extract summaries
    decoded_preds = [i.split("SUMMARY: ")[-1].split('DONE')[0].strip() if "SUMMARY:" in i else i.split("Assistant:")[-1].strip() if "Assistant:" in i else i for i in decoded_preds]
    decoded_labels = [i.split("SUMMARY: ")[-1].split('DONE')[0].strip() if "SUMMARY:" in i else i.split("Assistant:")[-1].strip() if "Assistant:" in i else i for i in decoded_labels]
    
    # Calculate SacreBLEU
    sacrebleu_scores = []
    for pred, label in zip(decoded_preds, decoded_labels):
        try:
            score = sacrebleu.sentence_bleu(pred, [label]).score
            sacrebleu_scores.append(score)
        except:
            sacrebleu_scores.append(0.0)
    
    # Generation length
    gen_lengths = [len(tokenizer.tokenize(p)) for p in decoded_preds]
    
    return {
        "eval_sacrebleu": np.mean(sacrebleu_scores),
        "eval_gen_len": np.mean(gen_lengths)
    }

def main():
    args = parse_args()
    
    print("="*80)
    print("Multilingual Code Summarization Training")
    print(f"Languages: Java + Python")
    print(f"Model: {args.base_model_name}")
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
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=hf_token,
        trust_remote_code=True
    )
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False, 
        r=8, 
        lora_alpha=16, 
        lora_dropout=0.1
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Load Java dataset
    print("Loading Java dataset...")
    ds_java = load_dataset("google/code_x_glue_ct_code_to_text", "java", cache_dir='datasets')
    
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
    
    # Load Python dataset
    print("Loading Python dataset...")
    ds_python = load_dataset("google/code_x_glue_ct_code_to_text", "python", cache_dir='datasets')
    
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
    
    # Merge datasets
    print("Merging multilingual datasets...")
    train_dataset = concatenate_datasets([train_java_processed, train_python_processed]).shuffle(seed=42)
    test_dataset = concatenate_datasets([test_java_processed, test_python_processed]).shuffle(seed=42)
    
    print(f"Total train size: {len(train_dataset)}")
    print(f"Total test size: {len(test_dataset)}")
    
    # Save processed datasets
    if args.save_processed_data:
        save_dir = f"./processed_datasets_multilingual"
        os.makedirs(save_dir, exist_ok=True)
        train_dataset.save_to_disk(f"{save_dir}/train")
        test_dataset.save_to_disk(f"{save_dir}/test")
        print(f"Datasets saved to {save_dir}")
    
    # Training configuration
    output_dir = f"/scratch/mhaque/results/{args.base_model_name.split('/')[-1]}_multilingual_cs_qlora"
    
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
        save_steps=500,
        eval_steps=500,
        metric_for_best_model="eval_sacrebleu",
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
        train_dataset=train_dataset,
        eval_dataset=test_dataset.select(range(min(args.eval_samples, len(test_dataset)))),
        peft_config=peft_config,
        dataset_text_field='text',
        compute_metrics=lambda eval_pred: compute_metrics_sacrebleu(eval_pred, tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    print("\nStarting multilingual training...")
    trainer.train()
    
    # Save the model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"\nMultilingual training completed! Model saved to: {output_dir}")

if __name__ == "__main__":
    main()

