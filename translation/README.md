# Code Translation

This folder contains scripts for training, inference, and evaluation of code translation models (Java ↔ C#).

## Folder Structure

```
code-translation/
├── fft_train.py                 # Full fine-tuning training script
├── qlora_train.py               # QLoRA training script
├── infer_translation_fft.py     # Inference + evaluation for FFT models
├── infer_translation_qlora.py   # Inference + evaluation for QLoRA models
└── wrap_java_methods.py         # Add class wrappers for static analysis
```

## Step 1: Training

### Full Fine-Tuning (FFT)

```bash
python fft_train.py \
    --base_model_name Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --device_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --sample_size -1 \
    --val_sample_size -1 \
    --eval_samples 625 \
    --num_train_epochs 5
```

### QLoRA

```bash
python qlora_train.py \
    --base_model_name Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --device_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --sample_size -1 \
    --val_sample_size -1 \
    --eval_samples 625 \
    --num_train_epochs 5
```

**Arguments:**
- `--base_model_name`: Base model from HuggingFace
- `--device_batch_size`: Per-device batch size (default: 4)
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: 8)
- `--sample_size`: Training samples (-1 for full dataset)
- `--val_sample_size`: Validation samples (-1 for full dataset)
- `--eval_samples`: Samples for evaluation during training
- `--num_train_epochs`: Number of training epochs (default: 5)

## Step 2: Inference and Evaluation

The inference scripts generate translations and compute CodeBLEU scores automatically.

### FFT Models

```bash
python infer_translation_fft.py \
    --model_path path/to/model \
    --direction cs_to_java \
    --output_dir path/to/output \
    --batch_size 8
```

### QLoRA Models

```bash
python infer_translation_qlora.py \
    --model_path path/to/qlora_adapter \
    --direction java_to_cs \
    --output_dir path/to/output \
    --batch_size 8
```

**Arguments:**
- `--model_path`: Path to trained model checkpoint
- `--direction`: Translation direction (`cs_to_java` or `java_to_cs`)
- `--output_dir`: Directory to save translated files and results
- `--batch_size`: Batch size for inference (default: 8)
- `--max_samples`: Limit samples for testing (optional)

**Output:**
- `<output_dir>/java_files/` or `<output_dir>/cs_files/`: Individual translated code files
- `<output_dir>/results_<direction>.json`: CodeBLEU scores

## Step 3: Wrap Java Files (For Static Analysis)

For PMD and SonarCloud analysis, Java methods need to be wrapped in compilable classes:

```bash
python wrap_java_methods.py \
    --input_dir path/to/java_files \
    --output_dir path/to/wrapped_output
```

**Arguments:**
- `--input_dir`: Directory containing Java files
- `--output_dir`: Directory to save wrapped Java files

## Complete Pipeline Example

```bash
# 1. Train model
python fft_train.py --base_model_name Qwen/Qwen2.5-Coder-1.5B-Instruct

# 2. Run inference + evaluation (C# to Java)
python infer_translation_fft.py \
    --model_path results/Qwen2.5-Coder-1.5B-Instruct_translation_fft \
    --direction cs_to_java \
    --output_dir results/cs_to_java

# 3. Run inference + evaluation (Java to C#)
python infer_translation_fft.py \
    --model_path results/Qwen2.5-Coder-1.5B-Instruct_translation_fft \
    --direction java_to_cs \
    --output_dir results/java_to_cs

# 4. Wrap Java files for static analysis
python wrap_java_methods.py \
    --input_dir results/cs_to_java/java_files \
    --output_dir results/cs_to_java/java_files_wrapped
```

## Metrics

The inference scripts automatically compute:

- **CodeBLEU**: Overall code similarity score
  - **N-gram Match**: Token-level similarity
  - **Weighted N-gram**: Weighted token similarity
  - **Syntax Match**: AST-based similarity
  - **Dataflow Match**: Semantic similarity

## Dataset

Uses `google/code_x_glue_cc_code_to_code_trans` dataset:
- Java ↔ C# parallel corpus
- Train/Validation/Test splits

## Requirements

```
torch
transformers
datasets
trl
peft
bitsandbytes
codebleu
```
