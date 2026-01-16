# Code Generation

This folder contains scripts for training, inference, and evaluation of code generation models.

## Folder Structure

```
code-generation/
├── fft_train.py                 # Full fine-tuning training script
├── qlora_train.py               # QLoRA training script
├── codereval/
│   ├── infer_generation_fft.py      # Inference for FFT models
│   ├── infer_generation_qlora.py    # Inference for QLoRA models
│   ├── filter_codereval_ids.py      # Filter unreliable test cases
│   ├── extract_code_from_jsonl.py   # Extract code to individual files
│   ├── add_java_wrappers_cg.py      # Add class wrappers for static analysis
│   ├── ids_to_discard.json          # IDs with unreliable tests
│   ├── CEJavaHumanLabel.jsonl       # CoderEval Java benchmark
│   └── CEPythonHumanLabel.jsonl     # CoderEval Python benchmark
└── dataset/
    └── codegen_codexglue/
        ├── java/
        └── python/
```

## Step 1: Training

### Full Fine-Tuning (FFT)

```bash
python fft_train.py \
    --base_model_name Qwen/Qwen2.5-Coder-0.5B-Instruct \
    --device_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --sample_size -1 \
    --val_sample_size -1 \
    --eval_samples 500 \
    --num_train_epochs 5
```

### QLoRA

```bash
python qlora_train.py \
    --base_model_name Qwen/Qwen2.5-Coder-0.5B-Instruct \
    --device_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --sample_size -1 \
    --val_sample_size -1 \
    --eval_samples 500 \
    --num_train_epochs 5
```

**Arguments:**
- `--base_model_name`: Base model from HuggingFace (e.g., `Qwen/Qwen2.5-Coder-0.5B-Instruct`, `Qwen/Qwen2.5-Coder-1.5B-Instruct`, `Qwen/Qwen2.5-Coder-3B-Instruct`)
- `--device_batch_size`: Per-device batch size (default: 2)
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: 16, effective batch size = 32)
- `--sample_size`: Training samples per language (-1 for full dataset)
- `--val_sample_size`: Validation samples per language (-1 for full dataset)
- `--eval_samples`: Samples for evaluation during training
- `--num_train_epochs`: Number of training epochs (default: 5)

## Step 2: Inference on CoderEval

### FFT Models

```bash
python codereval/infer_generation_fft.py \
    --model_path path/to/fft_model \
    --input_file codereval/CEJavaHumanLabel.jsonl \
    --output_file path/to/output.jsonl \
    --language java \
    --num_samples 1 \
    --temperature 0.0
```

### QLoRA Models

```bash
python codereval/infer_generation_qlora.py \
    --model_path path/to/qlora_adapter \
    --input_file codereval/CEJavaHumanLabel.jsonl \
    --output_file path/to/output.jsonl \
    --language java \
    --num_samples 1 \
    --temperature 0.0
```

**Arguments:**
- `--model_path`: Path to trained model checkpoint
- `--input_file`: Path to CoderEval benchmark file
- `--output_file`: Path to save predictions
- `--language`: Programming language (`java` or `python`)
- `--num_samples`: Number of samples to generate per task
- `--temperature`: Sampling temperature (0.0 for greedy decoding)

## Step 3: Filter Unreliable Tests

As described in the paper, some CoderEval test cases have unreliable tests. Filter them out:

```bash
python codereval/filter_codereval_ids.py \
    --input_dir path/to/jsonl_files \
    --discard_ids_file codereval/ids_to_discard.json
```

## Step 4: Extract Code to Files

Extract generated code from JSONL to individual files for static analysis:

1. Edit `BASE_PATH` in the script:
```python
BASE_PATH = "path/to/generation_results"
```

2. Run:
```bash
python codereval/extract_code_from_jsonl.py
```

This creates a folder with individual code files (`.java` or `.py`) for each prediction.

## Step 5: Add Java Wrappers (For Static Analysis)

For PMD and SonarCloud analysis, Java methods need to be wrapped in compilable classes:

```bash
python codereval/add_java_wrappers_cg.py \
    --input_dir path/to/java_files \
    --output_dir path/to/java_files_wrapped
```

The wrapped files can then be analyzed with:
- **PMD**: For code quality metrics
- **SonarCloud**: For static code analysis

## Complete Pipeline Example

```bash
# 1. Train model
python fft_train.py --base_model_name Qwen/Qwen2.5-Coder-1.5B-Instruct

# 2. Run inference
python codereval/infer_generation_fft.py \
    --model_path results/Qwen2.5-Coder-1.5B-Instruct_generation_fft \
    --input_file codereval/CEJavaHumanLabel.jsonl \
    --output_file results/java_predictions.jsonl \
    --language java

# 3. Filter unreliable tests
python codereval/filter_codereval_ids.py \
    --input_dir results \
    --discard_ids_file codereval/ids_to_discard.json

# 4. Extract code files (edit BASE_PATH in script first)
python codereval/extract_code_from_jsonl.py

# 5. Wrap Java files for static analysis
python codereval/add_java_wrappers_cg.py \
    --input_dir results/java_predictions_java_files \
    --output_dir results/java_predictions_java_files_wrapped
```
