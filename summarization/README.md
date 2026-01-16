# Code Summarization

This folder contains scripts for training, inference, and evaluation of code summarization models.

## Folder Structure

```
code-summarization/
├── fft_train.py                 # Full fine-tuning training script
├── qlora_train.py               # QLoRA training script
├── codereval/
    ├── infer_summarization_fft.py       # Inference for FFT models
    ├── infer_summarization_qlora.py     # Inference for QLoRA models
    ├── evaluate_summarization_metrics.py # Compute BLEU, ROUGE, METEOR, etc.
    ├── evaluate_summarization_llm_judge.py  # LLM-as-judge evaluation (GPT-5 mini)
    ├── aggregate_llm_judge_scores.py    # Aggregate LLM judge scores using mean
    ├── cs_codereval_eval_dataset_java_v2.jsonl  # CoderEval Java benchmark
    └── cs_codereval_eval_dataset_py_v2.jsonl    # CoderEval Python benchmark
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
- `--base_model_name`: Base model from HuggingFace
- `--device_batch_size`: Per-device batch size (default: 2)
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: 16)
- `--sample_size`: Training samples per language (-1 for full dataset)
- `--val_sample_size`: Validation samples per language (-1 for full dataset)
- `--eval_samples`: Samples for evaluation during training
- `--num_train_epochs`: Number of training epochs (default: 5)

## Step 2: Inference on CoderEval

### FFT Models

```bash
python codereval/infer_summarization_fft.py \
    --model_path path/to/model \
    --input_jsonl codereval/cs_codereval_eval_dataset_java_v2.jsonl \
    --output_file path/to/output.csv \
    --language java \
    --batch_size 8 \
    --max_new_tokens 128
```

### QLoRA Models

```bash
python codereval/infer_summarization_qlora.py \
    --model_path path/to/qlora_adapter \
    --input_jsonl codereval/cs_codereval_eval_dataset_java_v2.jsonl \
    --output_file path/to/output.csv \
    --language java \
    --batch_size 8 \
    --max_new_tokens 128
```

**Arguments:**
- `--model_path`: Path to trained model checkpoint
- `--input_jsonl`: Path to CoderEval benchmark file
- `--output_file`: Path to save predictions
- `--language`: Programming language (`java` or `python`)
- `--batch_size`: Batch size for inference
- `--max_new_tokens`: Maximum tokens to generate

## Step 3: Evaluate with Reference-Based Metrics

Compute BLEU, METEOR, ROUGE, chrF, BERTScore, and SIDE (Java only):

```bash
python codereval/evaluate_summarization_metrics.py \
    --dataset_path path/to/predictions.jsonl \
    --language java \
    --summary_field generated_summary \
    --output_file path/to/results
```

**Arguments:**
- `--dataset_path`: Path to JSONL file with predictions
- `--language`: Programming language (`java` or `python`)
- `--summary_field`: Field name containing generated summaries
- `--output_file`: Output path for results (without extension)
- `--side_checkpoint`: Path to SIDE model checkpoint (Java only)

**Output:**
- `<output_file>.txt`: Human-readable results
- `<output_file>.json`: Machine-readable results

## Step 4: Evaluate with LLM-as-Judge

Uses GPT-5-mini to evaluate summaries on Content Adequacy, Conciseness, and Fluency (1-5 scale).

1. Edit configuration variables in `evaluate_summarization_llm_judge.py`:
```python
INPUT_FILE = "path/to/predictions.jsonl"
OUTPUT_FOLDER = "path/to/output_folder"
LANGUAGE = "java"  # or "python"
SUMMARY_FIELD = "generated_summary"
MODEL_NAME = "model_name"
NUM_RUNS = 5
```

2. Set OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key"
```

3. Run:
```bash
python codereval/evaluate_summarization_llm_judge.py
```

**Output:**
- Individual run CSVs: `<output_folder>/<name>_1.csv`, `<name>_2.csv`, etc.
- Merged results: `<output_folder>/<name>_FINAL_MERGED.csv`

## Step 5: Aggregate LLM Judge Scores (Mean)

Recalculate final scores using mean instead of voting:

1. Edit configuration variables in `aggregate_llm_judge_scores.py`:
```python
INPUT_FOLDER = "path/to/llm_judge_results"
OUTPUT_FILE_NAME = "FINAL_MERGED_MEAN.csv"
NUM_RUNS = 5
```

2. Run:
```bash
python codereval/aggregate_llm_judge_scores.py
```

## Complete Pipeline Example

```bash
# 1. Train model
python fft_train.py --base_model_name Qwen/Qwen2.5-Coder-1.5B-Instruct

# 2. Run inference
python codereval/infer_summarization_fft.py \
    --model_path results/Qwen2.5-Coder-1.5B-Instruct_summarization_fft \
    --input_jsonl codereval/cs_codereval_eval_dataset_java_v2.jsonl \
    --output_file results/java_predictions.csv \
    --language java

# 3. Evaluate with reference-based metrics
python codereval/evaluate_summarization_metrics.py \
    --dataset_path results/java_predictions.jsonl \
    --language java \
    --summary_field generated_summary \
    --output_file results/java_metrics

# 4. Evaluate with LLM-as-judge (edit config in script first)
python codereval/evaluate_summarization_llm_judge.py

# 5. Aggregate LLM judge scores (edit config in script first)
python codereval/aggregate_llm_judge_scores.py
```

## Metrics

### Reference-Based Metrics
- **BLEU**: N-gram overlap
- **METEOR**: Semantic similarity with synonyms
- **ROUGE-1/2/L**: Recall-oriented metrics
- **chrF**: Character-level F-score
- **BERTScore**: Contextual embedding similarity
- **SIDE**: Semantic similarity for code summaries (Java only)

### LLM-as-Judge Metrics
- **Content Adequacy**: How well the summary captures code functionality
- **Conciseness**: Absence of unnecessary information
- **Fluency**: Readability and clarity
