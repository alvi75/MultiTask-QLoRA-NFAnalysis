# Exploring Multi-Task Parameter-Efficient Fine-Tuning Optimization for Code-Related Tasks

## Abstract

Large Language Models (LLMs) have proven highly effective in automating software engineering tasks. With billions of parameters, they bridge natural language and code semantics, achieving notable results in tasks such as code generation and summarization. This capability largely arises from their scale, which enables them to capture extensive domain-specific knowledge. However, larger models also incur substantial computational and financial costs, making full fine-tuning impractical for adapting LLMs in code-related tasks. Parameter-Efficient Fine-Tuning (PEFT) methods such as QLoRA address this limitation by enabling efficient model specialization with significantly lower resource demands. Recent studies show that QLoRA-optimized Large Code Models (LCMs)–LLMs specialized for code understanding and generation–perform strongly across diverse coding tasks. Yet, it remains unclear whether this effectiveness is preserved when a single model is QLoRA fine-tuned for multiple code-related tasks. The interaction between multi-task fine-tuning and QLoRA optimization, and the extent to which transfer learning contributes to the correctness and quality of the generated software artifacts, remain largely unexplored. To bridge this gap, we investigate multi-task QLoRA fine-tuning across three representative code-related tasks: (i) code generation, (ii) code summarization, and (iii) code translation. We evaluate functional correctness through execution-based and similarity-based metrics, complemented by a comprehensive analysis of code quality–an aspect largely overlooked in prior work. Our findings show that multi-task QLoRA effectively leverages transfer learning across tasks, achieving competitive or superior performance relative to both single-task QLoRA and multi-task full fine-tuning. Larger models demonstrate a more consistent balance between correctness and quality, whereas smaller models preserve functionality but exhibit a higher incidence of quality-related issues.

This repository contains experiments on **code summarization**, **code generation**, **code translation**, and **multitask training** using two training strategies:

- **FFT** (Full Fine-Tuning)  
- **QLoRA** (Quantized Low-Rank Adaptation)  

Each task has its own folder with separate scripts for FFT and QLoRA.

---

## Repository Structure
```
.
├── summarization/    # fft_train.py, qlora_train.py
├── generation/       # fft_train.py, qlora_train.py
├── translation/      # fft_train.py, qlora_train.py
├── multitask/        # fft_train.py, qlora_train.py          
├── requirements.txt
└── README.md
```

---

## Setup
Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Usage Examples

### Multitask — FFT
```bash
python multitask/fft_train.py   --base_model_name Qwen/Qwen2.5-Coder-1.5B-Instruct   --device_batch_size 2   --gradient_accumulation_steps 16   --save_processed_data True   --sample_size -1   --val_sample_size -1   --eval_samples 400   --early_stopping_patience 3   --early_stopping_threshold 0.001   --num_train_epochs 10
```

For a **generic FFT run**, remove the early-stopping flags:  
- `--early_stopping_patience`  
- `--early_stopping_threshold`

---

### Multitask — QLoRA
```bash
python multitask/qlora_train.py   --base_model_name Qwen/Qwen2.5-Coder-1.5B-Instruct   --device_batch_size 2   --gradient_accumulation_steps 16   --save_processed_data True   --sample_size -1   --val_sample_size -1   --eval_samples 400   --num_train_epochs 10
```

---

## Task Overview

| Task           | FFT Script                   | QLoRA Script                   |
|----------------|-----------------------------|--------------------------------|
| Summarization  | `summarization/fft_train.py` | `summarization/qlora_train.py` |
| Generation     | `generation/fft_train.py`    | `generation/qlora_train.py`    |
| Translation    | `translation/fft_train.py`   | `translation/qlora_train.py`   |
| Multitask      | `multitask/fft_train.py`     | `multitask/qlora_train.py`     |

---

## Notes
- **Batch size** = 2, **Grad accumulation** = 16 (default across all tasks).  
- **Validation samples**: recommendation, 5–10% of the validation set. To avoid OOM, capped to **250–400** depending on the task.  
- **Evaluation steps** are computed as *steps per epoch*, i.e., dataset size ÷ effective batch size. 
