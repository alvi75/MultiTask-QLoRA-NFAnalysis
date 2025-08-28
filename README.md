# Resource-Efficiency-SE

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
├── utils/            # shared helpers (data loader, evaluation, configs)
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
- **Validation samples**: by default, 5–10% of the validation set. To avoid OOM, capped to **250–400** depending on the task.  
- **Evaluation steps** are computed as *steps per epoch*, i.e., dataset size ÷ effective batch size.  
- Folder structure makes it easy to extend: simply add new task folders or training scripts as needed.
