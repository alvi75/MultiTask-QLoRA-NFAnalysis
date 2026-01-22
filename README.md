# Parameter-Efficient Multi-Task Fine-Tuning in Code-Related Tasks

[![arXiv](https://img.shields.io/badge/arXiv-2601.15094-b31b1b.svg)](https://arxiv.org/abs/2601.15094)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract

Large Language Models (LLMs) have proven highly effective in automating software engineering tasks, bridging natural language and code semantics to achieve notable results in code generation and summarization. However, their scale incurs substantial computational costs, making full fine-tuning impractical. Parameter-Efficient Fine-Tuning (PEFT) methods like QLoRA enable efficient specialization with lower resource demands. Recent studies show QLoRA-optimized Large Code Models (LCMs) perform strongly across diverse tasks, yet it remains unclear whether this effectiveness persists when a single model is QLoRA fine-tuned for multiple code-related tasks. The interaction between Multi-task fine-tuning and QLoRA optimization, and how transfer learning affects correctness and quality of generated artifacts, remains largely unexplored. We investigate Multi-task QLoRA fine-tuning across three representative tasks: code generation, translation, and summarization. We evaluate functional correctness through execution-based and similarity-based metrics, complemented by comprehensive code quality analysis--an aspect largely overlooked in prior work. Our findings show that Multi-task QLoRA effectively leverages transfer learning, achieving competitive or superior performance relative to both Single-task QLoRA and Multi-task full fine-tuning. Larger models demonstrate more consistent balance between correctness and quality, whereas smaller models preserve functionality but exhibit a higher incidence of quality-related issues.

---

## Repository Structure

```
.
├── generation/                   # Code generation scripts
├── summarization/                # Code summarization scripts
├── translation/                  # Code translation scripts
├── multitask/                    # Multi-task training scripts
├── non_functional_analysis/      # Static analysis tools (PMD, Pylint, SonarCloud, etc.)
├── statistical_tests/            # Statistical analysis (Wilcoxon signed-rank tests)
│   ├── R-scripts/                # R scripts and results
│   └── per-instance-value/       # Per-instance metric values
├── results/                      # Experimental results
├── dataset/                      # Training datasets
├── requirements.txt              # Python dependencies
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

Set environment variables:

```bash
export HF_TOKEN="your-huggingface-token"
export OPENAI_API_KEY="your-openai-api-key"  # For LLM-as-judge evaluation
```

---

## Task Overview

| Task | Folder | FFT Script | QLoRA Script |
|------|--------|------------|--------------|
| Code Generation | `generation/` | `fft_train.py` | `qlora_train.py` |
| Code Summarization | `summarization/` | `fft_train.py` | `qlora_train.py` |
| Code Translation | `translation/` | `fft_train.py` | `qlora_train.py` |
| Multi-Task | `multitask/` | `fft_train.py` | `qlora_train.py` |

Each task folder contains a **README.md** with detailed instructions for training, inference, and evaluation.

---

## Single-Task Training & Evaluation

See the README in each task folder:

- [generation/README.md](generation/README.md)
- [summarization/README.md](summarization/README.md)
- [translation/README.md](translation/README.md)

---

## Multi-Task Training

### FFT (Full Fine-Tuning)

```bash
python multitask/fft_train.py \
    --base_model_name Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --device_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --sample_size -1 \
    --val_sample_size -1 \
    --eval_samples 400 \
    --num_train_epochs 5
```

### QLoRA (Quantized Low-Rank Adaptation)

```bash
python multitask/qlora_train.py \
    --base_model_name Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --device_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --sample_size -1 \
    --val_sample_size -1 \
    --eval_samples 400 \
    --num_train_epochs 5
```

**Arguments:**
- `--base_model_name`: Base model (`Qwen/Qwen2.5-Coder-0.5B-Instruct`, `Qwen/Qwen2.5-Coder-1.5B-Instruct`, `Qwen/Qwen2.5-Coder-3B-Instruct`)
- `--device_batch_size`: Per-device batch size (default: 2)
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: 16, effective batch size = 32)
- `--sample_size`: Training samples per task (-1 for full dataset)
- `--val_sample_size`: Validation samples per task (-1 for full dataset)
- `--eval_samples`: Samples for evaluation during training
- `--num_train_epochs`: Number of training epochs

### Multi-Task Evaluation

After training, evaluate multi-task models using the inference scripts in each task folder:
- **Code Generation**: `generation/codereval/infer_generation_*.py`
- **Code Summarization**: `summarization/codereval/infer_summarization_*.py`
- **Code Translation**: `translation/infer_translation_*.py`

---

## Non-Functional Analysis

The `non_functional_analysis/` folder contains scripts for static code analysis:

| Script | Purpose |
|--------|---------|
| `analysis_PMD_checkstyle.py` | PMD and Checkstyle analysis (Java) |
| `analysis_pylint_flake8.py` | Pylint and Flake8 analysis (Python) |
| `lizard_analysis.py` | Cyclomatic complexity analysis |
| `RoslynAnalyzer/` | C# code analysis (translation task) |

---

## Statistical Tests

The `statistical_tests/` folder contains Wilcoxon signed-rank tests comparing single-task vs. multi-task performance.

```
statistical_tests/
├── R-scripts/                    # R scripts and results
└── per-instance-value/           # Raw per-instance metric values
    ├── pass1/
    ├── bleu_meteor_rouge_chrf_bertscore_side/summarization/
    ├── codebleu/translation/
    ├── llm_judge/
    ├── pmd/
    ├── pylint/generation/
    ├── sonarcloud/
    ├── lizard/
    └── roslyn/java_cs/
```

---

## Models

| Model | Parameters | HuggingFace |
|-------|------------|-------------|
| Qwen2.5-Coder-0.5B-Instruct | 0.5B | [Link](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct) |
| Qwen2.5-Coder-1.5B-Instruct | 1.5B | [Link](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) |
| Qwen2.5-Coder-3B-Instruct | 3B | [Link](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct) |

---

## Datasets

| Task | Dataset | Source |
|------|---------|--------|
| Code Generation | CodeXGLUE | [Link](https://drive.google.com/file/d/16udq-fCBuwr8lDN3Y_JSAK6S3tn98dut/view?usp=drive_link) |
| Code Summarization | CodeXGLUE Code-to-Text | [Link](https://github.com/microsoft/CodeXGLUE) |
| Code Translation | CodeXGLUE Java-C# | [Link](https://github.com/microsoft/CodeXGLUE) |
| Evaluation | CoderEval | [Link](https://github.com/CoderEval/CoderEval) |

---

## Notes

- **Batch size**: 2, **Gradient accumulation**: 16 (default across all tasks, effective batch size = 32)
- **Validation samples**: 5–10% of validation set recommended; capped at 250–400 to avoid OOM
- **Evaluation steps**: Computed as dataset size ÷ effective batch size (steps per epoch)

---

## Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{haque2026parameterefficientmultitaskfinetuningcoderelated,
      title={Parameter-Efficient Multi-Task Fine-Tuning in Code-Related Tasks}, 
      author={Md Zahidul Haque and Saima Afrin and Antonio Mastropaolo},
      year={2026},
      eprint={2601.15094},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2601.15094}, 
}
```

---

## License

This project is licensed under the MIT License.
