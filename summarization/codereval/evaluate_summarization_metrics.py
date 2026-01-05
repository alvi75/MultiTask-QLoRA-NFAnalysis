import os
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import util

import sacrebleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation script for CoderEval JSONL files")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to JSONL file")
    parser.add_argument('--language', type=str, required=True, choices=['java', 'python'], help="Programming language")
    parser.add_argument('--summary_field', type=str, default='summary_qwen0_5b', 
                       help="Field name for generated summary (e.g., summary_qwen0_5b, summary_qwen0_5b_qlora, summary_qwen1_5b)")
    parser.add_argument('--output_file', type=str, required=True, help="Output file for results")
    parser.add_argument('--side_checkpoint', type=str, default="path/to/SIDE/checkpoint", help="Path to SIDE model checkpoint (Java only)")
    parser.add_argument('--max_samples', type=int, default=None, help="Maximum samples to evaluate")
    args = parser.parse_args()
    return args

# === SIDE Score Functions (Java only) ===
def mean_pooling(model_output, attention_mask):
    """Mean pooling for SIDE embeddings"""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def compute_side_score(codes, predictions, checkpoint_path):
    """Compute SIDE score for Java code-summary pairs"""
    print("\nComputing SIDE score (Java only)...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        model = AutoModel.from_pretrained(checkpoint_path).to(device)
        model.eval()
    except Exception as e:
        print(f"Warning: Could not load SIDE model from {checkpoint_path}")
        print(f"Error: {e}")
        return None
    
    scores = []
    for code, summary in tqdm(zip(codes, predictions), total=len(codes), desc="SIDE"):
        pair = [code, summary]
        inputs = tokenizer(pair, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            output = model(**inputs)
        
        embeddings = mean_pooling(output, inputs['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
        scores.append(score)
    
    side_score = sum(scores) / len(scores)
    return side_score

def load_jsonl_data(file_path, summary_field, max_samples=None):
    """Load data from JSONL file"""
    data = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            data.append(json.loads(line.strip()))
    
    codes = []
    references = []
    predictions = []
    
    for item in data:
        codes.append(item['code'])
        references.append(item['reference_summary'])

        if summary_field in item:
            predictions.append(item[summary_field])
        else:
            print(f"Warning: Field '{summary_field}' not found in data. Available fields: {list(item.keys())}")
            predictions.append("")
    
    return codes, references, predictions

def main():
    args = parse_args()
    
    print("="*60)
    print("CODEREVAL EVALUATION SCRIPT")
    print("="*60)
    print(f"Language: {args.language}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Summary field: {args.summary_field}")
    
    print(f"\nLoading data from JSONL...")
    codes, references, predictions = load_jsonl_data(
        args.dataset_path, 
        args.summary_field,
        args.max_samples
    )
    
    print(f"Loaded {len(predictions)} samples")
    
    predictions = [pred.strip() if pred else "" for pred in predictions]
    references = [ref.strip() if ref else "" for ref in references]
    
    results = {}
    
    print("\nComputing BLEU...")
    try:
        bleu_scores = []
        for pred, ref in zip(predictions, references):
            if pred.strip() == '':
                bleu_scores.append(0.0)
            else:
                bleu = sacrebleu.sentence_bleu(pred, [ref])
                bleu_scores.append(bleu.score / 100)
        results['bleu'] = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
        print(f"BLEU: {results['bleu']:.4f}")
    except Exception as e:
        print(f"Error computing BLEU: {e}")
        results['bleu'] = 0.0
    
    print("Computing METEOR...")
    try:
        import nltk
        try:
            nltk.data.find('wordnet')
        except:
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
        
        from nltk.translate.meteor_score import meteor_score
        meteor_scores = []
        for pred, ref in zip(predictions, references):
            if pred.strip() == '':
                meteor_scores.append(0.0)
            else:
                score = meteor_score([ref.split()], pred.split())
                meteor_scores.append(score)
        results['meteor'] = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0
        print(f"METEOR: {results['meteor']:.4f}")
    except Exception as e:
        print(f"Error computing METEOR: {e}")
        results['meteor'] = 0.0

    print("Computing ROUGE...")
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for pred, ref in zip(predictions, references):
            if pred.strip() == '':
                rouge1_scores.append(0.0)
                rouge2_scores.append(0.0)
                rougeL_scores.append(0.0)
            else:
                scores = scorer.score(ref, pred)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)
        
        results['rouge-1'] = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0.0
        results['rouge-2'] = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0
        results['rouge-L'] = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0.0
        print(f"ROUGE-1: {results['rouge-1']:.4f}")
        print(f"ROUGE-2: {results['rouge-2']:.4f}")
        print(f"ROUGE-L: {results['rouge-L']:.4f}")
    except Exception as e:
        print(f"Error computing ROUGE: {e}")
        results['rouge-L'] = 0.0
        results['rouge-1'] = 0.0
        results['rouge-2'] = 0.0
    
    print("Computing chrF...")
    try:
        chrf_scores = []
        for pred, ref in zip(predictions, references):
            if pred.strip() == '':
                chrf_scores.append(0.0)
            else:
                chrf_score = sacrebleu.sentence_chrf(pred, [ref])
                chrf_scores.append(chrf_score.score / 100)
        results['chrf'] = sum(chrf_scores) / len(chrf_scores) if chrf_scores else 0.0
        print(f"chrF: {results['chrf']:.4f}")
    except Exception as e:
        print(f"Error computing chrF: {e}")
        results['chrf'] = 0.0
    
    print("Computing BERTScore...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Using device: {device}")
        
        valid_pairs = [(p, r) for p, r in zip(predictions, references) if p.strip() and r.strip()]
        
        if not valid_pairs:
            print("  Warning: No valid prediction-reference pairs for BERTScore")
            results['bertscore_f1'] = 0.0
            results['bertscore_precision'] = 0.0
            results['bertscore_recall'] = 0.0
        else:
            valid_preds, valid_refs = zip(*valid_pairs)
            
            try:
                P, R, F1 = bert_score(
                    list(valid_preds), 
                    list(valid_refs), 
                    model_type="bert-base-uncased",
                    device=device,
                    batch_size=8, 
                    verbose=True,
                    num_layers=12
                )
            except Exception as e1:
                print(f"  DeBERTa failed: {e1}")
                print("  Trying with roberta-large instead...")
                P, R, F1 = bert_score(
                    list(valid_preds), 
                    list(valid_refs), 
                    model_type="roberta-large",
                    device=device,
                    batch_size=16,
                    verbose=True
                )
            
            results['bertscore_f1'] = F1.mean().item()
            results['bertscore_precision'] = P.mean().item()
            results['bertscore_recall'] = R.mean().item()
            print(f"BERTScore Precision: {results['bertscore_precision']:.4f}")
            print(f"BERTScore Recall: {results['bertscore_recall']:.4f}")
            print(f"BERTScore F1: {results['bertscore_f1']:.4f}")
    except Exception as e:
        print(f"Error computing BERTScore: {e}")
        print("  Skipping BERTScore computation")
        results['bertscore_f1'] = 0.0
        results['bertscore_precision'] = 0.0
        results['bertscore_recall'] = 0.0

    if args.language == 'java':
        print("\n" + "="*40)
        print("Computing SIDE score (Java-specific)...")
        print("="*40)
        side_score = compute_side_score(codes, predictions, args.side_checkpoint)
        if side_score is not None:
            results['side_score'] = side_score
            print(f"SIDE Score: {side_score:.4f}")
        else:
            results['side_score'] = None
            print("SIDE score computation failed")
    else:
        print(f"\nSIDE score not computed (only available for Java, current language: {args.language})")
        results['side_score'] = None
    
    output_path = args.output_file if args.output_file.endswith('.txt') else f"{args.output_file}.txt"
    with open(output_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("EVALUATION RESULTS\n")
        f.write("="*60 + "\n")
        f.write(f"Language: {args.language}\n")
        f.write(f"Dataset file: {args.dataset_path}\n")
        f.write(f"Summary field: {args.summary_field}\n")
        f.write(f"Number of samples: {len(predictions)}\n")
        f.write("-"*60 + "\n")
        f.write(f"BLEU:                {results['bleu']:.4f}\n")
        f.write(f"METEOR:              {results['meteor']:.4f}\n")
        f.write(f"ROUGE-1:             {results['rouge-1']:.4f}\n")
        f.write(f"ROUGE-2:             {results['rouge-2']:.4f}\n")
        f.write(f"ROUGE-L:             {results['rouge-L']:.4f}\n")
        f.write(f"chrF:                {results['chrf']:.4f}\n")
        f.write(f"BERTScore Precision: {results['bertscore_precision']:.4f}\n")
        f.write(f"BERTScore Recall:    {results['bertscore_recall']:.4f}\n")
        f.write(f"BERTScore F1:        {results['bertscore_f1']:.4f}\n")
        if results['side_score'] is not None:
            f.write(f"SIDE Score:          {results['side_score']:.4f}\n")
        f.write("="*60 + "\n")
    
    json_path = output_path.replace('.txt', '.json')
    results_json = {
        'language': args.language,
        'dataset_file': args.dataset_path,
        'summary_field': args.summary_field,
        'num_samples': len(predictions),
        'metrics': results
    }
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"BLEU:                {results['bleu']:.4f}")
    print(f"METEOR:              {results['meteor']:.4f}")
    print(f"ROUGE-1:             {results['rouge-1']:.4f}")
    print(f"ROUGE-2:             {results['rouge-2']:.4f}")
    print(f"ROUGE-L:             {results['rouge-L']:.4f}")
    print(f"chrF:                {results['chrf']:.4f}")
    print(f"BERTScore F1:        {results['bertscore_f1']:.4f}")
    if results['side_score'] is not None:
        print(f"SIDE Score:          {results['side_score']:.4f}")
    print("="*60)
    print(f"\nResults saved to:")
    print(f"  - {output_path}")
    print(f"  - {json_path}")

if __name__ == "__main__":
    main()

# Usage:
#   python evaluate_summarization_metrics.py \
#       --dataset_path path/to/predictions.jsonl \
#       --language java \
#       --summary_field generated_summary \
#       --output_file path/to/results
