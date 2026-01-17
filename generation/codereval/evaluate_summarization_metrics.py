import os
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import util
import sys
import sacrebleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation script for CoderEval JSONL files")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to JSONL file")
    parser.add_argument('--language', type=str, required=True, choices=['java', 'python'], help="Programming language")
    parser.add_argument('--summary_field', type=str, default='summary_qwen0_5b', 
                       help="Field name for generated summary")
    parser.add_argument('--code_field', type=str, default='code',
                       help="Field name for code (default: code)")
    parser.add_argument('--reference_field', type=str, default='reference_summary',
                       help="Field name for reference summary (default: reference_summary)")
    parser.add_argument('--output_file', type=str, required=True, help="Output file for results")
    parser.add_argument('--side_checkpoint', type=str, default="/home/mhaque/QLoRA-Code-Summarization/SIDE/Models/baseline/103080", 
                       help="Path to SIDE model checkpoint (only for Java)")
    parser.add_argument('--max_samples', type=int, default=None, help="Maximum samples to evaluate")
    parser.add_argument('--validate_only', action='store_true', help="Only validate the JSONL file, don't evaluate")
    args = parser.parse_args()
    return args



def validate_jsonl(file_path, code_field, reference_field, summary_field):
    issues = []
    valid_count = 0
    total_lines = 0
    
    if not os.path.exists(file_path):
        return False, [f"File not found: {file_path}"], 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            pass
    except Exception as e:
        return False, [f"Cannot read file: {e}"], 0
    
    print(f"\nValidating JSONL file: {file_path}")
    print("-" * 60)
    
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        for line_num, line in enumerate(f, 1):
            total_lines += 1
            line = line.strip()
            
            if not line:
                issues.append(f"Line {line_num}: EMPTY LINE")
                continue
            
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                issues.append(f"Line {line_num}: JSON PARSE ERROR - {e}")
                continue
            
            missing_fields = []
            
            if code_field not in item:
                missing_fields.append(code_field)
            
            if reference_field not in item:
                missing_fields.append(reference_field)
            
            if summary_field not in item:
                missing_fields.append(summary_field)
            
            if missing_fields:
                available = list(item.keys())
                issues.append(f"Line {line_num}: MISSING FIELDS {missing_fields}. Available: {available}")
                continue
            
            empty_fields = []
            if not item.get(code_field, "").strip():
                empty_fields.append(code_field)
            if not item.get(reference_field, "").strip():
                empty_fields.append(reference_field)
            
            if empty_fields:
                issues.append(f"Line {line_num}: EMPTY VALUES for {empty_fields}")
            
            valid_count += 1
    
    print(f"Total lines:  {total_lines}")
    print(f"Valid lines:  {valid_count}")
    print(f"Issues found: {len(issues)}")
    
    if issues:
        print(f"\nFirst 10 issues:")
        for issue in issues[:10]:
            print(f"   {issue}")
        if len(issues) > 10:
            print(f"   ... and {len(issues) - 10} more issues")
    
    is_valid = valid_count > 0
    return is_valid, issues, valid_count


def load_jsonl_data_robust(file_path, code_field, reference_field, summary_field, max_samples=None):
    codes = []
    references = []
    predictions = []
    skipped = 0
    loaded = 0
    
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        for line_num, line in enumerate(f, 1):
            if max_samples and loaded >= max_samples:
                break
            
            line = line.strip()
            
            if not line:
                skipped += 1
                continue
            
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping line {line_num}: JSON parse error - {e}")
                skipped += 1
                continue
            
            code = None
            for field in [code_field, 'code', 'source_code', 'func_code', 'method_code', 'input']:
                if field in item and item[field]:
                    code = item[field]
                    break
            
            if code is None:
                print(f"Skipping line {line_num}: No code field found")
                skipped += 1
                continue
            
            reference = None
            for field in [reference_field, 'reference_summary', 'reference', 'summary', 'docstring']:
                if field in item and item[field]:
                    reference = item[field]
                    break
            
            if reference is None:
                print(f"Skipping line {line_num}: No reference field found")
                skipped += 1
                continue
            
            prediction = item.get(summary_field, "")
            if prediction is None:
                prediction = ""
            
            codes.append(str(code).strip())
            references.append(str(reference).strip())
            predictions.append(str(prediction).strip())
            loaded += 1
    
    print(f"\nLoading Summary:")
    print(f"   Loaded:  {loaded}")
    print(f"   Skipped: {skipped}")
    
    if loaded == 0:
        print("\nERROR: No valid samples loaded!")
        print("   Please check your JSONL file and field names.")
        print(f"   Expected fields: {code_field}, {reference_field}, {summary_field}")
        sys.exit(1)
    
    return codes, references, predictions


# ================
# SIDE (Java only)
# ================

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def compute_side_score(codes, predictions, checkpoint_path):
    print("\nComputing SIDE score (Java only)...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        model = AutoModel.from_pretrained(checkpoint_path).to(device)
        model.eval()
    except Exception as e:
        print(f"Could not load SIDE model from {checkpoint_path}")
        print(f"   Error: {e}")
        return None
    
    # Compute SIDE scores
    scores = []
    errors = 0
    
    for code, summary in tqdm(zip(codes, predictions), total=len(codes), desc="SIDE"):
        try:
            if not summary.strip():
                scores.append(0.0)
                continue
            
            pair = [code, summary]
            inputs = tokenizer(pair, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            
            with torch.no_grad():
                output = model(**inputs)
            
            embeddings = mean_pooling(output, inputs['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
            scores.append(score)
        except Exception as e:
            errors += 1
            scores.append(0.0)
    
    if errors > 0:
        print(f"SIDE computation errors: {errors}")
    
    side_score = sum(scores) / len(scores) if scores else 0.0
    return side_score



def compute_bleu(predictions, references):
    """Compute BLEU score with error handling"""
    print("\nComputing BLEU...")
    try:
        bleu_scores = []
        for pred, ref in zip(predictions, references):
            if not pred.strip():
                bleu_scores.append(0.0)
            else:
                try:
                    bleu = sacrebleu.sentence_bleu(pred, [ref])
                    bleu_scores.append(bleu.score / 100)
                except:
                    bleu_scores.append(0.0)
        result = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
        print(f"   BLEU: {result:.4f}")
        return result
    except Exception as e:
        print(f" Error: {e}")
        return 0.0


def compute_meteor(predictions, references):
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
            if not pred.strip():
                meteor_scores.append(0.0)
            else:
                try:
                    score = meteor_score([ref.split()], pred.split())
                    meteor_scores.append(score)
                except:
                    meteor_scores.append(0.0)
        result = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0
        print(f"   METEOR: {result:.4f}")
        return result
    except Exception as e:
        print(f" Error: {e}")
        return 0.0


def compute_rouge(predictions, references):
    print("Computing ROUGE...")
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
        
        for pred, ref in zip(predictions, references):
            if not pred.strip():
                rouge1_scores.append(0.0)
                rouge2_scores.append(0.0)
                rougeL_scores.append(0.0)
            else:
                try:
                    scores = scorer.score(ref, pred)
                    rouge1_scores.append(scores['rouge1'].fmeasure)
                    rouge2_scores.append(scores['rouge2'].fmeasure)
                    rougeL_scores.append(scores['rougeL'].fmeasure)
                except:
                    rouge1_scores.append(0.0)
                    rouge2_scores.append(0.0)
                    rougeL_scores.append(0.0)
        
        results = {
            'rouge-1': sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0.0,
            'rouge-2': sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0,
            'rouge-L': sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0.0,
        }
        print(f"   ROUGE-1: {results['rouge-1']:.4f}")
        print(f"   ROUGE-2: {results['rouge-2']:.4f}")
        print(f"   ROUGE-L: {results['rouge-L']:.4f}")
        return results
    except Exception as e:
        print(f"  Error: {e}")
        return {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-L': 0.0}


def compute_chrf(predictions, references):
    """Compute chrF score with error handling"""
    print("Computing chrF...")
    try:
        chrf_scores = []
        for pred, ref in zip(predictions, references):
            if not pred.strip():
                chrf_scores.append(0.0)
            else:
                try:
                    chrf_score = sacrebleu.sentence_chrf(pred, [ref])
                    chrf_scores.append(chrf_score.score / 100)
                except:
                    chrf_scores.append(0.0)
        result = sum(chrf_scores) / len(chrf_scores) if chrf_scores else 0.0
        print(f"   chrF: {result:.4f}")
        return result
    except Exception as e:
        print(f"  Error: {e}")
        return 0.0


def compute_bertscore(predictions, references):
    print("Computing BERTScore...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Using device: {device}")
        
        # Filter out empty predictions/references
        valid_pairs = [(p, r) for p, r in zip(predictions, references) if p.strip() and r.strip()]
        
        if not valid_pairs:
            print("  No valid prediction-reference pairs")
            return {'bertscore_f1': 0.0, 'bertscore_precision': 0.0, 'bertscore_recall': 0.0}
        
        valid_preds, valid_refs = zip(*valid_pairs)

        for model_type in ["bert-base-uncased", "roberta-base", "distilbert-base-uncased"]:
            try:
                P, R, F1 = bert_score(
                    list(valid_preds), 
                    list(valid_refs), 
                    model_type=model_type,
                    device=device,
                    batch_size=8,
                    verbose=False
                )
                results = {
                    'bertscore_f1': F1.mean().item(),
                    'bertscore_precision': P.mean().item(),
                    'bertscore_recall': R.mean().item(),
                }
                print(f"   BERTScore Precision: {results['bertscore_precision']:.4f}")
                print(f"   BERTScore Recall:    {results['bertscore_recall']:.4f}")
                print(f"   BERTScore F1:        {results['bertscore_f1']:.4f}")
                return results
            except Exception as e:
                print(f"  {model_type} failed: {e}")
                continue
        
        print("  All BERTScore models failed")
        return {'bertscore_f1': 0.0, 'bertscore_precision': 0.0, 'bertscore_recall': 0.0}
    except Exception as e:
        print(f"  Error: {e}")
        return {'bertscore_f1': 0.0, 'bertscore_precision': 0.0, 'bertscore_recall': 0.0}

def main():
    args = parse_args()
    
    print("=" * 60)
    print("CODEREVAL EVALUATION SCRIPT (BULLETPROOF VERSION)")
    print("=" * 60)
    print(f"Language:        {args.language}")
    print(f"Dataset:         {args.dataset_path}")
    print(f"Code field:      {args.code_field}")
    print(f"Reference field: {args.reference_field}")
    print(f"Summary field:   {args.summary_field}")
    
    is_valid, issues, valid_count = validate_jsonl(
        args.dataset_path, 
        args.code_field, 
        args.reference_field, 
        args.summary_field
    )
    
    if not is_valid:
        print("\n JSONL validation failed!")
        sys.exit(1)
    
    if args.validate_only:
        print("\n Validation complete. Exiting (--validate_only flag set)")
        sys.exit(0)
    
    # Load data
    print(f"\n Loading data...")
    codes, references, predictions = load_jsonl_data_robust(
        args.dataset_path,
        args.code_field,
        args.reference_field,
        args.summary_field,
        args.max_samples
    )
    
    print(f"\n Loaded {len(predictions)} samples")
    
    print(f"\n Sample (first item):")
    print(f"   Code:       {codes[0][:80]}...")
    print(f"   Reference:  {references[0][:80]}...")
    print(f"   Prediction: {predictions[0][:80]}...")
    
    empty_preds = sum(1 for p in predictions if not p.strip())
    if empty_preds > 0:
        print(f"\n  Warning: {empty_preds}/{len(predictions)} predictions are empty")
    
    print("\n" + "=" * 60)
    print("COMPUTING METRICS")
    print("=" * 60)
    
    results = {}
    
    results['bleu'] = compute_bleu(predictions, references)
    results['meteor'] = compute_meteor(predictions, references)
    
    rouge_results = compute_rouge(predictions, references)
    results.update(rouge_results)
    
    results['chrf'] = compute_chrf(predictions, references)
    
    bertscore_results = compute_bertscore(predictions, references)
    results.update(bertscore_results)
    
    if args.language == 'java':
        print("\n" + "-" * 40)
        side_score = compute_side_score(codes, predictions, args.side_checkpoint)
        results['side_score'] = side_score
        if side_score is not None:
            print(f"   SIDE Score: {side_score:.4f}")
    else:
        results['side_score'] = None

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    output_path = args.output_file if args.output_file.endswith('.txt') else f"{args.output_file}.txt"
    with open(output_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("EVALUATION RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Language:        {args.language}\n")
        f.write(f"Dataset file:    {args.dataset_path}\n")
        f.write(f"Summary field:   {args.summary_field}\n")
        f.write(f"Number of samples: {len(predictions)}\n")
        f.write(f"Empty predictions: {empty_preds}\n")
        f.write("-" * 60 + "\n")
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
        f.write("=" * 60 + "\n")
    
    # Save as JSON
    json_path = output_path.replace('.txt', '.json')
    results_json = {
        'language': args.language,
        'dataset_file': args.dataset_path,
        'summary_field': args.summary_field,
        'num_samples': len(predictions),
        'empty_predictions': empty_preds,
        'metrics': results
    }
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"BLEU:                {results['bleu']:.4f}")
    print(f"METEOR:              {results['meteor']:.4f}")
    print(f"ROUGE-1:             {results['rouge-1']:.4f}")
    print(f"ROUGE-2:             {results['rouge-2']:.4f}")
    print(f"ROUGE-L:             {results['rouge-L']:.4f}")
    print(f"chrF:                {results['chrf']:.4f}")
    print(f"BERTScore F1:        {results['bertscore_f1']:.4f}")
    if results['side_score'] is not None:
        print(f"SIDE Score:          {results['side_score']:.4f}")
    print("=" * 60)
    print(f"\n Results saved to:")
    print(f"   - {output_path}")
    print(f"   - {json_path}")


if __name__ == "__main__":
    main()

# Usage:
#   python evaluate_summarization_metrics.py \
#       --dataset_path path/to/predictions.jsonl \
#       --language java \
#       --summary_field generated_summary \
#       --output_file path/to/results
