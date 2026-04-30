#!/usr/bin/env python
"""
Re-evaluate three plant-related patent-family classifiers on the PINTO 2026-03-03 gold test.
Uses pre-selected thresholds from validation split.

Input:  metadata/gold_test_pinto_2026-03-03_reproducibility.parquet  (balanced: 97 pos / 97 neg)
   or:  metadata/gold_test_pinto_2026-03-03_positives_only.parquet   (97 positives, recall-only)

Output: metrics CSV and prediction probabilities CSV.
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument('--positives-only', action='store_true',
                    help='Evaluate on positives only (recall + probability distribution)')
args = parser.parse_args()

base = Path(__file__).resolve().parents[1]

if args.positives_only:
    parquet_path = base / 'metadata/gold_test_pinto_2026-03-03_positives_only.parquet'
    tag = 'pinto_2026-03-03_pos_only'
else:
    parquet_path = base / 'metadata/gold_test_pinto_2026-03-03_reproducibility.parquet'
    tag = 'pinto_2026-03-03'

# Load gold test
gold_df = pd.read_parquet(parquet_path)
has_title = gold_df['appln_title_en_final'].notna() & (gold_df['appln_title_en_final'].str.len() > 0)
has_abstract = gold_df['appln_abstract_en_final'].notna() & (gold_df['appln_abstract_en_final'].str.len() > 0)
full_text = gold_df[has_title & has_abstract].copy()
n_pos = (full_text['gold_label'] == 1).sum()
n_neg = (full_text['gold_label'] == 0).sum()
print(f'Evaluating {len(full_text)} families ({n_pos} pos, {n_neg} neg)')
if args.positives_only:
    print('  Mode: positives-only (recall + probability distribution)')

# Create input text
full_text['input_text'] = full_text['appln_title_en_final'] + '\n\n' + full_text['appln_abstract_en_final']

# Configuration — fine-tuned January models and their validation-selected thresholds
thresholds = {'bert-base-uncased': 0.038927, 'bert-for-patents': 0.994587, 'PaECTER': 0.992003}
checkpoints = {
    'bert-base-uncased': base / 'models/bert_base_uncased_2026-01-28/hf_export',
    'bert-for-patents': base / 'models/bert_for_patents_2026-01-29_safe/hf_export',
    'PaECTER': base / 'models/paecter_2026-01-29_rerun/hf_export',
}

results = []
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

for model_name in ['bert-base-uncased', 'bert-for-patents', 'PaECTER']:
    print(f'\nProcessing {model_name}...')
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoints[model_name]))
    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoints[model_name])).to(device).eval()

    probs = []
    with torch.no_grad():
        for i, text in enumerate(full_text['input_text']):
            if (i+1) % 40 == 0:
                print(f'  {i+1}/{len(full_text)}')
            inp = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True).to(device)
            out = model(**inp)
            p = torch.softmax(out.logits, dim=1)[0, 1].item()
            probs.append(p)

    y_true = full_text['gold_label'].values
    y_pred_proba = np.array(probs)
    threshold = thresholds[model_name]
    y_pred = (y_pred_proba >= threshold).astype(int)

    tp = ((y_pred==1) & (y_true==1)).sum()
    fp = ((y_pred==1) & (y_true==0)).sum()
    tn = ((y_pred==0) & (y_true==0)).sum()
    fn = ((y_pred==0) & (y_true==1)).sum()

    prec = tp/(tp+fp) if (tp+fp)>0 else float('nan')
    rec = tp/(tp+fn) if (tp+fn)>0 else float('nan')
    f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else float('nan')
    acc = (tp+tn)/len(y_true)

    row = {
        'Model': model_name,
        'Threshold': f'{threshold:.6f}',
        'N': len(full_text),
        'N_pos': n_pos,
        'N_neg': n_neg,
        'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn),
        'Prec': f'{prec:.3f}' if not np.isnan(prec) else 'N/A',
        'Rec': f'{rec:.3f}',
        'F1': f'{f1:.3f}' if not np.isnan(f1) else 'N/A',
        'Acc': f'{acc:.3f}',
        'Mean_prob_pos': f'{y_pred_proba[y_true==1].mean():.4f}',
    }

    if n_neg > 0:
        pr, re, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(re, pr)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        row['PR-AUC'] = f'{pr_auc:.3f}'
        row['ROC-AUC'] = f'{roc_auc:.3f}'
        row['Mean_prob_neg'] = f'{y_pred_proba[y_true==0].mean():.4f}'
    else:
        row['PR-AUC'] = 'N/A'
        row['ROC-AUC'] = 'N/A'
        row['Mean_prob_neg'] = 'N/A'

    results.append(row)
    print(f'  Rec: {rec:.3f} | Prec: {row["Prec"]} | F1: {row["F1"]} | PR-AUC: {row["PR-AUC"]} | ROC-AUC: {row["ROC-AUC"]}')

    full_text[f'{model_name}_prob'] = probs
    del model, tokenizer

# Save results
df = pd.DataFrame(results)
df.to_csv(base / f'metadata/gold_test_{tag}_results.csv', index=False)
full_text[['docdb_family_id', 'gold_label', 'bert-base-uncased_prob', 'bert-for-patents_prob', 'PaECTER_prob']].to_csv(
    base / f'metadata/gold_test_{tag}_predictions.csv', index=False)

print('\n' + '='*60)
print('FINAL RESULTS')
print('='*60)
print(df.to_string(index=False))
print(f'\nSaved to metadata/gold_test_{tag}_results.csv')
print(f'         metadata/gold_test_{tag}_predictions.csv')
