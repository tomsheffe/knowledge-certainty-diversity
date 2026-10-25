"""Fig. S6 — confidence calibration: higher self-rated (humans) or
self-consistency (LLMs) confidence predicts higher accuracy.

The paper states this qualitatively (no specific correlation value cited in
the main text), so this script checks the qualitative claim — a significant
positive confidence-accuracy relationship — for every agent type in every
experiment, plus a per-level accuracy table (the underlying data for the
Fig. S6 panels).
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from common import load_medcom1, load_medcom2, load_bot_bot_duo, load_bot_bot_trio


def per_level_accuracy(conf, correct, levels=(1, 2, 3, 4, 5)):
    d = pd.DataFrame({'conf': np.round(pd.to_numeric(conf)).clip(min(levels), max(levels)),
                      'correct': pd.to_numeric(correct)}).dropna()
    return d.groupby('conf')['correct'].agg(['mean', 'count'])


def run():
    results = {}

    _, subj3 = load_medcom1(0.62)
    results['human_exp3'] = (spearmanr(subj3['conf_val_start'], subj3['ans_start_correct']),
                             per_level_accuracy(subj3['conf_val_start'], subj3['ans_start_correct']))

    # S6 retains the original broader calibration sample across both arms,
    # independently of the main discussion-outcome population. Confidence
    # is recovered from the saved option counts; unavailable values are NaN.
    from pathlib import Path
    df4 = pd.read_csv(Path(__file__).resolve().parents[1] / 'derived_data' /
                      's6_pediatrics_calibration.csv').rename(columns={
        'human_confidence': 'doc.init_question_confidence',
        'human_correct': 'user_before_correct',
        'model_confidence': 'doc.answer_confidence_level',
        'model_correct': 'bot_before_correct'})
    results['human_exp4'] = (spearmanr(df4['doc.init_question_confidence'], df4['user_before_correct']),
                             per_level_accuracy(df4['doc.init_question_confidence'], df4['user_before_correct']))
    results['llm_exp4'] = (spearmanr(df4['doc.answer_confidence_level'], df4['bot_before_correct'], nan_policy='omit'),
                           per_level_accuracy(df4['doc.answer_confidence_level'], df4['bot_before_correct'], levels=(0, 1, 2, 3, 4, 5)))

    parsed1, _ = load_bot_bot_duo()
    allconf = pd.concat([parsed1['best_agent_confidence'], parsed1['second_agent_confidence']])
    allcorr = pd.concat([parsed1['best_agent_correct_before'], parsed1['second_agent_correct_before']])
    results['llm_exp1'] = (spearmanr(allconf, allcorr), per_level_accuracy(allconf, allcorr))

    _, parsed2 = load_bot_bot_trio()
    results['llm_exp2'] = (spearmanr(parsed2['conf_val_start'], parsed2['ans_start_correct']),
                           per_level_accuracy(parsed2['conf_val_start'], parsed2['ans_start_correct']))

    return results


if __name__ == '__main__':
    r = run()
    print("=== Fig. S6: confidence calibration (qualitative claim: r > 0, p < 0.05 everywhere) ===\n")
    labels = {
        'human_exp3': 'Human (Exp 3, trios)', 'human_exp4': 'Human (Exp 4, pairs)',
        'llm_exp4': 'LLM/GPT-4 Turbo (Exp 4, pairs)', 'llm_exp1': 'LLM (Exp 1, pairs)',
        'llm_exp2': 'LLM (Exp 2, trios)',
    }
    for key, label in labels.items():
        (rho, p), table = r[key]
        print(f"{label}: Spearman r={rho:.3f}, p={p:.2e}, n={int(table['count'].sum())}")
        print(table.rename(columns={'mean': 'accuracy', 'count': 'n'}).assign(
            accuracy=lambda d: (d['accuracy'] * 100).round(1)).to_string())
        print()
