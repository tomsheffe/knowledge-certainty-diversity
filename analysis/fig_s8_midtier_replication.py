"""Fig. S8 — replication with "a second set of widely used, mid-tier LLMs on
an extended question pool."

Data: ../data/fig_s8_midtier/ contains the 3-pair x 3-benchmark sweep
of GPT-3.5-turbo, Claude 3.5 Haiku, and Gemini 1.5 Flash, with 300 items
per benchmark (MedMCQA, MMLU, USMLE).

These raw per-iteration files don't carry precomputed majority_vote_{agent}
/ majority_vote_confidence_{agent} columns, so this script derives them via
common.compute_majority_vote_columns (mode of the 5 self-consistency
samples; confidence = agreement % with the mode, as defined in Methods).

Reports pre/post accuracy for the initially better and worse model in each
pair, paired tests across the 9 pair x benchmark cells, and PCS.
"""
import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
from common import DATA_DIR, compute_majority_vote_columns, _assign_duo_agents_for_duo, pair_diversity_gain

S8_DIR = os.path.join(DATA_DIR, 'fig_s8_midtier')

PAIRS = {
    'gpt35_gemini': ('gpt-3.5-turbo', 'gemini-1.5-flash-002'),
    'gpt35_haiku': ('gpt-3.5-turbo', 'claude-3-5-haiku'),
    'haiku_gemini': ('claude-3-5-haiku', 'gemini-1.5-flash-002'),
}
DATASETS = ['medmcqa', 'mmlu', 'usmle']


def load_one(pair_tag, dataset_tag):
    matches = glob.glob(os.path.join(S8_DIR, f'{pair_tag}_{dataset_tag}_*.csv'))
    assert len(matches) == 1, f"expected 1 file for {pair_tag}/{dataset_tag}, found {len(matches)}"
    df = pd.read_csv(matches[0])
    df = compute_majority_vote_columns(df)
    df['group'] = f'{pair_tag}_{dataset_tag}'  # these raw files are single-pair; no 'group' column to split on
    conf_cols = [c for c in df.columns if 'confidence' in c]
    for c in conf_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce') / 20  # 0-100 -> 0-5
    parsed, _ = _assign_duo_agents_for_duo(df)
    parsed['pair_tag'] = pair_tag
    parsed['dataset_tag'] = dataset_tag
    return parsed


def run():
    all_parsed = []
    for pair_tag in PAIRS:
        for dataset_tag in DATASETS:
            all_parsed.append(load_one(pair_tag, dataset_tag))
    combined = pd.concat(all_parsed, ignore_index=True)

    results = {'n_items': len(combined)}
    results['better_pre'] = combined['best_agent_correct_before'].mean() * 100
    results['better_post'] = combined['best_agent_correct_after'].mean() * 100
    results['worse_pre'] = combined['second_agent_correct_before'].mean() * 100
    results['worse_post'] = combined['second_agent_correct_after'].mean() * 100

    # Per-(pair, dataset) means -> SD and paired t-tests across the 9 cells
    agg = combined.groupby(['pair_tag', 'dataset_tag']).agg(
        best_pre=('best_agent_correct_before', 'mean'), best_post=('best_agent_correct_after', 'mean'),
        second_pre=('second_agent_correct_before', 'mean'), second_post=('second_agent_correct_after', 'mean'),
    ) * 100
    results['n_cells'] = len(agg)
    for k in ['best_pre', 'best_post', 'second_pre', 'second_post']:
        results[f'sd_{k}'] = agg[k].std(ddof=1)
    t_b, p_b = ttest_rel(agg['best_post'], agg['best_pre'])
    t_s, p_s = ttest_rel(agg['second_post'], agg['second_pre'])
    results['t_better'], results['p_better'] = t_b, p_b
    results['t_worse'], results['p_worse'] = t_s, p_s

    gain, agg_div = pair_diversity_gain(combined, filter_occurrences=0)
    results['pcs'] = gain * 100
    results['per_cell'] = agg

    return results


if __name__ == '__main__':
    r = run()
    print("=== Fig. S8: mid-tier model replication ===")
    print(f"Pairs: {list(PAIRS.values())}")
    print(f"n_items pooled: {r['n_items']} (9 pair x dataset cells, 300 items each)\n")
    print(f"Better agent: {r['better_pre']:.1f}% (SD {r['sd_best_pre']:.2f}) -> "
          f"{r['better_post']:.1f}% (SD {r['sd_best_post']:.2f}), "
          f"t({r['n_cells']-1})={r['t_better']:.2f}, p={r['p_better']:.3f}")
    print(f"Worse agent:  {r['worse_pre']:.1f}% (SD {r['sd_second_pre']:.2f}) -> "
          f"{r['worse_post']:.1f}% (SD {r['sd_second_post']:.2f}), "
          f"t({r['n_cells']-1})={r['t_worse']:.2f}, p={r['p_worse']:.3f}")
    print(f"PCS: {r['pcs']:.2f}%")
    print("\nPer pair x dataset cell:")
    print(r['per_cell'].round(1))
    print("\n(Qualitative claim to check: better agent's gain stays small/non-significant, "
          "PCS stays low -- same 'no-gain' pattern as the flagship-model Experiment 1.)")
