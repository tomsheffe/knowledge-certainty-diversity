"""Analyse additional question-set replications with the functions used for
Experiments 1–2.

Reads ../data/rerun_results/processed/ and reports, per run set and model
configuration: item-pooled pre/post accuracy, PCS with a 2,000-resample
item bootstrap and a 2,000-shuffle permutation null, and the initial
majority-vote accuracy for trios.

Usage from the repository root:
    python analysis/rerun_analysis.py [--root <processed directory>]
Results are printed to standard output.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from common import (_assign_duo_agents_for_duo, pair_diversity_gain, parse_bot_bot_bot,
                    aggregate_questions, aggregate_by_rebel_conf, trio_diversity_gain,
                    aggregate_bot_group_data)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.abspath(os.path.join(HERE, '..', 'data', 'rerun_results', 'processed'))
N_RESAMPLES = 2000


def _prep_duo(df):
    conf_cols = [c for c in df.columns if 'confidence' in c]
    df = df.copy()
    for c in conf_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce') / 20
    parsed, _ = _assign_duo_agents_for_duo(df)
    return parsed


def _boot_null_pairs(parsed, rng):
    obs, _ = pair_diversity_gain(parsed)
    boots = [pair_diversity_gain(parsed.sample(len(parsed), replace=True, random_state=int(rng.integers(1e9))))[0] for _ in range(N_RESAMPLES)]
    nulls = []
    for _ in range(N_RESAMPLES):
        p = parsed.copy(); p['best_agent_confidence'] = rng.permutation(p['best_agent_confidence'].values); nulls.append(pair_diversity_gain(p)[0])
    nulls = np.array(nulls)
    return obs * 100, np.percentile(boots, [2.5, 97.5]) * 100, nulls.mean() * 100, (nulls >= obs).mean()


def analyse_duo(path, rng):
    raw = pd.read_csv(path)
    print(f'\n=== {os.path.basename(path)} ===')
    for (models, ds), g in raw.groupby(['models', 'dataset_name']):
        parsed = _prep_duo(g)
        b_pre, b_post = parsed['best_agent_correct_before'].mean() * 100, parsed['best_agent_correct_after'].astype(float).mean() * 100
        w_pre, w_post = parsed['second_agent_correct_before'].mean() * 100, parsed['second_agent_correct_after'].astype(float).mean() * 100
        pcs, ci, null_mean, p = _boot_null_pairs(parsed, rng)
        print(f'{models:>16s} {ds:<12s} n={len(parsed):3d} | better {b_pre:5.1f}->{b_post:5.1f}  worse {w_pre:5.1f}->{w_post:5.1f} | '
              f'PCS {pcs:4.2f} pp [{ci[0]:.1f}, {ci[1]:.1f}] null {null_mean:.2f} p={p:.3f}')
    # pooled over datasets per model pair (as Fig. 2A pools the four benchmarks)
    for models, g in raw.groupby('models'):
        if g['dataset_name'].nunique() > 1:
            parsed = _prep_duo(g); pcs, ci, null_mean, p = _boot_null_pairs(parsed, rng)
            print(f'{models:>16s} {"POOLED":<12s} n={len(parsed):3d} | PCS {pcs:4.2f} pp [{ci[0]:.1f}, {ci[1]:.1f}] null {null_mean:.2f} p={p:.3f}')


def analyse_trio(path, rng):
    raw = pd.read_csv(path)
    print(f'\n=== {os.path.basename(path)} ===')
    groups = [(ds, g) for ds, g in raw.groupby('dataset_name')]
    if raw['dataset_name'].nunique() > 1:
        groups.append(('POOLED', raw))
    for ds, g in groups:
        g = g.reset_index(drop=True)
        parsed = parse_bot_bot_bot(g)
        maj = aggregate_bot_group_data(g)
        agg = aggregate_questions(parsed)
        three_way = parsed.groupby('question_number')['ans_start'].nunique()
        agg = agg[~agg['question_number'].isin(three_way[three_way == 3].index)]   # no coalition on three-way splits
        obs = trio_diversity_gain(aggregate_by_rebel_conf(agg)) * 100
        boots = [trio_diversity_gain(aggregate_by_rebel_conf(agg.sample(len(agg), replace=True, random_state=int(rng.integers(1e9))))) * 100 for _ in range(N_RESAMPLES)]
        nulls = []
        for _ in range(N_RESAMPLES):
            p = agg.copy(); p['rebel_confidence'] = rng.permutation(p['rebel_confidence'].values); nulls.append(trio_diversity_gain(aggregate_by_rebel_conf(p)) * 100)
        nulls = np.array(nulls)
        ranks = parsed.groupby('subject_rank').agg(pre=('ans_start_correct', 'mean'), post=('ans_end_correct', lambda x: pd.to_numeric(x, errors='coerce').mean()))
        rank_txt = '  '.join(f'{r} {v.pre*100:5.1f}->{v.post*100:5.1f}' for r, v in ranks.iterrows())
        print(f'{ds:<12s} n={len(g):3d} (coalition items {len(agg)}) | majority pre {maj.premajority.mean()*100:5.1f} post {maj.postmajority.mean()*100:5.1f} | {rank_txt} | '
              f'PCS {obs:4.2f} pp [{np.percentile(boots,2.5):.1f}, {np.percentile(boots,97.5):.1f}] null {nulls.mean():.2f} p={(nulls>=obs).mean():.3f}')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', default=DEFAULT_ROOT); args = ap.parse_args()
    rng = np.random.default_rng(0)
    files = sorted(glob.glob(os.path.join(args.root, '*.csv')))
    if not files:
        print(f'no processed files in {args.root}; check the packaged data/rerun_results/processed directory or --root'); return
    for f in files:
        (analyse_duo if f.endswith('_duo_per_group.csv') else analyse_trio)(f, rng)


if __name__ == '__main__':
    main()
