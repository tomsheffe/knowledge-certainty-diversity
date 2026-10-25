"""Fig. 5 — PCS and observed gain for each group (pair or trio), all four settings.

Uses the same PCS and gain definitions as Table 2
(reproduction/analysis/pcs_setting_level_cv.py), computed within each group:

  Exp 1  LLM pairs      12 units = 3 model pairs x 4 benchmarks (50 items each;
                        the reference agent is chosen per unit, as in the paper)
  Exp 2  LLM trio       4 units  = the single trio on each benchmark (50 items)
  Exp 3  human trios    10 trios (42 items each)
  Exp 4  human-LLM      30 participant-LLM pairs (their discussed items, ~19 each)

Gain: pairs, reference agent's post - pre accuracy; trios, the trio's majority
vote after minus before discussion. Both in percentage points.

Writes ../outputs/group_level_pcs_vs_gain.csv (read by
group_level_regression_cv.py) and ../outputs/figures/figure_5.png.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, '..', 'outputs')
from common import (load_bot_bot_duo, load_bot_bot_trio, load_exp3_human_trios,
                    load_exp4_human_llm, aggregate_questions,
                    aggregate_bot_group_data)
from pcs_setting_level_cv import pcs


def pairs(d, unit_col, label):
    rows = []
    for unit, g in d.groupby(unit_col):
        pre = g['best_agent_correct_before'].astype(float)
        post = g['best_agent_correct_after'].astype(float)
        rows.append(dict(setting=label, unit=unit, n_items=len(g),
                         pcs=pcs(g['best_agent_confidence'].astype(float), pre,
                                 g['second_agent_correct_before'].astype(float)),
                         gain=(post.mean() - pre.mean()) * 100))
    return rows


def trios(subjects, label, unit_of=None, maj_of=None):
    """maj_of: pre-discussion majority-vote correctness per question_number.
    Given for the LLM trio (majority of the three models' modal answers, as in
    Table 2); for human trios each answer is 0/1, so the coalition's accuracy
    is already the majority vote."""
    a = aggregate_questions(subjects)
    # Majority vote after discussion (at least two of three members right),
    # the same quantity as the pre-discussion majority (2026-10-12).
    post = (subjects.groupby(['group_num', 'question_number'])
            .agg(post=('ans_end_correct', lambda x: float(x.mean() > 0.5))).reset_index())
    a = a.merge(post, on=['group_num', 'question_number'], how='left')
    if maj_of is None:
        a['maj'] = (a['coalition_accuracy'].astype(float) > 0.5).astype(float)
    else:
        a['maj'] = a['question_number'].map(maj_of).astype(float)
    a['unit'] = a['group_num'] if unit_of is None else a['question_number'].map(unit_of)
    rows = []
    for unit, g in a.groupby('unit'):
        rows.append(dict(setting=label, unit=unit, n_items=len(g),
                         pcs=pcs(g['rebel_confidence'].astype(float),
                                 g['coalition_accuracy'].astype(float),
                                 g['rebel_accuracy'].astype(float),
                                 g['has_coalition'].astype(bool)),
                         gain=(g['post'].astype(float) - g['maj']).mean() * 100))
    return rows


def build():
    """Returns (per-group rows, pooled setting rows). The pooled row computes
    PCS and gain once over all items of the setting, as in Table 2; it is not
    the mean of the group values, because PCS fitted on few items is biased
    upward (it is an in-sample maximum)."""
    rows, pooled = [], []
    d1, _ = load_bot_bot_duo()
    rows += pairs(d1, 'group', 'Exp 1: LLM pairs')
    # Exp 1's reference agent is chosen per pair x benchmark, so pooling keeps those choices.
    pooled += pairs(d1.assign(all_items=1), 'all_items', 'Exp 1: LLM pairs')
    raw2, s2 = load_bot_bot_trio()
    # question_number in the parsed trio data is the row index of the raw file.
    bench = dict(enumerate(raw2['dataset_name']))
    assert set(s2['question_number'].unique()) <= set(bench)
    maj2 = dict(enumerate(aggregate_bot_group_data(raw2)['premajority']))
    rows += trios(s2, 'Exp 2: LLM trio', unit_of=bench, maj_of=maj2)
    pooled += trios(s2, 'Exp 2: LLM trio', maj_of=maj2)
    _, s3 = load_exp3_human_trios()
    rows += trios(s3, 'Exp 3: human trios')
    pooled += trios(s3, 'Exp 3: human trios', unit_of={q: 'all' for q in s3['question_number'].unique()})
    d4 = load_exp4_human_llm()
    rows += pairs(d4, 'participant_id', 'Exp 4: human-LLM pairs')
    pooled += pairs(d4.assign(all_items=1), 'all_items', 'Exp 4: human-LLM pairs')
    return pd.DataFrame(rows), pd.DataFrame(pooled)


STYLE = {  # marker, colour
    'Exp 1: LLM pairs': ('o', '#4C72B0'),
    'Exp 2: LLM trio': ('^', '#8172B2'),
    'Exp 3: human trios': ('s', '#DD8452'),
    'Exp 4: human-LLM pairs': ('D', '#55A868'),
}


def plot(df, pooled, path):
    fig, ax = plt.subplots(figsize=(7.5, 6.7))
    for label, (m, c) in STYLE.items():
        g = df[df['setting'] == label]
        ax.scatter(g['pcs'], g['gain'], marker=m, s=45, color=c, alpha=0.6,
                   edgecolor='white', linewidth=0.5,
                   label=f"{label} (n = {len(g)} units)")
        # Setting-level value, pooled over all items (Table 2 definition).
        p = pooled[pooled['setting'] == label].iloc[0]
        ax.scatter(p['pcs'], p['gain'], marker=m, s=220, color=c,
                   edgecolor='black', linewidth=1.2, zorder=5)
    ax.axhline(0, color='grey', lw=0.8, ls='--')
    # OLS fit across all groups (see group_level_regression_cv.py).
    slope, intercept = np.polyfit(df['pcs'], df['gain'], 1)
    xs = np.array([0, df['pcs'].max()])
    ax.plot(xs, intercept + slope * xs, color='black', lw=1.2,
            label=f'OLS fit, all groups (slope {slope:.2f})')
    ax.set_xlabel('Potential Conversation Synergy of the group (pp)')
    ax.set_ylabel('Observed gain after discussion (pp)')
    ax.legend(frameon=False, fontsize=9, loc='lower left',
              bbox_to_anchor=(0, 1.02), borderaxespad=0, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


if __name__ == '__main__':
    df, pooled = build()
    print("=== Fig. 5: group-level PCS and gain ===\nPooled over all items of each setting:")
    print(pooled[['setting', 'n_items', 'pcs', 'gain']].round(2).to_string(index=False))
    print()
    os.makedirs(os.path.join(OUT_DIR, 'figures'), exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, 'group_level_pcs_vs_gain.csv'), index=False)
    plot(df, pooled, os.path.join(OUT_DIR, 'figures', 'figure_5.png'))
    print("written to outputs/group_level_pcs_vs_gain.csv and outputs/figures/figure_5.png")
    print(df.groupby('setting').agg(units=('unit', 'size'), items=('n_items', 'median'),
                                    pcs_mean=('pcs', 'mean'), pcs_sd=('pcs', 'std'),
                                    gain_mean=('gain', 'mean'), gain_sd=('gain', 'std')).round(2))
    print()
    for name, sub in [('all units', df),
                      ('human settings only (Exp 3 + 4)', df[df.setting.str.startswith(('Exp 3', 'Exp 4'))]),
                      ('Exp 4 only', df[df.setting.str.startswith('Exp 4')]),
                      ('Exp 3 only', df[df.setting.str.startswith('Exp 3')])]:
        r, p = pearsonr(sub['pcs'], sub['gain'])
        rho, ps = spearmanr(sub['pcs'], sub['gain'])
        print(f"{name:32s} n = {len(sub):2d}: Pearson r = {r:+.2f} (p = {p:.3f}); "
              f"Spearman rho = {rho:+.2f} (p = {ps:.3f})")
