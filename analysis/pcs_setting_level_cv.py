"""Exploratory setting-level cross-validation and shared item preparation.

The reported group-level predictive analysis is in group_level_regression_cv.py.
Table 2 imports preparation functions from this module; the exploratory
cross-validation results here are not the paper's reported predictive test.
"""
import os
import numpy as np
import pandas as pd

from common import (load_bot_bot_duo, load_bot_bot_trio, load_medcom1_trios,
                    load_medcom2, aggregate_questions)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, '..', 'outputs')

K_FOLDS = 5
N_REP = 200
RNG_SEED = 0

# LLM per-item accuracy is the fraction of the five self-consistency samples
# that were correct. The observed pre/post change uses the same definition as
# Table 2's "Post-discussion change" row, so the two are directly comparable:
# item-pooled accuracy for pairs, and trio mean post vs. pre-discussion
# majority for trios.
_majority = lambda s: (pd.to_numeric(pd.Series(s).astype(float)) > 0.5).astype(int)


def pcs(conf, ref_acc, partner_acc, has_coalition=None):
    """PCS = sum_c (n_c/N) * max(0, a_P(c) - a_R(c)), in percentage points.

    N is every item in the set. Items with no coalition (all three members
    answering differently) have no rebel/coalition pair, so they are excluded
    from the per-level table but stay in N: they contribute no gain without
    being removed from the denominator."""
    df = pd.DataFrame(dict(c=conf, r=ref_acc, p=partner_acc))
    if has_coalition is not None:
        df['ok'] = np.asarray(has_coalition, dtype=bool)
    else:
        df['ok'] = True
    n_total = len(df.dropna(subset=['c', 'r', 'p']))
    df = df[df['ok']].dropna(subset=['c', 'r', 'p'])
    if n_total == 0 or len(df) == 0:
        return np.nan
    g = df.groupby('c').agg(n=('r', 'size'), a_r=('r', 'mean'), a_p=('p', 'mean'))
    return float((g['n'] / n_total * np.maximum(0.0, g['a_p'] - g['a_r'])).sum() * 100)


def _prep_pairs(load):
    d = load()
    return pd.DataFrame(dict(
        conf=d['best_agent_confidence'].astype(float).values,
        ref=pd.to_numeric(d['best_agent_correct_before'].astype(float)).values,
        partner=pd.to_numeric(d['second_agent_correct_before'].astype(float)).values,
        gain=(pd.to_numeric(d['best_agent_correct_after'].astype(float))
              - pd.to_numeric(d['best_agent_correct_before'].astype(float))).values,
        has_coalition=True))


def _prep_trios(load_subjects):
    s = load_subjects()
    a = aggregate_questions(s)
    post = (s.groupby(['group_num', 'question_number'])
             .agg(post=('ans_end_correct', 'mean')).reset_index())
    a = a.merge(post, on=['group_num', 'question_number'], how='left')
    a['maj'] = _majority(a['coalition_accuracy'])
    return pd.DataFrame(dict(
        conf=a['rebel_confidence'].astype(float).values,
        ref=pd.to_numeric(a['coalition_accuracy'].astype(float)).values,
        partner=pd.to_numeric(a['rebel_accuracy'].astype(float)).values,
        has_coalition=a['has_coalition'].astype(bool).values,
        gain=(a['post'].astype(float) - a['maj'].astype(float)).values))


SETTINGS = [
    ('Exp 1: LLM-LLM pairs', lambda: _prep_pairs(lambda: load_bot_bot_duo()[0])),
    ('Exp 2: LLM trios', lambda: _prep_trios(lambda: load_bot_bot_trio()[1])),
    ('Exp 3: human trios', lambda: _prep_trios(lambda: load_medcom1_trios()[1])),
    ('Exp 4: human-LLM pairs', lambda: _prep_pairs(load_medcom2)),
]


def run():
    rng = np.random.default_rng(RNG_SEED)
    rows = {}
    for label, prep in SETTINGS:
        df = prep().dropna(subset=['conf'])
        n = len(df)
        train_pcs, held_out_gain = [], []
        for _ in range(N_REP):
            fold = rng.permutation(n) % K_FOLDS
            for k in range(K_FOLDS):
                tr, te = df.iloc[fold != k], df.iloc[fold == k]
                train_pcs.append(pcs(tr['conf'], tr['ref'], tr['partner'], tr['has_coalition']))
                held_out_gain.append(te['gain'].mean() * 100)
        p, g = np.array(train_pcs), np.array(held_out_gain)
        rows[label] = dict(n_items=n, pcs=p.mean(), pcs_lo=np.percentile(p, 2.5),
                           pcs_hi=np.percentile(p, 97.5), gain=g.mean(),
                           # Reference agent's pre-discussion accuracy (better
                           # agent in pairs, coalition in trios): Table 2's
                           # "Reference pre-discussion accuracy" row.
                           baseline=float(df['ref'].mean() * 100))
    return rows


if __name__ == '__main__':
    r = run()
    df = pd.DataFrame(r).T
    df.index.name = ''
    os.makedirs(OUT_DIR, exist_ok=True)
    lines = [f"Setting-level {K_FOLDS}-fold cross-validation of PCS "
             f"({N_REP} repetitions, seed {RNG_SEED})", ""]
    lines.append(f"{'setting':26s} {'n':>5s} {'PCS (training folds)':>26s} {'gain (held-out fold)':>22s}")
    for k, v in r.items():
        lines.append(f"{k:26s} {v['n_items']:5.0f} "
                     f"{v['pcs']:8.2f} [{v['pcs_lo']:5.2f}, {v['pcs_hi']:5.2f}] pp"
                     f"{v['gain']:+18.2f} pp")
    text = '\n'.join(lines)
    print(text)
    with open(os.path.join(OUT_DIR, 'pcs_setting_level_cv_output.txt'), 'w') as f:
        f.write(text + '\n')
    df.to_csv(os.path.join(OUT_DIR, 'pcs_setting_level_cv.csv'))
    print("\nwritten to outputs/pcs_setting_level_cv_output.txt and .csv")
