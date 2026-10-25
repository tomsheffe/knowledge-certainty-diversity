"""Bootstrap 95% confidence intervals and a permutation null for PCS in each
experiment. PCS is a sum of max(0, .) terms and therefore never negative, so
a small value could be sampling noise; the permutation null (shuffle the
reference agent's confidence labels across items) makes "indistinguishable
from zero" a testable statement.

Bootstrap: cluster bootstrap (resample groups, then all of that group's
items) for Exp 1 (the 12 model pair x benchmark combinations), Exp 3 (10
trios) and Exp 4 (30 participant-LLM pairs). Item-level resampling for Exp 2,
a single trio configuration.

Permutation null: shuffle the reference agent's confidence label across
items (pairs) or the rebel's confidence label across disagreement items
(trios), recompute PCS on the shuffled data. Repeat 2,000 times per
experiment. p = P(PCS_null >= PCS_observed).
"""
import numpy as np
import pandas as pd
from common import (load_bot_bot_duo, load_bot_bot_trio, load_medcom1, load_medcom2,
                    aggregate_questions, aggregate_by_rebel_conf,
                    pair_diversity_gain, trio_diversity_gain)

N_RESAMPLES = 2000
SEED = 0


def bootstrap_pairs(df, n=N_RESAMPLES, seed=SEED):
    rng = np.random.default_rng(seed)
    m = len(df)
    gains = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, m, m)
        gains[i] = pair_diversity_gain(df.iloc[idx], filter_occurrences=0)[0] * 100
    return gains


def bootstrap_pairs_clustered(df, group_col, n=N_RESAMPLES, seed=SEED):
    rng = np.random.default_rng(seed)
    groups = df[group_col].unique()
    ng = len(groups)
    by_group = {g: df[df[group_col] == g] for g in groups}
    gains = np.empty(n)
    for i in range(n):
        chosen = rng.choice(groups, size=ng, replace=True)
        resampled = pd.concat([by_group[g] for g in chosen], ignore_index=True)
        gains[i] = pair_diversity_gain(resampled, filter_occurrences=0)[0] * 100
    return gains


def bootstrap_trios(agg_q, n=N_RESAMPLES, seed=SEED):
    rng = np.random.default_rng(seed)
    m = len(agg_q)
    gains = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, m, m)
        resampled = agg_q.iloc[idx]
        gains[i] = trio_diversity_gain(aggregate_by_rebel_conf(resampled)) * 100
    return gains


def bootstrap_trios_clustered(agg_q, group_col, n=N_RESAMPLES, seed=SEED):
    rng = np.random.default_rng(seed)
    groups = agg_q[group_col].unique()
    ng = len(groups)
    by_group = {g: agg_q[agg_q[group_col] == g] for g in groups}
    gains = np.empty(n)
    for i in range(n):
        chosen = rng.choice(groups, size=ng, replace=True)
        resampled = pd.concat([by_group[g] for g in chosen], ignore_index=True)
        gains[i] = trio_diversity_gain(aggregate_by_rebel_conf(resampled)) * 100
    return gains


def permutation_null_pairs(df, conf_col, n=N_RESAMPLES, seed=SEED):
    rng = np.random.default_rng(seed)
    df = df.reset_index(drop=True)
    conf_vals = df[conf_col].values.copy()
    nulls = np.empty(n)
    for i in range(n):
        shuffled = df.copy()
        shuffled[conf_col] = rng.permutation(conf_vals)
        nulls[i] = pair_diversity_gain(shuffled, filter_occurrences=0)[0] * 100
    return nulls


def permutation_null_trios(agg_q, n=N_RESAMPLES, seed=SEED):
    rng = np.random.default_rng(seed)
    agg_q = agg_q.reset_index(drop=True)
    conf_vals = agg_q['rebel_confidence'].values.copy()
    nulls = np.empty(n)
    for i in range(n):
        shuffled = agg_q.copy()
        shuffled['rebel_confidence'] = rng.permutation(conf_vals)
        nulls[i] = trio_diversity_gain(aggregate_by_rebel_conf(shuffled)) * 100
    return nulls


def run():
    results = {}

    # --- Exp 1: LLM pairs (resampling the 12 model pair x benchmark
    # combinations, the unit used for Exp 1's tests; 2026-10-12) ---
    p1, _ = load_bot_bot_duo()
    obs = pair_diversity_gain(p1, filter_occurrences=0)[0] * 100
    boot = bootstrap_pairs_clustered(p1, 'group')
    null = permutation_null_pairs(p1, 'best_agent_confidence')
    results['exp1'] = dict(obs=obs, boot=boot, null=null, n=len(p1))

    # --- Exp 2: LLM trios (item-level; single trio configuration) ---
    _, parsed2 = load_bot_bot_trio()
    aq2 = aggregate_questions(parsed2)
    obs = trio_diversity_gain(aggregate_by_rebel_conf(aq2)) * 100
    boot = bootstrap_trios(aq2)
    null = permutation_null_trios(aq2)
    results['exp2'] = dict(obs=obs, boot=boot, null=null, n=len(aq2))

    # --- Exp 3: human trios (cluster bootstrap by trio) ---
    from common import load_medcom1_trios
    _, interv3 = load_medcom1_trios()                # trio-level: all members (2026-09-15)
    aq3 = aggregate_questions(interv3)
    obs = trio_diversity_gain(aggregate_by_rebel_conf(aq3)) * 100
    boot = bootstrap_trios_clustered(aq3, 'group_num')
    null = permutation_null_trios(aq3)
    results['exp3'] = dict(obs=obs, boot=boot, null=null, n=len(aq3))

    # --- Exp 4: human-LLM pairs (cluster bootstrap by participant) ---
    d4 = load_medcom2()
    obs = pair_diversity_gain(d4, filter_occurrences=0)[0] * 100
    boot = bootstrap_pairs_clustered(d4, 'participant_id')
    null = permutation_null_pairs(d4, 'best_agent_confidence')
    results['exp4'] = dict(obs=obs, boot=boot, null=null, n=len(d4))

    return results


if __name__ == '__main__':
    r = run()
    print("=== PCS bootstrap CIs and permutation null ===")
    print(f"{N_RESAMPLES} resamples per experiment.\n")
    labels = {'exp1': 'Exp 1 (LLM pairs)', 'exp2': 'Exp 2 (LLM trios)',
             'exp3': 'Exp 3 (human trios)', 'exp4': 'Exp 4 (human-LLM pairs)'}
    for key, label in labels.items():
        d = r[key]
        boot_lo, boot_hi = np.percentile(d['boot'], [2.5, 97.5])
        null_p = (d['null'] >= d['obs']).mean()
        print(f"{label}  (n={d['n']} items)")
        print(f"  Observed PCS: {d['obs']:.2f}%")
        print(f"  Bootstrap 95% CI: [{boot_lo:.2f}%, {boot_hi:.2f}%]")
        print(f"  Permutation null: mean={d['null'].mean():.2f}%, "
              f"95th pct={np.percentile(d['null'],95):.2f}%, p={null_p:.4f}")
        print()
