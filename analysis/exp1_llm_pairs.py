"""exp1 llm pairs: compute results from the packaged inputs.
"""
import numpy as np
from scipy.stats import ttest_rel
from common import load_bot_bot_duo, pair_diversity_gain


def run():
    results = {}
    parsed, per_group = load_bot_bot_duo()

    # Pooled bar heights (Fig 2C) — item-level means
    results['better_pre_pooled'] = parsed['best_agent_correct_before'].mean() * 100
    results['better_post_pooled'] = parsed['best_agent_correct_after'].mean() * 100
    results['worse_pre_pooled'] = parsed['second_agent_correct_before'].mean() * 100
    results['worse_post_pooled'] = parsed['second_agent_correct_after'].mean() * 100

    # Per-unit means -> SD and paired t-tests across the 12 combinations of
    # model pair and benchmark ('group'). Decided 2026-10-12, as in the
    # original submission; the reference agent is chosen at this level too.
    agg = parsed.groupby('group').agg(
        best_pre=('best_agent_correct_before', 'mean'),
        best_post=('best_agent_correct_after', 'mean'),
        second_pre=('second_agent_correct_before', 'mean'),
        second_post=('second_agent_correct_after', 'mean'),
    ) * 100
    results['n_units'] = len(agg)
    for k in ['best_pre', 'best_post', 'second_pre', 'second_post']:
        results[f'sd_{k}'] = agg[k].std(ddof=1)
    t_b, p_b2 = ttest_rel(agg['best_post'], agg['best_pre'])
    t_s, p_s2 = ttest_rel(agg['second_post'], agg['second_pre'])
    results['t_better'], results['p_better'] = t_b, p_b2      # two-sided
    results['t_worse'], results['p_worse'] = t_s, p_s2

    # PCS at several minimum confidence-bin occurrence thresholds
    for occ in [0, 2, 5]:
        gain, agg_div = pair_diversity_gain(parsed, filter_occurrences=occ)
        results[f'pcs_pairs_occ{occ}'] = gain * 100
    results['pcs_table'] = agg_div

    return results


if __name__ == '__main__':
    r = run()
    print("=== Experiment 1: LLM-LLM pairs ===")
    print(f"Better: {r['better_pre_pooled']:.1f}% (SD {r['sd_best_pre']:.2f}) -> "
          f"{r['better_post_pooled']:.1f}% (SD {r['sd_best_post']:.2f})  "
          f"")
    print(f"Worse:  {r['worse_pre_pooled']:.1f}% (SD {r['sd_second_pre']:.2f}) -> "
          f"{r['worse_post_pooled']:.1f}% (SD {r['sd_second_post']:.2f})  "
          f"")
    print(f"Paired t (better): t({r['n_units']-1})={r['t_better']:.3f}, p(two-sided)={r['p_better']:.4f}")
    print(f"Paired t (worse):  t({r['n_units']-1})={r['t_worse']:.3f}, p(two-sided)={r['p_worse']:.4f}")
    for occ in [0, 2, 5]:
        print(f"PCS pairs (filter_occurrences={occ}): {r[f'pcs_pairs_occ{occ}']:.2f}%")
