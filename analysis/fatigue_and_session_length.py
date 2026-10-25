"""Session length and fatigue checks for Experiments 3 and 4.

Two questions:
  1. How long did sessions actually take? (from raw per-item reaction-time
     columns already in the read-only data copies -- not previously used
     for this purpose)
  2. Is there a within-session accuracy decline consistent with fatigue?
     (item-position check, both experiments; Exp 3's no-discussion control
     group re-used from exp3_human_trios.py as the cleanest fatigue check
     available -- a same-protocol retest with no discussion in between)

Caveat on the item-position check: `question_number` is the item's ID, not
a verified presentation order. If item order was randomized per
participant/trio (as Exp 1's discussion order was), a first-half/second-half
split by question_number is not the same as an early/late split by time and
this check is only weak evidence either way -- flagged in the printed
output rather than glossed over.

Timing columns: exp3_human_trios.csv's rt_start/rt_end and
exp4_human_llm_responses.csv's init_rt/final_rt are per-item response latencies in
milliseconds (time from item shown to answer submitted), not session-level
timestamps. Summing them per participant gives active on-task time, a lower
bound on true elapsed session time (it excludes reading instructions and any
idle time between items).
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, '..', 'data')


def _iqr(s):
    lo, hi = s.quantile([0.25, 0.75])
    return lo, hi


def exp3_timing():
    d = pd.read_csv(os.path.join(DATA_DIR, 'exp3_human_trios.csv'))
    per = d.groupby('subject').agg(n=('question_number', 'count'),
                                    pre_ms=('rt_start', 'sum'),
                                    post_ms=('rt_end', 'sum'))
    per['pre_min'] = per['pre_ms'] / 60000
    per['post_min'] = per['post_ms'] / 60000
    per['total_min'] = per['pre_min'] + per['post_min']
    return per, d


def exp4_timing():
    d = pd.read_csv(os.path.join(DATA_DIR, 'exp4_human_llm_responses.csv'))
    # All released rows belong to the Experiment 4 analysis cohort.
    c = d.copy()
    per = c.groupby('participant').agg(n=('question_number', 'count'),
                                    init_ms=('init_rt', 'sum'),
                                    final_ms=('final_rt', 'sum'))
    per['init_min'] = per['init_ms'] / 60000
    per['final_min'] = per['final_ms'] / 60000
    per['total_min'] = per['init_min'] + per['final_min']
    return per, c


def item_position_check(d, first_col, second_col, qcol='question_number'):
    half = np.where(d[qcol] <= d[qcol].median(), 'first', 'second')
    return d.assign(half=half).groupby('half')[[first_col, second_col]].mean() * 100


def run():
    results = {}

    per3, d3 = exp3_timing()
    results['exp3_per_participant'] = per3
    results['exp3_position'] = item_position_check(d3, 'ans_start_correct', 'ans_end_correct')

    per4, c4 = exp4_timing()
    results['exp4_per_participant'] = per4
    results['exp4_position'] = item_position_check(c4, 'is_correct_first_answer', 'is_correct_last_answer')

    return results


if __name__ == '__main__':
    r = run()

    print("=== Experiment 3 (human trios): active on-task time, minutes/participant ===")
    per3 = r['exp3_per_participant']
    for col, label in [('pre_min', 'Pre-session (42 items)'), ('post_min', 'Post-session (42 items, after discussion)'),
                       ('total_min', 'Total (excludes discussion-phase duration, not in this file)')]:
        lo, hi = _iqr(per3[col])
        print(f"  {label}: median {per3[col].median():.1f} min, IQR [{lo:.1f}, {hi:.1f}], "
              f"range [{per3[col].min():.1f}, {per3[col].max():.1f}] (n={len(per3)})")
    print()
    print("  Item-position check (question_number <= median vs. above; NOT verified")
    print("  to be presentation order -- see module docstring):")
    print(r['exp3_position'].to_string())
    print()

    print("=== Experiment 4 (human-LLM pairs): ===")
    per4 = r['exp4_per_participant']
    for col, label in [('init_min', 'Phase 1: solo answers (50 items)'),
                       ('final_min', 'Phase 2: conversation + final answer (50 items)'),
                       ('total_min', 'Total active on-task time')]:
        lo, hi = _iqr(per4[col])
        print(f"  {label}: median {per4[col].median():.1f} min, IQR [{lo:.1f}, {hi:.1f}], "
              f"range [{per4[col].min():.1f}, {per4[col].max():.1f}] (n={len(per4)})")
    print(f"  Participants exceeding 120 min total: {(per4['total_min'] > 120).sum()}/{len(per4)}")
    print(f"  Participants exceeding 180 min total: {(per4['total_min'] > 180).sum()}/{len(per4)}")
    print()
    print("  Item-position check (question_number <= median vs. above; same caveat as above):")
    print(r['exp4_position'].to_string())
    print()

    print("=== Fatigue-relevant result: Exp 3 no-discussion control group ===")
    # Computed from exp3_human_trios.run() rather than quoted. An earlier
    # version of this file quoted "t(9)=1.00", which is the test for the
    # *best-ranked* control participants (83.3% -> 82.1%, p = 0.343), not for
    # the control group overall.
    import exp3_human_trios
    e3 = exp3_human_trios.run()
    print(f"  Overall control, pre -> post: {e3['control_pre']:.1f}% (SD {e3['control_pre_sd']:.1f})"
          f" -> {e3['control_post']:.1f}% (SD {e3['control_post_sd']:.1f});"
          f" paired t({e3['control_df']}) = {e3['control_t']:.2f},"
          f" p = {e3['control_p']:.3f} (n = {e3['control_n']})")
    print("  A same-protocol retest with no discussion between phases; if fatigue drove a")
    print("  systematic decline over the session, it should appear here, and it does not.")
