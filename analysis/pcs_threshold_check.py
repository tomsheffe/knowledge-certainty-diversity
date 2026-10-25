"""How often is the PCS switching set a single threshold?

PCS fits each confidence level separately, so the levels at which the rule
switches to the partner P need not be adjacent. This script checks, for each
setting (all items pooled, as in Table 2) and for each group (as in Fig. 5),
whether the switching set is in fact a single threshold:

  pairs  (c = R's confidence):     switch at every level below some cut-off
  trios  (c = the rebel's):        switch at every level above some cut-off

(an empty set, or all levels, also counts). A set counts as threshold-shaped
when one cut-off in that direction achieves the full PCS; levels where R and P
are equally accurate contribute nothing either way, so they do not break a
threshold. Also reported: the share of PCS recovered by the best
single-threshold rule, fitted the same way (in-sample) but constrained to one
cut-off.

Writes ../outputs/pcs_threshold_check_output.txt.
"""
import os

import numpy as np
import pandas as pd

from common import (load_bot_bot_duo, load_bot_bot_trio, load_exp3_human_trios,
                    load_exp4_human_llm, aggregate_questions)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, '..', 'outputs')


def items():
    """One row per item: setting, group, direction, c, a_R, a_P, eligible."""
    frames = []
    d1, _ = load_bot_bot_duo()
    d4 = load_exp4_human_llm()
    for label, d, unit in [('Exp 1: LLM pairs', d1, 'group'),
                           ('Exp 4: human-LLM pairs', d4, 'participant_id')]:
        frames.append(pd.DataFrame(dict(
            setting=label, unit=d[unit].values, direction='below',
            c=d['best_agent_confidence'].astype(float).values,
            r=d['best_agent_correct_before'].astype(float).values,
            p=d['second_agent_correct_before'].astype(float).values,
            ok=True)))
    raw2, s2 = load_bot_bot_trio()
    bench = dict(enumerate(raw2['dataset_name']))
    _, s3 = load_exp3_human_trios()
    for label, s, unit_of in [('Exp 2: LLM trio', s2, bench), ('Exp 3: human trios', s3, None)]:
        a = aggregate_questions(s)
        frames.append(pd.DataFrame(dict(
            setting=label,
            unit=(a['group_num'] if unit_of is None else a['question_number'].map(unit_of)).values,
            direction='above',
            c=a['rebel_confidence'].astype(float).values,
            r=a['coalition_accuracy'].astype(float).values,
            p=a['rebel_accuracy'].astype(float).values,
            ok=a['has_coalition'].astype(bool).values)))
    return pd.concat(frames, ignore_index=True)


def check(g, direction):
    """(is_threshold, PCS, best single-threshold gain, levels, switching levels)."""
    n_total = len(g)
    lv = (g[g['ok']].groupby('c').agg(n=('r', 'size'), a_r=('r', 'mean'), a_p=('p', 'mean'))
          .sort_index())
    if direction == 'above':
        lv = lv.iloc[::-1]  # walk from the end where switching is expected
    diff = (lv['n'] / n_total * (lv['a_p'] - lv['a_r'])).values * 100
    switch = (lv['a_p'] > lv['a_r']).values
    pcs = diff[switch].sum()
    best_threshold = max([0.0] + list(np.cumsum(diff)))
    # Threshold-shaped: one cut-off achieves the full PCS (ties contribute 0).
    is_threshold = bool(best_threshold >= pcs - 1e-9)
    return is_threshold, pcs, best_threshold, list(lv.index), list(lv.index[switch])


def run():
    df = items()
    rows = []
    for setting, g in df.groupby('setting'):
        direction = g['direction'].iloc[0]
        t, pcs, thr, lv, sw = check(g, direction)
        rows.append(dict(setting=setting, unit='ALL (pooled)', n=len(g), threshold=t,
                         pcs=pcs, thr=thr, levels=lv, switch=sw))
        for unit, gu in g.groupby('unit'):
            t, pcs, thr, lv, sw = check(gu, direction)
            rows.append(dict(setting=setting, unit=unit, n=len(gu), threshold=t,
                             pcs=pcs, thr=thr, levels=lv, switch=sw))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    r = run()
    pooled, groups = r[r['unit'] == 'ALL (pooled)'], r[r['unit'] != 'ALL (pooled)']
    lines = ["=== Is the PCS switching set a single threshold? ===", "",
             "Pooled settings (Table 2):"]
    for _, x in pooled.iterrows():
        lines.append(f"  {x['setting']:24s} threshold = {x['threshold']!s:5s} PCS = {x['pcs']:5.2f}  "
                     f"best single threshold = {x['thr']:5.2f}  switch at {[int(v) for v in x['switch']]}")
    lines += ["", "Per group (Fig. 5):"]
    for s, g in groups.groupby('setting'):
        nz = g[g['pcs'] > 0]
        lines.append(f"  {s:24s} {int(g['threshold'].sum()):2d}/{len(g):2d} groups threshold-shaped; "
                     f"of the {len(nz)} with PCS > 0: {int(nz['threshold'].sum())}; "
                     f"single threshold recovers {100 * g['thr'].sum() / g['pcs'].sum():.0f}% of summed PCS")
    nz = groups[groups['pcs'] > 0]
    lines.append(f"  {'All groups':24s} {int(groups['threshold'].sum()):2d}/{len(groups):2d} threshold-shaped; "
                 f"of the {len(nz)} with PCS > 0: {int(nz['threshold'].sum())}; "
                 f"single threshold recovers {100 * groups['thr'].sum() / groups['pcs'].sum():.0f}% of summed PCS")
    lines += ["", "Groups that are not threshold-shaped:"]
    for _, x in groups[~groups['threshold']].iterrows():
        lines.append(f"  {x['setting']:24s} {str(x['unit']):22s} n = {x['n']:3d}  levels {[int(v) for v in x['levels']]}  "
                     f"switch at {[int(v) for v in x['switch']]}  PCS = {x['pcs']:.1f}, threshold rule = {x['thr']:.1f}")
    text = '\n'.join(lines)
    print(text)
    with open(os.path.join(OUT_DIR, 'pcs_threshold_check_output.txt'), 'w') as f:
        f.write(text + '\n')
