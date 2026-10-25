"""Table 2 — Potential Conversation Synergy (PCS) with uncertainty, null
model, and the observed post-discussion change of the reference agent, for
all four experiments.

Emits the table as it appears in the paper. Assembles results from
pcs_uncertainty_and_null.py (bootstrap CI, permutation null),
fig_s5_multiconfidence.py (multi-confidence PCS), and the experiment scripts
(reference agent's observed pre/post accuracy). Writes ../outputs/table2.md,
.csv and .html. Takes a few minutes (the resampling scripts run inside).

The paper's held-out validation is at the level of groups, not items:
group_level_regression_cv.py (Fig. 5). The item-fold scripts
pcs_setting_level_cv.py and pcs_out_of_sample_validation.py are not reported.
"""
import os
import numpy as np
import pandas as pd

import pcs_uncertainty_and_null as unc
import pcs_setting_level_cv as cv
import fig_s5_multiconfidence as s5
import exp1_llm_pairs, exp2_llm_trios, exp3_human_trios
from common import load_exp4_human_llm, load_exp3_human_trios
from scipy.stats import ttest_rel

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, '..', 'outputs')


def _fmt_p(p):
    return 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'


# Reference agent's pre-discussion accuracy uses the per-item preparation in
# pcs_setting_level_cv (better agent in pairs, coalition in trios), keyed by
# setting label; map the labels to the short names used below.
SETTING_KEY = {'Exp 1: LLM-LLM pairs': 'exp1', 'Exp 2: LLM trios': 'exp2',
               'Exp 3: human trios': 'exp3', 'Exp 4: human-LLM pairs': 'exp4'}


def reference_accuracy():
    return {SETTING_KEY[label]: float(prep().dropna(subset=['conf'])['ref'].mean() * 100)
            for label, prep in cv.SETTINGS}


def build():
    u = unc.run()
    base = reference_accuracy()
    m = s5.run()
    e1, e2, e3 = exp1_llm_pairs.run(), exp2_llm_trios.run(), exp3_human_trios.run()
    d4 = load_exp4_human_llm()
    # Exp 4 reference-agent change: item-pooled means (as in Fig. 3C); paired test across the 30 pairs
    from scipy.stats import ttest_rel
    agg4 = d4.groupby('participant_id').agg(pre=('best_agent_correct_before', 'mean'),
                                            post=('best_agent_correct_after', 'mean'))
    t4, p4 = ttest_rel(agg4['post'], agg4['pre'])
    e4_pre, e4_post = d4['best_agent_correct_before'].mean() * 100, d4['best_agent_correct_after'].mean() * 100

    # Trio post-discussion change: the trio's majority vote after discussion
    # vs. its majority vote before (an item scores 1 when at least two of the
    # three members are right), the same quantity at both time points and the
    # reference PCS is measured against. Changed 2026-10-12; it previously
    # compared the members' mean accuracy after discussion with the majority
    # before, which mixes two different quantities. Exp 3: paired across the
    # 10 trios. Exp 2: one configuration, no test.
    e2_post_mean = e2['majority_post']
    e2_maj = e2['majority_pre']
    _, interv3 = load_exp3_human_trios()                # trio-level: all members (2026-09-15)
    per_item3 = interv3.groupby(['group_num', 'question_number']).agg(
        post=('ans_end_correct', lambda x: float(x.mean() > 0.5)),
        maj=('ans_start_correct', lambda x: float(x.mean() > 0.5)))
    per_trio3 = per_item3.groupby(level='group_num').mean() * 100
    t3, p3 = ttest_rel(per_trio3['post'], per_trio3['maj'])
    e3_post_mean, e3_maj = per_item3['post'].mean() * 100, per_item3['maj'].mean() * 100

    # Multi-confidence PCS: baseline='max' (same reference as single-confidence
    # PCS: better agent in pairs, coalition in trios). See fig_s5 / make_figures.
    rows = [
        dict(setting='Exp 1: LLM–LLM pairs', items='600 (3 model pairs × 200)', reference='Better agent',
             ref_pre=base['exp1'],
             pcs=u['exp1']['obs'], boot=u['exp1']['boot'], null=u['exp1']['null'], multi=m['exp1_max'],
             change=e1['better_post_pooled'] - e1['better_pre_pooled'],
             change_txt=f"{e1['better_pre_pooled']:.1f} → {e1['better_post_pooled']:.1f}", change_p=e1['p_better'],
             change_note='paired across 12 model pair x benchmark combinations'),
        dict(setting='Exp 2: LLM trios', items='200 (one trio configuration)', reference='Coalition (majority of trio)',
             ref_pre=base['exp2'],
             pcs=u['exp2']['obs'], boot=u['exp2']['boot'], null=u['exp2']['null'], multi=m['exp2_max'],
             change=e2_post_mean - e2_maj, change_txt=f"{e2_maj:.1f} → {e2_post_mean:.1f}",
             change_p=None, change_note='majority vote after vs. before; single configuration, no test'),
        dict(setting='Exp 3: human trios', items='420 trio-items (10 trios × 42)', reference='Coalition (majority of trio)',
             ref_pre=base['exp3'],
             pcs=u['exp3']['obs'], boot=u['exp3']['boot'], null=u['exp3']['null'], multi=m['exp3_max'],
             change=e3_post_mean - e3_maj, change_txt=f"{e3_maj:.1f} → {e3_post_mean:.1f}",
             change_p=p3, change_note='majority vote after vs. before, paired across 10 trios'),
        dict(setting='Exp 4: human–LLM pairs', items='561 discussed items (30 pairs)', reference='Better agent',
             ref_pre=base['exp4'],
             pcs=u['exp4']['obs'], boot=u['exp4']['boot'], null=u['exp4']['null'], multi=m['exp4_max'],
             change=e4_post - e4_pre, change_txt=f"{e4_pre:.1f} → {e4_post:.1f}", change_p=p4,
             change_note='paired across 30 pairs'),
    ]

    # Transposed layout (metrics as rows, experiments as columns): five
    # columns, fits a portrait page. Combined cells are split into rows and
    # the notes move to footnotes so every cell stays short.
    COLS = ['LLM–LLM pairs (Exp 1)', 'LLM trios (Exp 2)', 'Human trios (Exp 3)', 'Human–LLM pairs (Exp 4)']
    ITEMS = ['600 (3 pairs × 200)', '200 (1 trio; 192 with a majority)', '420 (10 trios × 42)', '561 (30 pairs)']
    REF = ['Better agent', 'Coalition', 'Coalition', 'Better agent']
    CHANGE_FN = ['', '‡', '†', '']
    out = {}
    for i, r in enumerate(rows):
        lo, hi = np.percentile(r['boot'], [2.5, 97.5])
        # Remove floating-point noise before rounding half-up for publication.
        from decimal import Decimal, ROUND_HALF_UP
        lo_text, hi_text = [str(Decimal(str(round(float(x), 10))).quantize(
            Decimal('0.1'), rounding=ROUND_HALF_UP)) for x in (lo, hi)]
        null_p = (r['null'] >= r['pcs']).mean()
        # A PCS of exactly zero is matched or exceeded by every shuffle, so
        # the null test carries no information there; flagged rather than
        # reported as a significance result.
        null_txt = _fmt_p(null_p).replace('p = ', '').replace('p < ', '< ')
        if r['pcs'] == 0:
            null_txt += '§'
        out[COLS[i]] = {
            'Items': ITEMS[i],
            'Reference agent': REF[i],
            'Reference pre-discussion accuracy (%)': f"{r['ref_pre']:.1f}",
            'PCS (pp) [95% CI]': f"{r['pcs']:.1f} [{lo_text}, {hi_text}]",
            'Permutation null: mean PCS (pp)': f"{r['null'].mean():.2f}",
            'Permutation null: p': null_txt,
            'Multi-confidence PCS (pp)': f"{r['multi']:.1f}",
            'Post-discussion change (pp), pre → post': f"{r['change']:+.1f} ({r['change_txt']}){CHANGE_FN[i]}",
            'Paired test, p': ('—' if r['change_p'] is None
                               else _fmt_p(r['change_p']).replace('p = ', '').replace('p < ', '< ')),
        }
    df = pd.DataFrame(out)
    df.index.name = ''
    return df


TITLE = "**Table 2. Potential Conversation Synergy across the four experiments**"
LEGEND_BODY = ("PCS, percentage points the reference agent would gain by taking the partner's answer at those levels of the designated confidence variable where the partner is more accurate. "
               "Reference agent, the initially better agent (pairs) or the majority answer (trios). "
               "95% CI, percentile bootstrap. "
               "Permutation null, average PCS after shuffling the designated confidence labels; p, fraction of shuffles reaching the observed value. "
               "Multi-confidence PCS, PCS calculated over all combinations defined by both agents' levels (Fig. S5). "
               "Post-discussion change, the reference agent's accuracy before and after discussion (pairs), or the accuracy of the trio's majority vote before and after discussion (trios), with two-sided paired-test p across units (12 model pair–benchmark combinations, 10 trios, 30 pairs). "
               "All resampling procedures are detailed in Methods.")
FOOTNOTES = ["† Majority vote before vs. after discussion, paired across the ten trios.",
             "‡ Single configuration, no test possible.",
             "§ PCS is exactly zero, so every shuffle matches or exceeds it and the null test carries no information."]
LEGEND = TITLE + '\n' + LEGEND_BODY + '\n' + ' '.join(FOOTNOTES)


def _html(df):
    """Self-contained HTML (Word and Google Docs open it and keep the table)."""
    esc = lambda s: str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    head = ''.join(f'<th style="text-align:center;padding:4px 8px;border-bottom:1px solid #000">{esc(c)}</th>' for c in df.columns)
    body = ''
    for idx, row in df.iterrows():
        cells = ''.join(f'<td style="text-align:center;padding:3px 8px;white-space:nowrap">{esc(v)}</td>' for v in row)
        body += f'<tr><td style="text-align:left;padding:3px 8px">{esc(idx)}</td>{cells}</tr>\n'
    title_html = TITLE.replace('**', '')
    return f"""<html><head><meta charset="utf-8"><style>
body{{font-family:Arial,Helvetica,sans-serif;font-size:9pt;max-width:18cm}}
table{{border-collapse:collapse;border-top:1.5px solid #000;border-bottom:1.5px solid #000;width:100%}}
</style></head><body>
<p><b>{esc(title_html)}</b></p>
<table>
<thead><tr><th style="text-align:left;padding:4px 8px;border-bottom:1px solid #000"></th>{head}</tr></thead>
<tbody>
{body}</tbody></table>
<p style="font-size:8pt">{esc(LEGEND_BODY)}<br>{'<br>'.join(esc(f) for f in FOOTNOTES)}</p>
</body></html>
"""


if __name__ == '__main__':
    df = build()
    os.makedirs(OUT_DIR, exist_ok=True)
    md = ['# Table 2', '', df.to_markdown(disable_numparse=True), '', LEGEND, '']
    with open(os.path.join(OUT_DIR, 'table2.md'), 'w') as f:
        f.write('\n'.join(md))
    df.to_csv(os.path.join(OUT_DIR, 'table2.csv'))
    with open(os.path.join(OUT_DIR, 'table2.html'), 'w') as f:
        f.write(_html(df))
    print(df.to_string())
    print("\nwritten to outputs/table2.md, table2.csv and table2.html")
