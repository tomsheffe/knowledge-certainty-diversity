"""Generates the paper's figures (main text Figs. 2 and 3; supplementary
S3/S4 robustness panels, S5 heatmaps, S6 calibration, S7 switching, S8
mid-tier panels, S9 worse-agent view) from the data in ../data/ and ../derived_data/, using the
plotting functions in plotting.py. Also writes captions.md with the n,
interval type, and test statistics behind each panel.

Writes everything to ../outputs/figures/; reads packaged ../data/ and ../derived_data/ inputs.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import (load_bot_bot_duo, load_bot_bot_trio, load_exp3_human, load_exp3_human_trios, load_exp4_human_llm,
                    aggregate_questions, aggregate_by_rebel_conf, aggregate_bot_group_data,
                    get_rank_accuracies, multi_confidence_pcs, paired_sign_test)
from plotting import (plot_diversity_plots, plot_agent_accuracies,
                      plot_rebel_coalition_accuracies, plot_rank_accuracies)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, '..', 'outputs', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)


def savefig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote outputs/figures/{os.path.basename(path)}")


# ---------------------------------------------------------------------------
# Caption support: crossover tests + a captions.md written from the same
# numbers the panels were drawn with, so figures and captions cannot drift apart.
# ---------------------------------------------------------------------------

def _items(df, kind):
    """Item-level (c, ref, partner) for the crossover tests. Pairs: the better
    agent's confidence, the better (ref) and worse (partner) agent's accuracy.
    Trios: 2-1 split items only, the rebel's confidence, the coalition (ref)
    and rebel (partner) accuracy -- the same items the panel plots."""
    if kind == 'pair':
        out = pd.DataFrame(dict(c=df['best_agent_confidence'], ref=df['best_agent_correct_before'],
                                partner=df['second_agent_correct_before']))
    else:
        df = df[df['has_coalition']]
        out = pd.DataFrame(dict(c=df['rebel_confidence'], ref=df['coalition_accuracy'],
                                partner=df['rebel_accuracy']))
    out = out.astype(float).dropna()
    out['c'] = out['c'].round().astype(int)
    return out


def _crossover_test(items, levels):
    """Exact paired sign test of partner vs. reference on the items at `levels`.
    The two accuracies come from the same items, so they are compared item by
    item rather than as independent proportions."""
    sub = items[items['c'].isin(levels)]
    k, m, p = paired_sign_test(sub['ref'], sub['partner'])
    return dict(levels=list(levels), n=len(sub), ref_pct=100 * sub['ref'].mean(),
                partner_pct=100 * sub['partner'].mean(), partner_better=k, ref_better=m, p=p)


def _per_level_lines(items, kind):
    ref_name, partner_name = ('better', 'worse') if kind == 'pair' else ('coalition', 'rebel')
    lines = []
    for c in sorted(items['c'].unique()):
        t = _crossover_test(items, [c])
        lines.append(f"    level {c}: n={t['n']}, {ref_name}={t['ref_pct']:.1f}%, {partner_name}={t['partner_pct']:.1f}%; "
                     f"{partner_name} more accurate on {t['partner_better']} items, {ref_name} on {t['ref_better']}; "
                     f"sign test p={t['p']:.2g}")
    return lines


def _fmt_p(p):
    return 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'


def _caption_block(title, A, B, C, D, items_A, items_B, cross_A=None, cross_B=None):
    out = [f"## {title}", ""]
    out.append(f"**A.** Error bars: {A['interval']}; purple bars show items per level "
               f"(n = {A['n_items']} items; per level: {A['n_per_level']}).")
    if cross_A:
        out.append(f"  Crossover levels {cross_A['levels']} ({cross_A['n']} items): worse agent {cross_A['partner_pct']:.1f}% vs. "
                   f"better agent {cross_A['ref_pct']:.1f}%; worse more accurate on {cross_A['partner_better']} items, "
                   f"better on {cross_A['ref_better']} (exact sign test, {_fmt_p(cross_A['p'])}).")
    out.append("  Per-level paired sign tests (worse vs. better):")
    out += _per_level_lines(items_A, 'pair')
    out.append("")
    out.append(f"**B.** Error bars: {B['interval']}; purple bars show items per level "
               f"(n = {B['n_items']} trio-items with a 2-1 split; per level: {B['n_per_level']}).")
    if cross_B:
        out.append(f"  Crossover level(s) {cross_B['levels']} ({cross_B['n']} items): rebel {cross_B['partner_pct']:.1f}% vs. "
                   f"coalition {cross_B['ref_pct']:.1f}%; rebel more accurate on {cross_B['partner_better']} items, "
                   f"coalition on {cross_B['ref_better']} (exact sign test, {_fmt_p(cross_B['p'])}).")
    out.append("  Per-level paired sign tests (rebel vs. coalition):")
    out += _per_level_lines(items_B, 'trio')
    out.append("")
    out.append(f"**C.** Bars: mean accuracy pooled over items. Error bars: {C['interval']}. "
               f"{C['test']}: worse {C['worse']['pre']:.1f}→{C['worse']['post']:.1f}% "
               f"(t = {C['worse']['t']:.2f}, two-sided {_fmt_p(C['worse']['p_two_sided'])}{', ' + C['worse']['star'] if C['worse']['star'] else ''}); "
               f"better {C['better']['pre']:.1f}→{C['better']['post']:.1f}% "
               f"(t = {C['better']['t']:.2f}, two-sided {_fmt_p(C['better']['p_two_sided'])}{', ' + C['better']['star'] if C['better']['star'] else ''}).")
    out.append("")
    maj_n = D['majority']['n']
    maj_str = '' if maj_n is None else f" (n = {maj_n}; groups for human trios, items for the single LLM trio configuration)"
    star_d = (', ' + D['overall']['star']) if D['overall']['star'] else ''
    out.append(f"**D.** Bars: mean accuracy by initial rank. Error bars: {D['interval']} "
               f"(n per rank: {D['n_per_rank']}). Majority bar: pre-discussion majority vote{maj_str}. "
               f"Bracket: {D['test']}: n = {D['overall']['n']}, t = {D['overall']['t']:.2f}, "
               f"two-sided {_fmt_p(D['overall']['p_two_sided'])}{star_d}.")
    out.append("")
    return out


CAPTION_LINES = ["# Figure captions — error bars, n, and tests",
                 "",
                 "Generated by make_figures.py from the same numbers used to draw each panel.",
                 "Stars: * p<0.05, ** p<0.01, *** p<0.001 (two-sided paired tests).",
                 ""]


# ---------------------------------------------------------------------------
# Figure 2 — LLM-only interactions: pairs and trios
# ---------------------------------------------------------------------------

def make_figure_2():
    print("Figure 2 (LLM pairs + trios)...")
    parsed1, _ = load_bot_bot_duo()
    raw2, parsed2 = load_bot_bot_trio()
    agg_q2 = aggregate_questions(parsed2)
    agg_conf2 = aggregate_by_rebel_conf(agg_q2)
    rank_acc2 = get_rank_accuracies(parsed2)
    majority2 = aggregate_bot_group_data(raw2)

    A, B, C, D = {}, {}, {}, {}
    fig, axs = plt.subplots(2, 2, figsize=(20, 18))
    # Shared axis labels across each row: left panels keep "Accuracy (%)", right panels keep
    # "Number of Samples" (top row) / drop the duplicate y-label (bottom row).
    # LLM accuracies before discussion are fractions of self-consistency
    # samples, so Fig. 2's intervals are percentile bootstraps over items
    # rather than Wilson (binomial) intervals (2026-10-14).
    plot_diversity_plots(parsed1, ax=axs[0, 0], include_ytitle_right=False, info=A, interval='bootstrap')
    axs[0, 0].set_title('A. LLM Pairs', fontsize=27, fontweight='bold')
    plot_rebel_coalition_accuracies(agg_conf2, ax=axs[0, 1], include_ytitle_left=False, info=B,
                                    items=agg_q2, interval='bootstrap')
    axs[0, 1].set_title('B. LLM Trios', fontsize=27, fontweight='bold')
    # Unit: the 12 combinations of model pair and benchmark ('group'), as in
    # the original submission and at which the reference agent is chosen
    # (decided 2026-10-12; briefly the 3 model pairs, 2026-09-13 to 10-12).
    plot_agent_accuracies(parsed1, ax=axs[1, 0], include_legend=False, group_col='group', info=C)
    axs[1, 0].set_title('C. LLM Pairs pre/post', fontsize=27, fontweight='bold')
    items2 = {r: {'pre': parsed2.loc[parsed2['subject_rank'] == r, 'ans_start_correct'].astype(float).values,
                  'post': parsed2.loc[parsed2['subject_rank'] == r, 'ans_end_correct'].astype(float).values}
              for r in ['Least', 'Middle', 'Best']}
    items2['majority'] = majority2['premajority'].astype(float).values
    plot_rank_accuracies(rank_acc2, list(majority2['premajority']), ax=axs[1, 1], include_ytitle=False, info=D,
                         single_config_items=items2)
    axs[1, 1].set_title('D. LLM Trios pre/post', fontsize=27, fontweight='bold')
    fig.tight_layout()
    savefig(fig, 'figure_2.png')
    CAPTION_LINES.extend(_caption_block('Figure 2 — LLM-only interactions (no crossover expected; none shaded)', A, B, C, D,
                                        _items(parsed1, 'pair'), _items(agg_q2, 'trio')))


# ---------------------------------------------------------------------------
# Figure 3 — human-involving interactions: trios and human-LLM pairs
# ---------------------------------------------------------------------------

def make_figure_3():
    print("Figure 3 (human trios + human-LLM pairs)...")
    df4 = load_exp4_human_llm()
    group_summary3, subj3 = load_exp3_human(0.62)
    interv3 = subj3[subj3['control'] == 0]          # individual-level (panel D)
    _, trios_all3 = load_exp3_human_trios()             # trio-level, all members (panel B)
    agg_q3 = aggregate_questions(trios_all3)
    agg_conf3 = aggregate_by_rebel_conf(agg_q3)
    rank_acc3 = get_rank_accuracies(interv3)
    majority3 = list(group_summary3[group_summary3['control'] == 0]['mean_premajority'])

    A, B, C, D = {}, {}, {}, {}
    CROSS_A, CROSS_B = [0, 1, 2], [5]   # crossover levels claimed in the text (tested, not shaded)
    fig, axs = plt.subplots(2, 2, figsize=(20, 18))
    plot_diversity_plots(df4, ax=axs[0, 0], include_ytitle_right=False, info=A)
    axs[0, 0].set_title('A. Human-LLM Pairs', fontsize=27, fontweight='bold')
    plot_rebel_coalition_accuracies(agg_conf3, ax=axs[0, 1], include_ytitle_left=False, info=B,
                                    items=agg_q3, interval='wilson')
    axs[0, 1].set_title('B. Human Trios', fontsize=27, fontweight='bold')
    plot_agent_accuracies(df4, ax=axs[1, 0], include_legend=False, info=C)
    axs[1, 0].set_title('C. Human-LLM Pairs pre/post', fontsize=27, fontweight='bold')
    plot_rank_accuracies(rank_acc3, majority3, ax=axs[1, 1], include_ytitle=False, info=D)
    axs[1, 1].set_title('D. Human Trios pre/post', fontsize=27, fontweight='bold')
    fig.tight_layout()
    savefig(fig, 'figure_3.png')
    CAPTION_LINES.extend(_caption_block(
        'Figure 3 — Human-involving interactions (crossover levels tested)', A, B, C, D,
        _items(df4, 'pair'), _items(agg_q3, 'trio'),
        cross_A=_crossover_test(_items(df4, 'pair'), CROSS_A),
        cross_B=_crossover_test(_items(agg_q3, 'trio'), CROSS_B)))


# ---------------------------------------------------------------------------
# Fig. S5 — multi-confidence PCS heatmaps
# ---------------------------------------------------------------------------

def _heatmap(ref_mean, partner_mean, counts, title, ax):
    levels = list(ref_mean.index)
    ref_vals = ref_mean.values.astype(float)
    partner_vals = partner_mean.values.astype(float)
    oracle = np.nanmax(np.dstack([ref_vals, partner_vals]), axis=2)
    delta = np.nan_to_num((ref_vals - partner_vals), nan=0.0)
    im = ax.imshow(delta, vmin=-0.4, vmax=0.4, cmap='coolwarm', origin='lower', aspect='auto')
    ax.set_xticks(range(len(levels))); ax.set_yticks(range(len(levels)))
    ax.set_xticklabels(levels); ax.set_yticklabels(levels)
    ax.set_title(title, fontsize=21, fontweight='bold')
    ax.set_xlabel('Partner confidence', fontsize=20)
    ax.set_ylabel('Reference confidence', fontsize=20)
    for i in range(len(levels)):
        for j in range(len(levels)):
            n = counts.values[i, j]
            if n > 0:
                ax.text(j, i, f'n={int(n)}', ha='center', va='center', fontsize=17, color='white')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Δ (ref-partner)')


def make_figure_s5():
    print("Fig. S5 (multi-confidence PCS heatmaps)...")
    parsed1, _ = load_bot_bot_duo()
    raw2, parsed2 = load_bot_bot_trio()
    agg_q2 = aggregate_questions(parsed2)
    _, trios_all3 = load_exp3_human_trios()             # trio-level: all members
    agg_q3 = aggregate_questions(trios_all3)
    df4 = load_exp4_human_llm()
    # Trios: only 2-1 split items have a rebel and a coalition; the full item
    # count is the denominator N, exactly as in fig_s5_multiconfidence.py, so
    # the panel titles match the values reported in Table 2.
    c2, c3 = agg_q2[agg_q2['has_coalition']], agg_q3[agg_q3['has_coalition']]
    totals = {'LLM trios (Exp 2)': len(agg_q2), 'Human trios (Exp 3)': len(agg_q3)}

    fig, axs = plt.subplots(2, 2, figsize=(12, 11))
    specs = [
        (parsed1['best_agent_confidence'], parsed1['best_agent_correct_before'],
         parsed1['second_agent_confidence'], parsed1['second_agent_correct_before'], 'LLM pairs (Exp 1)', axs[0, 0]),
        (c2['rebel_confidence'], c2['rebel_accuracy'],
         c2['coalition_max_confidence'], c2['coalition_accuracy'], 'LLM trios (Exp 2)', axs[0, 1]),
        (c3['rebel_confidence'], c3['rebel_accuracy'],
         c3['coalition_max_confidence'], c3['coalition_accuracy'], 'Human trios (Exp 3)', axs[1, 0]),
        (df4['best_agent_confidence'], df4['best_agent_correct_before'],
         df4['second_agent_confidence'], df4['second_agent_correct_before'], 'Human-LLM pairs (Exp 4)', axs[1, 1]),
    ]
    # Use baseline='max': subtract the larger of the reference and partner
    # weighted accuracies from the two-confidence oracle accuracy.
    for ref_conf, ref_acc, partner_conf, partner_acc, title, ax in specs:
        gain, info = multi_confidence_pcs(ref_conf, ref_acc, partner_conf, partner_acc, baseline='max',
                                          total_items=totals.get(title))
        _heatmap(info['ref_mean'], info['partner_mean'], info['counts'],
                f'{title}\nPCS(multi-conf)={gain*100:.1f} pp', ax)
    fig.suptitle('Fig. S5 —multi-confidence PCS oracle grids', fontsize=25, fontweight='bold', y=1.02)
    fig.tight_layout()
    savefig(fig, 'fig_s5_multiconfidence.png')


# ---------------------------------------------------------------------------
# Fig. S6 — calibration curves
# ---------------------------------------------------------------------------

def make_figure_s6():
    from supplement_original_style import figure_s6
    fig = figure_s6()
    from pathlib import Path
    fig.savefig(os.path.join(FIG_DIR, 'fig_s6_calibration.png'), dpi=300,
                bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig. S7 — confidence-sensitive switching (AME forest plot)
# ---------------------------------------------------------------------------

def make_figure_s7():
    from supplement_original_style import figure_s7
    fig = figure_s7()
    from pathlib import Path
    fig.savefig(os.path.join(FIG_DIR, 'fig_s7_switching.png'), dpi=300,
                bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig. S8 — mid-tier model replication bar chart
# ---------------------------------------------------------------------------

def make_figure_s8():
    from supplementary_replications import save
    save(8, FIG_DIR)


# ---------------------------------------------------------------------------
# Fig. S9 — Exp 4 from the worse agent's perspective
# ---------------------------------------------------------------------------

def make_figure_s9():
    print("Fig. S9 (worse-agent perspective)...")
    df4 = load_exp4_human_llm()
    swapped = df4.rename(columns={
        'best_agent_confidence': '_tmp_bc', 'second_agent_confidence': 'best_agent_confidence',
        'best_agent_correct_before': '_tmp_bcb', 'second_agent_correct_before': 'best_agent_correct_before',
    })
    swapped = swapped.rename(columns={'_tmp_bc': 'second_agent_confidence', '_tmp_bcb': 'second_agent_correct_before'})
    fig, ax = plt.subplots(figsize=(7, 6))
    plot_diversity_plots(swapped, ax=ax)
    ax.set_title('Fig. S9 —Exp 4, worse agent as reference\n'
                "(labels: 'Better' line = worse agent's own accuracy, 'Worse' line = best agent's accuracy)",
                fontsize=21, fontweight='bold')
    fig.tight_layout()
    savefig(fig, 'fig_s9_worse_agent.png')


# ---------------------------------------------------------------------------
# Figs. S1-S4 — temperature robustness
# ---------------------------------------------------------------------------

def make_figure_s1_s4():
    print("Figs. S1-S4 (temperature robustness)...")
    from fig_s1_s4_temperature_robustness import run as run_temp
    r = run_temp()
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    temps_duo = ['temp02', 'temp05']
    x = np.arange(len(temps_duo))
    better = [r[f'duo_{t}']['best_pre'] for t in temps_duo]
    better_post = [r[f'duo_{t}']['best_post'] for t in temps_duo]
    worse = [r[f'duo_{t}']['second_pre'] for t in temps_duo]
    worse_post = [r[f'duo_{t}']['second_post'] for t in temps_duo]
    axs[0].plot(x, better, 'o-', label='Better, pre', color='#4E95D9')
    axs[0].plot(x, better_post, 'o--', label='Better, post', color='#4E95D9', alpha=0.6)
    axs[0].plot(x, worse, 's-', label='Worse, pre', color='#C00000')
    axs[0].plot(x, worse_post, 's--', label='Worse, post', color='#C00000', alpha=0.6)
    axs[0].set_xticks(x); axs[0].set_xticklabels(['T=0.2', 'T=0.5'])
    axs[0].set_ylabel('Accuracy (%)', fontsize=21, fontweight='bold')
    axs[0].set_title('LLM pairs across temperatures', fontsize=22, fontweight='bold')
    axs[0].legend(fontsize=18); axs[0].set_ylim(0, 100)

    # The saved T=.8 trio_temp08 file is a legacy single-model export.
    # S1/S2 use the separate complete three-model calibration records.
    temps_trio = ['temp02', 'temp05']
    x2 = np.arange(len(temps_trio))
    best_pre = [r[f'trio_{t}']['best_pre'] for t in temps_trio]
    best_post = [r[f'trio_{t}']['best_post'] for t in temps_trio]
    axs[1].plot(x2, best_pre, 'o-', label='Best, pre', color='#4E95D9')
    axs[1].plot(x2, best_post, 'o--', label='Best, post', color='#4E95D9', alpha=0.6)
    axs[1].set_xticks(x2); axs[1].set_xticklabels(['T=0.2', 'T=0.5'])
    axs[1].set_title('LLM trios across temperatures', fontsize=22, fontweight='bold')
    axs[1].legend(fontsize=18); axs[1].set_ylim(0, 100)

    fig.suptitle('Figs. S1-S4 —temperature robustness', fontsize=27, fontweight='bold')
    fig.tight_layout()
    savefig(fig, 'fig_s1_s4_temperature.png')
    from supplementary_replications import save
    save(3, FIG_DIR)
    save(4, FIG_DIR)


if __name__ == '__main__':
    make_figure_2()
    make_figure_3()
    cap_path = os.path.join(FIG_DIR, 'captions.md')
    with open(cap_path, 'w') as f:
        f.write('\n'.join(CAPTION_LINES))
    print(f"  wrote {cap_path}")
    make_figure_s5()
    make_figure_s6()
    make_figure_s7()
    make_figure_s8()
    make_figure_s9()
    make_figure_s1_s4()
    print(f"\nAll figures written to {FIG_DIR}")
