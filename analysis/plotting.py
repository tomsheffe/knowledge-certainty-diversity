"""Plotting functions for the paper's Figure 2 and Figure 3 panels, plus the
supplementary figures (see make_figures.py).

Error bars (standardized across panels):
  - Diversity/crossover panels (Fig 2A/B, 3A/B): 95% Wilson score intervals
    on per-confidence-level accuracy, for pair and trio panels alike.
  - Pair pre/post bars (Fig 2C/3C): t-based 95% CI of the mean across the
    grouping unit (`group_col`), n stated in the caption.
  - Trio rank bars (Fig 2D/3D): t-based 95% CI across subjects; when a rank
    has a single sample (Exp 2's one trio configuration) the interval is a
    95% Wilson score interval over `single_trials` items. Changed 2026-09-20
    from a Wald/normal-approximation interval so every binomial interval in
    Figs. 2-3 uses the same estimator as the diversity panels.

Each function accepts an optional `info` dict and fills it with the exact
n / interval type / test statistics used to draw the panel, so captions can
be generated from the same numbers (see make_figures.py -> captions.md).

Every function here either takes an `ax` (to be embedded in a multi-panel
figure) or creates its own. None of these call plt.show(); make_figures.py
saves everything to PNG under ../outputs/figures/.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Softer typeface than the DejaVu Sans default; falls back gracefully on machines
# without Avenir (macOS ships it). Applies to every figure that imports this module.
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Avenir Next', 'Avenir', 'Helvetica Neue', 'Helvetica', 'DejaVu Sans']
# Larger tick labels and default text everywhere (readability at journal column width).
plt.rcParams['font.size'] = 21
plt.rcParams['xtick.labelsize'] = 21
plt.rcParams['ytick.labelsize'] = 21
from math import sqrt
from scipy.stats import ttest_rel, t as t_dist, norm

COLORS = {'majority': 'dimgray'}
SHADE_KW = dict(color='gold', alpha=0.15, zorder=0, linewidth=0)
_Z = {0.90: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}


def wilson_ci(k, n, z=1.959963984540054):
    """Wilson score interval for k successes in n binary (0/1) trials. Not
    used for fractional self-consistency scores: those panels use
    `bootstrap_mean_ci` (2026-10-14)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + (z ** 2) / n
    center = (p + (z ** 2) / (2 * n)) / denom
    half = (z * sqrt(max(p * (1 - p), 0.0) / n + (z ** 2) / (4 * n * n))) / denom
    return max(0.0, center - half), min(1.0, center + half)


N_BOOT, BOOT_SEED = 2000, 0


def bootstrap_mean_ci(values, ci=0.95, n_boot=N_BOOT, seed=BOOT_SEED):
    """Percentile bootstrap interval for the mean of item scores, resampling
    items. Used where scores are not 0/1 (an LLM's accuracy on an item is the
    fraction of its self-consistency samples that were correct), for which a
    binomial (Wilson) interval does not apply."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(axis=1)
    a = (1 - ci) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def _item_yerr(means_pct, value_lists, interval, z, ci=0.95):
    """Asymmetric error bars per point from item-level scores: Wilson for 0/1
    scores, percentile bootstrap over items otherwise."""
    lo, hi = [], []
    for v, m in zip(value_lists, means_pct):
        v = np.asarray(v, dtype=float)
        if interval == 'bootstrap':
            l, h = bootstrap_mean_ci(v, ci)
        else:
            l, h = wilson_ci(v.sum(), len(v), z)
        lo.append(m - 100.0 * l); hi.append(100.0 * h - m)
    return np.vstack([lo, hi])


def _asym_yerr(means_pct, ks, ns, z):
    lo, hi = [], []
    for k, n, m in zip(ks, ns, means_pct):
        l, h = wilson_ci(k, n, z)
        lo.append(m - 100.0 * l); hi.append(100.0 * h - m)
    return np.vstack([lo, hi])


def _shade(ax, levels):
    for lvl in levels or []:
        ax.axvspan(lvl - 0.5, lvl + 0.5, **SHADE_KW)


# ---------------------------------------------------------------------------
# Fig. 2A / 3A — pair diversity plot (better/worse accuracy vs better's confidence)
# ---------------------------------------------------------------------------

def plot_diversity_plots(df, filter_occurrences=0, ax=None, ci=0.95, capsize=4,
                         include_ytitle_left=True, include_ytitle_right=True,
                         shade_levels=None, info=None, interval='wilson'):
    """interval: 'wilson' for 0/1 accuracies (Fig. 3A), 'bootstrap' for
    fractional self-consistency accuracies (Fig. 2A)."""
    z = _Z.get(ci, 1.959963984540054)
    d = df.copy()
    d['best_agent_confidence'] = d['best_agent_confidence'].astype(int)
    agg = d.groupby('best_agent_confidence').agg(
        mean_best_accuracy=('best_agent_correct_before', 'mean'),
        mean_second_accuracy=('second_agent_correct_before', 'mean'),
        count=('best_agent_correct_before', 'count'),
        best_correct=('best_agent_correct_before', 'sum'),
        second_correct=('second_agent_correct_before', 'sum'),
    ).reset_index()
    if filter_occurrences > 0:
        agg = agg[agg['count'] >= filter_occurrences]
    agg['mean_best_accuracy'] *= 100.0
    agg['mean_second_accuracy'] *= 100.0
    levels = list(agg['best_agent_confidence'])
    by_lvl = {c: g for c, g in d.groupby('best_agent_confidence')}
    best_yerr = _item_yerr(agg['mean_best_accuracy'],
                           [by_lvl[c]['best_agent_correct_before'] for c in levels], interval, z, ci)
    sec_yerr = _item_yerr(agg['mean_second_accuracy'],
                          [by_lvl[c]['second_agent_correct_before'] for c in levels], interval, z, ci)

    if ax is None:
        fig, ax1 = plt.subplots(figsize=(8, 6))
    else:
        ax1 = ax; fig = ax1.figure
    _shade(ax1, shade_levels)
    x = agg['best_agent_confidence']
    line1, = ax1.plot(x, agg['mean_best_accuracy'], 'o-', label='Better Performer', color='#4E95D9')
    line2, = ax1.plot(x, agg['mean_second_accuracy'], 's-', label='Worse Performer', color='#C00000')
    ax1.errorbar(x, agg['mean_best_accuracy'], yerr=best_yerr, fmt='none',
                ecolor=line1.get_color(), elinewidth=1.5, capsize=capsize, capthick=1.5, zorder=3)
    ax1.errorbar(x, agg['mean_second_accuracy'], yerr=sec_yerr, fmt='none',
                ecolor=line2.get_color(), elinewidth=1.5, capsize=capsize, capthick=1.5, zorder=3)
    ax1.set_xlabel('Confidence of Better', fontsize=23, fontweight='bold')
    if include_ytitle_left:
        ax1.set_ylabel('Accuracy (%)', fontsize=23, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax2 = ax1.twinx()
    ax2.bar(x, agg['count'], color='purple', alpha=0.3, width=0.3, align='center')
    if include_ytitle_right:
        ax2.set_ylabel('Number of Samples', fontsize=23, fontweight='bold', color='purple')
    ax2.tick_params(axis='y', colors='purple')
    ax1.legend(prop={'size': 21, 'weight': 'bold'}, loc='upper left')

    if info is not None:
        info.update(kind='pair_diversity',
                    interval=(f'{int(ci*100)}% percentile bootstrap over items per confidence level'
                              if interval == 'bootstrap' else f'{int(ci*100)}% Wilson score interval per confidence level'),
                    n_items=int(agg['count'].sum()), n_per_level=dict(zip(agg['best_agent_confidence'].astype(int), agg['count'].astype(int))),
                    shaded_levels=list(shade_levels or []), per_level=agg)
    return fig, ax1


# ---------------------------------------------------------------------------
# Fig. 2C / 3C — pair pre/post bar chart
# ---------------------------------------------------------------------------

def plot_agent_accuracies(df, ax=None, include_legend=True, include_ytitle=True, add_sigstars=True,
                          group_col='participant_id', ci=0.95, info=None):
    best_before = df['best_agent_correct_before'].mean() * 100
    best_after = df['best_agent_correct_after'].mean() * 100
    second_before = df['second_agent_correct_before'].mean() * 100
    second_after = df['second_agent_correct_after'].mean() * 100

    agg = (df.groupby(group_col)
           .agg(best_before_mean=('best_agent_correct_before', lambda x: x.mean() * 100),
                best_after_mean=('best_agent_correct_after', lambda x: x.mean() * 100),
                second_before_mean=('second_agent_correct_before', lambda x: x.mean() * 100),
                second_after_mean=('second_agent_correct_after', lambda x: x.mean() * 100))
           .reset_index())
    n = len(agg)
    df_dof = max(n - 1, 0)
    sds = {k: (agg[k].std(ddof=1) if n >= 2 else 0.0)
           for k in ['best_before_mean', 'best_after_mean', 'second_before_mean', 'second_after_mean']}
    if n >= 2:
        t_best, p_best_2 = ttest_rel(agg['best_after_mean'], agg['best_before_mean'])
        p_best = p_best_2
        t_second, p_second_2 = ttest_rel(agg['second_after_mean'], agg['second_before_mean'])
        p_second = p_second_2
    else:
        t_best = t_second = np.nan; p_best = p_second = 1.0

    # Error bars: the bars are item-pooled accuracies, so their intervals come
    # from a cluster bootstrap that resamples units (group_col) and recomputes
    # the pooled accuracy, which weights units by their number of items exactly
    # as the bar does (2026-10-14; previously a t-interval of the unweighted
    # per-unit means). The paired t-test across units is kept for the stars.
    cols = ['second_agent_correct_before', 'best_agent_correct_before',
            'second_agent_correct_after', 'best_agent_correct_after']
    g = df.groupby(group_col)
    sums = g[cols].sum().astype(float).values            # units x 4
    cnts = g[cols].count().astype(float).values
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(sums), (N_BOOT, len(sums)))
    boot = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1) * 100     # N_BOOT x 4
    a_ = (1 - ci) / 2
    lo_q, hi_q = np.quantile(boot, a_, axis=0), np.quantile(boot, 1 - a_, axis=0)
    point = np.array([second_before, best_before, second_after, best_after])
    lower, upper = point - lo_q, hi_q - point
    pre_errs = np.vstack([lower[:2], upper[:2]])        # 2 x 2 (lower; upper), worse then better
    post_errs = np.vstack([lower[2:], upper[2:]])

    x = np.array([0, 1]); width = 0.35
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure
    pre_heights = [second_before, best_before]
    post_heights = [second_after, best_after]
    ax.bar(x - width / 2, pre_heights, width, yerr=pre_errs, capsize=6, ecolor='black',
          error_kw={'elinewidth': 1.5, 'capthick': 1.5}, color=['#C00000', '#4E95D9'],
          edgecolor='black', linewidth=2)
    rects_post = ax.bar(x + width / 2, post_heights, width, yerr=post_errs, capsize=6, ecolor='black',
                        error_kw={'elinewidth': 1.5, 'capthick': 1.5}, color=['#C00000', '#4E95D9'],
                        edgecolor='none')
    for r in rects_post:
        ax.add_patch(plt.Rectangle((r.get_x(), r.get_y()), r.get_width(), r.get_height(),
                                   fill=False, hatch='////', edgecolor='white', linewidth=1))
        ax.add_patch(plt.Rectangle((r.get_x(), r.get_y()), r.get_width(), r.get_height(),
                                   fill=False, edgecolor='black', linewidth=2))
    ax.set_xticks([0 - width / 2, 0 + width / 2, 1 - width / 2, 1 + width / 2])
    ax.set_xticklabels(['Pre', 'Post', 'Pre', 'Post'], fontsize=24, fontweight='bold', color='#444444')
    if include_ytitle:
        ax.set_ylabel('Accuracy (%)', fontsize=23, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.text(0, -0.16, 'Worse', ha='center', va='top', transform=ax.get_xaxis_transform(),
           fontsize=22, fontweight='bold')
    ax.text(1, -0.16, 'Better', ha='center', va='top', transform=ax.get_xaxis_transform(),
           fontsize=22, fontweight='bold')
    if include_legend:
        import matplotlib.patches as mpatches
        ax.legend(handles=[mpatches.Patch(facecolor='#9e9e9e', edgecolor='black', linewidth=1.5, label='Pre'),
                          mpatches.Patch(facecolor='#9e9e9e', edgecolor='black', linewidth=1.5,
                                         hatch='////', label='Post')],
                 loc='upper center', ncol=2, frameon=True, prop={'weight': 'bold', 'size': 21})

    label_pad, bracket_pad = 1.0, 9.0
    def _annotate(x_c, y, err, val):
        ax.annotate(f'{val:.1f}', (x_c, y + err + label_pad), ha='center', va='bottom',
                   fontsize=22, fontweight='bold')
    for xc, val, err in zip([0 - width / 2, 1 - width / 2], pre_heights, pre_errs[1]):
        _annotate(xc, val, err, val)
    for xc, val, err in zip([0 + width / 2, 1 + width / 2], post_heights, post_errs[1]):
        _annotate(xc, val, err, val)

    def star_for(p):
        if p < 0.001: return '***'
        if p < 0.01: return '**'
        if p < 0.05: return '*'
        return ''
    top_needed = max(max(pre_heights[i] + pre_errs[1][i], post_heights[i] + post_errs[1][i]) for i in range(2)) + label_pad + 5
    if add_sigstars:
        for xc, p, top in [(0, p_second, max(second_before + pre_errs[1][0], second_after + post_errs[1][0])),
                           (1, p_best, max(best_before + pre_errs[1][1], best_after + post_errs[1][1]))]:
            s = star_for(p)
            if s:
                y = top + bracket_pad
                ax.plot([xc - width / 2, xc + width / 2], [y, y], lw=1.5, c='black')
                ax.annotate(s, (xc, y), xytext=(0, 1), textcoords='offset points',
                          ha='center', va='bottom', fontsize=24, fontweight='bold')
                top_needed = max(top_needed, y + 7)
    ax.set_ylim(0, max(100, top_needed + (16 if include_legend else 0)))

    if info is not None:
        info.update(kind='pair_bars', group_col=group_col, n_groups=int(n),
                    interval=f'{int(ci*100)}% percentile bootstrap of the pooled accuracy, resampling the {n} {group_col} units',
                    worse=dict(pre=second_before, post=second_after, t=t_second, p_two_sided=p_second, star=star_for(p_second)),
                    better=dict(pre=best_before, post=best_after, t=t_best, p_two_sided=p_best, star=star_for(p_best)),
                    test='paired t-test on per-unit accuracies, two-sided')
    return fig, ax


# ---------------------------------------------------------------------------
# Fig. 2B / 3B — rebel/coalition (trio) diversity plot
# ---------------------------------------------------------------------------

def plot_rebel_coalition_accuracies(agg_df_with_error, ax=None, include_ytitle_left=True,
                                    include_ytitle_right=True, ci=0.95, capsize=5,
                                    shade_levels=None, info=None, items=None, interval='wilson'):
    """Error bars are 95% Wilson intervals computed from the per-level mean and
    item count (k = mean * n). When item-level scores are supplied, the
    uncertainty method also accounts for fractional scores (see below)."""
    z = _Z.get(ci, 1.959963984540054)
    a = agg_df_with_error.copy()
    n_col = 'count_coalition' if 'count_coalition' in a.columns else 'count'
    ns = a[n_col].astype(float)
    reb_pct = a['mean_rebel_accuracy'] * 100
    coal_pct = a['mean_coalition_accuracy'] * 100
    if items is not None:
        # Item-level scores (2-1 split items only): Wilson for 0/1 scores
        # (Fig. 3B), percentile bootstrap for fractional LLM scores (Fig. 2B).
        it = items[items['has_coalition']] if 'has_coalition' in items.columns else items
        by_lvl = {c: g for c, g in it.groupby('rebel_confidence')}
        lv = list(a['rebel_confidence'])
        reb_yerr = _item_yerr(reb_pct, [by_lvl[c]['rebel_accuracy'] for c in lv], interval, z, ci)
        coal_yerr = _item_yerr(coal_pct, [by_lvl[c]['coalition_accuracy'] for c in lv], interval, z, ci)
    else:
        reb_yerr = _asym_yerr(reb_pct, a['mean_rebel_accuracy'] * ns, ns, z)
        coal_yerr = _asym_yerr(coal_pct, a['mean_coalition_accuracy'] * ns, ns, z)

    if ax is None:
        fig, ax1 = plt.subplots(figsize=(8, 6))
    else:
        ax1 = ax; fig = ax1.figure
    _shade(ax1, shade_levels)
    ax1.errorbar(a['rebel_confidence'], reb_pct, yerr=reb_yerr,
                fmt='o-', capsize=capsize, label='Rebel Accuracy', color='darkorange')
    ax1.errorbar(a['rebel_confidence'], coal_pct, yerr=coal_yerr,
                fmt='s-', capsize=capsize, label='Coalition Accuracy', color='seagreen')
    ax1.set_xlabel('Rebel Confidence', fontsize=23, fontweight='bold')
    if include_ytitle_left:
        ax1.set_ylabel('Accuracy (%)', fontsize=23, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax2 = ax1.twinx()
    ax2.bar(a['rebel_confidence'], a[n_col], color='purple', alpha=0.3, width=0.3, align='center')
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))   # item counts
    if include_ytitle_right:
        ax2.set_ylabel('Number of Samples', fontsize=23, fontweight='bold', color='purple')
    ax2.tick_params(axis='y', colors='purple')
    ax1.legend(prop={'size': 20, 'weight': 'bold'}, loc='upper left')

    if info is not None:
        info.update(kind='trio_diversity',
                    interval=(f'{int(ci*100)}% percentile bootstrap over items per rebel-confidence level'
                              if interval == 'bootstrap' else f'{int(ci*100)}% Wilson score interval per rebel-confidence level'),
                    n_items=int(ns.sum()), n_per_level=dict(zip(a['rebel_confidence'].astype(int), ns.astype(int))),
                    shaded_levels=list(shade_levels or []), per_level=a)
    return fig, ax1


# ---------------------------------------------------------------------------
# Fig. 2D / 3D — trio rank (Best/Middle/Worst) pre/post bar chart + majority
# ---------------------------------------------------------------------------

def plot_rank_accuracies(rank_dict, majority_accuracies=None, ax=None, add_sigstars=True,
                         include_ytitle=True, fallback_when_single=True, single_trials=200, ci=0.95,
                         info=None, single_config_items=None):
    """single_config_items: for a single trio configuration (Exp 2), item-level
    scores {'Least'|'Middle'|'Best': {'pre': [...], 'post': [...]}, 'majority': [...]}.
    When given, every bar's interval is a percentile bootstrap over items
    (pre-discussion LLM scores are fractional self-consistency accuracies)."""
    groups = ['Least', 'Middle', 'Best']
    pre_colors = ['#C00000', '#2E8B57', '#4E95D9']

    def mean_ci(arr01):
        """Returns (mean_pct, lower_halfwidth_pct, upper_halfwidth_pct, n, used_fallback).

        With >= 2 units: symmetric t-based CI of the mean across units.
        With a single unit (Exp 2's one trio configuration): a Wilson score
        interval over `single_trials` items. Changed 2026-09-20 from a Wald
        (normal-approximation) interval so that every binomial interval in
        Figs. 2-3 uses the same estimator as the diversity panels; Wilson is
        asymmetric, hence the separate lower/upper half-widths."""
        if not arr01:
            return 0.0, 0.0, 0.0, 0, False
        a = np.asarray(arr01, dtype=float)
        n = len(a); p = float(a.mean()); mean_pct = p * 100.0
        alpha = 1.0 - ci
        z = norm.ppf(1 - alpha / 2)
        if n >= 2:
            se = (a.std(ddof=1) * 100.0) / np.sqrt(n)
            h = t_dist.ppf(1 - alpha / 2, n - 1) * se if se > 0 else 0.0
            return mean_pct, h, h, n, False
        if fallback_when_single and single_trials:
            lo, hi = wilson_ci(p * single_trials, single_trials, z)
            return mean_pct, mean_pct - 100.0 * lo, 100.0 * hi - mean_pct, 1, True
        return mean_pct, 0.0, 0.0, 1, False

    def item_boot(values):
        v = np.asarray(values, dtype=float)
        m = v.mean() * 100.0
        lo, hi = bootstrap_mean_ci(v, ci)
        return m, m - 100.0 * lo, 100.0 * hi - m, 1, True

    pre_means, post_means, pre_errs, post_errs, ns, fallbacks = [], [], [], [], [], []
    for g in groups:
        if single_config_items is not None:
            m_pre, lo_pre, hi_pre, n_pre, fb_pre = item_boot(single_config_items[g]['pre'])
            m_post, lo_post, hi_post, n_post, fb_post = item_boot(single_config_items[g]['post'])
        else:
            m_pre, lo_pre, hi_pre, n_pre, fb_pre = mean_ci(rank_dict[g]['pre'])
            m_post, lo_post, hi_post, n_post, fb_post = mean_ci(rank_dict[g]['post'])
        pre_means.append(m_pre); post_means.append(m_post); ns.append(n_pre); fallbacks.append(fb_pre)
        pre_errs.append((lo_pre, hi_pre))
        post_errs.append((lo_post, hi_post))
    pre_means, post_means = np.array(pre_means), np.array(post_means)
    # 2 x N (lower, upper) for matplotlib's asymmetric yerr
    pre_errs, post_errs = np.array(pre_errs).T, np.array(post_errs).T

    all_pre, all_post = [], []
    for rank in rank_dict:
        all_pre.extend(rank_dict[rank]['pre']); all_post.extend(rank_dict[rank]['post'])

    x_groups = np.arange(3); width = 0.35
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    ax.bar(x_groups - width / 2, pre_means, width, yerr=pre_errs, capsize=6, ecolor='black',
          error_kw={'elinewidth': 1.5, 'capthick': 1.5}, color=pre_colors, edgecolor='black', linewidth=1.5)
    ax.bar(x_groups + width / 2, post_means, width, yerr=post_errs, capsize=6, ecolor='black',
          error_kw={'elinewidth': 1.5, 'capthick': 1.5}, color=pre_colors, hatch='////',
          edgecolor='black', linewidth=1.5)

    have_majority = majority_accuracies is not None and len(majority_accuracies) > 0
    maj_mean = maj_err = maj_n = None
    if have_majority:
        if single_config_items is not None:
            maj_mean, maj_lo, maj_hi, maj_n, maj_fb = item_boot(single_config_items['majority'])
            maj_n = len(single_config_items['majority'])
        else:
            maj_mean, maj_lo, maj_hi, maj_n, maj_fb = mean_ci(majority_accuracies)
        maj_err = np.array([[maj_lo], [maj_hi]])
        x_maj = 3
        ax.bar(x_maj, maj_mean, width * 1.5, yerr=maj_err, capsize=6, ecolor='black',
              error_kw={'elinewidth': 1.5, 'capthick': 1.5}, color=COLORS['majority'], edgecolor='black', linewidth=1.5)
        ax.annotate(f'{maj_mean:.1f}', (x_maj, maj_mean + maj_hi + 1), ha='center', va='bottom',
                   fontsize=21, fontweight='bold')

    label_pad, bracket_pad = 1.0, 9.0
    top_needed = max(np.max(pre_means + pre_errs[1]), np.max(post_means + post_errs[1])) + label_pad + 5
    t_all = p_all = np.nan; s = ''
    if add_sigstars and len(all_pre) == len(all_post) and len(all_pre) >= 2:
        t_all, p_two = ttest_rel(all_post, all_pre)
        p_all = p_two
        s = '***' if p_all < 0.001 else ('**' if p_all < 0.01 else ('*' if p_all < 0.05 else ''))
        if s:
            left, right = x_groups[0] - width / 2, x_groups[-1] + width / 2
            y = max(np.max(pre_means + pre_errs[1]), np.max(post_means + post_errs[1])) + bracket_pad
            ax.plot([left, right], [y, y], lw=1.5, c='black')
            ax.annotate(s, ((left + right) / 2, y), xytext=(0, 1), textcoords='offset points',
                      ha='center', va='bottom', fontsize=24, fontweight='bold')
            top_needed = max(top_needed, y + 7)

    ax.set_ylim(0, max(100, top_needed))
    if include_ytitle:
        ax.set_ylabel('Accuracy (%)', fontsize=23, fontweight='bold')
    ax.set_xlabel('Agent Rank in Group', fontsize=22, fontweight='bold', labelpad=78)
    xticks, xtick_labels = [], []
    for i in range(3):
        xticks.extend([i - width / 2, i + width / 2]); xtick_labels.extend(['Pre', 'Post'])
    if have_majority:
        xticks.append(3); xtick_labels.append('Majority')
    ax.set_xticks(xticks); ax.set_xticklabels(xtick_labels, fontsize=23, fontweight='bold', color='#444444')
    for i, g in enumerate(['Worse', 'Middle', 'Better']):
        ax.text(i, -0.16, g, ha='center', va='top', transform=ax.get_xaxis_transform(),
               fontsize=21, fontweight='bold')
    if have_majority:
        ax.text(3, -0.16, 'Majority', ha='center', va='top', transform=ax.get_xaxis_transform(),
               fontsize=21, fontweight='bold')
    for xc, val, err in zip(x_groups - width / 2, pre_means, pre_errs[1]):
        ax.annotate(f'{val:.1f}', (xc, val + err + label_pad), ha='center', va='bottom', fontsize=21, fontweight='bold')
    for xc, val, err in zip(x_groups + width / 2, post_means, post_errs[1]):
        ax.annotate(f'{val:.1f}', (xc, val + err + label_pad), ha='center', va='bottom', fontsize=21, fontweight='bold')

    if info is not None:
        used_fallback = any(fallbacks)
        info.update(kind='rank_bars', n_per_rank=dict(zip(['Worse', 'Middle', 'Better'], ns)),
                    interval=(f'{int(ci*100)}% percentile bootstrap over items (single trio configuration)'
                              if single_config_items is not None else
                              f'{int(ci*100)}% Wilson score interval over {single_trials} items (single trio configuration)'
                              if used_fallback else f'{int(ci*100)}% t-based CI across subjects per rank'),
                    majority=dict(mean=maj_mean, n=maj_n),
                    overall=dict(n=len(all_pre), t=t_all, p_two_sided=p_all, star=s),
                    test='paired t-test on per-subject accuracies pooled across ranks, two-sided')
    return fig, ax
