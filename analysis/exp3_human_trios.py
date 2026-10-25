"""exp3 human trios: compute results from the packaged inputs.
"""
import numpy as np
from scipy.stats import ttest_rel, ttest_ind
from common import (load_exp3_human, load_exp3_human_trios, get_rank_accuracies, aggregate_questions,
                    aggregate_by_rebel_conf, trio_diversity_gain, paired_sign_test)

N_BOOT, SEED = 2000, 0


def run(filter_threshold=0.62):
    results = {}
    group_summary, subjects = load_exp3_human(filter_threshold)

    interv = subjects[subjects['control'] == 0]
    control = subjects[subjects['control'] == 1]
    results['n_interv_subjects'] = interv['subject'].nunique()
    results['n_control_subjects'] = control['subject'].nunique()

    def per_subject(df):
        return df.groupby('subject').agg(
            pre=('ans_start_correct', 'mean'), post=('ans_end_correct', 'mean')) * 100

    # Overall pre/post, intervention
    ps_i = per_subject(interv)
    results['interv_pre'], results['interv_pre_sd'] = ps_i['pre'].mean(), ps_i['pre'].std(ddof=1)
    results['interv_post'], results['interv_post_sd'] = ps_i['post'].mean(), ps_i['post'].std(ddof=1)
    t, p = ttest_rel(ps_i['pre'], ps_i['post'])
    results['interv_t'], results['interv_df'], results['interv_p'] = t, len(ps_i) - 1, p

    # Overall pre/post, control. The paired test here is the one to quote for
    # "the control group did not improve"; `best_control_*` below is a
    # different test, over the best-ranked control participants only.
    ps_c = per_subject(control)
    results['control_pre'], results['control_pre_sd'] = ps_c['pre'].mean(), ps_c['pre'].std(ddof=1)
    results['control_post'], results['control_post_sd'] = ps_c['post'].mean(), ps_c['post'].std(ddof=1)
    t, p = ttest_rel(ps_c['post'], ps_c['pre'])
    results['control_t'], results['control_df'], results['control_p'] = t, len(ps_c) - 1, p
    results['control_n'] = len(ps_c)

    # Best participants, intervention
    rank_i = get_rank_accuracies(interv)
    best_pre = np.array(rank_i['Best']['pre']) * 100
    best_post = np.array(rank_i['Best']['post']) * 100
    results['best_pre'], results['best_pre_sd'] = best_pre.mean(), best_pre.std(ddof=1)
    results['best_post'], results['best_post_sd'] = best_post.mean(), best_post.std(ddof=1)
    t, p = ttest_rel(best_pre, best_post)
    results['best_t'], results['best_df'], results['best_p'] = t, len(best_pre) - 1, p

    # Majority vote (pre), intervention groups
    maj_i = group_summary[group_summary['control'] == 0]['mean_premajority'] * 100
    results['majority_pre'] = maj_i.mean()
    # Best post vs majority: paired PER SUBJECT (n=11, matching paper's t(10) and
    # the 'best_t' n above) — NOT per group_num, which would collapse a trio's
    # two tied "Best" subjects into one row and lose a degree of freedom.
    best_subjects = interv[interv['subject_rank'] == 'Best'].groupby(['group_num', 'subject']).agg(
        best_post=('ans_end_correct', 'mean'))['best_post'] * 100
    maj_lookup = group_summary.set_index('group_num')['mean_premajority'] * 100
    maj_per_subject = best_subjects.reset_index()['group_num'].map(maj_lookup)
    t, p = ttest_rel(best_subjects.values, maj_per_subject.values)
    results['bestpost_vs_majority_t'], results['bestpost_vs_majority_df'], results['bestpost_vs_majority_p'] = \
        t, len(best_subjects) - 1, p

    # Best control participants
    rank_c = get_rank_accuracies(control)
    bc_pre = np.array(rank_c['Best']['pre']) * 100
    bc_post = np.array(rank_c['Best']['post']) * 100
    results['best_control_pre'], results['best_control_post'] = bc_pre.mean(), bc_post.mean()
    t, p = ttest_rel(bc_pre, bc_post)
    results['best_control_t'], results['best_control_df'], results['best_control_p'] = t, len(bc_pre) - 1, p

    # Between-group improvements
    imp_i = ps_i['post'] - ps_i['pre']
    imp_c = ps_c['post'] - ps_c['pre']
    results['interv_improvement'], results['control_improvement'] = imp_i.mean(), imp_c.mean()
    t, p = ttest_ind(imp_i, imp_c)
    results['between_t'], results['between_df'], results['between_p'] = t, len(imp_i) + len(imp_c) - 2, p

    imp_best_i = best_post - best_pre
    imp_best_c = bc_post - bc_pre
    results['best_interv_improvement'], results['best_control_improvement'] = imp_best_i.mean(), imp_best_c.mean()
    t, p = ttest_ind(imp_best_i, imp_best_c)
    results['best_between_t'], results['best_between_df'], results['best_between_p'] = \
        t, len(imp_best_i) + len(imp_best_c) - 2, p

    # Rebel/coalition diversity (Fig 3B) — trio-level, so ALL THREE members
    # of each discussion trio (see load_exp3_human_trios; decided 2026-09-15).
    _, trios_all = load_exp3_human_trios()
    results['n_trio_members'] = trios_all['subject'].nunique()
    agg_q = aggregate_questions(trios_all)
    agg_conf = aggregate_by_rebel_conf(agg_q)
    results['pcs_trio'] = trio_diversity_gain(agg_conf) * 100
    results['rebel_table'] = agg_conf
    lvl5 = agg_conf[agg_conf['rebel_confidence'] == 5]
    if len(lvl5):
        results['rebel5_acc'] = lvl5['mean_rebel_accuracy'].iloc[0] * 100
        results['rebel5_se'] = lvl5['se_rebel_accuracy'].iloc[0] * 100
        results['coalition5_acc'] = lvl5['mean_coalition_accuracy'].iloc[0] * 100
        results['coalition5_se'] = lvl5['se_coalition_accuracy'].iloc[0] * 100
        results['n_lvl5'] = int(lvl5['count_coalition'].iloc[0])
        results['n_split_items'] = int(agg_q['has_coalition'].sum())
        # Rebel vs. coalition at level 5, on the 2-1 split items only (the
        # only items with a rebel), compared item by item: exact sign test,
        # the same test make_figures.py uses for the Fig. 3B caption. Changed
        # 2026-10-03 from a two-proportion z-test over all 62 level-5 items,
        # 34 of which were unanimous and had been binned by an arbitrary
        # member's confidence.
        sub5 = agg_q[agg_q['has_coalition'] & (agg_q['rebel_confidence'] == 5)]
        k, m, p = paired_sign_test(sub5['coalition_accuracy'], sub5['rebel_accuracy'])
        results['rebel5_rebel_better'], results['rebel5_coalition_better'] = k, m
        results['rebel5_p'] = p
        # Items within a trio are not independent: percentile CI for the
        # rebel's accuracy at level 5, resampling whole trios.
        rng = np.random.default_rng(SEED)
        trios = sub5['group_num'].unique()
        boot = []
        for _ in range(N_BOOT):
            pick = rng.choice(trios, size=len(trios), replace=True)
            b = np.concatenate([sub5.loc[sub5['group_num'] == t, 'rebel_accuracy'].values for t in pick])
            boot.append(b.mean() * 100)
        results['rebel5_ci'] = tuple(np.percentile(boot, [2.5, 97.5]))
        results['rebel5_n_trios'] = len(trios)
    return results


if __name__ == '__main__':
    r = run()
    print("=== Experiment 3: human trios ===")
    print(f"Subjects: intervention {r['n_interv_subjects']}, control {r['n_control_subjects']}")
    print(f"Intervention overall: {r['interv_pre']:.1f} (SD {r['interv_pre_sd']:.1f}) -> "
          f"{r['interv_post']:.1f} (SD {r['interv_post_sd']:.2f}), t({r['interv_df']})={r['interv_t']:.2f}, p={r['interv_p']:.4g}  "
          f"")
    print(f"Best: {r['best_pre']:.1f} (SD {r['best_pre_sd']:.1f}) -> {r['best_post']:.1f} (SD {r['best_post_sd']:.1f}), "
          f"t({r['best_df']})={r['best_t']:.2f}, p={r['best_p']:.4g}")
    print(f"Majority pre (intervention): {r['majority_pre']:.1f}")
    print(f"Best post vs majority: t({r['bestpost_vs_majority_df']})={r['bestpost_vs_majority_t']:.2f}, "
          f"p={r['bestpost_vs_majority_p']:.4g}")
    print(f"Control overall: {r['control_pre']:.1f} (SD {r['control_pre_sd']:.1f}) -> "
          f"{r['control_post']:.1f} (SD {r['control_post_sd']:.1f}), "
          f"t({r['control_df']})={r['control_t']:.2f}, p={r['control_p']:.3f} (n={r['control_n']})")
    print(f"Best control: {r['best_control_pre']:.1f} -> {r['best_control_post']:.1f}, "
          f"t({r['best_control_df']})={r['best_control_t']:.2f}, p={r['best_control_p']:.3f}")
    print(f"Improvement: intervention {r['interv_improvement']:.2f} vs control {r['control_improvement']:.2f}, "
          f"t({r['between_df']})={r['between_t']:.2f}, p={r['between_p']:.4g}")
    print(f"Best improvement: {r['best_interv_improvement']:.2f} vs {r['best_control_improvement']:.2f}, "
          f"t({r['best_between_df']})={r['best_between_t']:.2f}, p={r['best_between_p']:.4g}")
    if 'rebel5_acc' in r:
        lo, hi = r['rebel5_ci']
        print(f"Rebels@5 (2-1 split items only, n={r['n_lvl5']} of {r['n_split_items']} split items, "
              f"{r['rebel5_n_trios']} trios): rebel {r['rebel5_acc']:.1f}% vs coalition {r['coalition5_acc']:.1f}%; "
              f"rebel right on {r['rebel5_rebel_better']}, coalition on {r['rebel5_coalition_better']}, "
              f"exact sign test p={r['rebel5_p']:.4g}; rebel accuracy 95% CI (trio bootstrap) {lo:.1f}-{hi:.1f}%")
    print(f"PCS human trios: {r['pcs_trio']:.2f}%")
