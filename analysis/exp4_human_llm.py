"""exp4 human llm: compute results from the packaged inputs.
"""
import numpy as np
from scipy.stats import ttest_rel
from statsmodels.stats.proportion import proportions_ztest
from common import load_exp4_human_llm, pair_diversity_gain


def run():
    results = {}
    df = load_exp4_human_llm()
    results['n_items'] = len(df)
    results['n_participants'] = df['participant_id'].nunique()
    per_p_n = df.groupby('participant_id').size()
    results['items_per_participant_mean'] = per_p_n.mean()
    results['items_per_participant_sd'] = per_p_n.std(ddof=1)

    # Better/worse agent, paired across participants
    agg = df.groupby('participant_id').agg(
        best_pre=('best_agent_correct_before', 'mean'),
        best_post=('best_agent_correct_after', 'mean'),
        second_pre=('second_agent_correct_before', 'mean'),
        second_post=('second_agent_correct_after', 'mean'),
    ) * 100
    results['n_pairs'] = len(agg)
    for k in ['best_pre', 'best_post', 'second_pre', 'second_post']:
        results[f'{k}_mean'], results[f'{k}_sd'] = agg[k].mean(), agg[k].std(ddof=1)
    # Manuscript bars are pooled over items; SDs and paired tests describe
    # participant-level accuracies. Keep both estimands explicitly available.
    for k, col in [('best_pre', 'best_agent_correct_before'),
                   ('best_post', 'best_agent_correct_after'),
                   ('second_pre', 'second_agent_correct_before'),
                   ('second_post', 'second_agent_correct_after')]:
        results[f'{k}_participant_mean'] = results[f'{k}_mean']
        results[f'{k}_mean'] = df[col].mean() * 100
    t_b, p_b = ttest_rel(agg['best_post'], agg['best_pre'])
    t_s, p_s = ttest_rel(agg['second_post'], agg['second_pre'])
    results['t_better'], results['df_better'], results['p_better'] = t_b, len(agg) - 1, p_b
    results['t_worse'], results['df_worse'], results['p_worse'] = t_s, len(agg) - 1, p_s

    # Who started better: bot vs human. Uses the SAME definition as
    # _assign_agent_columns's 'human_better' (mean_user_acc > mean_bot_acc,
    # strict), so a tie counts as bot-better — consistent with best_agent
    # assignment. One participant (id 24) is an exact tie (80.6% both); with
    # a strict '>' on the bot side this participant would be silently
    # dropped from "bot better", undercounting by 1 (25 instead of 26).
    part_acc = df.groupby('participant_id').agg(
        mean_user_acc=('user_before_correct', 'mean'),
        mean_bot_acc=('bot_before_correct', 'mean'),
    ) * 100
    bot_better = part_acc[~(part_acc['mean_user_acc'] > part_acc['mean_bot_acc'])]
    results['n_bot_better'] = len(bot_better)
    results['n_total_pairs'] = len(part_acc)
    results['bot_better_bot_acc'] = bot_better['mean_bot_acc'].mean()
    results['bot_better_human_acc'] = bot_better['mean_user_acc'].mean()

    # GPT-4 Turbo accuracy when it was the better agent (item-level, paired by participant)
    sub = df.merge(bot_better.reset_index()[['participant_id']], on='participant_id')
    agg_bot = sub.groupby('participant_id').agg(
        bot_pre=('bot_before_correct', 'mean'), bot_post=('bot_correct_after', 'mean')) * 100
    t, p = ttest_rel(agg_bot['bot_post'], agg_bot['bot_pre'])
    results['bot_when_better_pre'], results['bot_when_better_pre_sd'] = agg_bot['bot_pre'].mean(), agg_bot['bot_pre'].std(ddof=1)
    results['bot_when_better_post'], results['bot_when_better_post_sd'] = agg_bot['bot_post'].mean(), agg_bot['bot_post'].std(ddof=1)
    results['bot_when_better_pre'] = sub['bot_before_correct'].mean() * 100
    results['bot_when_better_post'] = sub['bot_correct_after'].mean() * 100
    results['overall_bot_initial_accuracy'] = df['bot_before_correct'].mean() * 100
    results['overall_human_initial_accuracy'] = df['user_before_correct'].mean() * 100
    results['bot_when_better_t'], results['bot_when_better_df'], results['bot_when_better_p'] = t, len(agg_bot) - 1, p

    # Confidence-guided switching (best agent, on disagreement items)
    df_stats = df.copy()
    df_stats['best_agent_conf_rounded'] = df_stats['best_agent_confidence'].round().astype(int)
    low = df_stats[df_stats['best_agent_conf_rounded'].isin([0, 1, 2])]
    high = df_stats[df_stats['best_agent_conf_rounded'].isin([3, 4, 5])]

    def switch_stats(group):
        diff = group[group['best_agent_answer_before'] != group['second_agent_answer_before']]
        flips = diff[diff['best_agent_answer_after'] == diff['second_agent_answer_before']]
        return len(flips), len(diff)

    low_flip, low_total = switch_stats(low)
    high_flip, high_total = switch_stats(high)
    results['low_flip_rate'] = low_flip / low_total * 100 if low_total else np.nan
    results['high_flip_rate'] = high_flip / high_total * 100 if high_total else np.nan
    results['low_total'], results['high_total'] = low_total, high_total
    stat, pval = proportions_ztest(count=np.array([low_flip, high_flip]),
                                   nobs=np.array([low_total, high_total]), alternative='two-sided')
    results['switch_z'], results['switch_p'] = stat, pval

    # PCS
    gain, agg_div = pair_diversity_gain(df, filter_occurrences=0)
    results['pcs'] = gain * 100
    results['pcs_table'] = agg_div

    return results


if __name__ == '__main__':
    r = run()
    print("=== Experiment 4: human-LLM pairs ===")
    print("Accuracy means are item-pooled; SDs and paired tests are across participants.")
    print(f"N={r['n_participants']} participants, {r['n_items']} discussed items, "
          f"{r['items_per_participant_mean']:.1f}/subj (SD {r['items_per_participant_sd']:.1f})  "
          f"")
    print(f"Better: {r['best_pre_mean']:.1f}% (SD {r['best_pre_sd']:.1f}) -> "
          f"{r['best_post_mean']:.1f}% (SD {r['best_post_sd']:.1f}), t({r['df_better']})={r['t_better']:.2f}, p={r['p_better']:.2e}  "
          f"")
    print(f"Worse:  {r['second_pre_mean']:.1f}% (SD {r['second_pre_sd']:.1f}) -> "
          f"{r['second_post_mean']:.1f}% (SD {r['second_post_sd']:.1f}), t({r['df_worse']})={r['t_worse']:.2f}, p={r['p_worse']:.2e}  "
          f"")
    print(f"Bot better (including ties) in {r['n_bot_better']}/{r['n_total_pairs']} pairs.")
    print(f"Across all analyzed discussions: initial bot accuracy {r['overall_bot_initial_accuracy']:.1f}%, "
          f"human accuracy {r['overall_human_initial_accuracy']:.1f}%")
    print(f"Bot (when better) pre->post: {r['bot_when_better_pre']:.1f}% (SD {r['bot_when_better_pre_sd']:.1f}) -> "
          f"{r['bot_when_better_post']:.1f}% (SD {r['bot_when_better_post_sd']:.1f}), "
          f"t({r['bot_when_better_df']})={r['bot_when_better_t']:.2f}, p={r['bot_when_better_p']:.2e}  "
          f"")
    print(f"Switch rate: low-conf {r['low_flip_rate']:.1f}% (n={r['low_total']}) vs "
          f"high-conf {r['high_flip_rate']:.1f}% (n={r['high_total']}), Z={r['switch_z']:.3f}, p={r['switch_p']:.2e}  "
          f"")
    print(f"PCS: {r['pcs']:.2f}%")
