"""exp2 llm trios: compute results from the packaged inputs.
"""
import numpy as np
from common import (load_bot_bot_trio, aggregate_bot_group_data, get_rank_accuracies,
                    aggregate_questions, aggregate_by_rebel_conf, trio_diversity_gain)


def run():
    results = {}
    raw, parsed = load_bot_bot_trio()

    rank_acc = get_rank_accuracies(parsed)
    for rank, paper_label in [('Best', 'best'), ('Middle', 'middle'), ('Least', 'worst')]:
        results[f'{paper_label}_pre'] = np.mean(rank_acc[rank]['pre']) * 100 if rank_acc[rank]['pre'] else np.nan
        results[f'{paper_label}_post'] = np.mean(rank_acc[rank]['post']) * 100 if rank_acc[rank]['post'] else np.nan

    majority = aggregate_bot_group_data(raw)
    results['majority_pre'] = majority['premajority'].mean() * 100
    results['majority_post'] = majority['postmajority'].mean() * 100

    # Rebel/coalition diversity (Fig 2B) — uses majority-vote answers per model
    agg_q = aggregate_questions(parsed)
    agg_conf = aggregate_by_rebel_conf(agg_q)
    results['pcs_trio'] = trio_diversity_gain(agg_conf) * 100
    results['n_disagreements'] = len(agg_q)
    results['n_questions'] = raw.shape[0]
    results['rebel_table'] = agg_conf

    return results


if __name__ == '__main__':
    r = run()
    print("=== Experiment 2: LLM trios ===")
    print(f"Best:   {r['best_pre']:.1f}% -> {r['best_post']:.1f}%")
    print(f"Middle: {r['middle_pre']:.1f}% -> {r['middle_post']:.1f}%")
    print(f"Worst:  {r['worst_pre']:.1f}% -> {r['worst_post']:.1f}%")
    print(f"Majority vote pre: {r['majority_pre']:.1f}%")
    print(f"PCS trio: {r['pcs_trio']:.2f}%  "
          f"({r['n_disagreements']} disagreement items of {r['n_questions']} questions)")
    print("\nRebel vs coalition by rebel confidence:")
    print(r['rebel_table'][['rebel_confidence', 'mean_rebel_accuracy', 'mean_coalition_accuracy',
                            'count_coalition']].to_string(index=False))
