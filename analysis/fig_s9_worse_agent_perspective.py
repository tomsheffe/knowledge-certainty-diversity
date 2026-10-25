"""Fig. S9: condition on the worse agent's confidence, retaining the better
agent as the accuracy reference. The two-panel plot compares Experiments 1
and 4; this numerical helper reports Experiment 4's PCS and per-level table.
"""
from common import load_medcom2, pair_diversity_gain, multi_confidence_pcs


def run():
    df = load_medcom2()
    # S9 changes the conditioning confidence, while retaining the initially
    # better agent as the accuracy reference. Swapping accuracies changes
    # the estimand and incorrectly gives 26.74 pp instead of 2.50 pp.
    swapped = df.copy()
    swapped['best_agent_confidence'] = df['second_agent_confidence']

    gain, agg = pair_diversity_gain(swapped, filter_occurrences=0)
    multi_gain, _ = multi_confidence_pcs(
        df['second_agent_confidence'], df['best_agent_correct_before'],
        df['best_agent_confidence'], df['second_agent_correct_before'])
    return gain * 100, multi_gain * 100, agg


if __name__ == '__main__':
    single, multi, table = run()
    print("=== Fig. S9: Exp 4 diversity from the worse agent's perspective ===")
    print(f"Single-confidence PCS (better-agent baseline, worse-agent confidence): {single:.2f} pp")
    print(f"Multi-confidence PCS (better-agent baseline): {multi:.2f} pp")
    print("\nWorse agent's own accuracy vs. best agent's accuracy, by worse agent's confidence level:")
    print(table[['best_agent_confidence', 'mean_best', 'mean_second', 'count']].rename(columns={
        'best_agent_confidence': 'worse_agents_own_confidence',
        'mean_best': 'best_agent_accuracy', 'mean_second': 'worse_agent_accuracy',
    }).assign(worse_agent_accuracy=lambda d: (d['worse_agent_accuracy'] * 100).round(1),
             best_agent_accuracy=lambda d: (d['best_agent_accuracy'] * 100).round(1)).to_string(index=False))
    
