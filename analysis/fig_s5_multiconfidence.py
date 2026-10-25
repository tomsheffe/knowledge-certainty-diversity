"""Fig. S5: PCS conditioned on both agents' confidence.

Prints both baseline conventions implemented in common.multi_confidence_pcs:
'rebel' subtracts the supplied reference agent's weighted accuracy; 'max'
subtracts the larger of the reference and partner weighted accuracies.
The finalized figure pipeline uses 'max'. Values are computed from the
packaged data rather than repeated as hardcoded numerical summaries.
"""
import numpy as np
from common import (load_bot_bot_duo, load_bot_bot_trio, load_exp3_human, load_exp3_human_trios, load_exp4_human_llm,
                    aggregate_questions, multi_confidence_pcs)


def run():
    results = {}

    # --- Exp 1: LLM pairs (reference = best agent) ---
    parsed1, _ = load_bot_bot_duo()
    for baseline in ['rebel', 'max']:
        gain, _ = multi_confidence_pcs(
            parsed1['best_agent_confidence'], parsed1['best_agent_correct_before'],
            parsed1['second_agent_confidence'], parsed1['second_agent_correct_before'],
            baseline=baseline)
        results[f'exp1_{baseline}'] = gain * 100

    # --- Exp 2: LLM trios (reference = rebel, per the heatmap's default rebel_role) ---
    raw2, parsed2 = load_bot_bot_trio()
    agg_q2 = aggregate_questions(parsed2)
    _c2 = agg_q2[agg_q2['has_coalition']]
    for baseline in ['rebel', 'max']:
        gain, _ = multi_confidence_pcs(
            _c2['rebel_confidence'], _c2['rebel_accuracy'],
            _c2['coalition_max_confidence'], _c2['coalition_accuracy'],
            total_items=len(agg_q2),
            baseline=baseline)
        results[f'exp2_{baseline}'] = gain * 100

    # --- Exp 3: human trios (reference = rebel) ---
    _, trios_all3 = load_exp3_human_trios()             # trio-level: all members (2026-09-15)
    agg_q3 = aggregate_questions(trios_all3)
    _c3 = agg_q3[agg_q3['has_coalition']]
    for baseline in ['rebel', 'max']:
        gain, _ = multi_confidence_pcs(
            _c3['rebel_confidence'], _c3['rebel_accuracy'],
            _c3['coalition_max_confidence'], _c3['coalition_accuracy'],
            total_items=len(agg_q3),
            baseline=baseline)
        results[f'exp3_{baseline}'] = gain * 100

    # --- Exp 4: human-LLM pairs (reference = best agent) ---
    df4 = load_exp4_human_llm()
    for baseline in ['rebel', 'max']:
        gain, _ = multi_confidence_pcs(
            df4['best_agent_confidence'], df4['best_agent_correct_before'],
            df4['second_agent_confidence'], df4['second_agent_correct_before'],
            baseline=baseline)
        results[f'exp4_{baseline}'] = gain * 100

    return results


if __name__ == '__main__':
    r = run()
    print("=== Fig. S5: multi-confidence PCS ===")
    print(f"Exp1 LLM pairs:      rebel={r['exp1_rebel']:.2f}%  max={r['exp1_max']:.2f}%")
    print(f"Exp2 LLM trios:      rebel={r['exp2_rebel']:.2f}%  max={r['exp2_max']:.2f}%")
    print(f"Exp3 human trios:    rebel={r['exp3_rebel']:.2f}%  max={r['exp3_max']:.2f}%")
    print(f"Exp4 human-LLM pairs: rebel={r['exp4_rebel']:.2f}%  max={r['exp4_max']:.2f}%")
