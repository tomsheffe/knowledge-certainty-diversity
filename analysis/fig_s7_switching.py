"""Fig. S7: confidence-sensitive switching after initial disagreement.

Uses common.build_switch_dataframe and common.compute_confidence_ame to
estimate confidence's average marginal effect in logistic switching models
across human/model agents and pair/trio settings. Experiment 4's separate
low-versus-high-confidence comparison is computed in exp4_human_llm.py.
"""
from common import (load_exp3_human, load_exp4_human_llm, load_bot_bot_duo, load_bot_bot_trio,
                    build_switch_dataframe, summarise_switch_dataframe, compute_confidence_ame)


def run():
    # Exp 3 (2026-09-15): discussion trios only (controls were previously
    # pooled in), own-switching rows for the included participants only
    # (individual-level exclusion applies), partners from all three members.
    from common import load_exp3_human_trios
    _, subj3 = load_exp3_human(0.62)
    included3 = set(subj3.loc[subj3['control'] == 0, 'subject'])
    _, trios_all3 = load_exp3_human_trios()
    df4 = load_exp4_human_llm()
    parsed1, _ = load_bot_bot_duo()
    _, parsed2 = load_bot_bot_trio()
    # Preserve the original S7 covariate: any partner with a correct
    # pre-discussion response. For fractional LLM scores, positive means
    # at least one of the five samples was correct. The caption now states
    # this explicitly; no modal-answer-correctness substitution is made.

    rows_df = build_switch_dataframe(trios_all3, df4, parsed1, parsed2,
                                     exp3_human_include_subjects=included3)
    summary = summarise_switch_dataframe(rows_df)
    ame = compute_confidence_ame(rows_df.loc[rows_df['disagreed'] == 1])
    return rows_df, summary, ame


if __name__ == '__main__':
    rows_df, summary, ame = run()
    print("=== Fig. S7: confidence-sensitive switching ===")
    print(f"\nRow-level switch dataset: {len(rows_df)} agent-question observations "
          f"({rows_df['disagreed'].sum()} in disagreement)")
    print("\nAverage marginal effect of a 1-unit rise in confidence on P(switch), "
          "conditional on disagreement (negative & significant = paper's claim):")
    print(ame.to_string(index=False))
    print("\n% switched by confidence level, agent x context (n=total disagreements per cell):")
    print(summary.pivot_table(index='conf', columns=['agent', 'context'], values='pct_switched').round(1))
