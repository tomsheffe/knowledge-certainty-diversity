"""Experiment 4: which items did participants choose to discuss?

Selection table for the human-LLM arm (1,550 participant x item instances,
31 participants), in three groups:

  discussed        the 562 instances in the paper's analysed set (chatbot flag;
                   all 562, including the one later dropped from PCS because
                   required confidence information was unavailable)
  LLM answer shown the LLM sent its opening answer and the item is not in the
                   discussed set (760 with no reply, plus 14 with a brief
                   participant reply that the chatbot did not record as a
                   discussion, so no post-discussion LLM answer exists)
  LLM never spoke  the conversation is empty; no LLM answer was shown or recorded

All 1,550 instances fall into one of the three groups. Model accuracy and disagreement use only instances with a
recorded LLM answer (valid-answer denominator), so a missing answer is never
counted as wrong or as a disagreement.

Tests: each measure is regressed on group by OLS with participant-clustered
standard errors (instances from one participant are not independent);
p-values compare the discussed group with each of the other two.

Writes ../outputs/exp4_selection_table.{md,csv}.
"""
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from common import DATA_DIR

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, '..', 'outputs')
GROUPS = ['LLM never spoke', 'LLM answer shown', 'discussed']


def load():
    df = pd.read_csv(os.path.join(DATA_DIR, 'exp4_human_llm_responses.csv'))
    # The release already contains only the Experiment 4 analysis cohort.
    raw = pd.read_csv(os.path.join(DATA_DIR, 'exp4_human_llm_discussions.csv'))
    key = (raw[['doc.question_number', 'opening_message_id', 'doc.init_question_answer_mapped']]
           .rename(columns={'doc.question_number': 'question_number',
                            'opening_message_id': 'opening_message_id',
                            'doc.init_question_answer_mapped': 'user_init_question_answer_mapped'})
           .drop_duplicates().assign(in_paper_set=1))
    df = df.merge(key, on=['question_number', 'opening_message_id', 'user_init_question_answer_mapped'], how='left')
    df['in_paper_set'] = df['in_paper_set'].fillna(0).astype(int)
    empty = df['conversation_empty']
    df['grp'] = np.select(
        [df['in_paper_set'] == 1, empty],
        ['discussed', 'LLM never spoke'], default='LLM answer shown')

    df['participant'] = df['participant'].astype('category').cat.codes   # anonymous id
    df['conf'] = df['user_init_question_confidence'].astype(float)
    df['acc'] = df['is_correct_first_answer'].astype(float)
    # Cohort accuracy on the question: mean initial accuracy of all participants
    # in this arm on that question (all 1,550 instances).
    df['cohort'] = df.groupby('question_number')['acc'].transform('mean')
    valid = df['first_bot_answer_mapped'].notna()
    df['model_acc'] = np.where(valid, (df['first_bot_answer_mapped'] == df['target']).astype(float), np.nan)
    df['disagree'] = np.where(valid, (df['user_init_question_answer_mapped']
                                      != df['first_bot_answer_mapped']).astype(float), np.nan)
    return df


def compare(df, col):
    """p-values, discussed vs. each other group, OLS with participant clusters."""
    d = df[df['grp'].isin(GROUPS)].dropna(subset=[col])
    out = {}
    for other in ['LLM answer shown', 'LLM never spoke']:
        sub = d[d['grp'].isin(['discussed', other])].copy()
        if sub.loc[sub['grp'] == other, col].empty:
            out[other] = np.nan
            continue
        sub['is_disc'] = (sub['grp'] == 'discussed').astype(float)
        m = smf.ols(f'{col} ~ is_disc', data=sub).fit(
            cov_type='cluster', cov_kwds={'groups': sub['participant']})
        out[other] = float(m.pvalues['is_disc'])
    return out


def fmt_p(p):
    if p != p:
        return '—'
    return '< 0.001' if p < 0.001 else f'{p:.3f}'


def build():
    df = load()
    g = {k: df[df['grp'] == k] for k in GROUPS}
    rows = []
    rows.append(('Participant–question instances',
                 [f"{len(g[k])} ({g[k]['participant'].nunique()} participants)" for k in GROUPS], None))
    rows.append(('Initial participant confidence (1–5), mean',
                 [f"{g[k]['conf'].mean():.2f}" for k in GROUPS], compare(df, 'conf')))
    rows.append(('Initial participant accuracy',
                 [f"{100 * g[k]['acc'].mean():.1f}%" for k in GROUPS], compare(df, 'acc')))
    rows.append(('Mean cohort accuracy on the question',
                 [f"{100 * g[k]['cohort'].mean():.1f}%" for k in GROUPS], compare(df, 'cohort')))

    def valid_cell(k, col):
        v = g[k][col].dropna()
        return '— (no LLM answer recorded)' if v.empty else f"{100 * v.mean():.1f}% (n = {len(v)})"
    rows.append(('Initial LLM accuracy (valid answers)',
                 [valid_cell(k, 'model_acc') for k in GROUPS], compare(df, 'model_acc')))
    rows.append(('Initial disagreement (valid answers)',
                 [valid_cell(k, 'disagree') for k in GROUPS], compare(df, 'disagree')))

    table = pd.DataFrame(
        [[name] + cells + ([fmt_p(p['LLM answer shown']), fmt_p(p['LLM never spoke'])] if p else ['', ''])
         for name, cells, p in rows],
        columns=['Measure', 'Non-discussed: LLM never spoke', "Non-discussed: LLM's answer shown", 'Discussed',
                 "p, discussed vs. answer shown", 'p, discussed vs. never spoke'])
    return table, len(df)


if __name__ == '__main__':
    table, n_all = build()
    note = (f"{n_all} instances in total. p-values: OLS of each measure on group, "
            f"participant-clustered standard errors.")
    print("=== Exp 4: selection of discussed items ===\n")
    print(table.to_string(index=False))
    print('\n' + note)
    os.makedirs(OUT_DIR, exist_ok=True)
    table.to_csv(os.path.join(OUT_DIR, 'exp4_selection_table.csv'), index=False)
    with open(os.path.join(OUT_DIR, 'exp4_selection_table.md'), 'w') as f:
        f.write('# Experiment 4: discussed vs. non-discussed items\n\n')
        f.write(table.to_markdown(index=False) + '\n\n' + note + '\n')
    print("\nwritten to outputs/exp4_selection_table.md and .csv")
