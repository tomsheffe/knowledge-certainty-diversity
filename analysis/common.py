"""Shared preprocessing and metric functions for Experiments 1–4 and the
supplementary analyses. These functions do not modify input files.

Packaged inputs in ../data/ (see ../data/PROVENANCE.md):
  exp1_llm_duo_per_group.csv: finalized Experiment 1 model-pair results.
  exp2_llm_trio.csv: Experiment 2 model-trio results.
  exp3_human_trios.csv: Experiment 3 human-trio and control responses.
  exp4_human_llm_discussions.csv: Experiment 4 discussion records.

Temperature replications are in ../data/temperature_robustness/.
The completed three-model S1/S2 calibration records are in
../derived_data/s1_s2_temperature_confidence.csv.
"""
import os
import warnings
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

# ---------------------------------------------------------------------------
# Experiment 3 (human trios) preprocessing
# ---------------------------------------------------------------------------

def _aggregate_group_data(exp3_human_raw_data):
    group_data = exp3_human_raw_data.groupby(['group_num', 'question_number']).agg(
        premajority=('ans_start_correct', lambda x: int(x.sum() > (len(x) / 2))),
        postmajority=('ans_end_correct', lambda x: int(x.sum() > (len(x) / 2)))
    ).reset_index()
    group_mean = group_data.groupby('group_num').agg(
        mean_premajority=('premajority', 'mean'),
        mean_postmajority=('postmajority', 'mean')
    ).reset_index()
    group_mean['control'] = group_mean['group_num'].apply(
        lambda x: 1 if 'c' in str(x).lower() else 0)
    return group_mean


def _add_subject_rank(exp3_human_raw_data):
    subject_avg = exp3_human_raw_data.groupby(['group_num', 'subject'])['ans_start_correct'] \
        .mean().reset_index(name='avg_ans_start')

    def assign_rank(group):
        group_sorted = group.sort_values(by='avg_ans_start', ascending=False).reset_index(drop=True)
        if len(group_sorted) == 3:
            if group_sorted.loc[0, 'avg_ans_start'] == group_sorted.loc[1, 'avg_ans_start']:
                group_sorted.loc[0, 'rank'] = 'Best'
                group_sorted.loc[1, 'rank'] = 'Best'
                group_sorted.loc[2, 'rank'] = 'Middle'
            elif group_sorted.loc[1, 'avg_ans_start'] == group_sorted.loc[2, 'avg_ans_start']:
                group_sorted.loc[0, 'rank'] = 'Best'
                group_sorted.loc[1, 'rank'] = 'Middle'
                group_sorted.loc[2, 'rank'] = 'Middle'
            else:
                group_sorted.loc[0, 'rank'] = 'Best'
                group_sorted.loc[1, 'rank'] = 'Middle'
                group_sorted.loc[2, 'rank'] = 'Least'
        return group_sorted[['group_num', 'subject', 'rank']]

    rank_df = subject_avg.groupby('group_num').apply(assign_rank).reset_index(drop=True)
    exp3_human_raw_data = exp3_human_raw_data.merge(rank_df, on=['group_num', 'subject'], how='left')
    return exp3_human_raw_data.rename(columns={'rank': 'subject_rank'})


def _filter_threshold(df, threshold):
    subject_accuracy = df.groupby('subject')['ans_start_correct'].mean().reset_index(name='mean_accuracy')
    valid_subjects = subject_accuracy[subject_accuracy['mean_accuracy'] >= threshold]['subject']
    return df[df['subject'].isin(valid_subjects)]


def pre_process_exp3_human(exp3_human_raw_data, filter_threshold=0.62):
    """Compute group summaries and retain individuals meeting the accuracy threshold."""
    exp3_human_group_summary = _aggregate_group_data(exp3_human_raw_data)
    exp3_human_processed = _add_subject_rank(exp3_human_raw_data)
    exp3_human_processed_filtered = _filter_threshold(exp3_human_processed, filter_threshold)
    return exp3_human_group_summary, exp3_human_processed_filtered


def load_exp3_human(filter_threshold=0.62):
    raw = pd.read_csv(os.path.join(DATA_DIR, 'exp3_human_trios.csv'))
    return pre_process_exp3_human(raw, filter_threshold)


def load_exp3_human_trios():
    """Exp 3 discussion trios with ALL THREE original members (no accuracy
    exclusion), for trio-level analyses: rebel/coalition split, PCS, Fig. 3B,
    the Exp 3 column of Table 2, Fig. S5, and the partners in Fig. S7.

    Decided 2026-09-15. The individual-level exclusion (pre-discussion
    accuracy < 62%, 7 of 60 participants) applies to individual-level
    analyses only (per-participant pre/post, ranks, Fig. 3D). Applying it
    before the trio-level split left three of the ten discussion trios with
    two members, for which no majority exists; on their 1-1 splits
    `aggregate_questions` assigned rebel/coalition by row order, which in all
    three trios made the more accurate member the "rebel" (PCS 3.57 ->
    2.86 once all members are used). The excluded participants took part in
    the discussion and their answers formed the majority the trio faced,
    which is also how the majority-vote baseline (`_aggregate_group_data`
    on the raw data) has always been computed.

    Returns (group_summary, subjects) with subjects restricted to the
    discussion condition."""
    group_summary, subjects = load_exp3_human(filter_threshold=0.0)
    return group_summary, subjects[subjects['control'] == 0]


# ---------------------------------------------------------------------------
# Experiment 4 (human-LLM) preprocessing
# ---------------------------------------------------------------------------

def _add_participant_id(df):
    participant_ids = []
    current_id = 1
    prev_q = None
    for _, row in df.iterrows():
        current_q = row['doc.question_number']
        if prev_q is not None and current_q < prev_q:
            current_id += 1
        participant_ids.append(current_id)
        prev_q = current_q
    df = df.copy()
    df['participant_id'] = participant_ids
    return df


def _assign_agent_columns(exp4_human_llm_df):
    df = exp4_human_llm_df.copy()
    participant_acc = df.groupby('participant_id').agg(
        mean_user_acc=('user_before_correct', 'mean'),
        mean_bot_acc=('bot_before_correct', 'mean')
    ).reset_index()
    df = df.merge(participant_acc, on='participant_id', how='left')
    df['human_better'] = df['mean_user_acc'] > df['mean_bot_acc']

    # Bot confidence: a separately-run 10-rep self-consistency batch
    # (confidence_A-D, vote counts summing to 10) merged into this file;
    # answer_confidence_level = votes on whichever option the bot actually
    # answered with. Fixed 2026-09-19: for 16 of 562 discussed items this
    # merged field was NaN even though confidence_A-D and the chosen answer
    # were both present and would have given a valid, mostly nonzero value
    # (a gap in the upstream merge, not missing self-consistency data --
    # documented in data/PROVENANCE.md). Fill missing merged confidence
    # from the chosen-answer vote counts, so 15 of the 16 recover their value
    # (14 nonzero, 1 a genuine 0); only 1 of 562 is truly unrecoverable
    # (its chosen answer itself is missing) and is dropped. This raises
    # Experiment 4's single-confidence PCS from 7.3 to 8.0 pp.
    _recomputed = pd.Series(np.nan, index=df.index)
    for _letter in ['A', 'B', 'C', 'D']:
        _mask = df['doc.first_bot_answer_mapped'].astype(str).str.strip('()') == _letter
        _recomputed = _recomputed.where(~_mask, df[f'doc.confidence_{_letter}'])
    df['doc.answer_confidence_level'] = df['doc.answer_confidence_level'].combine_first(_recomputed)
    df = df[df['doc.answer_confidence_level'].notna()].copy()
    df['doc.answer_confidence_level'] = np.ceil(df['doc.answer_confidence_level'].fillna(0) / 2).astype(int)
    df['bot_answer_after'] = df['filtered_resps']
    df['bot_correct_after'] = df['filtered_resps'] == df['doc.target']

    hb = df['mean_user_acc'] > df['mean_bot_acc']
    df['best_agent_answer_before'] = np.where(hb, df['doc.init_question_answer_mapped'], df['doc.first_bot_answer_mapped'])
    df['best_agent_correct_before'] = np.where(hb, df['user_before_correct'], df['bot_before_correct'])
    df['best_agent_answer_after'] = np.where(hb, df['doc.final_question_answer_mapped'], df['bot_answer_after'])
    df['best_agent_correct_after'] = np.where(hb, df['user_after_correct'], df['bot_correct_after'])
    df['best_agent_confidence'] = np.where(hb, df['doc.init_question_confidence'], df['doc.answer_confidence_level'])
    df['second_agent_answer_before'] = np.where(hb, df['doc.first_bot_answer_mapped'], df['doc.init_question_answer_mapped'])
    df['second_agent_correct_before'] = np.where(hb, df['bot_before_correct'], df['user_before_correct'])
    df['second_agent_answer_after'] = np.where(hb, df['bot_answer_after'], df['doc.final_question_answer_mapped'])
    df['second_agent_correct_after'] = np.where(hb, df['bot_correct_after'], df['user_after_correct'])
    df['second_agent_confidence'] = np.where(hb, df['doc.answer_confidence_level'], df['doc.init_question_confidence'])
    return df


def load_exp4_human_llm():
    raw = pd.read_csv(os.path.join(DATA_DIR, 'exp4_human_llm_discussions.csv'))
    df = _add_participant_id(raw)
    return _assign_agent_columns(df)


# ---------------------------------------------------------------------------
# Experiment 1 (LLM pairs) preprocessing
# ---------------------------------------------------------------------------

def _assign_duo_agents_for_duo(df):
    agents = ["Alice", "David", "John"]
    group_list = []
    for grp, grp_df in df.groupby('group'):
        agent_scores = {}
        for agent in agents:
            col_ans = f"majority_vote_{agent}"
            col_conf = f"majority_vote_confidence_{agent}"
            if col_ans not in grp_df.columns or col_conf not in grp_df.columns:
                continue
            correct_mask = grp_df[col_ans] == grp_df['Correct_answer']
            agent_scores[agent] = correct_mask.mean()
        sorted_agents = sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)
        best_agent, second_agent = sorted_agents[0][0], sorted_agents[1][0]

        grp_df = grp_df.copy()
        grp_df['best_agent'] = best_agent
        grp_df['second_agent'] = second_agent
        grp_df['participant_id'] = grp
        grp_df['best_agent_answer_before'] = grp_df[f"majority_vote_{best_agent}"]
        best_cols = [f"question_start_i{i}_{best_agent}_is_correct" for i in range(5)]
        grp_df['best_agent_correct_before'] = grp_df[best_cols].mean(axis=1)
        grp_df['best_agent_confidence'] = grp_df[f"majority_vote_confidence_{best_agent}"]
        grp_df['second_agent_answer_before'] = grp_df[f"majority_vote_{second_agent}"]
        second_cols = [f"question_start_i{i}_{second_agent}_is_correct" for i in range(5)]
        grp_df['second_agent_correct_before'] = grp_df[second_cols].mean(axis=1)
        grp_df['second_agent_confidence'] = grp_df[f"majority_vote_confidence_{second_agent}"]
        grp_df['best_agent_answer_after'] = grp_df[f"question_end_{best_agent}_extracted_answer"]
        grp_df['best_agent_correct_after'] = grp_df[f"question_end_{best_agent}_is_correct"]
        grp_df['second_agent_answer_after'] = grp_df[f"question_end_{second_agent}_extracted_answer"]
        grp_df['second_agent_correct_after'] = grp_df[f"question_end_{second_agent}_is_correct"]
        group_list.append(grp_df)
    return pd.concat(group_list, ignore_index=True), group_list


def compute_majority_vote_columns(df, agents=('Alice', 'David', 'John')):
    """Derives majority_vote_{agent} (modal answer across the 5
    question_start_i0-i4_{agent}_extracted_answer self-consistency samples)
    and majority_vote_confidence_{agent} (agreement % with the mode, i.e. the
    self-consistency confidence definition stated in the paper's Methods) for
    raw per-iteration files that don't already carry these precomputed
    columns (unlike the main new_models_agg_reextract/duo_per_group.csv).
    Returns a copy of df with these columns added (0-100 scale, matching the
    scale used elsewhere before the /20 conversion in load_bot_bot_duo).
    """
    df = df.copy()
    for agent in agents:
        cols = [f'question_start_i{i}_{agent}_extracted_answer' for i in range(5)]
        if not all(c in df.columns for c in cols):
            continue
        ans_matrix = df[cols]

        def _mode_and_conf(row):
            vals = row.dropna()
            if len(vals) == 0:
                return pd.Series({'ans': np.nan, 'conf': np.nan})
            counts = vals.value_counts()
            top = counts.index[0]
            return pd.Series({'ans': top, 'conf': counts.iloc[0] / len(vals) * 100})

        res = ans_matrix.apply(_mode_and_conf, axis=1)
        df[f'majority_vote_{agent}'] = res['ans']
        df[f'majority_vote_confidence_{agent}'] = res['conf']
    return df


def load_bot_bot_duo():
    """Loads Exp 1 LLM-LLM pair data.

    NOTE: in this canonical file (new_models_agg_reextract/duo_per_group.csv),
    the 'group' column concatenates pair identity + benchmark dataset name
    (e.g. 'gpt_geminiMEDBULLETS'), giving 12 groups (3 pairs x 4 benchmarks)
    rather than 3. _assign_duo_agents_for_duo's groupby('group') therefore
    picks best/second agent per pair-per-dataset. These 12 combinations are
    the paper's unit of inference for Exp 1 (tests, Fig. 2C error bars,
    bootstrap). A 'pair_id' column (group name with the dataset_name suffix
    stripped) identifies the 3 model pairs.
    """
    # USMLE pre-discussion answers corrected; see data/PROVENANCE.md.
    df = pd.read_csv(os.path.join(DATA_DIR, 'exp1_llm_duo_per_group.csv'))
    conf_cols = [c for c in df.columns if 'confidence' in c]
    for c in conf_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce') / 20  # 0-100 -> 0-5
    parsed, per_group = _assign_duo_agents_for_duo(df)
    parsed['pair_id'] = parsed.apply(
        lambda r: r['group'][:-len(r['dataset_name'])] if r['group'].endswith(r['dataset_name']) else r['group'],
        axis=1)
    return parsed, per_group


# ---------------------------------------------------------------------------
# Experiment 2 (LLM trios) preprocessing
# ---------------------------------------------------------------------------

def parse_bot_bot_bot(bot_raw_data):
    subjects = ['David', 'John', 'Alice']
    rows = []
    for idx, row in bot_raw_data.iterrows():
        for subj in subjects:
            start_cols = [f'question_start_i{i}_{subj}_is_correct' for i in range(5)]
            vals = []
            for col in start_cols:
                if col in bot_raw_data.columns:
                    try:
                        vals.append(float(row[col]))
                    except Exception:
                        vals.append(0)
            avg_start = np.mean(vals) if vals else np.nan
            ans_start = row.get(f'majority_vote_{subj}')
            conf_val_start = row.get(f'majority_vote_confidence_{subj}', np.nan)
            ans_end = row.get(f'question_end_{subj}_extracted_answer')
            ans_end_correct = row.get(f'question_end_{subj}_is_correct')
            rows.append({
                'subject': subj, 'group_num': 'bot_group', 'control': 0,
                'question_number': idx,
                'ans_start': ans_start, 'ans_start_correct': avg_start,
                'conf_val_start': conf_val_start / 20,
                'ans_end': ans_end, 'ans_end_correct': ans_end_correct,
                '_avg_start': avg_start,
            })
    parsed_df = pd.DataFrame(rows)
    subj_avg = parsed_df.groupby('subject')['_avg_start'].mean().reset_index()
    subj_avg = subj_avg.sort_values('_avg_start', ascending=False).reset_index(drop=True)
    if subj_avg.loc[0, '_avg_start'] == subj_avg.loc[1, '_avg_start']:
        ranks = ['Best', 'Best', 'Middle']
    elif subj_avg.loc[1, '_avg_start'] == subj_avg.loc[2, '_avg_start']:
        ranks = ['Best', 'Middle', 'Middle']
    else:
        ranks = ['Best', 'Middle', 'Least']
    subj_avg['subject_rank'] = ranks
    parsed_df = parsed_df.merge(subj_avg[['subject', 'subject_rank']], on='subject', how='left')
    return parsed_df.drop(columns=['_avg_start'])


def load_bot_bot_trio():
    raw = pd.read_csv(os.path.join(DATA_DIR, 'exp2_llm_trio.csv'))
    return raw, parse_bot_bot_bot(raw)


def aggregate_bot_group_data(bot_raw_data):
    """Pre/post majority vote over LLM trio conversations."""
    subjects = ['David', 'John', 'Alice']
    rows = []
    for _, row in bot_raw_data.iterrows():
        pre_votes, post_votes = [], []
        for subj in subjects:
            vals = []
            for i in range(5):
                col = f'question_start_i{i}_{subj}_is_correct'
                if col in bot_raw_data.columns:
                    try:
                        vals.append(float(row[col]))
                    except Exception:
                        vals.append(0)
            pre_votes.append(1 if (vals and np.mean(vals) > 0.5) else 0)
            end_col = f'question_end_{subj}_is_correct'
            try:
                post_votes.append(1 if float(row[end_col]) == 1 else 0)
            except Exception:
                post_votes.append(0)
        rows.append({'premajority': 1 if sum(pre_votes) > 1 else 0,
                     'postmajority': 1 if sum(post_votes) > 1 else 0})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def pair_diversity_gain(df, filter_occurrences=0):
    """PCS for pairs: oracle picks max(best, second) accuracy per confidence
    level of the better agent."""
    d = df.copy()
    d['best_agent_confidence'] = d['best_agent_confidence'].astype(int)
    agg = d.groupby('best_agent_confidence').agg(
        mean_best=('best_agent_correct_before', 'mean'),
        mean_second=('second_agent_correct_before', 'mean'),
        count=('best_agent_correct_before', 'count'),
    ).reset_index()
    if filter_occurrences > 0:
        agg = agg[agg['count'] >= filter_occurrences]
    agg['combined'] = agg[['mean_best', 'mean_second']].max(axis=1)
    total = agg['count'].sum()
    best_points = (agg['mean_best'] * agg['count']).sum()
    combined_points = (agg['combined'] * agg['count']).sum()
    return (combined_points - best_points) / total, agg


def aggregate_questions(subjects_df):
    """Rebel/coalition split per (group, question)."""
    def process_group(g):
        unique_answers = g['ans_start'].unique()
        counts = g['ans_start'].value_counts()
        if len(unique_answers) == 3:
            coalition_answer, rebel_answer = counts.index[0], counts.index[1]
        elif len(unique_answers) != 2:
            coalition_answer = rebel_answer = counts.index[0]
        else:
            if counts.iloc[0] == 2:
                coalition_answer, rebel_answer = counts.index[0], counts.index[1]
            else:
                coalition_answer, rebel_answer = counts.index[1], counts.index[0]
        rebel_rows = g[g['ans_start'] == rebel_answer]
        coalition_rows = g[g['ans_start'] == coalition_answer]
        if rebel_rows.empty or coalition_rows.empty:
            return None
        rebel_row = rebel_rows.iloc[0]
        return {
            'rebel_accuracy': rebel_row['ans_start_correct'],
            'rebel_confidence': rebel_row['conf_val_start'],
            'coalition_accuracy': coalition_rows['ans_start_correct'].mean(),
            'coalition_max_confidence': coalition_rows['conf_val_start'].max(),
            # True only for a 2-1 split, the one case in which a coalition and
            # a rebel exist. Unanimous items (one answer) and items on which
            # all three members answered differently (8 of 200 in Experiment
            # 2; impossible in binary Experiment 3) have no rebel: the columns
            # above then come from an arbitrary member, so these items are
            # left out of every rebel/coalition table, but stay in the item
            # count N and in every observed-outcome measure. Three-way items
            # excluded 2026-09-21; unanimous items 2026-10-03 (they had been
            # binned by an arbitrary member's confidence, which left PCS
            # unchanged but diluted the per-level rebel/coalition accuracies
            # and the crossover tests).
            'has_coalition': len(unique_answers) == 2,
        }
    grouped = subjects_df.groupby(['group_num', 'question_number']).apply(process_group)
    grouped = grouped[grouped.notnull()]
    return pd.DataFrame(grouped.tolist(), index=grouped.index).reset_index()


def paired_sign_test(ref_acc, partner_acc):
    """Exact two-sided sign test of partner vs. reference accuracy on the same
    items. Counts items where the partner was more accurate and items where
    the reference was; ties carry no information and are left out. For 0/1
    accuracies with different answers (a 2-1 split in binary Experiment 3)
    this is McNemar's exact test. Returns (n_partner_better, n_ref_better, p)."""
    from scipy.stats import binomtest
    d = np.asarray(partner_acc, dtype=float) - np.asarray(ref_acc, dtype=float)
    k, m = int((d > 0).sum()), int((d < 0).sum())
    p = binomtest(k, k + m, 0.5).pvalue if k + m else float('nan')
    return k, m, float(p)


def aggregate_by_rebel_conf(agg_df):
    # Only 2-1 split items have a rebel and a coalition (see
    # `aggregate_questions`), so only they enter the per-level table. They remain part of the setting, so the full
    # item count travels with the aggregate as the PCS denominator N: such
    # items contribute no gain but are not removed from it. See
    # `aggregate_questions`'s `has_coalition` and `trio_diversity_gain`.
    n_total = len(agg_df)
    if 'has_coalition' in agg_df.columns:
        agg_df = agg_df[agg_df['has_coalition']]
    grouped = agg_df.groupby('rebel_confidence').agg(
        mean_rebel_accuracy=('rebel_accuracy', 'mean'),
        std_rebel_accuracy=('rebel_accuracy', 'std'),
        count_rebel=('rebel_accuracy', 'count'),
        mean_coalition_accuracy=('coalition_accuracy', 'mean'),
        std_coalition_accuracy=('coalition_accuracy', 'std'),
        count_coalition=('coalition_accuracy', 'count'),
    ).reset_index()
    grouped['se_rebel_accuracy'] = grouped['std_rebel_accuracy'] / np.sqrt(grouped['count_rebel'])
    grouped['se_coalition_accuracy'] = grouped['std_coalition_accuracy'] / np.sqrt(grouped['count_coalition'])
    grouped.attrs['total_items'] = n_total
    return grouped


def trio_diversity_gain(agg_with_err, total_items=None):
    """PCS for trios relative to the fixed coalition reference.


    `total_items` is the denominator N. Pass the full item count of the
    setting when `agg_with_err` was built from 2-1 split items only, so that
    items with no coalition dilute PCS rather than vanishing from it (they
    contribute no gain but remain in N). Defaults to the count in the
    aggregate, i.e. no dilution."""
    a = agg_with_err.copy()
    a['combined'] = a[['mean_rebel_accuracy', 'mean_coalition_accuracy']].max(axis=1)
    if total_items is None:
        total_items = agg_with_err.attrs.get('total_items')
    total = float(total_items) if total_items is not None else a['count_coalition'].sum()
    coal_pts = (a['mean_coalition_accuracy'] * a['count_coalition']).sum()
    comb_pts = (a['combined'] * a['count_coalition']).sum()
    return (comb_pts - coal_pts) / total


def multi_confidence_pcs(ref_conf, ref_acc, partner_conf, partner_acc, levels=(1, 2, 3, 4, 5),
                         baseline='rebel', total_items=None):
    """Multi-confidence (Fig. S5) PCS variant: an oracle grid conditioned on
    BOTH agents' confidence (a `len(levels) x len(levels)` cell mean-accuracy
    table), rather than only the reference agent's confidence.

    The cell-weighted calculation for baseline='rebel' is:
        oracle_mean(cell)   = max(ref_mean(cell), partner_mean(cell))
        diversity_gain_pp   = sum(oracle_mean * counts) - sum(ref_mean * counts)
                               ------------------------------------------------
                                              sum(counts)
    i.e. baseline='rebel' — gain is always measured against the reference
    agent's own cell-conditional accuracy. Single-confidence trio
    PCS likewise uses a fixed coalition reference. The legacy baseline='max'
    option is retained for the separately specified multi-confidence analysis.

    Parameters mirror the 1-D case: ref_* is the agent PCS is measured
    relative to; partner_* is the other agent. The caller selects the
    reference arrays and baseline convention.
    """
    d = pd.DataFrame({'ref_conf': ref_conf, 'partner_conf': partner_conf,
                      'ref_acc': ref_acc, 'partner_acc': partner_acc}).dropna(
        subset=['ref_conf', 'partner_conf'])
    minL, maxL = min(levels), max(levels)
    d['ref_conf'] = np.round(pd.to_numeric(d['ref_conf'])).clip(minL, maxL)
    d['partner_conf'] = np.round(pd.to_numeric(d['partner_conf'])).clip(minL, maxL)

    ref_mean = d.pivot_table(index='ref_conf', columns='partner_conf', values='ref_acc',
                             aggfunc='mean').reindex(index=levels, columns=levels)
    partner_mean = d.pivot_table(index='ref_conf', columns='partner_conf', values='partner_acc',
                                 aggfunc='mean').reindex(index=levels, columns=levels)
    counts = d.pivot_table(index='ref_conf', columns='partner_conf', values='ref_acc',
                           aggfunc='count').reindex(index=levels, columns=levels).astype(float).fillna(0.0)

    stacked = np.dstack([ref_mean.values, partner_mean.values])
    # Cells of the level x level grid that hold no items are all-NaN, which
    # nanmax warns about. That is expected here -- sparse cells are the norm
    # in a 5x5 grid -- and those cells carry a count of 0, so they contribute
    # nothing to the sum below. Suppress the warning rather than let it
    # clutter every run.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        oracle_mean = np.nanmax(stacked, axis=2)

    # `total_items` is the PCS denominator N. Pass the setting's full item
    # count when items with no coalition have been filtered out beforehand,
    # so they contribute no gain without leaving the denominator -- matching
    # the single-confidence definition. Added 2026-09-21.
    total_n = float(total_items) if total_items is not None else float(np.nansum(counts.values))
    ref_corrects = float(np.nansum(ref_mean.values * counts.values))
    oracle_corrects = float(np.nansum(oracle_mean * counts.values))

    if baseline == 'rebel':
        gain_corrects = oracle_corrects - ref_corrects
    elif baseline == 'max':
        partner_corrects = float(np.nansum(partner_mean.values * counts.values))
        gain_corrects = oracle_corrects - max(ref_corrects, partner_corrects)
    else:
        raise ValueError("baseline must be 'rebel' or 'max'")

    return (gain_corrects / total_n) if total_n > 0 else np.nan, {
        'ref_mean': ref_mean, 'partner_mean': partner_mean, 'counts': counts, 'total_n': total_n,
    }


def build_switch_dataframe(exp3_human_subjects, exp4_human_llm_df, bot_bot_duo_parsed, bot_bot_bot_parsed,
                           exp3_human_include_subjects=None):
    """Row-level (agent, context, switched, disagreed, conf, other_correct)
    table across all four experiments, for the confidence-sensitive-switching
    analysis (Fig. S7).

    exp3_human_include_subjects: optional set of Exp 3 subject ids. When given,
    only these participants contribute their OWN switching rows (switching is
    an individual-level behaviour, so the accuracy exclusion applies), while
    partners' initial answers are taken from every member of the trio
    (pass `load_exp3_human_trios()` subjects as exp3_human_subjects). Added
    2026-09-15; previously `subj3` was passed unfiltered, which pooled the
    no-discussion controls into the Human-trio row.
    """
    def _make_record(start, end, conf, partner_init, partner_init_correct, agent, context):
        disagreed = any(start != p for p in partner_init)
        other_correct = any(partner_init_correct)
        switched = disagreed and start != end and any(end == p for p in partner_init)
        return {'agent': agent, 'context': context, 'conf': pd.to_numeric(conf, errors='coerce'),
                'switched': int(switched), 'other_correct': int(other_correct), 'disagreed': int(disagreed)}

    records = []

    # 1. human - trio (Exp 3)
    for _, grp in exp3_human_subjects.groupby(['group_num', 'question_number']):
        for idx, row in grp.iterrows():
            if exp3_human_include_subjects is not None and row['subject'] not in exp3_human_include_subjects:
                continue
            partners = grp.loc[grp.index != idx, 'ans_start'].tolist()
            partners_correct = grp.loc[grp.index != idx, 'ans_start_correct'].tolist()
            records.append(_make_record(row['ans_start'], row['ans_end'], row['conf_val_start'],
                                        partners, partners_correct, 'Human', 'trio'))

    # 2. human - duo (Exp 4, human's own switching)
    for _, r in exp4_human_llm_df.iterrows():
        records.append(_make_record(
            r['doc.init_question_answer_mapped'], r['doc.final_question_answer_mapped'],
            r['doc.init_question_confidence'],
            [r['doc.first_bot_answer_mapped']], [r['bot_before_correct']], 'Human', 'duo'))

    # 3. LLM - duo (Exp 4, GPT-4 Turbo's own switching)
    for _, r in exp4_human_llm_df.iterrows():
        records.append(_make_record(
            r['doc.first_bot_answer_mapped'], r['filtered_resps'], r['doc.answer_confidence_level'],
            [r['doc.init_question_answer_mapped']], [r['user_before_correct']], 'LLM', 'duo'))

    # 4. LLM - duo (Exp 1, both roles)
    for pref, partner_pref in [('best_agent', 'second_agent'), ('second_agent', 'best_agent')]:
        for _, r in bot_bot_duo_parsed.iterrows():
            records.append(_make_record(
                r[f'{pref}_answer_before'], r[f'{pref}_answer_after'], r[f'{pref}_confidence'],
                [r[f'{partner_pref}_answer_before']], [r[f'{partner_pref}_correct_before']], 'LLM', 'duo'))

    # 5. LLM - trio (Exp 2)
    for _, grp in bot_bot_bot_parsed.groupby(['group_num', 'question_number']):
        for idx, row in grp.iterrows():
            partners = grp.loc[grp.index != idx, 'ans_start'].tolist()
            partners_correct = grp.loc[grp.index != idx, 'ans_start_correct'].tolist()
            records.append(_make_record(row['ans_start'], row['ans_end'], row['conf_val_start'],
                                        partners, partners_correct, 'LLM', 'trio'))

    return pd.DataFrame(records).dropna(subset=['conf'])


def summarise_switch_dataframe(df):
    return (df[df['disagreed'] == 1]
            .groupby(['agent', 'context', 'conf'], as_index=False)
            .agg(total_disagreed=('disagreed', 'sum'), total_switched=('switched', 'sum'))
            .assign(pct_switched=lambda d: 100 * d.total_switched / d.total_disagreed))


def compute_confidence_ame(rows_df, alpha=0.05):
    """AME (average marginal effect) of confidence on P(switch), fit via
    `switched ~ conf + other_correct` logit, separately per (agent, context)."""
    import statsmodels.formula.api as smf
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    recs = []
    for agent in ['Human', 'LLM']:
        for ctx in ['duo', 'trio']:
            sub = rows_df.query('agent == @agent and context == @ctx')
            if sub.empty:
                continue
            mdl = smf.logit('switched ~ conf + other_correct', data=sub).fit(disp=False)
            mfx = mdl.get_margeff(at='overall', method='dydx')
            ame, se_ame, p_ame = map(float, (mfx.margeff[0], mfx.margeff_se[0], mfx.pvalues[0]))
            recs.append({'Condition': f'{agent}-{ctx}', 'n': len(sub), 'AME_pp': ame * 100,
                        'CI_low_pp': (ame - z * se_ame) * 100, 'CI_high_pp': (ame + z * se_ame) * 100,
                        'p': p_ame})
    return pd.DataFrame(recs).sort_values('Condition')


def get_rank_accuracies(df):
    """Per-subject pre/post accuracy grouped by rank."""
    subject_stats = df.groupby(['group_num', 'subject']).agg(
        pre_accuracy=('ans_start_correct', 'mean'),
        post_accuracy=('ans_end_correct', 'mean'),
        subject_rank=('subject_rank', 'first'),
    ).reset_index()
    return {
        rank: {
            'pre': subject_stats.loc[subject_stats['subject_rank'] == rank, 'pre_accuracy'].tolist(),
            'post': subject_stats.loc[subject_stats['subject_rank'] == rank, 'post_accuracy'].tolist(),
        }
        for rank in ['Best', 'Middle', 'Least']
    }
