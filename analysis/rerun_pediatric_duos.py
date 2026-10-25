"""Additional LLM-pair comparisons using Experiment 4's paediatric items.

Pairs answer questions from the same 50-item, four-option set used with
human participants, under the Experiment 1 protocol: five self-consistency
samples with shuffled choices, six discussion messages, then a final answer.
The script uses the available saved rows in each run; a run may be incomplete.
These additional analyses are separate from the main experiments.

PCS uses the more accurate initial agent as reference, conditions on that
agent's confidence, and adopts the partner's answer at confidence levels
where the partner is more accurate. Initial accuracy is the fraction of
five samples answered correctly; confidence is the number agreeing with
the modal answer (1–5).

Reads  ../data/rerun_results/duo_pediatric/output_*/results.csv
Writes ../outputs/rerun_pediatric_duos.{txt,csv}
"""
import os
import glob
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_GLOB = os.path.join(HERE, '..', 'data', 'rerun_results', 'duo_pediatric',
                            'output_*', 'results.csv')
OUT_DIR = os.path.join(HERE, '..', 'outputs')
N_REPS = 5
EXPECTED_ITEMS = 50   # the paediatric set; shorter runs are incomplete
N_BOOT = 2000
RNG_SEED = 0


def _agent_tags(results_csv):
    """Map the config's person names to their model tags."""
    cfg = os.path.join(os.path.dirname(results_csv), 'base_config.json')
    with open(cfg) as f:
        persons = json.load(f)['persons']
    return {p['name']: p['tag'] for p in persons}


def _per_agent(d, name):
    """Pre-discussion accuracy (fraction of samples correct), confidence
    (samples agreeing with the modal answer), and post-discussion accuracy."""
    correct = pd.concat([d[f'question_start_i{i}_{name}_is_correct'].astype(float)
                         for i in range(N_REPS)], axis=1)
    answers = pd.concat([d[f'question_start_i{i}_{name}_extracted_answer'].astype(str)
                         for i in range(N_REPS)], axis=1)
    modal = answers.mode(axis=1)[0]
    return dict(pre=correct.mean(axis=1),
                conf=answers.eq(modal, axis=0).sum(axis=1),
                post=d[f'question_end_{name}_is_correct'].astype(float),
                first=answers.iloc[:, 0])


def pcs(conf, ref_acc, partner_acc):
    """PCS = sum_c (n_c/N) * max(0, a_P(c) - a_R(c)), in percentage points."""
    df = pd.DataFrame(dict(c=conf, r=ref_acc, p=partner_acc)).dropna()
    if len(df) == 0:
        return np.nan
    g = df.groupby('c').agg(n=('r', 'size'), a_r=('r', 'mean'), a_p=('p', 'mean'))
    return float((g['n'] / len(df) * np.maximum(0.0, g['a_p'] - g['a_r'])).sum() * 100)


def analyse(results_csv, rng):
    d = pd.read_csv(results_csv)
    tags = _agent_tags(results_csv)
    names = list(tags)
    a = {n: _per_agent(d, n) for n in names}

    # Reference agent = higher pre-discussion accuracy, as in `common.py`.
    ref, par = sorted(names, key=lambda n: -a[n]['pre'].mean())
    df = pd.DataFrame(dict(c=a[ref]['conf'], r=a[ref]['pre'], p=a[par]['pre']))
    obs = pcs(df['c'], df['r'], df['p'])
    boots = np.array([pcs(*(s := df.sample(len(df), replace=True,
                                           random_state=int(rng.integers(1e9))))[['c', 'r', 'p']].T.values)
                      for _ in range(N_BOOT)])

    disagree = (a[names[0]]['first'] != a[names[1]]['first']).mean() * 100
    return dict(
        config=os.path.basename(os.path.dirname(results_csv)),
        n_items=len(d),
        reference=tags[ref], partner=tags[par],
        ref_pre=a[ref]['pre'].mean() * 100, ref_post=a[ref]['post'].mean() * 100,
        par_pre=a[par]['pre'].mean() * 100, par_post=a[par]['post'].mean() * 100,
        ref_conf_top=float((a[ref]['conf'] == N_REPS).mean() * 100),
        disagreement=disagree,
        pcs=obs, pcs_lo=np.percentile(boots, 2.5), pcs_hi=np.percentile(boots, 97.5),
    )


def run():
    rng = np.random.default_rng(RNG_SEED)
    # A run still in progress has an empty results.csv; skip those rather
    # than failing, so this can be run while later configs are still going.
    files = [f for f in sorted(glob.glob(RESULTS_GLOB)) if os.path.getsize(f) > 0]
    if not files:
        raise SystemExit(f"no completed results found under {RESULTS_GLOB}")
    rows = []
    for f in files:
        name = os.path.basename(os.path.dirname(f))
        try:
            r = analyse(f, rng)
        except (pd.errors.EmptyDataError, KeyError) as e:
            print(f"skipping {name}: unreadable ({e})")
            continue
        # A config that died partway leaves a short results.csv. Reporting it
        # beside complete runs would put a 13-item PCS next to a 50-item one,
        # so drop it loudly rather than averaging it in.
        if r['n_items'] < EXPECTED_ITEMS:
            print(f"skipping {name}: incomplete, {r['n_items']:.0f}/{EXPECTED_ITEMS} items")
            continue
        rows.append(r)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    df = run()
    os.makedirs(OUT_DIR, exist_ok=True)
    lines = ["Matched LLM-LLM pairs on Experiment 4's 50 paediatric items", ""]
    for _, r in df.iterrows():
        lines += [
            f"{r['reference']}  +  {r['partner']}   (n = {r['n_items']:.0f} items)",
            f"  reference {r['reference']}: {r['ref_pre']:.1f}% -> {r['ref_post']:.1f}%"
            f"   ({r['ref_conf_top']:.0f}% of items at top confidence)",
            f"  partner   {r['partner']}: {r['par_pre']:.1f}% -> {r['par_post']:.1f}%",
            f"  initial disagreement: {r['disagreement']:.0f}%",
            f"  PCS = {r['pcs']:.2f} pp  [95% CI {r['pcs_lo']:.2f}, {r['pcs_hi']:.2f}]",
            "",
        ]
    lines += [f"Mean PCS across {len(df)} LLM-LLM pairs: {df['pcs'].mean():.2f} pp",
              "For comparison, human-LLM pairs on this item pool: PCS = 8.0 pp "
              "[5.9, 10.3] (Table 2).", ""]
    text = '\n'.join(lines)
    print(text)
    with open(os.path.join(OUT_DIR, 'rerun_pediatric_duos.txt'), 'w') as f:
        f.write(text)
    df.to_csv(os.path.join(OUT_DIR, 'rerun_pediatric_duos.csv'), index=False)
    print("written to outputs/rerun_pediatric_duos.{txt,csv}")
