"""Additional manuscript/response-letter descriptives from released records.

Durations sum response latencies, not elapsed wall-clock times. Correlations
are descriptive associations; they do not identify effects of fatigue.
"""
from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr
from exp4_selection_table import load

ROOT = Path(__file__).resolve().parents[1]


def main():
    d = load()
    print('=== Experiment 4 exposure categories (item-pooled human accuracy) ===')
    for label, g in d.groupby('grp'):
        print(f'{label}: n={len(g)}, '
              f'{100*g.is_correct_first_answer.mean():.1f}% -> '
              f'{100*g.is_correct_last_answer.mean():.1f}%')
    counts = d[d.grp == 'discussed'].groupby('participant').size().reindex(
        d.participant.unique(), fill_value=0)
    print(f'Full response cohort: {len(counts)} participants, '
          f'{(counts == 0).sum()} without qualifying discussions; '
          f'discussed items mean={counts.mean():.2f}, SD={counts.std(ddof=1):.2f}.')
    per = d.groupby('participant').agg(
        pre=('is_correct_first_answer', 'mean'),
        post=('is_correct_last_answer', 'mean'),
        initial_ms=('init_rt', 'sum'), final_ms=('final_rt', 'sum'))
    rho, p = spearmanr(per.initial_ms + per.final_ms, per.post - per.pre)
    print(f'Exp 4 total active time vs human gain: rho={rho:.4f}, p={p:.5f}, n={len(per)}')

    h = pd.read_csv(ROOT / 'data/medcom1_fulldata.csv')
    per = h.groupby(['control', 'group_num', 'subject']).agg(
        initial_ms=('rt_start', 'sum'), final_ms=('rt_end', 'sum'),
        pre=('ans_start_correct', 'mean'), post=('ans_end_correct', 'mean'))
    print('=== Experiment 3 durations (all recorded participants, before accuracy exclusions) ===')
    for condition, label in [(0, 'Discussion'), (1, 'No-discussion control')]:
        g = per.xs(condition, level='control')
        minutes = g.final_ms / 60000
        lo, hi = minutes.quantile([.25, .75])
        print(f'{label}: second-phase median={minutes.median():.2f} min, '
              f'IQR={lo:.2f}-{hi:.2f}, n={len(g)} participants')
        rho, p = spearmanr(g.initial_ms + g.final_ms, g.post - g.pre)
        print(f'{label} total active time vs gain: rho={rho:.4f}, p={p:.5f}')
        if condition == 0:
            trio_minutes = minutes.groupby('group_num').mean()
            print(f'{(trio_minutes > 60).sum()} of {len(trio_minutes)} trios '
                  'exceeded 60 min in the discussion/final-answer phase '
                  '(mean recorded duration across each trio).')


if __name__ == '__main__':
    main()
