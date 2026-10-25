"""fig s1 s4 temperature robustness: compute results from the packaged inputs.
"""
import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
from common import DATA_DIR, _assign_duo_agents_for_duo, pair_diversity_gain, parse_bot_bot_bot, get_rank_accuracies

TEMP_DIR = os.path.join(DATA_DIR, 'temperature_robustness')


def load_duo_at_temp(temp_tag):
    df = pd.read_csv(os.path.join(TEMP_DIR, f'duo_{temp_tag}.csv'))
    conf_cols = [c for c in df.columns if 'confidence' in c]
    for c in conf_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce') / 20
    # Match Methods and the main Experiment 1 analysis: rank within each
    # model-pair × benchmark, then retain those twelve units for inference.
    df['group'] = df['group'].astype(str) + '::' + df['dataset_name'].astype(str)
    parsed, _ = _assign_duo_agents_for_duo(df)
    return parsed


def load_trio_at_temp(temp_tag):
    raw = pd.read_csv(os.path.join(TEMP_DIR, f'trio_{temp_tag}.csv'))
    return raw, parse_bot_bot_bot(raw)


def run():
    results = {}

    for temp_tag in ['temp02', 'temp05']:
        parsed = load_duo_at_temp(temp_tag)
        gain, _ = pair_diversity_gain(parsed, filter_occurrences=0)
        agg = parsed.groupby('participant_id').agg(
            best_pre=('best_agent_correct_before', 'mean'), best_post=('best_agent_correct_after', 'mean'),
            second_pre=('second_agent_correct_before', 'mean'), second_post=('second_agent_correct_after', 'mean'),
        ) * 100
        t_b, p_b = ttest_rel(agg['best_post'], agg['best_pre']) if len(agg) > 1 else (np.nan, np.nan)
        t_s, p_s = ttest_rel(agg['second_post'], agg['second_pre']) if len(agg) > 1 else (np.nan, np.nan)
        results[f'duo_{temp_tag}'] = {
            'pcs': gain * 100, 'n_pairs': len(agg),
            'best_pre': agg['best_pre'].mean(), 'best_post': agg['best_post'].mean(),
            'second_pre': agg['second_pre'].mean(), 'second_post': agg['second_post'].mean(),
            't_best': t_b, 'p_best': p_b,
            't_worse': t_s, 'p_worse': p_s,
        }

    for temp_tag in ['temp02', 'temp05']:
        raw, parsed = load_trio_at_temp(temp_tag)
        rank_acc = get_rank_accuracies(parsed)
        results[f'trio_{temp_tag}'] = {
            'best_pre': np.mean(rank_acc['Best']['pre']) * 100 if rank_acc['Best']['pre'] else np.nan,
            'best_post': np.mean(rank_acc['Best']['post']) * 100 if rank_acc['Best']['post'] else np.nan,
            'worst_pre': np.mean(rank_acc['Least']['pre']) * 100 if rank_acc['Least']['pre'] else np.nan,
            'worst_post': np.mean(rank_acc['Least']['post']) * 100 if rank_acc['Least']['post'] else np.nan,
        }

    return results


if __name__ == '__main__':
    r = run()
    print("=== Figs. S1-S4: temperature robustness (LLM-only settings) ===\n")
    print("LLM pairs — PCS and better-agent gain across temperatures:")
    for temp_tag in ['temp02', 'temp05']:
        d = r[f'duo_{temp_tag}']
        print(f"  T={temp_tag[-2]}.{temp_tag[-1]}: PCS={d['pcs']:.2f}%, n_pairs={d['n_pairs']}, "
              f"better {d['best_pre']:.1f}%->{d['best_post']:.1f}% (t={d['t_best']:.2f}, p={d['p_best']:.3f}), "
              f"worse {d['second_pre']:.1f}%->{d['second_post']:.1f}%")
    print("\nLLM trios — best/worst rank accuracy across temperatures:")
    for temp_tag in ['temp02', 'temp05']:
        d = r[f'trio_{temp_tag}']
        print(f"  T={temp_tag[-2]}.{temp_tag[-1]}: best {d['best_pre']:.1f}%->{d['best_post']:.1f}%, "
              f"worst {d['worst_pre']:.1f}%->{d['worst_post']:.1f}%")
    print("\nThe legacy single-model T=0.8 file is not a trio replication; S1/S2 use the completed three-model export.")
    print("\n(Qualitative claim to check: best-agent gain stays ~flat/non-significant, and no "
          "temperature produces the large, significant gains seen for human-involving groups.)")
