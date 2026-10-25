"""Submission-style four-panel S3, S4 and S8, using packaged inputs only.

Run: python analysis/supplementary_replications.py (from reproduction/).
"""
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import (aggregate_questions, aggregate_by_rebel_conf,
                    aggregate_bot_group_data, get_rank_accuracies,
                    pair_diversity_gain, trio_diversity_gain, parse_bot_bot_bot)
from plotting import (plot_diversity_plots, plot_rebel_coalition_accuracies,
                      plot_agent_accuracies, plot_rank_accuracies)

ROOT = Path(__file__).resolve().parents[1]


def composite(pair, raw, trio):
    agg = aggregate_questions(trio)
    by = aggregate_by_rebel_conf(agg)
    majority = aggregate_bot_group_data(raw)
    fig, axs = plt.subplots(2, 2, figsize=(20, 18))
    A, B, C, D = {}, {}, {}, {}
    plot_diversity_plots(pair, ax=axs[0, 0], include_ytitle_right=False,
                         info=A, interval='bootstrap')
    plot_rebel_coalition_accuracies(by, ax=axs[0, 1], items=agg,
        interval='bootstrap', include_ytitle_left=False, info=B)
    plot_agent_accuracies(pair, ax=axs[1, 0], include_legend=False,
                         group_col='participant_id', info=C)
    items = {r: {'pre': trio.loc[trio.subject_rank == r, 'ans_start_correct'].astype(float).values,
                 'post': trio.loc[trio.subject_rank == r, 'ans_end_correct'].astype(float).values}
             for r in ['Least', 'Middle', 'Best']}
    items['majority'] = majority.premajority.values
    plot_rank_accuracies(get_rank_accuracies(trio), list(majority.premajority),
        ax=axs[1, 1], include_ytitle=False, single_config_items=items,
        add_sigstars=False, info=D)
    for ax, title in zip(axs.flat, ['A. LLM Pairs', 'B. LLM Trios',
                                  'C. LLM Pairs pre/post', 'D. LLM Trios pre/post']):
        ax.set_title(title, fontsize=27, fontweight='bold')
    fig.tight_layout()
    info = dict(A=A, B=B, C=C, D=D,
                pcs_pair=pair_diversity_gain(pair, filter_occurrences=0)[0]*100,
                pcs_trio=trio_diversity_gain(by)*100,
                n_pair_units=int(pair.participant_id.nunique()),
                n_trio=len(agg), n_split=int(agg.has_coalition.sum()))
    return fig, info


def build(number):
    if number in (3, 4):
        from fig_s1_s4_temperature_robustness import load_duo_at_temp, load_trio_at_temp
        tag = {3: 'temp02', 4: 'temp05'}[number]
        raw, trio = load_trio_at_temp(tag)
        pair = load_duo_at_temp(tag)
    elif number == 8:
        from fig_s8_midtier_replication import PAIRS, DATASETS, load_one
        pair = pd.concat([load_one(p, d) for p in PAIRS for d in DATASETS], ignore_index=True)
        raw = pd.read_csv(ROOT / 'derived_data/s8_midtier_trio.csv')
        trio = parse_bot_bot_bot(raw)
    else:
        raise ValueError(number)
    return composite(pair, raw, trio)


def save(number, out=None):
    out = Path(out) if out else ROOT / 'outputs/figures'
    out.mkdir(parents=True, exist_ok=True)
    fig, info = build(number)
    stem = out / f'figure_s{number}'
    fig.savefig(stem.with_suffix('.png'), dpi=150, bbox_inches='tight', facecolor='white')
    fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    stem.with_suffix('.json').write_text(json.dumps(info, indent=2,
        default=lambda x: x.item() if hasattr(x, 'item') else str(x)))
    return info


if __name__ == '__main__':
    for number in (3, 4, 8):
        info = save(number)
        print(f'S{number}: pair PCS={info["pcs_pair"]:.6f} pp; '
              f'{info["n_pair_units"]} pair–benchmark units; {info["n_trio"]} trio items')
