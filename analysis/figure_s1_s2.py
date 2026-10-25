"""Reproduce S1/S2 from the complete item-matched temperature records."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/figures'
LABELS = {'Alice': 'Gemini 2.5 Pro', 'David': 'Claude Opus 4', 'John': 'GPT-4.1'}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(ROOT / 'derived_data/s1_s2_temperature_confidence.csv')
    assert not data.duplicated(['item_id', 'model', 'temperature']).any()
    assert data.groupby(['model', 'temperature']).size().eq(200).all()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, (model, label) in zip(axes, LABELS.items()):
        for temp, frame in data[data.model.eq(model)].groupby('temperature'):
            stats = frame.groupby('confidence').is_correct.agg(['mean', 'count'])
            ax.plot(stats.index, stats['mean'] * 100, 'o-', label=f'T={temp:g}')
        ax.set(title=label, xlabel='Confidence', xlim=(1, 5), ylim=(0, 100))
        ax.set_xticks(range(1, 6))
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Accuracy (%)')
    axes[-1].legend()
    fig.tight_layout()
    fig.savefig(OUT / 'figure_s1.png', dpi=300)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    results = []
    for col, (model, label) in enumerate(LABELS.items()):
        wide = data[data.model.eq(model)].pivot(index='item_id', columns='temperature', values='confidence')
        for row, temp in enumerate([.2, .5]):
            pairs = wide[[temp, .8]].dropna()
            r, p = spearmanr(pairs[temp], pairs[.8])
            results.append(dict(model=label, temperature=temp, reference_temperature=.8, n=len(pairs), rho=r, p=p))
            counts = pairs.groupby([temp, .8]).size().reset_index(name='n')
            ax = axes[row, col]
            ax.scatter(counts[temp], counts[.8], s=30 + 30 * counts.n, alpha=.55, edgecolors='black')
            ax.plot([1, 5], [1, 5], '--', color='grey')
            ax.set(title=f'{label}: rho={r:.2f}', xlabel=f'Confidence (T={temp:g})',
                   ylabel='Confidence (T=0.8)', xlim=(.8, 5.2), ylim=(.8, 5.2))
            ax.set_xticks(range(1, 6)); ax.set_yticks(range(1, 6)); ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT / 'figure_s2.png', dpi=300)
    plt.close(fig)
    pd.DataFrame(results).to_csv(ROOT / 'outputs/s1_s2_correlations.csv', index=False)
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == '__main__':
    main()
