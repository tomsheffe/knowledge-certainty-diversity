"""Fig. 5 statistics — group-level regression of gain on PCS, with
cross-validated explained variance. Answers R2's request to estimate on one subset and evaluate on
held-out data. The held-out units are whole groups (pairs or trios), so a
group's PCS and gain never enter the fit that predicts it.

Input: ../outputs/group_level_pcs_vs_gain.csv (run group_level_pcs_vs_gain.py
first): one row per group, 56 groups -- Exp 1 (12 LLM pair x benchmark), Exp 2
(4 LLM trio x benchmark), Exp 3 (10 human trios), Exp 4 (30 human-LLM pairs).

1. OLS, gain ~ PCS, HC3 standard errors.
2. Out-of-sample R^2 = 1 - sum (y - yhat)^2 / sum (y - mean(y_train))^2 on the
   held-out groups, i.e. the improvement over predicting the training mean.
   Negative means worse than that. Two schemes, both stratified by setting:
     - 5-fold (80/20), 200 random fold assignments; sums pooled across folds
     - 70/30 splits, 1,000 repetitions
3. Permutation null for the 5-fold R^2: PCS shuffled across groups (1,000).
4. The same within the human settings, and within Exp 4, to separate
   between-setting from within-setting prediction; plus setting-only and
   setting + PCS models for comparison.
5. Share of the between-group variance in gain that item sampling alone
   would produce (Exp 3, Exp 4): mean over groups of var(item-level change) / n
   against the observed variance of the group gains.

Writes ../outputs/group_level_regression_cv_output.txt.
"""
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from common import load_medcom1_trios, load_medcom2, aggregate_questions

warnings.filterwarnings('ignore')  # sklearn: Exp 2 has 4 groups < 5 folds
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, '..', 'outputs')
SEED = 0
N_KFOLD_REP, N_SPLIT_REP, N_PERM = 200, 1000, 1000


def design(d, model):
    X = [np.ones(len(d))]
    if 'setting' in model:
        for s in SETTINGS[1:]:
            X.append((d['setting'] == s).astype(float).values)
    if 'pcs' in model:
        X.append(d['pcs'].values)
    return np.column_stack(X)


def fit_predict(train, test, model):
    beta, *_ = np.linalg.lstsq(design(train, model), train['gain'].values, rcond=None)
    return design(test, model) @ beta


def cv_r2(d, model, splitter, strata, pcs=None):
    """Mean over repetitions of the out-of-sample R^2 (sums pooled over the
    folds of one repetition for K-fold; one split per repetition otherwise)."""
    if pcs is not None:
        d = d.assign(pcs=pcs)
    y = d['gain'].values
    sse = sst = 0.0
    for tr, te in splitter.split(d, strata):
        pred = fit_predict(d.iloc[tr], d.iloc[te], model)
        sse += ((y[te] - pred) ** 2).sum()
        sst += ((y[te] - y[tr].mean()) ** 2).sum()
    return 1 - sse / sst


def repeated(d, model, scheme, strata, rng, n_rep, pcs=None):
    out = []
    for _ in range(n_rep):
        seed = int(rng.integers(1 << 31))
        if scheme == 'kfold':
            sp = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        else:
            sp = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
        out.append(cv_r2(d, model, sp, strata, pcs))
    return np.array(out)


def sampling_noise():
    """(observed variance of group gains, variance expected from item sampling)."""
    out = {}
    d4 = load_medcom2()
    d4['diff'] = (d4['best_agent_correct_after'].astype(float)
                  - d4['best_agent_correct_before'].astype(float)) * 100
    _, s3 = load_medcom1_trios()
    a3 = aggregate_questions(s3)
    # Trio gain: majority vote after minus before (same as the figure).
    post = (s3.groupby(['group_num', 'question_number'])['ans_end_correct']
            .agg(lambda x: float(x.mean() > 0.5)).rename('post').reset_index())
    a3 = a3.merge(post, on=['group_num', 'question_number'])
    a3['diff'] = (a3['post'] - (a3['coalition_accuracy'].astype(float) > 0.5)) * 100
    for label, d, unit in [('Exp 3: human trios', a3, 'group_num'),
                           ('Exp 4: human-LLM pairs', d4, 'participant_id')]:
        g = d.groupby(unit)['diff']
        n, v = g.size(), g.var(ddof=1)
        ok = n >= 2  # a group with one item has no within-group variance
        out[label] = (g.mean()[ok].var(ddof=1), (v[ok] / n[ok]).mean())
    return out


def summarise(r):
    return f"{r.mean():+.2f} [{np.percentile(r, 2.5):+.2f}, {np.percentile(r, 97.5):+.2f}]"


if __name__ == '__main__':
    df = pd.read_csv(os.path.join(OUT_DIR, 'group_level_pcs_vs_gain.csv'))
    SETTINGS = sorted(df['setting'].unique())
    rng = np.random.default_rng(SEED)
    lines = ["=== Fig. 5 statistics: group-level regression and cross-validation ==="]

    lines.append("=== 1. OLS, gain ~ PCS (HC3 SEs) ===")
    for name, d in [('All 56 groups', df),
                    ('Human settings (Exp 3 + 4)', df[df.setting.str.startswith(('Exp 3', 'Exp 4'))]),
                    ('Exp 4 only', df[df.setting.str.startswith('Exp 4')])]:
        m = smf.ols('gain ~ pcs', data=d).fit(cov_type='HC3')
        ci = m.conf_int().loc['pcs']
        lines.append(f"{name:28s} n = {len(d):2d}: slope = {m.params['pcs']:+.2f} pp per pp "
                     f"[{ci[0]:+.2f}, {ci[1]:+.2f}], p = {m.pvalues['pcs']:.3f}, "
                     f"in-sample R^2 = {m.rsquared:.2f}")
    m = smf.ols('gain ~ C(setting) + pcs', data=df).fit(cov_type='HC3')
    lines.append(f"{'All, controlling for setting':28s} n = 56: slope = {m.params['pcs']:+.2f} "
                 f"[{m.conf_int().loc['pcs'][0]:+.2f}, {m.conf_int().loc['pcs'][1]:+.2f}], "
                 f"p = {m.pvalues['pcs']:.3f}")

    lines.append("")
    lines.append("=== 2. Out-of-sample R^2 on held-out groups: mean [2.5th, 97.5th pct] over repetitions ===")
    lines.append(f"{'groups / model':44s} {'5-fold (80/20)':>24s} {'70/30 splits':>24s}")
    subsets = [('All 56', df),
               ('Human (Exp 3 + 4), 40', df[df.setting.str.startswith(('Exp 3', 'Exp 4'))]),
               ('Exp 4 only, 30', df[df.setting.str.startswith('Exp 4')])]
    kfold_all = None
    for name, d in subsets:
        d = d.reset_index(drop=True)
        strata = d['setting'].values
        models = [('PCS', 'pcs')]
        if d['setting'].nunique() > 1:
            models += [('setting only', 'setting'), ('setting + PCS', 'setting+pcs')]
        for mlabel, model in models:
            k = repeated(d, model, 'kfold', strata, rng, N_KFOLD_REP)
            s = repeated(d, model, 'split', strata, rng, N_SPLIT_REP)
            if name.startswith('All') and model == 'pcs':
                kfold_all = k
            lines.append(f"{name + ': ' + mlabel:44s} {summarise(k):>24s} {summarise(s):>24s}")

    lines.append("")
    lines.append("=== 3. Permutation null, all 56 groups, gain ~ PCS, 5-fold ===")
    d = df.reset_index(drop=True)
    strata = d['setting'].values
    obs = kfold_all.mean()
    null = np.array([repeated(d, 'pcs', 'kfold', strata, rng, 5,
                              pcs=rng.permutation(d['pcs'].values)).mean()
                     for _ in range(N_PERM)])
    p = (1 + (null >= obs).sum()) / (1 + N_PERM)
    lines.append(f"observed mean R^2 = {obs:+.3f}; null mean = {null.mean():+.3f}, "
                 f"95th pct = {np.percentile(null, 95):+.3f}; one-sided p = {p:.3f} ({N_PERM} shuffles)")

    lines.append("")
    lines.append("=== 4. Between-group variance in gain vs. item-sampling noise ===")
    for label, (obs_var, noise_var) in sampling_noise().items():
        lines.append(f"{label:24s} observed variance {obs_var:5.1f}; expected from item sampling "
                     f"{noise_var:5.1f}; share beyond sampling {max(0.0, 1 - noise_var / obs_var):.2f}")

    text = '\n'.join(lines)
    print(text)
    with open(os.path.join(OUT_DIR, 'group_level_regression_cv_output.txt'), 'w') as f:
        f.write(text + '\n')
