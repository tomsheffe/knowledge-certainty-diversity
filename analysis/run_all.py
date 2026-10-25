"""Runs the main analysis scripts (Experiments 1-4 and the supplementary
figures' statistics) and writes their combined output to
../outputs/run_all_output.txt. Reads ../data/ only; writes to ../outputs/.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, '..', 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

SCRIPTS = [
    'exp1_llm_pairs.py',
    'exp2_llm_trios.py',
    'exp3_human_trios.py',
    'exp4_human_llm.py',
    'reported_descriptive_checks.py',
    # Fig. 5: group-level regression and cross-validation (~15 s).
    'group_level_pcs_vs_gain.py',
    'group_level_regression_cv.py',
    # Methods: how often the PCS switching set is a single threshold.
    'pcs_threshold_check.py',
    'fig_s5_multiconfidence.py',
    'fig_s6_calibration.py',
    'fig_s7_switching.py',
    'fig_s8_midtier_replication.py',
    'fig_s9_worse_agent_perspective.py',
    'fig_s1_s4_temperature_robustness.py',
    'exp4_selection_table.py',
]

chunks = []
failed = []
for script in SCRIPTS:
    result = subprocess.run([sys.executable, script], cwd=HERE, capture_output=True, text=True)
    chunks.append(result.stdout)
    if result.returncode != 0:
        failed.append(script)
        chunks.append(f"\n!!! {script} exited with code {result.returncode} !!!\n{result.stderr}\n")

output = '\n'.join(chunks)
print(output)

out_path = os.path.join(OUT_DIR, 'run_all_output.txt')
with open(out_path, 'w') as f:
    f.write(output)
print("\n[written to outputs/run_all_output.txt]")
print("\nTo generate quantitative figures: python analysis/make_figures.py")
print("For the S1/S2 temperature-confidence panels: python analysis/figure_s1_s2.py")
print("To regenerate Table 2, run: python3 make_table2.py  (a few minutes; it")
print("re-runs the bootstrap, permutation null and multi-confidence scripts)")
if failed:
    print('FAILED: ' + ', '.join(failed), file=sys.stderr)
    sys.exit(1)
