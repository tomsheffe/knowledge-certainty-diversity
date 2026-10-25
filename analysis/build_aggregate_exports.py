"""Export finalized result tables; run the named analyses first."""
from pathlib import Path
import shutil
ROOT = Path(__file__).resolve().parents[1]
FILES = {
    'table2.csv': 'make_table2.py',
    'table2.md': 'make_table2.py',
    'exp4_selection_table.csv': 'exp4_selection_table.py',
    'exp4_selection_table.md': 'exp4_selection_table.py',
    'group_level_pcs_vs_gain.csv': 'group_level_pcs_vs_gain.py',
    'rerun_pediatric_duos.csv': 'rerun_pediatric_duos.py',
}
def main():
    missing = [f'{name}: run analysis/{script}' for name, script in FILES.items()
               if not (ROOT/'outputs'/name).is_file()]
    if missing:
        raise SystemExit('Missing analysis outputs:\n' + '\n'.join(missing))
    target = ROOT/'data/aggregated'
    target.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        shutil.copyfile(ROOT/'outputs'/name, target/name)
    print(f'Exported {len(FILES)} aggregate/result files to {target}')
if __name__ == '__main__':
    main()
