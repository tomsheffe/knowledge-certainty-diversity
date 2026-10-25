# Knowledge-certainty diversity

Analysis code and de-identified research data for **Knowledge-Certainty Diversity Predicts Human-Human and Human-AI Open-Ended Discussion Synergy and Reveals Pure-AI Discussion Shortfalls**, by Tom Sheffer, Alon Miron, Asael Sklar, Yaniv Dover, and Ariel Goldstein.

This package reproduces the analyses from saved experimental records without API keys or model calls. The LLM experiment framework is released separately and is not included here. The code and data are separated from the original development repository and its history.

## Run the analyses

Run these commands from the repository root. The validated environment is Python 3.9.13; `requirements-lock.txt` pins the full dependency set. `requirements.txt` lists the direct analysis dependencies.

```sh
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python analysis/verify_release.py
python analysis/run_all.py
python analysis/make_table2.py
python analysis/exp4_selection_table.py
python analysis/make_figures.py
python analysis/make_figure_conversations.py
python analysis/figure_s1_s2.py
python analysis/rerun_pediatric_duos.py
python analysis/build_aggregate_exports.py
```

Analysis outputs are written to `outputs/`; the underlying experimental records are not modified. The aggregate exporter refreshes the summary copies in `data/aggregated/`. Table 2 uses 2,000 resamples and can take several minutes. The default analysis runner exits with a nonzero status if a component fails.

`make_figures.py` produces the main quantitative figures and supplementary figures S3–S9, plus a summary of the temperature analyses. `figure_s1_s2.py` produces the temperature-confidence calibration and item-matched scatter figures. Figure 4 is produced by `make_figure_conversations.py`. `reported_descriptive_checks.py` (included in the main runner) reproduces the exposure, phase-duration and fatigue-correlation summaries. Figure 5 is produced by `group_level_pcs_vs_gain.py` (called by the main runner). The conceptual Figure 1 and questionnaire screenshot Figure S10 are illustrations, not statistical analyses.

## Package contents

- `data/aggregated/`: finalized result tables and anonymous group-level PCS/gain records.
- `analysis/`: preprocessing, PCS definitions, confidence-based comparisons, paired tests, bootstrap/permutation analyses, cross-validation and figures.
- `data/`: categorical and numeric experimental records, including all five solo model answers where available; author-supplied question sets.
- `derived_data/`: anonymous sufficient records for the broader Figure S6 calibration sample, the mid-tier trio replication, and item-matched temperature confidence analyses.
- `PROTOCOLS.md`: historical model prompts and participant instructions.
- `data/conversations/`: de-identified human-trio and human–LLM conversations. Redactions are marked in the text and in `privacy_redacted`. The Experiment 3 transcript export covers 41 of 42 questions per trio; the ECG-item transcript is absent. Ten Experiment 4 transcripts with incomplete source text are explicitly flagged (see `DATA_DICTIONARY.md` and `data/PROVENANCE.md`).
- `data/PROVENANCE.md` and `DATA_DICTIONARY.md`: populations, transformations, variables and interpretation.

Experiment 4 selection analyses use 1,550 participant–question instances: 214 with no model response, 774 with an opening model answer but no qualifying discussion, and 562 qualifying discussions. The main PCS/performance analysis uses the 561 records with the required confidence information. Figure S6 intentionally uses the original broader calibration sample; it is not restricted to these discussions.

Demographic summaries cannot be independently regenerated from this public package because demographic fields are withheld for privacy.

The release excludes direct participant identifiers, demographics, wall-clock timestamps, private crosswalks, credentials, repository history and unreviewed raw exports. Third-party benchmark question wording and unrestricted copies of raw model responses are not included; the numerical model records are sufficient for the reported analyses. See `THIRD_PARTY_NOTICES.md` for benchmark sources and licence boundaries.

## Licensing and citation

Original code is licensed under the MIT License (`LICENSE`). Original research data and the authors' question materials are licensed under CC BY 4.0 (`DATA_LICENSE.txt`). Attribute the authors and paper, link to the licence and indicate modifications. These licences do not relicense third-party question banks or software dependencies.

See `CITATION.cff` for manuscript citation metadata. Publication identifiers should be added when assigned.

## Validation

The release was installed and the full analysis runner passed in a fresh Python environment. Table 2, the Experiment 4 selection table, and the 56 group-level numerical records matched the source analysis outputs. Structured non-identifying fields were checked against the source records. Human conversation text was reviewed for identifying disclosures, with explicit redaction markers; transcripts are separate from statistical inputs. `analysis/verify_release.py` checks the principal populations, joins and PCS values.

See `PRIVACY.md` for the de-identification process and `CODE_INVENTORY.md` for inclusion boundaries. The human-study collection platform is not included.

Additional response-letter checks are available in `analysis/fatigue_and_session_length.py`, `analysis/rerun_analysis.py` and `analysis/pcs_setting_level_cv.py`. They use the same saved data. The fatigue and rerun scripts print to the terminal; the setting-level validation script also writes files under `outputs/`. Keep the directory structure intact; the scripts resolve inputs relative to their own location.

## Uploading this folder

Use this folder as the repository root. It is self-contained: no parent-workspace files, API keys or original identifier mappings are required. Upload the contents, including the licence and documentation files. `MANIFEST.sha256` records the distributed file checksums; on macOS/Linux run `shasum -a 256 -c MANIFEST.sha256` before generating new outputs. Generated `outputs/`, environments and caches are ignored by Git. No Git history is included.
