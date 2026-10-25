# Analysis code inventory

`analysis/` contains the finalized numerical and figure pipelines: PCS, baseline comparisons, paired tests, bootstrap/permutation analyses, cross-validation, selection, fatigue/duration and replication analyses. `PROTOCOLS.md` records the historical prompts and participant instructions.

The separately released LLM experiment framework and the human-study collection platform are excluded. Exploratory notebooks and obsolete analyses are superseded by the finalized scripts; notebook outputs, unrelated projects, development environments, private audit scripts and repository history are excluded.

The README lists the reproduction commands. Analysis dependencies were validated in a fresh environment; no API services or credentials are required. `analysis/build_aggregate_exports.py` exports finalized tables into `data/aggregated/` after their generating analyses have run.
