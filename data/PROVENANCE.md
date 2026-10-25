# Data provenance

The files in this directory are release copies of the finalized analysis inputs. Numeric/categorical outcomes, confidence values, missingness and row ordering were retained. Original personal identifiers and free-text copies embedded in analysis tables were removed. No private crosswalk is distributed.

| Input | Role |
| --- | --- |
| `exp1_llm_duo_per_group.csv` | Finalized Experiment 1 model-pair results; includes the corrected extraction of USMLE initial answers. |
| `exp2_llm_trio.csv` | Experiment 2 trio records. |
| `exp3_human_trios.csv` | Experiment 3 individual responses, including intervention and no-discussion controls. Participant/group IDs are replaced consistently; group suffix `i`/`c` preserves condition. |
| `exp4_human_llm_responses.csv` | All 1,550 instances in the Experiment 4 analysis cohort. The redundant constant study-arm column is omitted. |
| `exp4_human_llm_discussions.csv` | 562 discussion records; the analysis loader applies its confidence completeness rule to obtain 561. `release_participant` links each record to the anonymous participant in the full table. The redundant constant study-arm column is omitted. |
| `temperature_robustness/` | Saved records used for temperature replications; T=0.8 `trio_temp08.csv` contains the older single-model export and is not the three-model source for S1/S2. |
| `fig_s8_midtier/` | Model-pair mid-tier replications. The corresponding trio records are in `derived_data/`. |
| `rerun_results/` | Additional saved question-set replications; retained configuration files contain model identifiers only. |
| `questions_datasets/` | Authors' English question sets: 50 pediatrics questions and 41 internal-medicine text questions in two-/four-option forms. The 42nd internal-medicine item (original question 25) requires two ECG images and is absent from the text-only exports. |

`derived_data/s1_s2_temperature_confidence.csv` contains the 200 items × three models × three temperatures from the original plotting inputs, including the completed three-model T=0.8 export. `item_id` hashes the original question text to permit matching without redistributing that text. The hashes are not participant identifiers.

`derived_data/s6_pediatrics_calibration.csv` preserves the broader Figure S6 calibration population across both arms. Its anonymous rows contain the human/model confidence and correctness fields needed for that analysis; missing model confidence remains missing.

Transcript redactions do not change analytical outcomes or joining keys. Relative response latencies are retained where required for the duration/fatigue checks; calendar timestamps are not released. Reproduction scripts compute results from these files rather than from the edited transcript text.

The transcript release preserves the saved source text after privacy redaction. Ten Experiment 4 source cells are visibly truncated; the release flags these with `source_text_incomplete` rather than implying that their endings are recoverable. The source discussion/postprocessing exports contain the same missing endings. The analysis uses recorded answers, confidences and outcomes, not reparsed transcript text.

The Experiment 3 transcript export contains 41 questions for each of ten trios (410 records). It contains no conversation record for question 25, the ECG-image item. Numerical responses for all 42 Experiment 3 questions remain in the analysis data. The release therefore provides all available records from the supplied transcript export, not a guaranteed complete chat archive.

## Experiment 4 cohort reconciliation

The original response export contains 31 participants in the main analysis arm and 35 in the other arm. A separate enrollment/demographics export labels the same 66 participants as 32 and 34: one participant is assigned to different arms in the two exports. The release follows the response-file cohort used by the finalized analysis and does not reassign that participant or invent missing responses. The two main Experiment 4 tables contain only this cohort, yielding 31 participants with responses, 30 with qualifying discussions, and one with none; their constant arm labels are omitted. Enrollment counts and analysis counts should not be treated as interchangeable; the historical arm discrepancy remains unresolved.

LLM item IDs and speaking order were recovered from the same source rows after checking exact equality of every previously released analytical column. Upstream IDs are populated only when saved in the source document (including MedBullets URL qid); no IDs were inferred from row numbers.
