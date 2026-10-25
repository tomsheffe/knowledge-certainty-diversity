# Data dictionary

Blank numeric cells mean unavailable data, not incorrect answers or zero confidence. Accuracy indicators are 1/True for correct and 0/False for incorrect. Answer labels are local to each question and should not be interpreted as clinical text.

## Human trios

`subject`: anonymous participant; `group_num`: anonymous assigned trio; suffix `c` denotes no-discussion control and `i` intervention. `control`: original condition indicator. `question_number`: original task item. `ans_start`/`ans_end`: initial/final answer; corresponding `_correct` fields: correctness. `conf_val_start`/`conf_val_end`: self-rated confidence (1–5). `rt_start`/`rt_end`: relative response durations in milliseconds. The file includes all 60 participants × 42 items before the analysis-specific exclusion rules in `common.py`.

## Human–LLM pairs

`participant` and `release_participant`: anonymous participant identifiers shared across the full selection table, discussion table and transcript file. `opening_message_id`: anonymous identifier for a recorded opening message; joins do not depend on transcript wording. `question_number`: original pediatrics item number.

`user_init_question_answer_mapped`/`user_final_question_answer_mapped`: initial/final participant answer labels. `user_init_question_confidence`/`user_final_question_confidence`: confidence (1–5). `target`: correct answer. `is_correct_first_answer`/`is_correct_last_answer`: correctness. `answer_changed`: answer-change indicator. `init_rt`/`final_rt`: relative response durations in milliseconds.

`first_bot_answer_mapped`: extracted opening model answer, missing if unavailable. `confidence_A` through `confidence_D`: counts from repeated independent model answers. `answer_confidence_level` and `answer_confidence_category`: saved confidence representations; use `common.py` for the final rounding/missingness rule. `had_conversation`: recorded discussion classification. `user_turn_count`/`assistant_turn_count`: counts derived from the raw transcript before redaction. `conversation_empty`: no recorded conversational text. The 14 unregistered participant replies remain in the opening-answer-shown category, consistent with the final selection analysis.

In the discussion table, `doc.` prefixes mark fields inherited from the saved discussion record; `doc.init_question_answer_mapped` is the participant's initial answer. `bot_before_correct`, `user_before_correct`, `user_after_correct` are correctness indicators. `filtered_resps` is the extracted model answer after the discussion. `doc.conv_length` is the saved conversation length. `doc.had_long_conversation` is a saved length category, not the selection-analysis definition of discussion.

## LLM-only records

`group`: model-pair/configuration label; `dataset_name`: source benchmark; `Correct_answer`: correct answer label. `question_start_i0_*` through `question_start_i4_*`: the five independent solo responses, with `_extracted_answer` and `_is_correct` fields. `question_end_*`: final response after discussion. Agent aliases Alice, David and John are model roles, not human participant names. Their models depend on experiment/replication; configurations are documented in the manuscript and retained rerun configuration files.

`majority_vote_*`: modal solo answer. `majority_vote_confidence_*`: agreement with that answer, usually stored as a percentage before loader rescaling. `answers_votes_*`: saved answer-vote counts. Main Experiment 1 and temperature replications rank the better agent within each model pair and benchmark. Trio PCS conditions on rebel confidence for 2–1 splits while retaining all items in the denominator.

## Derived records and transcripts

`s1_s2_temperature_confidence.csv`: one row per item/model/temperature, anonymous question-text hash, 1–5 confidence and binary correctness. `s6_pediatrics_calibration.csv`: anonymous broader-cohort `study_arm`, `human_confidence`, `human_correct`, `model_confidence`, `model_correct`. See `data/PROVENANCE.md` for why populations differ.

Transcript rows contain anonymous group/participant IDs, question number, conversation text and `privacy_redacted`. Explicit markers such as `[name redacted]` and `[personal information redacted]` replace identifying information. Empty conversations are retained to preserve the full Experiment 4 population. Some source conversations are serialized lists; these are data, never executable code.

In `data/conversations/experiment4.csv`, `source_text_incomplete` marks the ten source records with an unterminated serialized conversation. Treat `conversation` as text, not executable code; a list parser will fail on these records. A False flag means no such truncation was detected, not proof that every original message was recorded.

## LLM item linkage

Every LLM answer table includes `item_id` (SHA-256 of the exact UTF-8 `Original_Question` string in the saved run), `upstream_id` (the original dataset ID or MedBullets URL qid, where present; otherwise blank), and `Speaker_Order` (model aliases in speaking order). Hashes match exact question strings across runs; changes to wording or whitespace produce different hashes. These are benchmark item identifiers, never human participant identifiers. The source question wording is not redistributed.

One Experiment 1 source row contains response prose in `Speaker_Order` rather than a model order. Its release value is blank; no order is inferred. Some MedBullets source links are shortened URLs with no saved qid; these also retain a blank `upstream_id` and a complete question hash. A numeric upstream ID of zero is retained as a valid ID.
