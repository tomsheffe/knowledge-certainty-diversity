# Data de-identification

The human data use replacement participant/group identifiers. Direct names, phone numbers, chat identifiers, private crosswalks, demographics and calendar timestamps are excluded. Response durations are retained for timing analyses. Participant IDs are replacement labels used to join records within an experiment. Their numbering preserves the analysis ordering; original identifiers and the private mapping are not distributed.

Conversation exports are in `data/conversations/`:

- Experiment 3: 410 conversation records; 41 have privacy redactions.
- Experiment 4: 1,550 participant–question records, including 214 with no conversation text; 3 records have privacy redactions.

Conversation text was reviewed again with conservative removal of borderline personal health, age and clinical-training disclosures, and identifiable names and personal disclosures were replaced by explicit redaction markers. `privacy_redacted` marks affected records. Medical case descriptions are study-question content and should not be mistaken for participants' personal histories. Pseudonymous joining keys and statistical outcomes were retained; transcript redactions do not change the analysis inputs.

Checks included source-identifier matching, contact/credential/path pattern scans and contextual review of human dialogue. These measures reduce disclosure risk; no automated scan or textual review can guarantee that unrestricted free text is impossible to re-identify. Do not attempt to identify participants or combine the data with external identifying records.

Private review notes, original transcripts and identifier mappings are not part of this release. Third-party dataset rights remain separate from de-identification; see `THIRD_PARTY_NOTICES.md`.

Ten Experiment 4 records already end prematurely in the source export. They are retained with `source_text_incomplete=True`; no missing text has been invented or reconstructed. This limitation is separate from privacy redaction.

The Experiment 3 transcript export contains 41 questions for each of ten trios (410 records). It contains no conversation record for question 25, the ECG-image item. Numerical responses for all 42 Experiment 3 questions remain in the analysis data. The release therefore provides all available records from the supplied transcript export, not a guaranteed complete chat archive.

Hebrew grammar may convey gender even though demographic columns are excluded. Figure 4 links its trio speaker labels to replacement participant IDs to permit verification of the plotted responses; this is within-release linkage, not anonymity between records.
