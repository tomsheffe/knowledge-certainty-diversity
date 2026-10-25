# Table 2

|                                         | LLM–LLM pairs (Exp 1)   | LLM trios (Exp 2)                 | Human trios (Exp 3)   | Human–LLM pairs (Exp 4)   |
|:----------------------------------------|:------------------------|:----------------------------------|:----------------------|:--------------------------|
| Items                                   | 600 (3 pairs × 200)     | 200 (1 trio; 192 with a majority) | 420 (10 trios × 42)   | 561 (30 pairs)            |
| Reference agent                         | Better agent            | Coalition                         | Coalition             | Better agent              |
| Reference pre-discussion accuracy (%)   | 80.3                    | 77.1                              | 78.6                  | 76.8                      |
| PCS (pp) [95% CI]                       | 0.0 [0.0, 0.6]          | 0.0 [0.0, 2.9]                    | 2.9 [0.5, 5.5]        | 8.0 [5.9, 10.3]           |
| Permutation null: mean PCS (pp)         | 0.01                    | 0.27                              | 0.04                  | 0.02                      |
| Permutation null: p                     | 1.000§                  | 1.000§                            | < 0.001               | < 0.001                   |
| Multi-confidence PCS (pp)               | 0.5                     | 1.0                               | 5.7                   | 10.7                      |
| Post-discussion change (pp), pre → post | -2.7 (80.3 → 77.7)      | +0.5 (78.0 → 78.5)‡               | +6.9 (78.6 → 85.5)†   | +5.9 (76.8 → 82.7)        |
| Paired test, p                          | 0.022                   | —                                 | 0.010                 | < 0.001                   |

**Table 2. Potential Conversation Synergy across the four experiments**
PCS, percentage points the reference agent would gain by taking the partner's answer at those levels of the designated confidence variable where the partner is more accurate. Reference agent, the initially better agent (pairs) or the majority answer (trios). 95% CI, percentile bootstrap. Permutation null, average PCS after shuffling the designated confidence labels; p, fraction of shuffles reaching the observed value. Multi-confidence PCS, PCS calculated over all combinations defined by both agents' levels (Fig. S5). Post-discussion change, the reference agent's accuracy before and after discussion (pairs), or the accuracy of the trio's majority vote before and after discussion (trios), with two-sided paired-test p across units (12 model pair–benchmark combinations, 10 trios, 30 pairs). All resampling procedures are detailed in Methods.
† Majority vote before vs. after discussion, paired across the ten trios. ‡ Single configuration, no test possible. § PCS is exactly zero, so every shuffle matches or exceeds it and the null test carries no information.
