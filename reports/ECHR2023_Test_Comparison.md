# ECHR2023 test results compared with PILOT

The proposed model was evaluated on 2,999 processed test cases. The saved model selected by validation came from epoch 5. The values below come from the user's completed test console output. Published reference values come from Section 5 and Table 1 of *PILOT: Legal Case Outcome Prediction with Case Law* (haneesh1.pdf, NAACL 2024). They describe different models.

PILOT reports a mean and standard deviation across five seeds. The project result is from one training run. Differences below are percentage points. They describe observed values; they do not establish statistical significance or repeatability.

| Measure | Published PILOT mean ± SD | Proposed model | Observed difference |
| --- | ---: | ---: | ---: |
| Micro F1 | 71.50% ± 0.80 | 70.8269% | -0.6731 points |
| Micro Jaccard | 55.70% ± 1.00 | 54.8310% | -0.8690 points |
| Micro PR-AUC | 54.30% ± 1.40 | 75.6654% | +21.3654 points, subject to metric audit |
| Micro ROC-AUC | 83.10% ± 0.70 | 94.5364% | +11.4364 points, subject to metric audit |

The model has not shown the planned 2–3 percentage-point increase in micro F1. Its observed micro F1 is 0.67 points below the PILOT mean. Its micro Jaccard is also lower. The two AUC values are much higher numerically, but an overall superiority claim is not supported before checking equivalent score inputs, micro aggregation, metric definitions, labels, splits, and preprocessing. In particular, this evaluator uses continuous probabilities. It reports trapezoidal PR-AUC and average precision as separate measures. The PDF does not provide enough detail to verify the original AUC implementation.

The following measures are available for the proposed model. PILOT Table 1 does not supply matching values for these measures, so no numerical improvement can be calculated from that table.

| Additional measure | Proposed model | PILOT Table 1 |
| --- | ---: | --- |
| Micro precision | 72.13% | Not reported |
| Micro recall | 69.57% | Not reported |
| Macro precision | 54.82% | Not reported |
| Macro recall | 46.96% | Not reported |
| Macro F1 | 49.89% | Not reported |
| Exact-match accuracy | 45.62% | Not reported |
| Hamming loss | 5.51% | Not reported |
| Micro average precision | 75.67% | Not reported separately |
| Evaluation loss | 0.2193 | Not reported |

Precision, recall, and F1 are better when higher. Exact-match accuracy is better when higher. Hamming loss is better when lower. Evaluation loss can be compared only when the loss definition, weighting, and data are equivalent; its scale is not a percentage.

PILOT Table 1 does not supply per-article precision, recall, or F1. The article results below therefore describe the proposed model alone. Support is the number of positive examples for that article; a case can contain several positive articles.

| Article | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| 10 | 194 | 82.61% | 58.76% | 68.67% |
| 11 | 148 | 65.41% | 58.78% | 61.92% |
| 13 | 520 | 42.64% | 54.62% | 47.89% |
| 14 | 61 | 69.70% | 37.70% | 48.94% |
| 18 | 10 | 0.00% | 0.00% | 0.00% |
| 2 | 188 | 93.87% | 81.38% | 87.18% |
| 3 | 834 | 75.65% | 83.81% | 79.52% |
| 34 | 60 | 48.00% | 20.00% | 28.24% |
| 38 | 7 | 0.00% | 0.00% | 0.00% |
| 5 | 690 | 84.23% | 68.12% | 75.32% |
| 6 | 1081 | 74.07% | 73.73% | 73.90% |
| 7 | 9 | 0.00% | 0.00% | 0.00% |
| 8 | 421 | 76.38% | 69.12% | 72.57% |
| 9 | 26 | 85.71% | 69.23% | 76.60% |
| P1-1 | 347 | 78.81% | 76.08% | 77.42% |
| P4-7 | 21 | 0.00% | 0.00% | 0.00% |

The results support these inferences:

1. **Decisions at the saved threshold remain below the reference on F1 and Jaccard.** A good probability ranking can still produce incorrect binary decisions. Threshold calibration on validation is a possible future experiment, not a proven improvement. Do not choose thresholds by maximizing the known test results.
2. **Positive article decisions are still missed.** Micro recall 69.57% means about 30.43% of true article violations were missed. Micro precision 72.13% means about 27.87% of predicted article violations were incorrect. These percentages use different denominators.
3. **Performance varies substantially by article.** Macro F1 is 20.94 points below micro F1. Macro F1 gives each article equal weight; micro F1 pools all decisions. Articles 18, 38, 7, and P4-7 have no true-positive predictions. Together, they contain 47 positive article occurrences, which can overlap within cases. Article 13 is also weak despite 520 positive examples, so limited rare-class data is not the sole possible cause.
4. **Some article predictions are strong.** Article 2 has F1 87.18%, Article 3 has 79.52%, and P1-1 has 77.42%. Article 34 has F1 28.24%; its recall is only 20.00%. These are useful targets for an error review.
5. **Complete-case correctness remains limited.** Exact-match accuracy 45.62% means the entire article set was correct in about 46 of 100 cases. About 54 of 100 cases had at least one incorrect or missing article. Hamming loss measures individual case/article errors. Its low value can coexist with low complete-case accuracy when negative decisions are common. ROC-AUC measures ranking and does not replace case accuracy.
6. **These test scores alone do not establish overfitting.** Compare training and validation measures for the same selected epoch 5. The earlier epoch-8 validation snapshot and these epoch-5 test predictions come from different model states and splits; their difference is not a controlled overfitting measure.
7. **The AUC gap needs an audit.** Verify that the paper uses the same continuous scores, pooled micro aggregation, label set, and PR-AUC definition. Average precision and trapezoidal PR-AUC both round to 75.67% here, but their definitions are distinct. A metric-definition difference can change a comparison without changing the model.

Keep the following protocol differences in the research report:

- The project has 2,999 processed test cases; the paper reports 3,000. Record which case was removed and why when the original processing record is available. The reason is not established by this result table.
- The project checkpoint reads at most 382 content tokens per case. PILOT Appendix B describes LLM-generated FACT summaries with a maximum output length of 512 tokens. The 8,192-token reference in the older generic report note belongs to the separate Haneesh3/ECtHR-A experiment. That wording was incorrect for this first dataset. The metric values are unchanged by this correction.
- The proposed model uses dual token/span evidence. PILOT uses precedent retrieval and temporal-shift handling. This evaluation is a comparison with the published reference, not a PILOT reproduction.
- The published values average five seeds; the project result uses one run. A difference smaller than a reported standard deviation does not prove equivalence, and a positive numerical difference does not by itself prove a statistically significant improvement.

For the current delivery, retain the existing test scores and report the F1/Jaccard shortfall honestly. Treat the higher AUC values as provisional reference comparisons until the original evaluation method is verified. For a future experiment, inspect errors for Articles 13 and 34, review rare-label training support, and select any threshold or loss changes on validation. The current test results have already been observed and should not drive configuration selection.
