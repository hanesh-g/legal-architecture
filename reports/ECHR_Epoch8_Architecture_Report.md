# ECHR Legal Outcome Prediction

Epoch 8 results, model approach, and improvement plan

Prepared: 16 October 2026

Style: simplified technical English inspired by ASD-STE100. No formal compliance score is claimed.

## 1. Main finding

The model fits the training cases much better than the validation cases. Generalization remains the main problem.

Training micro F1 is 98.23%. Validation micro F1 is 74.51%. The gap is 23.72 percentage points. Validation macro F1 is 51.66%. Rare labels and some common labels remain difficult.

These figures describe epoch 8. They are not final test scores. The pasted record does not identify the selected checkpoint.

### Evidence used

This report uses the latest user-pasted epoch record, the earlier uploaded dataset audit, and the reviewed legal-architecture source. It describes the designed token-and-span model. It does not describe the separate TRACE precedent-retrieval model.

The pasted training record ends during the P1-1 metrics. Missing training fields were not reconstructed. The validation section and the training summary measures used here are complete.

## 2. Dataset

The task predicts ECHR article labels from legal case text. ECHR means the European Convention on Human Rights. A case can have several positive article labels. The task is therefore multi-label classification.

| Partition | Cases | Evidence |
| --- | --- | --- |
| Training | 8,138 | Confirmed in the supplied epoch record |
| Validation | 3,000 | Confirmed in the supplied epoch record |
| Test | 2,999 | Earlier uploaded split audit only; no new test results supplied |

The earlier files contain id, text, labels, and date fields. The earlier date audit placed training cases in 1968–2014, validation cases in 2014–2018, and test cases in 2018–2023. These periods were not rechecked for the current local run.

The 16 labels are 10, 11, 13, 14, 18, 2, 3, 34, 38, 5, 6, 7, 8, 9, P1-1, and P4-7. Keep their documented meanings and fixed order. Article support counts can exceed the number of cases because a case can have several labels.

The earlier training audit found only four Article 18 positives and five P4-7 positives. Article 6 had 4,969 positives. This class imbalance limits rare-label learning.

## 3. Model approach

The implementation uses two evidence paths. One path reads token states. The other path reads span states and their graph relations.

| Stage | What the implementation does | Why it is used |
| --- | --- | --- |
| Input preparation | Divide text into at most eight ordered spans. Pack one sequence with special tokens. | Keep a token-to-span map. |
| LegalBERT | Produce contextual token states and a CLS document state. | Represent legal text in context. |
| Dual Token-LCA | Use two learned label-query attention branches and a branch gate. | Produce token evidence for each label. |
| Span pooling | Calculate the mean of content token states in each retained span. | Create one state per span. |
| QRASR | Count 12 regex cue types. Add a learned, gated cue projection to each span. | Let legal cues influence span states. |
| Adaptive span graph | Mix semantic, span-order, and self-connection priors. Propagate span states. | Share information between related spans. |
| Span-LCA | Use learned label queries over graph-enhanced span states. | Produce span evidence for each label. |
| Gated fusion | Concatenate token evidence, gated span evidence, and CLS context. | Combine the two evidence paths. |
| Prediction head | Produce one score per label. Use sigmoid outputs and binary cross-entropy. | Allow several predicted articles per case. |

LCA means label-conditioned attention. QRASR is the cue-guided span-refinement stage. CLS is the special token used for document context.

MiniLM embeddings build semantic graph edges only. They are not classifier input features. MiniLM stays frozen. The graph connects spans within one case. It does not retrieve precedent cases.

The source PDF ends with a single-class softmax head. This ECHR experiment uses the explicit multi-label extension. Each sigmoid output is independent. The default decision threshold is 0.5; the supplied record does not confirm the actual threshold setting.

### Document coverage

The implementation permits at most 384 packed tokens, including two special tokens. It retains at most 382 content tokens. Leading truncation can remove later evidence. Some spans can disappear after truncation. Cue counts and MiniLM embeddings use only retained text.

Do not describe this implementation as unlimited full-document processing. Longer coverage requires a separate chunking design and a controlled experiment.

### Training controls

The repository supports AdamW, gradient clipping, dropout, optional layer freezing, separate learning rates, validation-based checkpoint selection, and epoch recovery saves. The pasted result confirms epoch 8 and multi-label mode. It does not confirm learning rates, seed, dropout, frozen layers, threshold, or the selected epoch. Obtain metadata.json before recording those values as verified settings.

## 4. Overall epoch-8 results

| Measure | Training | Validation |
| --- | --- | --- |
| Cases | 8138 | 3000 |
| Micro precision | 98.73% | 74.90% |
| Micro recall | 97.75% | 74.14% |
| Micro F1 | 98.23% | 74.51% |
| Macro precision | 94.64% | 53.51% |
| Macro recall | 91.03% | 50.87% |
| Macro F1 | 91.65% | 51.66% |
| Exact-match accuracy | 95.56% | 51.83% |
| Hamming loss — lower is better | 0.32% | 4.60% |

| Loss field | Value | Meaning |
| --- | --- | --- |
| training_loss | 0.017850 | Mean training-mode loss reported for epoch 8 |
| validation.loss | 0.304807 | Evaluation loss on the validation set |
| training.loss | Not supplied | The pasted record ends before this field |

Training-mode loss and evaluation loss use different dropout conditions. Do not treat their difference as a controlled loss comparison. The training and validation evaluation F1 measures provide the clearer comparison here.

### How to read the measures

- Precision measures how many predicted positive labels are correct.
- Recall measures how many true positive labels are found.
- F1 combines precision and recall. Micro F1 pools label decisions across cases.
- Macro F1 averages the 16 article F1 scores. Every article has equal weight.
- Exact-match accuracy requires the complete predicted article set to match the target set.
- Hamming loss measures the fraction of incorrect article decisions. It includes missed positives and false positives.

The validation micro F1 of 74.51% is not accuracy. Exact-match accuracy is 51.83%. Hamming loss is 4.60%. A low Hamming loss can coexist with weak rare-label results because many article decisions are negative.

### Measured generalization gaps

| Measure | Training minus validation |
| --- | --- |
| Micro F1 | 23.72 percentage points |
| Macro F1 | 39.99 percentage points |
| Exact-match accuracy | 43.73 percentage points |

These gaps show a substantial generalization problem. They are consistent with overfitting. Class imbalance and changes over time can also contribute. One epoch record cannot establish when validation stopped improving.

## 5. Validation results by article

Support is the number of validation cases with that positive article label. All values below come from the complete validation section.

| Article | Support | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| 10 | 151 | 80.95% | 67.55% | 73.65% |
| 11 | 80 | 72.50% | 72.50% | 72.50% |
| 13 | 506 | 41.28% | 31.82% | 35.94% |
| 14 | 48 | 50.00% | 54.17% | 52.00% |
| 18 | 9 | 0.00% | 0.00% | 0.00% |
| 2 | 180 | 85.14% | 82.78% | 83.94% |
| 3 | 796 | 82.28% | 88.07% | 85.07% |
| 34 | 46 | 16.67% | 4.35% | 6.90% |
| 38 | 4 | 20.00% | 25.00% | 22.22% |
| 5 | 608 | 83.06% | 75.82% | 79.28% |
| 6 | 1165 | 73.89% | 87.21% | 80.00% |
| 7 | 7 | 40.00% | 28.57% | 33.33% |
| 8 | 287 | 77.21% | 57.84% | 66.14% |
| 9 | 17 | 50.00% | 52.94% | 51.43% |
| P1-1 | 435 | 83.18% | 85.29% | 84.22% |
| P4-7 | 11 | 0.00% | 0.00% | 0.00% |

### What the results show

Article 3 has the highest validation F1 at 85.07%. P1-1 reaches 84.22%. Article 2 reaches 83.94%. These labels provide the strongest results in this snapshot.

Article 13 has 506 positive validation cases, but its F1 is 35.94%. Its recall is 31.82%. This weakness cannot be explained by a tiny validation sample alone. Inspect missing document evidence, label definitions, and the selected decision threshold.

Article 34 has recall of 4.35% and F1 of 6.90%. Its training F1 is 83.50%. Article 18 has training F1 of 80.00% but validation F1 of zero. These cases show weak transfer beyond the training set.

Articles 18 and P4-7 have zero recall. Article 38 has only four positive validation cases. Its 25% recall corresponds to one recovered positive case. Do not treat this small result as stable evidence of improvement.

Article 6 has recall of 87.21% and precision of 73.89%. It finds many positives but also produces false positives. Raising its threshold could trade recall for precision. Measure that trade on validation data only.

## 6. Improvement plan

Keep one unchanged baseline. Change one factor per experiment. Select settings from validation results, not test results.

| Priority | What to do | Why | How to check |
| --- | --- | --- | --- |
| 1. Verify the split | Audit IDs, normalized text, related cases, and prediction-time information. | A shared case can inflate performance. | Save a split manifest and file hashes. |
| 2. Select the right epoch | Read all epoch results and selected_epoch from metadata.json. | The final epoch need not generalize best. | Compare validation loss and the agreed primary F1 metric. |
| 3. Test smaller updates | Compare encoder rates 2e-5 and 1e-5. Keep the head rate fixed. | Smaller updates may preserve pretrained features. | Keep all other settings and splits unchanged. |
| 4. Test regularization | Compare dropout 0.1 and 0.2. Then test freezing zero or four lower layers. | The model fits training cases very closely. | Accept a setting only if validation improves. |
| 5. Address rare positives | Add labeled training cases. Separately test capped positive weights. | Four or five positives give a weak learning signal. | Track per-label recall and false positives. |
| 6. Test thresholds | Fit thresholds on validation predictions after model selection. | The default threshold may not fit every article. | Require enough positives and negatives for per-label tuning. |
| 7. Test document coverage | Measure truncation, then compare a chunk-based extension. | Later evidence may be omitted. | Use identical cases and training budgets. |
| 8. Test each component | Compare encoder-only, token attention, span attention, cues, graph, and full fusion. | An added module may not help this dataset. | Use the same metric, seeds, and split. |

A prior audit found one normalized text overlap between the original validation and test files. The cleaned validation copy had 2,999 records. This run reports 3,000. Confirm which files were used. The size alone does not prove that the old overlap remains, but it requires a new check.

Audit the Article 18 reference mapping. An earlier reference file used the title for Article 15. Preserve outcome codes until the source documentation confirms their meaning. Do not infer that the case labels themselves are wrong.

Class weighting can improve recall and also increase false positives. The current code offers capped positive weights. Automatic per-label threshold tuning and long-document chunking are proposed extensions, not verified features of this run.

Changing the seed is a repeatability check, not a way to select a favorable test result. Confirm promising settings with at least three fixed seeds. Report the mean and spread.

## 7. Recommended next experiment

First obtain the complete metadata.json and history.json for this run. Confirm the saved checkpoint and its selected epoch. Do not assume that model.pt contains the epoch-8 weights.

- Fix and record the data split before starting any comparison.
- Choose validation micro F1 as the primary objective if overall article decisions are the priority. Keep macro F1 as a required secondary measure.
- Repeat the baseline with fixed splits and the same eight-epoch budget.
- Change only the encoder learning rate from 2e-5 to 1e-5. Keep the head rate at the baseline value.
- Compare validation loss, micro F1, macro F1, and article-level errors.
- If the change helps, confirm it with additional seeds before testing another change.
- Evaluate the final selected model once on untouched test data. Use new held-out cases if repeated test feedback has guided the experiments.

If the faculty requires eight epochs, complete all eight and retain the best validation checkpoint. Do not claim that checkpoint selection removes overfitting. It avoids using a worse final epoch when an earlier model is better.

### What the software already provides

The updated repository saves training-state.pt and history.json after each completed epoch and its evaluations. The recovery state includes the latest model, optimizer, random states, history, and best model. The final model.pt is the selected model. Resume uses the recovery state, not the final prediction file.

These recovery features protect completed work. They do not increase model accuracy by themselves.

## 8. Conclusion

At epoch 8, the model achieved validation micro F1 of 74.51%, macro F1 of 51.66%, and exact-match accuracy of 51.83%. Training scores were much higher. The model therefore needs better generalization, not simply more training epochs.

Prioritize split verification, best-epoch selection, document coverage, and rare-label data. Use controlled experiments to test learning rates, dropout, and class weighting. No proposed change guarantees higher performance.

### Evidence and limits

- Primary result: the latest user-pasted epoch-8 JSON fragment supplied in this conversation.
- Dataset context: the earlier uploaded ECHR train.jsonl, val.jsonl, and test.jsonl audit.
- Approach: Proposed Architecture for Legal Outcome.pdf and the reviewed legal-architecture implementation, repository revision 447e12b.
- Not supplied: complete current training history, current metadata, selected checkpoint result, test evaluation, and the full training per-label tail.
- No new model training, test evaluation, or causal improvement experiment was performed to create this report.

The report uses short sentences, active verbs, and consistent terms. The requested style is approximated. ASD-STE100 conformity has not been measured or certified.

