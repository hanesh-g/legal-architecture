# Reduce overfitting in the ECHR experiment

## Evidence from the supplied TRACE report

The report describes a different model with precedent retrieval. This repository does not implement that retrieval system. Its new multi-label option supports the article prediction task, but it does not reproduce TRACE.

Training loss fell from 0.15273 at epoch 1 to 0.00734 at epoch 8. Validation micro F1 reached 0.76440 at epoch 4 and fell to 0.75036 at epoch 8. This suggests overfitting for the micro objective. Validation macro F1 still improved to 0.56555 at epoch 7. The two metrics select different checkpoints.

Test micro F1 was 0.69539. Test macro F1 was 0.51868. These are F1 scores, not accuracy. The report has no training F1, validation loss, test accuracy, prediction scores, or confirmed selected checkpoint. It cannot establish all causes of the generalization gap. Its statement that `best.pt` was probably saved at epoch 7 is an inference.

Article 13 has precision 0.369 and recall 0.656. It produces many false positives. P4-7 has precision 0.500 and recall 0.095. It misses most positives. One threshold change will not solve both patterns. Article 18 and Article 38 have F1 of zero. The report gives test support, not training counts. Inspect the actual training counts before attributing these failures to insufficient training data.

## Prepare the data

Use the original case texts and article labels. The uploaded `ECHR.json` is a reference file. It has no labeled case records.

Save records as JSON Lines:

```json
{"case_id":"case-001","text":"Facts available before judgment.","labels":["3","6"]}
{"case_id":"case-002","text":"Facts from another case.","labels":[]}
```

A document can have several labels. An empty list means no listed violation. It must not mean that labels are missing or unknown. Use the same label order for every experiment. Train, validation, and test sets must contain separate cases. Group related proceedings and duplicate documents in one split. Use time-based splits when predicting future cases. Do not use outcome text as input.

Check article identities before training. The supplied reference file calls Article 18 "Derogation in time of emergency". That title belongs to Article 15. Article 18 concerns limits on restrictions of rights. Check whether P4-7 is the intended label, or whether the source dataset means P7-4. Do not rename labels without checking the original dataset.

When reproducing TRACE, confirm that precedent retrieval uses training cases only. Do not use validation or test judgments as precedent labels. For a future-case task, precedents must also be available before the target case. This repository has no precedent retrieval to audit.

## Run an eight-epoch experiment

Use the same data splits as the baseline. First run the same settings with the new reporting controls. Then change one factor at a time. The settings below are a starting candidate, not proven best values.

```sh
legal-outcome train \
  --data data/train.jsonl \
  --validation data/validation.jsonl \
  --labels 10 11 13 14 18 2 3 34 38 5 6 7 8 9 P1-1 P4-7 \
  --task-type multi_label \
  --epochs 8 \
  --dropout 0.3 \
  --weight-decay 0.01 \
  --learning-rate 1e-5 \
  --head-learning-rate 1e-4 \
  --freeze-lower-layers 4 \
  --selection-metric macro_f1 \
  --report-training-metrics \
  --output checkpoints/echr-regularized-1 \
  > training-history.jsonl
```

Use `--device cuda` only when CUDA is available. Apply the public model network settings before the first download. Each experiment needs a new output directory.

Training completes all eight epochs. The checkpoint contains the epoch with the highest validation macro F1. Ties keep the first best epoch. Metadata records the selected epoch and all epoch results. Use `--selection-metric micro_f1` if overall positive-label F1 is the agreed primary target. Decide this before comparing test results. The automatic default selects macro F1 when validation exists. Without validation, it saves the last epoch.

Dropout applies to LegalBERT dropout modules, graph propagation, and final fusion. It does not add attention-weight dropout to the custom attention branches. Weight decay excludes one-dimensional weights and biases. The separate encoder rate limits updates to pretrained features. Freezing four lower layers also freezes embeddings.

## Test the selected model

```sh
legal-outcome evaluate \
  --checkpoint checkpoints/echr-regularized-1 \
  --data data/test.jsonl \
  > results.json
```

The report includes micro and macro precision, recall, and F1. It also includes per-article support, precision, recall, and F1, exact-match accuracy, Hamming loss, and loss.

Exact-match accuracy counts a case as correct only when every predicted article matches its target. Hamming loss counts incorrect label decisions. Independent sigmoid probabilities do not sum to one. A case can have no predicted articles. Do not compare these probabilities with the old single-class softmax outputs.

Weighted training loss is recorded when positive weights are enabled. Compare micro and macro F1 across experiments; weighted and unweighted loss values do not have the same scale.

## Additional experiments

- Compare dropout 0.1, 0.2, and 0.3 on validation data. More dropout is not always better.
- Compare freezing zero and four lower layers. Freezing can also cause underfitting.
- Try `--positive-weight-cap 3` only after measuring training class counts. Weights use training positives and negatives only. The cap limits excessive rare-class weights. This can increase false positives. A class with no training positives cannot use this weighting option.
- Keep the default threshold 0.5 for the first comparison. Tune global or per-article thresholds on validation predictions only in the original TRACE pipeline. This repository exposes a global `--threshold` setting but does not implement automatic threshold optimization.
- Compare several fixed seeds, such as 42, 43, and 44. Use validation results to select settings. Reserve a fresh test set or nested evaluation if the original test results have influenced repeated decisions.
- Investigate document coverage. This architecture retains at most 382 content tokens. Longer document coverage needs a separate chunking design and a controlled comparison.

No model change guarantees higher accuracy. Report the actual held-out results and the objective used to select the checkpoint.
