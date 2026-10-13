# What changed, why, and how

## Split audit

**What:** The trainer checks normalized document text. The preparation tool can make separate cleaned split copies.

**Why:** Whitespace and case changes can hide a document shared between validation and test.

**How:** The audit normalizes whitespace and case before comparison. It reports overlapping IDs. The optional preparation step excludes validation copies and records every exclusion. It preserves training and test records.

## Checkpoint selection

**What:** The user can select micro F1, macro F1, or validation loss. Training still completes eight epochs.

**Why:** The original run chose macro F1. That choice can differ from the best micro-F1 epoch.

**How:** Each epoch uses threshold 0.5 for selection. The trainer stores the best weights. The saved report states the selected epoch and metric. Threshold tuning occurs after selection.

## Rare-label threshold guard

**What:** Very small validation classes retain threshold 0.5.

**Why:** A threshold fitted to a few positives can fail on later cases.

**How:** The trainer requires a minimum positive and negative count before tuning a label. The default count is 20. Grid-search ties prefer 0.5. All tuning uses validation predictions.

## Optimization controls

**What:** LegalBERT and new layers can use different learning rates. Lower LegalBERT layers can be frozen. Encoder dropout has its own setting.

**Why:** Smaller encoder updates can preserve pretrained information. New graph and prediction layers need to learn their task.

**How:** The optimizer creates separate parameter groups. Frozen parameters receive no updates. Biases and one-dimensional scales receive no weight decay. The original warmup and linear schedule remain in use.

## Generalization diagnostics

**What:** Epoch results include validation BCE. An option also evaluates the training set.

**Why:** Training minibatch loss alone cannot measure overfitting reliably.

**How:** Evaluation disables dropout. It records comparable unweighted BCE and fixed-threshold metrics. These traces show whether training improves while validation gets worse.

## Complete metrics

**What:** Evaluation reports micro, macro, and weighted precision, recall, and F1. It also reports exact-match accuracy and Hamming loss.

**Why:** One F1 score can hide false positives or missed rare labels.

**How:** The evaluator compares each article decision with its target. It saves prediction matrices for later verification. It reports class support and zero-safe scores.

## No-neighbor graph behavior

**What:** A case with no eligible precedent keeps its original case representation.

**Why:** It should behave the same when processed alone or with cases that have neighbors.

**How:** The graph layer returns the case vector for each row with no valid neighbors. It applies graph fusion only to rows with eligible precedents.

## Run safety

**What:** The trainer rejects an existing output directory and records data support before training.

**Why:** A new experiment must not overwrite an earlier checkpoint or conceal the label distribution.

**How:** The trainer creates a new directory. It derives the label vocabulary from training only. It rejects unknown validation or test labels. Checkpoints load with restricted weight loading.
