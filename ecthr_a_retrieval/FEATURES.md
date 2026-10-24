# What, why, and how

## Frozen retrieval

**What:** Predict article labels from similar training documents.

**Why:** The selected paper studies a retrieval alternative to task-specific model training.

**How:** Qwen encodes each document. Cosine similarity selects neighbors. Their labels receive similarity weights. The encoder receives no gradient updates.

## Validation selection

**What:** Select the neighbor count and threshold on validation only.

**Why:** Test labels must not guide parameter selection.

**How:** The runner embeds training and validation cases first. It saves the selected settings before processing the test split. The selection function receives no test vectors or targets.

## Correct pooling and labels

**What:** Keep the source article mapping and use the last visible token state.

**Why:** Padding positions are not document content. Dataset class indices are not article numbers.

**How:** The attention mask locates the last visible position. Label metadata checks the ten article names. Empty label lists remain valid examples.

## Recovery files

**What:** Save completed embedding shards and the validation choice.

**Why:** A stopped terminal must not discard all completed encoding work.

**How:** Temporary files are replaced only after writing succeeds. Rerunning the same command reuses completed shards. Model and data identities prevent incompatible cache reuse.

## Clear reports

**What:** Save metrics, settings, prediction matrices, and resource measurements.

**Why:** A result needs its data split and selection settings to be reviewable.

**How:** The final JSON distinguishes smoke and full runs. It records micro and macro scores, exact-match accuracy, class support, revisions, context length, cache reuse, and current-invocation timing.
