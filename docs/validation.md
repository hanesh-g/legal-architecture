# Validation record

Validation used Python 3.12 and the CPU package versions in `requirements-lock.txt`.

## Passed

- 57 main-project tests were verified across the full regression suite and targeted follow-up runs after the proposed ECtHR-A workflow update.
- The single-label and multi-label command workflows each trained for two epochs on synthetic records. It saved a checkpoint, loaded it, evaluated two records, and predicted one document. The workflow used local small BERT and SentenceTransformer models.
- Checkpoint recovery preserved predictions, class order, tokenizer settings, and cue patterns.
- Both token attention branches received gradients. Span attention, graph mixing, fusion, and the encoder received gradients.
- MiniLM features received no gradients. Changes to those features changed graph edges but did not change token attention.
- Padding had zero attention. Changes to padded values did not change predictions.
- A separate smoke command completed a BERT forward pass, backpropagation, and an optimizer step.
- `pip check` found no broken dependency requirements.
- Repeating the locked installation preserved the installed package versions.

- Validation checkpoint selection restored an earlier best epoch, while last-epoch selection retained the final epoch.
- Known multi-label examples verified micro and macro precision, recall, F1, exact-match accuracy, and Hamming loss.
- Class-weight caps, optimizer groups, multi-label target checks, and normalized split-overlap checks passed.

- Interrupted and resumed single-label and multi-label runs exactly matched uninterrupted CPU runs, including dropout, history, final training weights, and best-epoch selection.
- A simulated checkpoint write failure preserved the previous complete file. Changed resume settings were rejected. GPU resume was not exercised in this CPU-only environment.

- Micro PR-AUC, average precision, and ROC-AUC matched scikit-learn reference calculations, including tied scores. Undefined targets returned explicit nulls. Micro Jaccard matched its F1 relationship.

- The proposed ECtHR-A adapter preserved class-index mapping and empty label lists on local fixtures. An eight-epoch tiny-BERT workflow resumed after epoch 3 and regenerated a final model from a saved epoch-8 state.
- Readable tables, logs, history CSV, per-article CSV, text reports, prediction export, data hashes, and repeated-test result reuse were verified.
- Frozen MiniLM cache reuse and random-state preservation were verified. Exact remote revisions identify reusable caches. Old recovery files without the new optional revision fields remained readable.

## Blocked

An earlier instance could not download the named public LegalBERT tokenizer. The cloud network proxy returned HTTP 403 for `huggingface.co`. The public LegalBERT and MiniLM weight integration check did not run. The local checks do not replace that check. The later ECtHR-A preparation attempt could read Hugging Face metadata, but its dataset file download to `us.aws.cdn.hf.co` was blocked by the cloud proxy. Full official-data preparation and training were not validated in this instance.

## Not measured

No real dataset or target class list was supplied. Legal outcome accuracy, calibration, fairness, and jurisdiction coverage were not measured. The synthetic labels have no legal meaning.

## Known warning

The installed SentenceTransformer version reports a deprecation warning for its `models` import path in the integration test. The test completed. The older import path supports the declared package version range.

## GitHub checks

The repository includes a GitHub Actions workflow. A local pass does not establish that the remote workflow has completed. Check the repository Actions tab after the push.
