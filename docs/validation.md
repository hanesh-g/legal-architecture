# Validation record

Validation used Python 3.12 and the CPU package versions in `requirements-lock.txt`.

## Passed

- 39 automated tests completed after the multi-label and regularization update.
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

## Blocked

The named public LegalBERT tokenizer could not download. The cloud network proxy returned HTTP 403 for `huggingface.co`. The public LegalBERT and MiniLM weight integration check did not run. The local checks do not replace that check.

## Not measured

No real dataset or target class list was supplied. Legal outcome accuracy, calibration, fairness, and jurisdiction coverage were not measured. The synthetic labels have no legal meaning.

## Known warning

The installed SentenceTransformer version reports a deprecation warning for its `models` import path in the integration test. The test completed. The older import path supports the declared package version range.

## GitHub checks

The repository includes a GitHub Actions workflow. A local pass does not establish that the remote workflow has completed. Check the repository Actions tab after the push.
