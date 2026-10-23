# Completion plan: Haneesh1 and Haneesh3

## Scope

The uploaded haneesh1.pdf is *PILOT: Legal Case Outcome Prediction with Case Law*, NAACL 2024, pages 609–621. The uploaded Haneesh3.pdf is *Retrieval-Based Multi-Label Legal Annotation: Extensible, Data-Efficient and Hallucination-Free*.

Use the first paper's evaluation protocol for the completed ECHR2023 experiment. Changing the metric report does not require training again. The current dual-token/span architecture is not a reproduction of PILOT's retrieval and temporal-shift model. Identify the method correctly in the final report.

## Track 1: evaluate the completed ECHR2023 model

The paper's four primary measures, in its Section 5 and Table 1, are:

| Measure | Saved predictions needed |
| --- | --- |
| Micro F1 | Binary article decisions and true labels |
| Micro Jaccard | Binary article decisions and true labels |
| Micro PR-AUC | Continuous article probabilities and true labels |
| Micro ROC-AUC | Continuous article probabilities and true labels |

The evaluator now reports all four for multi-label tasks. It retains the existing precision, recall, macro F1, exact-match accuracy, and Hamming loss.

PR-AUC has two common definitions. This implementation reports trapezoidal PR-AUC as `micro_pr_auc` and average precision separately as `micro_average_precision`. Confirm the original PILOT evaluation code's definition before claiming an exact paper reproduction. The PDF does not resolve that detail. Both are pooled over case/article decisions; neither uses only hard predictions. Score ties are handled as groups.

Steps:

1. Locate the completed ECHR checkpoint containing model.pt, metadata.json, and tokenizer/. Do not use the Realistic LJP checkpoint.
2. Read metadata.json. Confirm multi_label mode and the expected 16 ECHR2023 article labels. Record selected_epoch.
3. Preserve the training split, label order, threshold, and selected checkpoint. Do not select settings from the test set.
4. Update the legal_outcome source and install it in the CUDA Python environment with `python -m pip install -e .`. Do not use the CPU lock file on the GPU machine.
5. Run evaluation once with the selected ECHR checkpoint and the ECHR2023 test file. Change the paths below to the actual locations.

```powershell
.\.venv\Scripts\python.exe -m legal_outcome.cli evaluate --checkpoint checkpoints/baseline_safe --data "C:\24071a3224\Datasets\Datasets\PILOT\test.jsonl" --device cuda > echr2023-paper-test.json
```

6. Read echr2023-paper-test.json and attach metadata.json to the final results. Report the actual test example count.

The full test scores for this saved model have not been supplied. The last reviewed epoch-8 snapshot had validation micro F1 0.74515, macro F1 0.51664, and exact-match accuracy 0.51833 on 3,000 validation cases. These are not test results and are not necessarily the selected model's results.

Paper Table 1 reports PILOT micro F1 0.715 ± 0.008, micro Jaccard 0.557 ± 0.010, PR-AUC 0.543 ± 0.014, and ROC-AUC 0.831 ± 0.007. These are published reference values, not results achieved by this project. The paper averages five seeds. A first delivery can identify itself as a single-run result; a five-seed claim requires five actual runs.

The paper uses 8,138 training, 3,000 validation, and 3,000 test cases. The earlier uploaded project data had 2,999 test cases. Report the difference and its preprocessing cause. Do not silently describe the modified test split as the exact paper benchmark.

## Track 2: ECtHR-A for Haneesh3

Choose `coastalcph/lex_glue`, configuration `ecthr_a`.

Verified source: https://huggingface.co/datasets/coastalcph/lex_glue

| Item | Value |
| --- | --- |
| Task | Predict violated ECHR articles from factual paragraphs |
| Official train / validation / test | 9,000 / 1,000 / 1,000 |
| Article labels | 2, 3, 5, 6, 8, 9, 10, 11, 14, P1-1 |
| Source fields | text: list of paragraphs; labels: list of integer class indices |
| No positive article | Empty labels list; keep these examples |

Dataset indices 0–9 map to those article names in the stated order. Index 0 is Article 2; it is not Article 0. Read the ClassLabel metadata and preserve the mapping.

ECtHR-B predicts alleged/considered articles rather than judged violations. ECtHR-A is selected because it aligns with outcome prediction. ECtHR-A is not the previous 16-label ECHR2023 dataset. Do not reuse the old 16-label classification head or mix their test scores.

Released code: https://github.com/lizhang-AIandLaw/Retrieval-based-Multi-label-Legal-Annotation

Its README and eval_rag.py implement nearest training-document retrieval with similarity-weighted label voting. This differs from the paper's description of a label-description index. State explicitly that the initial implementation follows the released training-document k-NN baseline.

Fast initial experiment:

1. Use a separate environment and output directory. Keep the completed ECHR2023 model unchanged.
2. Load official ECtHR-A splits. Join paragraph text in original order. Keep empty label lists.
3. Start with Qwen/Qwen3-Embedding-0.6B, a frozen encoder. Use batch size 1 first on a 16 GB GPU. Test memory with a short smoke run.
4. Use the paper's 8,192-token context where feasible. Record any change to context length. The current architecture's 382-content-token limit would be a different experiment.
5. Encode and cache the training index and validation queries. Store model revision, context settings, label mapping, and data fingerprints with each cache. Do not reuse caches with different settings.
6. Retrieve training documents only. Tune k in {5, 10, 15, 20} and a small threshold grid on validation micro F1. Keep macro F1 as a secondary result.
7. Freeze k, threshold, model, and input settings. Encode test queries and evaluate once.
8. Save micro F1 and macro F1, the two main Table 3 measures. Add micro/macro precision and recall, exact-match accuracy, runtime, and peak GPU memory.

Do not run eval_rag.py with `--search_params true` unchanged: that branch explicitly searches using test labels. Move its search to validation before using it for a defensible result. Also verify padding-aware last-token pooling and cast scores to float32 before NumPy conversion.

No eight-epoch gradient training is required for this frozen retrieval experiment. Index construction uses labeled training documents, so it is not zero-supervision. If faculty instead requests the paper's hierarchical Legal-BERT baseline, that is a separate training experiment. It is not reproduced by simply changing the dataset in the 384-token architecture.

The paper reports Qwen-0.6B ECtHR-A micro/macro F1 of 67.47/59.02 and Legal-BERT 76.25/65.62 (percent scale). Treat these as reference values only. Matching labels alone does not establish protocol equivalence.

## Execution order

If only one GPU is available, run the ECHR2023 evaluation first. Then run the ECtHR-A retrieval experiment. Do not start both GPU jobs together. If two GPU machines are available, one person can evaluate ECHR2023 while the other prepares and runs ECtHR-A.

The first delivery should contain:

- ECHR2023 test table with the four PILOT measures, selected epoch, test count, metric definitions, and single-run status.
- ECtHR-A test table with micro/macro F1, chosen k and threshold, encoder revision, context limit, and measured resource use.
- A clear distinction between published baselines, local test results, and validation results.

Start additional models, seeds, or scaling studies only after this minimum delivery is complete. No model score or completion time is guaranteed in advance.
