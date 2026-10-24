# Legal outcome architecture

This project implements the architecture in *Proposed Architecture for Legal Outcome.pdf*. It uses PyTorch, LegalBERT, and MiniLM. The source document defines a model. It does not supply training data or trained prediction heads.

## Install

Use Python 3.10 or later. Python 3.12 was used for validation. The commands below install the tested CPU environment.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-deps -e .
```

For a GPU environment, install the PyTorch build for your CUDA version. Then use `python -m pip install -e '.[test]'`. The lock file describes the CPU environment only.

## Check the installation

```sh
OMP_NUM_THREADS=2 pytest -q
OMP_NUM_THREADS=2 legal-outcome smoke
```

These checks use small test encoders. They require no model downloads. The smoke command runs a BERT encoder, both evidence paths, the prediction head, backpropagation, and one optimizer step. It does not test legal accuracy.

## Train

Supply a UTF-8 JSON Lines file. Each line must contain `text` and `label`.

```json
{"text": "Facts and arguments available before the decision.", "label": "allowed"}
```

Use your actual class names. The names below are examples only.

```sh
legal-outcome train --data data/train.jsonl --validation data/validation.jsonl \
  --labels allowed dismissed --output checkpoints/run-1 --epochs 3 --batch-size 2
```

The command downloads the two public model checkpoints on first use. Permit `huggingface.co` and its model download destinations in restricted environments. Set `HF_HOME` to a writable cache directory. No access token is required for these public models. Use `--device cuda` when CUDA is available. Use `--legalbert-model` and `--minilm-model` to select local model directories or other compatible checkpoints.

Use separate cases in the training, validation, and test sets. Split linked cases together. Use the same time limit for all input documents. Remove the final order and other text that reveals the target judgment when the task predicts a future outcome. Otherwise, the score can measure target leakage.

The command trains all LegalBERT layers by default. Use `--freeze-lower-layers 4` for the alternative shown on page 2 of the source. With validation data, the best validation macro-F1 epoch is saved after all requested epochs. Use `--selection-metric micro_f1`, `loss`, or `last` to change this rule. Without validation, the last epoch is saved. The command does not select the best validation epoch. It rejects an existing output directory.

## Evaluate and predict

```sh
legal-outcome evaluate --checkpoint checkpoints/run-1 --data data/test.jsonl
legal-outcome predict --checkpoint checkpoints/run-1 --text-file data/case.txt --evidence
```

Single-label evaluation reports loss, accuracy, micro and macro precision, recall, F1, per-class metrics, and a confusion matrix. Matrix rows are actual classes. Matrix columns are predicted classes. Class order is stored in the checkpoint.

Prediction reports one class and its class probabilities. The optional evidence output reports span attention for that class. Attention weights show model behavior. They do not prove that a span caused a judgment. Softmax values are not calibrated confidence estimates.

## Implementation notes

Read [Feature explanations](docs/features.md) for what, why, and how for each feature. Read [Architecture decisions](docs/architecture.md) for the equations, unspecified details, and source conflicts. Read [Validation](docs/validation.md) for the checks and their limits.

The explanations use short sentences, active verbs, and consistent terms. They follow the requested style close to ASD-STE100. Formal ASD-STE100 compliance has not been certified.

## Reduce overfitting and use ECHR article labels

Read [the eight-epoch experiment guide](docs/overfitting.md). It explains the supplied TRACE results, multi-label data format, best-checkpoint selection, and regularization settings. Use `--task-type multi_label` for article label lists. This is an explicit extension of the PDF single-class prediction head. It does not reproduce TRACE precedent retrieval.

## Automatic checkpoints and resume

Training now displays batch progress and saves `training-state.pt` and `history.json` inside `--output` after each completed epoch, including its evaluations. The recovery checkpoint contains the latest model, optimizer, random states, epoch history, and best-validation model. The previous complete checkpoint stays in place until its replacement has been written. Keep enough free disk space for both the existing and temporary checkpoint files; full LegalBERT recovery files can be several GB.

To resume, repeat the original command with the same settings and output directory. Add:

```sh
--resume checkpoints/run-1/training-state.pt
```

`--epochs 8` still means eight epochs in total. A checkpoint saved after epoch 5 resumes at epoch 6. A stop during epoch 6 restarts that epoch from the saved epoch-5 state. Changed data, labels, optimizer settings, seed, device, or reporting settings are rejected. The total epoch target may increase. The checkpoint is a local training artifact; it does not contain raw case text. Keep access to the original datasets and encoder models.

PowerShell and Jupyter both work. Closing PowerShell, stopping the notebook kernel, restarting the computer, or running out of runtime can stop training. The recovery file survives these events if it was saved to persistent local storage. Neither interface recovers the old run that stopped before this feature was added.

On successful completion, the usual `model.pt`, `metadata.json`, and tokenizer are saved at the output root for evaluation and prediction. `training-state.pt` remains available for continuation and keeps the latest training weights, even when `model.pt` uses an earlier best epoch.

## Faculty paper evaluation

See [the Haneesh1/Haneesh3 completion plan](docs/faculty_completion_plan.md). Multi-label evaluation now includes micro Jaccard, trapezoidal PR-AUC, average precision, and ROC-AUC from continuous model probabilities. Existing saved prediction checkpoints do not need retraining for these additional metrics.

For the completed first dataset, use the [ECHR2023 test guide](first_dataset/README.md) and [evaluation package](downloads/echr2023-evaluation.zip). `python -m legal_outcome.echr2023 inspect` checks the saved task and epoch count. `evaluate` uses the original selected model, can export its saved best weights from a completed recovery checkpoint, and saves readable paper measures without retraining. Keep the original processed ECHR2023 test split; this is separate from LexGLUE ECtHR-A.

## Separate ECtHR-A retrieval experiment

The standalone [ECtHR-A runner and PowerShell guide](ecthr_a_retrieval/README.md) implement the frozen Qwen3-0.6B document-retrieval baseline for Haneesh3. It has its own dependencies and output files. The [downloadable ZIP](downloads/ecthr-a-retrieval.zip) contains code, tests, and instructions; it downloads the public dataset on first use. Validation selects k and threshold before test evaluation.

## Train the proposed architecture on ECtHR-A

Use the dedicated [preparation, training, recovery, and test guide](ecthr_a_proposed/README.md). The three commands are `python -m legal_outcome.ecthr_a prepare`, `train`, and `test`. This workflow trains the designed LegalBERT/token/span model with a new ten-label head. It has readable console summaries, persistent logs, JSON/CSV reports, pinned data/model revisions, a truncation audit, and reusable frozen MiniLM embeddings. It is separate from the optional Qwen retrieval baseline. The original 382-content-token limit is stated in every test report.

## A single predeclared long-document experiment

The [one-run protocol](one_run_ecTHR/README.md) adds an explicit long-document extension: 64 chunks × 128 packed tokens, global token/span evidence, mean chunk-CLS context, gradient checkpointing, effective batch size eight, and a fixed eight-epoch warmup/decay schedule. Use `python -m legal_outcome.one_run prepare`, `preflight`, `train`, and `test`. The protocol fixes the metric and references before training and reports the observed differences without claiming significance. It remains distinct from the original 382-content-token variant.
