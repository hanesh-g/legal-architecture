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

The command trains all LegalBERT layers by default. Use `--freeze-lower-layers 4` for the alternative shown on page 2 of the source. The last epoch is saved. The command does not select the best validation epoch. It rejects an existing output directory.

## Evaluate and predict

```sh
legal-outcome evaluate --checkpoint checkpoints/run-1 --data data/test.jsonl
legal-outcome predict --checkpoint checkpoints/run-1 --text-file data/case.txt --evidence
```

Evaluation reports loss, accuracy, macro F1, and a confusion matrix. Matrix rows are actual classes. Matrix columns are predicted classes. Class order is stored in the checkpoint.

Prediction reports one class and its class probabilities. The optional evidence output reports span attention for that class. Attention weights show model behavior. They do not prove that a span caused a judgment. Softmax values are not calibrated confidence estimates.

## Implementation notes

Read [Feature explanations](docs/features.md) for what, why, and how for each feature. Read [Architecture decisions](docs/architecture.md) for the equations, unspecified details, and source conflicts. Read [Validation](docs/validation.md) for the checks and their limits.

The explanations use short sentences, active verbs, and consistent terms. They follow the requested style close to ASD-STE100. Formal ASD-STE100 compliance has not been certified.
