# ECtHR-A: train the proposed architecture

This workflow trains the proposed LegalBERT, Dual Token-LCA, QRASR, adaptive span graph, Span-LCA, and gated-fusion model. It does not run the frozen Qwen retrieval baseline. Keep this project in a separate folder from your old experiments.

The first experiment preserves the designed input limit: 384 packed tokens, including two special tokens. It retains at most 382 content tokens. The preparation command measures truncation and retained span counts. This is a truncated-input variant, not an 8,192-token reproduction of Haneesh3. Do not claim a controlled improvement over a longer-context model from score differences alone.

## What is prepared

- Official LexGLUE ECtHR-A data: 9,000 training, 1,000 validation, and 1,000 test cases.
- Ten article labels: 2, 3, 5, 6, 8, 9, 10, 11, 14, and P1-1.
- Empty target lists retained as no-positive-article cases.
- Fixed data and model revisions, file hashes, and a token coverage audit.
- Eight epochs, best validation micro-F1 selection, threshold 0.5, seed 42.
- Recovery after each completed epoch and its evaluations.
- Frozen MiniLM embeddings cached once; trainable LegalBERT features are not cached.
- Console tables, append-only logs, JSON results, Excel-readable CSV files, and a plain-text test report.

## Windows setup: run once on the separate system

Copy only commands inside code blocks. Do not type step numbers or `PS C:\...>` prompts.

### 1. Extract the proposed-model package

Extract `ecthr-a-proposed.zip` into a new folder. Open the folder containing `pyproject.toml` and `legal_outcome`. Right-click an empty area and select **Open in Terminal**. Use PowerShell.

```powershell
Get-ChildItem -Name
py -3.12 --version
nvidia-smi
```

If Python 3.12 is missing:

```powershell
winget install --exact --id Python.Python.3.12
```

After installation, close PowerShell and reopen it in the extracted folder. If the machine has no NVIDIA GPU or nvidia-smi fails, share the output before selecting a CUDA build. The commands below fit the previously shown CUDA 13.2-capable NVIDIA driver. An older driver may require a different PyTorch CUDA build.

### 2. Install a separate environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e ".[data]"
.\.venv\Scripts\python.exe -m pip check
```

Do not install the old CPU-only lock file in this GPU environment.

### 3. Check the environment

```powershell
.\.venv\Scripts\python.exe -c "import torch, transformers, sentence_transformers, datasets; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Unavailable')"
```

Continue only if the imports succeed and CUDA is True. If Windows Application Control blocks a trusted dependency such as a SciPy DLL, ask the administrator for an approved environment. Do not disable the policy.

Use one GPU job at a time. Pause mining or other intensive GPU programs. Allow ample disk space for pretrained model downloads and recovery checkpoints. A full LegalBERT recovery state can be several GB. Atomic replacement also needs space for the temporary file.

### 4. Run the architecture software smoke check

```powershell
.\.venv\Scripts\python.exe -m legal_outcome.cli smoke
```

This uses a tiny random BERT. It checks forward computation and an optimizer step. It does not measure legal accuracy or verify public model download access.

## Prepare and audit ECtHR-A

### 5. Download and prepare the official data

```powershell
.\.venv\Scripts\python.exe -u -m legal_outcome.ecthr_a prepare
```

No CSV or Excel conversion is required. Do not copy PILOT, Realistic LJP, or IJJD records into this workflow. The command downloads the official dataset and the LegalBERT tokenizer. It resolves and records model revisions. Full LegalBERT and MiniLM weights load at training time.

Expect a table with these case counts:

```text
Split   Cases
train   9000
val     1000
test    1000
```

The full table also shows empty-label counts, truncated-case counts, retained token fraction, and mean retained span count. Open these files to inspect coverage:

```text
data/ecthr_a/dataset-manifest.json
data/ecthr_a/token-coverage.csv
```

The converter preserves paragraph order and official splits. It maps source class indices 0-9 to the actual article names. Generated case_id values refer to source rows; they are not verified court identifiers.

If the audit finds identical normalized text in training and validation, it stops before training. Preserve the files and share the audit. Do not delete cases or change the official split automatically. Other reported cross-split overlaps also require inspection before performance claims.

Public downloads require Hugging Face and its file CDN. Proxy HTTP 403 means access is blocked; it does not mean the dataset needs paid access or a token. The cloud development check could read metadata but its file CDN request was blocked. Do not disable TLS verification. Use an approved network or allow the required download destination through environment settings.

## Train the proposed model

### 6. Start eight-epoch training

Run this complete command once:

```powershell
.\.venv\Scripts\python.exe -u -m legal_outcome.ecthr_a train
```

Defaults are: batch size 2, encoder/head learning rate 2e-5, dropout 0.1, weight decay 0.01, no frozen lower layers, seed 42, and fixed threshold 0.5. The target is eight epochs in total. All settings are recorded. The trainer uses training and validation files only.

Before the first epoch, MiniLM encodes the retained spans once and saves reusable frozen features. Later epochs reuse these features. LegalBERT still trains normally. Cache identity includes the retained text and model identity.

The screen shows preparation messages, batch progress with loss, evaluation progress, and a readable epoch table:

```text
Epoch | Train loss | Val loss | Train micro F1 | Val micro F1 | Val macro F1 | New best
```

Values printed with `%` are percentages. JSON and CSV metrics use fractions from 0 to 1. `New best` means an improvement at that epoch; the final selected epoch is stated after training. Training minibatch loss uses dropout, so compare training evaluation loss with validation loss when assessing generalization.

Keep PowerShell open. Prevent sleep while plugged in. The default output is `checkpoints/ecthr_a_proposed`.

### 7. Resume after an interruption

Use the same folder, files, dependencies, and settings:

```powershell
.\.venv\Scripts\python.exe -u -m legal_outcome.ecthr_a train --resume
```

A checkpoint saved after epoch 5 resumes at epoch 6. A stop during epoch 6 can lose that incomplete epoch. Completed epochs remain saved. If the final epoch was saved but export was interrupted, this command can finish exporting the selected model without repeating the epochs.

If no epoch checkpoint was saved yet, recovery is not available. Start epoch 1 in a new output folder, for example `train --output checkpoints/ecthr_a_retry`, and retain that folder argument for later resume and test commands.

Do not start a second training process while the first is active. Do not delete the checkpoint folder. Changed training settings or data are rejected during resume. For a new experiment, use a new output folder.

If batch size 2 causes GPU out-of-memory before the first completed checkpoint, use a separate experiment:

```powershell
.\.venv\Scripts\python.exe -u -m legal_outcome.ecthr_a train --batch-size 1 --output checkpoints/ecthr_a_batch1
```

For that experiment, retain `--batch-size 1 --output checkpoints/ecthr_a_batch1` and add `--resume` when restarting. Also use that output folder when testing. Batch size changes the training configuration and must be recorded.

## Review training, then extract final test results

### 8. Confirm eight completed epochs

```powershell
Get-Content .\checkpoints\ecthr_a_proposed\training-summary.txt
```

The history should contain epochs 1 through 8. The selected epoch may be earlier than 8. That is expected. The folder contains:

```text
history.json             Full epoch metrics
history.csv              Spreadsheet-friendly epoch metrics
training-summary.txt     Readable table
training-state.pt        Latest recovery state and best model
metadata.json            Final selected model configuration and history
model.pt                 Selected prediction weights
tokenizer/               Saved tokenizer
```

The persistent console log is `logs/ecthr_a_proposed-training.log`. The recipe is `logs/ecthr_a_proposed-recipe.json`.

If you plan to tune settings, compare validation results now and keep the test set unused. Use the next step only when this configuration has been chosen for final evaluation. Do not select the configuration by whichever test metric improves.

### 9. Evaluate the saved selected model

```powershell
.\.venv\Scripts\python.exe -u -m legal_outcome.ecthr_a test
```

This command verifies the dataset hashes and the checkpoint's training/validation fingerprints. It requires eight completed epochs by default. It evaluates 1,000 test cases with the selected model, not automatically the epoch-8 weights.

The screen prints a neat overall metrics table and a ten-article results table. Primary Haneesh3 measures are micro F1 and macro F1. Other results include micro/macro precision and recall, exact-match accuracy, Hamming loss, micro Jaccard, PR-AUC, average precision, ROC-AUC, and evaluation loss.

The default results folder is `results/ecthr_a_proposed`. Repeating the test command for the same model and data reopens the existing report. It does not evaluate the test set again. A different model requires a new results directory.

### 10. Send the final files

```text
results/ecthr_a_proposed/test-results.json
results/ecthr_a_proposed/test-report.txt
results/ecthr_a_proposed/per-article-results.csv
checkpoints/ecthr_a_proposed/metadata.json
data/ecthr_a/dataset-manifest.json
```

`test-summary.csv` contains overall scores. `predictions.pt` stores probabilities, decisions, targets, and label order for verification. `published-reference-comparison.csv` keeps paper-reported values separate from measured results. Read `comparison-notes.txt` before using that table.

## What may and may not be claimed

This experiment trains the proposed architecture with a new ten-label output head. It does not reuse the old 16-label ECHR2023 prediction head. Its main evidence paths remain the same as the designed architecture.

The paper reference values are Qwen-0.6B micro/macro F1 67.47%/59.02% and Legal-BERT 76.25%/65.62%. Those are published values, not locally reproduced baselines. The proposed variant has a much shorter input limit. A numerical gain alone does not establish a controlled improvement. No 2-3 percentage-point increase is assumed or guaranteed.

The implementation has been checked with local small-model fixtures, including eight epochs and recovery. A full public-data training run was not completed in the development environment because its Hugging Face file CDN was blocked. Your completed test report is the evidence for real ECtHR-A performance.

Data source: https://huggingface.co/datasets/coastalcph/lex_glue

Paper reference: uploaded Haneesh3.pdf, *Retrieval-Based Multi-Label Legal Annotation: Extensible, Data-Efficient and Hallucination-Free*.
