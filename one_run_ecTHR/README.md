# One fixed ECtHR-A experiment

## Purpose

Train one fixed long-document extension of the proposed architecture. Select the checkpoint using validation data. Evaluate the selected model on the official test split once. Record the observed difference from published reference scores.

No procedure guarantees improvement. One run cannot establish statistical significance or seed stability. A negative result must be reported as a negative result. Do not choose a different paper baseline or metric after viewing the test score.

## What this version changes

The older package retained 382 content tokens. This separate extension permits 64 non-overlapping LegalBERT chunks of 128 packed tokens. Each chunk has two special tokens, so the retained content budget is at most 8,064 tokens in 8,192 packed positions.

LegalBERT encodes chunks independently. Their states are restored to document order. Dual Token-LCA attends over all visible token states. Eight ordered global spans pool content across chunks. QRASR, the adaptive graph, Span-LCA, gated fusion, and the ten-label prediction head remain in use. Document context is the mean of valid chunk CLS states. Padding has no span membership. MiniLM still provides graph similarities only.

This is a long-document extension of the original proposal. The original PDF's short packed-sequence model remains available in the repository. The new context cap aligns with the paper's nominal cap, but chunking, aggregation, optimization, and scoring details can still differ. It is a single-run comparison to published references, not a controlled local reproduction.

## Predeclared protocol

| Setting | Fixed value |
| --- | --- |
| Data | Official LexGLUE ECtHR-A, 9,000 train / 1,000 validation / 1,000 test |
| Outputs | Ten articles; empty label lists retained |
| Epochs | 8 in total |
| Seed | 42 |
| Packed context | 64 chunks × 128 tokens = 8,192 |
| Content cap | 8,064 tokens |
| Global spans | At most 8 |
| Encoder | nlpaueb/legal-bert-base-uncased; pinned revision |
| Semantic graph encoder | all-MiniLM-L6-v2; frozen, pinned revision |
| Encoder chunk microbatch | 4 valid chunks |
| Training case microbatch | 1 |
| Gradient accumulation | 8 case microbatches |
| Nominal effective batch | 8 cases |
| Encoder learning rate | 1e-5 |
| New-layer learning rate | 1e-4 |
| Dropout | 0.2 |
| Weight decay | 0.01; matrix weights only |
| Frozen lower layers | Embeddings and first four LegalBERT layers |
| Memory control | Non-reentrant gradient checkpointing |
| Learning-rate schedule | Linear; 10% optimizer-update warmup |
| Checkpoint selection | Highest validation micro F1 at fixed threshold 0.5 |
| Threshold search | None |
| Test evaluation | Once, after all eight epochs and final checkpoint selection |

These are reasonable preset values, not values proven optimal for this dataset. Do not change them after seeing test performance.

## Reference choice and target

Haneesh3 Table 3 reports the following ECtHR-A values:

| Published model | Micro F1 | Macro F1 |
| --- | --- | --- |
| GPT-5.2, best ECtHR-A row in that table | 79.33% | 76.31% |
| Legal-BERT, closest encoder-family reference | 76.25% | 65.62% |
| Qwen-0.6B retrieval | 67.47% | 59.02% |

The primary comparison is fixed to the best ECtHR-A row, GPT-5.2. A 2–3 percentage-point micro-F1 increase would require 81.33–82.33%. The Legal-BERT comparison is also reported, by name. Its 2–3 point target would be 78.25–79.25%. Beating that row alone does not mean beating the best row in the paper.

The program reports actual differences, including negative ones. It does not assume either target will be reached. If micro F1 rises while macro F1 falls, report the tradeoff instead of claiming an unqualified overall improvement. Precision and recall are additional measures; the paper does not supply equivalent precision/recall reference values in its main table.

## Exact Windows PowerShell procedure

Use a separate new folder. Do not replace files in an active training project. Copy only the commands inside the code blocks, without step numbers or PowerShell prompts.

### 1. Extract and open the package

Extract `ecthr-a-one-run.zip`. Open the folder containing `pyproject.toml` and `legal_outcome`. Right-click an empty area and select **Open in Terminal**. Use PowerShell.

```powershell
Get-ChildItem -Name
py -3.12 --version
nvidia-smi
```

Install Python 3.12 if it is missing:

```powershell
winget install --exact --id Python.Python.3.12
```

After installation, close PowerShell and reopen it in the package folder. The CUDA commands below suit the CUDA 13.2-capable NVIDIA driver previously shown. If this system has a different driver or no NVIDIA GPU, resolve that before selecting a PyTorch CUDA build.

### 2. Install dependencies

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e ".[data]"
.\.venv\Scripts\python.exe -m pip check
```

Check imports and GPU access:

```powershell
.\.venv\Scripts\python.exe -c "import torch, transformers, sentence_transformers, datasets; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Unavailable')"
```

Continue only when imports succeed and CUDA is True. Use an administrator-approved environment if Application Control blocks trusted dependencies. Do not disable the policy. Do not install the old CPU-only lock file.

### 3. Prepare data and freeze the plan

```powershell
.\.venv\Scripts\python.exe -u -m legal_outcome.one_run prepare
```

This downloads official ECtHR-A and the tokenizer, records model revisions, converts the ten article indices, audits split text overlap, and measures coverage. No model has been trained yet. No CSV from your earlier datasets is needed.

Expected case counts are 9,000 train, 1,000 validation, and 1,000 test. Review:

```text
data/ecthr_a_one_run/dataset-manifest.json
data/ecthr_a_one_run/token-coverage.csv
logs/ecthr_a_one_run/ecthr-a-one-run-plan.json
```

The plan records source/model revisions, data hashes, software versions, code hashes, settings, metric, and named references. It rejects changes to this experiment. Do not edit code or upgrade packages after freezing the plan. Do not delete or reassign official cases to make an audit pass. If the audit stops, share its message.

### 4. Check full-context hardware operation

```powershell
.\.venv\Scripts\python.exe -u -m legal_outcome.one_run preflight
```

This uses synthetic text covering the full packed window. It performs a forward/backward memory check with space reserved for approximate Adam moment buffers. It does not use dataset examples, calculate research metrics, or update model parameters. It is not a second research training run.

Continue only after `status: passed`. The report is:

```text
logs/ecthr_a_one_run/ecthr-a-one-run-preflight.json
```

The check cannot guarantee every runtime condition. If it fails, stop and share the error. Do not reduce the context or change settings silently. First close competing GPU programs or use adequate hardware before the research run starts.

### 5. Start the one research training run

```powershell
.\.venv\Scripts\python.exe -u -m legal_outcome.one_run train
```

Keep PowerShell open. Prevent sleep while plugged in. Use one GPU job at a time. The run is much heavier than the old 382-token variant; no completion time is guaranteed.

The screen shows preparation, cached MiniLM features, batch progress, training loss, validation loss, validation micro/macro F1, and checkpoint-save messages. Gradient accumulation gives a nominal batch of eight cases while processing one case at a time. The linear schedule counts optimizer updates, not document microbatches.

All eight epochs belong to this one run. Best-epoch selection uses validation only. The test file is not passed to the trainer.

### 6. Resume the same run if interrupted

```powershell
.\.venv\Scripts\python.exe -u -m legal_outcome.one_run train --resume
```

This continues the same fixed eight-epoch run. It restores model weights, optimizer, scheduler, random states, history, and best weights. A checkpoint saved after epoch 5 resumes at epoch 6. The latest incomplete epoch may repeat. There is no recovery before the first successful epoch save.

Do not change the epoch target, schedule, accumulation, dataset, or environment during resume. The schedule was fixed for eight epochs before training. If final export was interrupted after epoch 8 was saved, the same resume command can finish export without repeating the epochs.

### 7. Confirm completion

```powershell
Get-Content .\checkpoints\ecthr_a_one_run\training-summary.txt
```

Verify that epochs 1–8 are present. The best selected epoch can be earlier than 8. Both the recovery file and selected prediction model must exist:

```powershell
Test-Path .\checkpoints\ecthr_a_one_run\training-state.pt
Test-Path .\checkpoints\ecthr_a_one_run\model.pt
Test-Path .\checkpoints\ecthr_a_one_run\metadata.json
```

All three should be True. Do not evaluate test cases before this step is complete.

### 8. Evaluate the test set once

```powershell
.\.venv\Scripts\python.exe -u -m legal_outcome.one_run test
```

The command verifies the fixed experiment and evaluates the selected model on 1,000 test cases. It prints an overall table, per-article results, and the observed differences from named paper references. It saves the result. Repeating the same completed command reads the saved result instead of evaluating again.

Do not change the threshold, choose a different epoch from test scores, change the seed, or rerun a new configuration after viewing the test result if you are claiming a one-run experiment.

### 9. Extract the reports

```powershell
Get-Content .\results\ecthr_a_one_run\one-run-comparison.txt
Get-Content .\results\ecthr_a_one_run\test-report.txt
```

Send these files:

```text
results/ecthr_a_one_run/test-results.json
results/ecthr_a_one_run/one-run-comparison.json
results/ecthr_a_one_run/one-run-comparison.txt
checkpoints/ecthr_a_one_run/metadata.json
logs/ecthr_a_one_run/ecthr-a-one-run-plan.json
```

Scores in console tables use percentages. JSON/CSV use fractions. A difference of +2.00 pp means two percentage points. Micro F1 is not exact-match accuracy.

## Interpret the result honestly

- A higher test score is an observed single-run difference from a published reference.
- A lower score is a negative result, not a reason to change the reference or metric.
- A micro-F1 gain with a macro-F1 loss is a tradeoff.
- The same nominal context cap does not prove identical preprocessing, segmentation, training, or scoring.
- One run cannot isolate which new component caused a change.
- Statistical significance and seed stability require additional evidence, such as paired baseline predictions or repeated training runs. This protocol does not claim them.

The source paper's earlier released test-based parameter-search option is not used. No API-based generative baseline is called. No pretrained Qwen model is required for this proposed-architecture run.

## Verification status

The extension is verified using local small-model fixtures: full-window packing, tail sensitivity, chunk-position limits, padding masks, gradient checkpointing, accumulation, warmup/decay, and interrupted-run recovery. The development machine has CPU PyTorch. It has not profiled full LegalBERT on your GPU or measured a new full ECtHR-A score. Its Hugging Face dataset-file CDN request was blocked. Run the hardware preflight on the target system and obtain actual performance from your completed test report.

Data: https://huggingface.co/datasets/coastalcph/lex_glue

Paper reference: supplied Haneesh3.pdf, Table 3.
