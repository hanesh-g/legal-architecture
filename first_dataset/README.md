# First dataset: test the saved ECHR2023/PILOT model

Use this guide for **dataset 1 / haneesh1.pdf**. This is the earlier 16-label ECHR2023 experiment. The newer LexGLUE ECtHR-A experiment has 10 labels and uses a different test split.

The model is already trained. Evaluate its saved, validation-selected weights. You do not need another training run to add measures to the report. Evaluation runs the test cases through the model and saves predictions. It does not update model weights.

The epoch 8 values previously shared were validation measures: micro F1 74.51%, macro F1 51.66%, and exact-match accuracy 51.83%. These are not test results. The selected model may come from an earlier epoch.

## 1. Use the machine that holds the original checkpoint

Open PowerShell. Keep the original ECHR project and its checkpoint folder. If you use a different machine, copy the entire original checkpoint folder and the original processed test file to it first.

Run one GPU job at a time. If ECtHR-A is training on this GPU, complete that job before this evaluation.

Use a separate folder for this evaluation code. The commands below do not replace the old project or change its Python installation.

```powershell
$work = Join-Path $env:USERPROFILE "Downloads\echr2023-evaluation"
New-Item -ItemType Directory -Path $work -Force | Out-Null
$archive = Join-Path $work "echr2023-evaluation.zip"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/hanesh-g/legal-architecture/main/downloads/echr2023-evaluation.zip" -OutFile $archive
$codeDir = Join-Path $work "code"
Expand-Archive -LiteralPath $archive -DestinationPath $codeDir -Force
Set-Location -LiteralPath $codeDir
Test-Path -LiteralPath ".\legal_outcome\echr2023.py"
```

The last command must print `True`. Copy only code from a command block. Do not copy list numbers, `PS C:\...>`, or the expected output.

## 2. Select the original project and CUDA Python

Run this command. At its prompt, paste the full path to the project that trained ECHR2023. Do not paste the new ECtHR-A project path.

```powershell
$oldProject = Read-Host "Paste the full path to the original ECHR project folder"
if (-not (Test-Path -LiteralPath $oldProject -PathType Container)) { throw "The original project folder was not found." }
```

Earlier project paths in this session included:

- `C:\Users\BSR\Downloads\legal-architecture-main\legal-architecture-main`
- `C:\24071a3224\legal-architecture-main\legal-architecture-main`

Use the location that actually contains the ECHR checkpoint on this machine.

Run the next block to use its CUDA Python. If the project has no `.venv`, the command asks for the full path to the Python executable you previously checked with CUDA.

```powershell
$gpuPython = Join-Path $oldProject ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $gpuPython -PathType Leaf)) {
    $gpuPython = Read-Host "Paste the full path to your working CUDA python.exe"
}
if (-not (Test-Path -LiteralPath $gpuPython -PathType Leaf)) { throw "Python executable was not found." }
& $gpuPython -c "import torch, transformers, sentence_transformers; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); assert torch.cuda.is_available(), 'Use the Python environment where CUDA was True'"
if ($LASTEXITCODE -ne 0) { throw "Fix the Python environment before you continue." }
```

It must print `CUDA: True`. This uses dependencies from your existing GPU environment. Running a module from the new code folder does not require another editable installation.

## 3. Find the saved ECHR checkpoint and check its epoch count

```powershell
& $gpuPython -u -m legal_outcome.echr2023 inspect --root "$oldProject"
if ($LASTEXITCODE -ne 0) { throw "Checkpoint inspection failed. Read the message above." }
```

The table shows compatible 16-label multi-label checkpoints. It excludes the 2-label Realistic LJP task and the 10-label LexGLUE ECtHR-A task. Label compatibility alone does not prove the dataset source. Choose the original ECHR run by your records and its validation selection rule.

Read these columns:

| Column | Meaning |
| --- | --- |
| Epochs | Number of completed epochs saved in the checkpoint |
| Selected | Best epoch chosen by the original validation rule |
| Prediction files | Whether metadata.json, model.pt, and tokenizer/ are present |
| Recovery | Whether training-state.pt is present |

Eight completed epochs with selected epoch 3, for example, means all eight epochs finished and epoch 3 was selected. These two numbers serve different purposes.

- **Epochs = 8 and prediction files = YES:** use the saved model directly.
- **Epochs = 8, prediction files = NO, recovery = YES:** the helper exports the saved best weights into a separate result folder. It does not retrain the model.
- **Epochs less than 8:** the original run is incomplete. Preserve its files and resume that same run with its original settings. Do not present it as an eight-epoch experiment. This helper will stop before testing.
- **No compatible checkpoint:** confirm that you scanned the original project. A log or history.json alone cannot restore model weights.

Do not choose between several checkpoints by their test scores. Use the run and selection rule already chosen from validation.

Paste the checkpoint folder from the table when prompted:

```powershell
$checkpoint = Read-Host "Paste the full ECHR checkpoint folder from the table"
if (-not (Test-Path -LiteralPath $checkpoint -PathType Container)) { throw "Checkpoint folder was not found." }
```

Paste the folder path alone. Do not paste a path to an individual .pt file.

## 4. Select the original processed test file

Use the held-out ECHR2023 test file associated with that training run. Do not make another split. Keep the original labels and preprocessing.

The usual location is `data\test.jsonl` inside the original project. This block checks that location and asks for a path if it is missing:

```powershell
$testFile = Join-Path $oldProject "data\test.jsonl"
if (-not (Test-Path -LiteralPath $testFile -PathType Leaf)) {
    $testFile = Read-Host "Paste the full path to the original processed ECHR test.jsonl"
}
if (-not (Test-Path -LiteralPath $testFile -PathType Leaf)) { throw "Test file was not found." }
$firstCase = Get-Content -LiteralPath $testFile -TotalCount 1 | ConvertFrom-Json
$firstCase.PSObject.Properties.Name
```

The printed field names must include `text` and `labels`. Each `labels` value must be a list of article strings. A record can have an empty list.

```json
{"text": "Case facts...", "labels": ["3", "6"]}
```

Keep any other original fields. The evaluator uses the checkpoint's saved label order. It expects the original 16 article names: `10, 11, 13, 14, 18, 2, 3, 34, 38, 5, 6, 7, 8, 9, P1-1, P4-7`.

Some raw PILOT files use `facts` and `violated_articles`. They require the original preprocessing. Do not pass such a file directly or replace it with the LexGLUE ECtHR-A test file. Do not use train.jsonl or val.jsonl as test data. The helper rejects a file that exactly matches a saved training or validation fingerprint; this is not a complete text-overlap audit.

## 5. Run the test evaluation once

Remain in the new evaluation code folder. Run this entire command:

```powershell
& $gpuPython -u -m legal_outcome.echr2023 evaluate --checkpoint "$checkpoint" --test-data "$testFile" --results results/echr2023_final --cache cache/echr2023_semantics --device cuda --batch-size 2
if ($LASTEXITCODE -ne 0) { throw "Evaluation did not finish. Read the error above before continuing." }
```

The helper checks the task and saved epoch count. It then loads the selected model. If final export was interrupted, it exports the best model saved in training-state.pt. It preserves the original checkpoint.

The model and tokenizer can require their original Hugging Face files. If those files are absent from the local cache, internet access is needed. Initial loading and span preparation can take time before the evaluation progress bar starts. Keep PowerShell open and keep the computer awake.

You will see the evaluation progress and clear result tables. The four main paper measures are printed together. Precision, recall, macro F1, exact-match accuracy, and Hamming loss are also saved. The evaluator keeps the trained model's threshold and input limit.

No training, threshold search, or model selection uses the test labels. Running the exact same command after a completed evaluation reuses its saved report. It does not test another model. If an interrupted report write left predictions complete, the helper repairs the report without another evaluation.

## 6. Open the saved results

```powershell
Get-Content -LiteralPath ".\results\echr2023_final\paper-metrics.txt"
notepad ".\results\echr2023_final\test-report.txt"
Invoke-Item ".\results\echr2023_final"
```

| File | Use |
| --- | --- |
| paper-metrics.txt | Read the four required measures together |
| paper-metrics.csv | Copy the four values into your report |
| test-report.txt | Read all measures and per-article results |
| test-results.json | Preserve full numerical results and model details |
| test-summary.csv | Open scalar results in Excel |
| per-article-results.csv | Check precision, recall, F1, and support by article |
| predictions.pt | Preserve targets, probabilities, and decisions |
| published-reference-comparison.csv | Read the observed differences from the published PILOT row |
| first-dataset-identity.json | Preserve the checkpoint and test-file identity |
| evaluation.log | Read the evaluation output after the window closes |
| recovered-selected-model/ | Present only if recovery export was needed |

Tables use the percent scale. JSON and CSV values are fractions. For example, `0.735` means `73.5%`.

## 7. Make the paper comparison with the correct limits

The uploaded haneesh1.pdf is *PILOT: Legal Case Outcome Prediction with Case Law*. Section 5 / Table 1 reports these PILOT results:

| Main measure | Published five-seed mean |
| --- | --- |
| Micro F1 | 0.715, or 71.5% |
| Micro Jaccard | 0.557, or 55.7% |
| Micro PR-AUC | 0.543, or 54.3% |
| Micro ROC-AUC | 0.831, or 83.1% |

These numbers are published references. They are not your test results. This checkpoint uses the proposed dual token/span architecture, not PILOT's retrieval and temporal-shift method.

Use percentage points to describe a difference. For example, a measured test micro F1 of 0.735 would be 2.0 percentage points above 0.715. This is an example, not a measured project result. A 2–3 point micro F1 increase over the PILOT mean would require a test micro F1 of 73.5–74.5% under comparable evaluation conditions. No increase is guaranteed.

State these limits in the report:

1. The paper averages five seeds. This project evaluation is one training run. Its observed difference does not establish significance or repeatability.
2. The paper has 3,000 test cases. An earlier uploaded processed file had 2,999. Report the actual count and any preprocessing difference. Do not silently claim an identical split.
3. This evaluator reports trapezoidal micro PR-AUC and micro average precision separately. Confirm the original paper code's definition before treating either as an exact match.
4. Preserve the saved input limit. The earlier model reads at most 382 content tokens by default. Evaluation cannot make that checkpoint into a whole-document model. Report its actual limit, which is saved in test-results.json. A long-document model would be a separate training experiment.
5. Micro F1 and exact-match accuracy are different measures. Report each with its correct name.

Save the result folder and the original selected checkpoint. Send test-results.json, paper-metrics.txt, and the original history.json for the final research document. Do not publish private case data or model weights to the code repository.

## What this update implements

- **Checkpoint inspection:** identifies the 16-label task and saved epoch count. This prevents mixing ECHR2023, Realistic LJP, and ECtHR-A results.
- **Recovery export:** restores the saved validation-selected weights if the final export is absent. This avoids repeating a completed training run.
- **Paper measures and readable reports:** calculates the four requested measures from test labels, probabilities, and predictions. It keeps the model settings fixed.
- **Evaluation identity and report reuse:** records source hashes and preserves completed predictions. This helps recover from a closed terminal without choosing another test configuration.

The new helper is tested with small local models. The original trained checkpoint and its test score are on your machine. No full real-data ECHR test score is claimed by this package.
