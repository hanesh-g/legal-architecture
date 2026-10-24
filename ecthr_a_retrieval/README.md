# ECtHR-A: frozen Qwen retrieval baseline

Use this package on the separate system. It runs Haneesh3's released-code-style document-retrieval baseline. It does not run the earlier ECHR2023 classifier. No task-specific gradient training or eight-epoch loop is required.

## What the experiment does

1. Load `coastalcph/lex_glue`, configuration `ecthr_a`.
2. Preserve 9,000 training, 1,000 validation, and 1,000 test cases.
3. Keep the ten article outputs and all cases with empty label lists.
4. Join each case's factual paragraphs in order.
5. Use frozen `Qwen/Qwen3-Embedding-0.6B` to encode cases.
6. Index training documents only.
7. Tune the neighbor count and decision threshold on validation micro F1.
8. Use the fixed settings for one final test evaluation.

The paper's main measures are micro F1 and macro F1. This runner also reports precision, recall, exact-match accuracy, Hamming loss, per-label scores, runtime, and peak allocated GPU memory.

## Before starting

Use a separate folder and virtual environment. Keep the other project and its weights unchanged. The CUDA commands below require an NVIDIA GPU and a compatible driver. A 16 GB GPU is a reasonable starting point for the 0.6B model at batch size 1. Actual memory use must be tested. Allow several GB of free disk space for the model and dataset caches. Internet access must permit Hugging Face downloads.

Do not install the other project's CPU-only requirements lock file in this environment. No SciPy, scikit-learn, or Excel reader is required by this runner. If Windows Application Control blocks a trusted dependency, ask the administrator for an approved environment; do not disable the policy.

## Windows PowerShell: exact steps

### 1. Open the extracted folder

Extract the download. Open the folder containing `run_ecthr_a.py` and `requirements.txt`. In File Explorer, right-click an empty area and select **Open in Terminal**. Use PowerShell.

```powershell
Get-ChildItem -Name
nvidia-smi
py -3.12 --version
```

The file list must include `run_ecthr_a.py`. If Python 3.12 is missing:

```powershell
winget install --exact --id Python.Python.3.12
```

Close PowerShell and reopen it in the package folder after installation. On a managed machine, use an administrator-approved installation if needed.

### 2. Create and install the environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

The CUDA 12.8 build works with the previously shown CUDA 13.2-capable NVIDIA driver. If this system has a different or older driver, use the compatible build from https://pytorch.org/get-started/locally/.

### 3. Confirm GPU access

```powershell
.\.venv\Scripts\python.exe -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Unavailable')"
```

Continue only if CUDA is True. Pause other GPU-intensive programs before the run. Use one GPU job at a time on this system.

### 4. Run the small smoke check

```powershell
.\.venv\Scripts\python.exe -u .\run_ecthr_a.py --output runs/ecthr_a_smoke --device cuda --batch-size 1 --max-length 8192 --smoke
```

The first invocation downloads the dataset and model. It may take time before embedding starts. The smoke check uses only 20 training, 5 validation, and 5 test cases. Its scores are not final results. Success means the command exits without error and creates `runs\ecthr_a_smoke\results.json`.

### 5. Run the full experiment

After the smoke check finishes, run this complete command once:

```powershell
.\.venv\Scripts\python.exe -u .\run_ecthr_a.py --output runs/ecthr_a_full --device cuda --batch-size 1 --max-length 8192
```

Do not add a number before the command. Do not run it in the Python `>>>` prompt. Keep PowerShell open. Prevent sleep while the system is plugged in.

The runner shows progress for training embeddings, validation embeddings, parameter selection, and test embeddings. The order is:

- Embed the 9,000 training cases.
- Embed the 1,000 validation cases.
- Select k from 5, 10, 15, 20 and threshold from 0.2, 0.3, 0.4, 0.5, 0.6.
- Save `selection.json`.
- Embed the 1,000 test cases and evaluate once.
- Save `results.json`.

There are no training epochs. The encoder remains frozen. The training set supplies labeled neighbor examples, so this is not a zero-supervision method.

### 6. Recover after a closed terminal or interruption

Reopen PowerShell in the same package folder. Run the exact full command from step 5 again. There is no `--resume` flag.

Completed embedding shards are reused automatically. By default, a shard contains 64 cases. A stop can lose the current incomplete shard, but previously completed shards remain saved. The validation choice is also reused after it has been saved. A completed run prints the existing results rather than evaluating test again.

Keep the same model, context length, batch size, shard size, grids, and dependencies for a resumed run. Revisions and data fingerprints are checked. If you intentionally change settings, use a new output directory. Do not delete old caches without checking what they contain.

### 7. Read the full results

```powershell
Get-Content .\runs\ecthr_a_full\results.json
```

Send these three files for the final report:

```text
runs/ecthr_a_full/results.json
runs/ecthr_a_full/selection.json
runs/ecthr_a_full/run.json
```

Ensure results.json says `smoke: false`, `dataset_mode: official`, and `test.examples: 1000`. Scores are on a 0–1 scale. Multiply by 100 to express percentages. Micro F1 is not accuracy; exact-match accuracy requires the complete predicted article set to match.

## Output files

```text
runs/ecthr_a_full/
  run.json                  # Data/model revisions, fingerprints, settings, mapping
  selection.json            # Validation grid and chosen k/threshold
  results.json              # Final test results
  test-predictions.pt        # Probabilities and test targets for verification
  cache/train/*.pt          # Completed training embedding shards
  cache/validation/*.pt     # Completed validation embedding shards
  cache/test/*.pt           # Completed test embedding shards
```

The dataset loader uses the Hugging Face cache. You do not need to create CSV or JSONL files. Do not copy PILOT/ECHR2023, Realistic LJP, or IJJD inputs into this experiment.

## If GPU memory is insufficient

Batch size is already 1. Close other GPU programs and rerun the same command first. If 8,192 tokens still do not fit, use a separate run with a shorter context:

```powershell
.\.venv\Scripts\python.exe -u .\run_ecthr_a.py --output runs/ecthr_a_4096 --device cuda --batch-size 1 --max-length 4096
```

A 4,096-token result is a different experiment. Record the limit in the report. Do not reuse the 8,192-token output folder or claim identical context to the paper.

## Reproducibility and limits

Source paper: *Retrieval-Based Multi-Label Legal Annotation: Extensible, Data-Efficient and Hallucination-Free* (uploaded as Haneesh3.pdf).

Data: https://huggingface.co/datasets/coastalcph/lex_glue

Released reference code: https://github.com/lizhang-AIandLaw/Retrieval-based-Multi-label-Legal-Annotation

This implementation follows the released training-document k-NN baseline, with validation-only parameter selection, last-visible-token pooling, atomic shard saves, strict cache identities, and explicit reports. It does not implement a label-description index, reranking, GPT inference, fine-tuning, or every experiment in the paper. The reference eval_rag.py test-based grid search is not used here.

The output labels, in source index order, are 2, 3, 5, 6, 8, 9, 10, 11, 14, and P1-1. Index 0 maps to Article 2. No-positive-label cases remain empty; no eleventh output is added. Macro scores average the ten article labels, including zero-support labels as zero when undefined. No instruction prefix is added to document embeddings, matching the simple released-code baseline. Tokenizer truncation is capped at the recorded length.

The runner records normalized-text overlaps but preserves official splits. Inspect any nonzero overlap before making claims about generalization. It does not establish separation of related cases with different text. It does not apply PILOT's chronological precedent filter; this is the distinct LexGLUE document-retrieval task.

Runtime and peak GPU allocation refer to the current invocation. Cache reuse can omit earlier encoding time. Do not compare a warm-cache invocation with a cold full run as if they measured the same work.

Software validation uses small local fixtures and a tiny Qwen3 model. It does not establish a full Qwen3-0.6B ECtHR-A score. No accuracy improvement or runtime is guaranteed.
