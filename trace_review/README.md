# Revised TRACE training code

This is the reviewed version of the uploaded TRACE trainer. It preserves the TRACE architecture. It adds explicit checkpoint selection, validation loss, rare-label threshold guards, optimization controls, and complete metrics. Read FEATURES.md for what changed, why, and how.

This package contains code only. It contains no case documents, trained weights, or new ECHR accuracy results. Nine checks passed, including eight epochs with a small local encoder. Full LegalBERT training still needs model access and suitable compute.

## Run

1. Extract the ZIP and open its trace-training-review directory.
2. Install a CUDA-compatible PyTorch build for your GPU, then run `python -m pip install -r requirements.txt`. For CPU, use `python -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu`.
3. Create a data directory. Put your original train.jsonl, val.jsonl, and test.jsonl there.
4. Open train_8_epochs_revised.ipynb and run the cells. Alternatively, run the commands below.

```sh
python prepare_data.py --source data --output data-clean --exclude-validation-duplicates
python train.py --data_dir data-clean --output_dir runs/audit --epochs 8 --dry_run
python -u train.py --data_dir data-clean --output_dir runs/candidate_clean \
  --epochs 8 --max_chunks 8 --max_length 256 --batch_size 2 --neighbors 5 \
  --lr 1e-5 --head_lr 1e-4 --freeze_lower_layers 4 \
  --dropout 0.2 --encoder_dropout 0.1 --weight_decay 0.01 \
  --selection_metric micro_f1 --calibration_min_support 20 --report_train_metrics
```

The preparation command creates separate cleaned copies. It excludes validation records with normalized text also found in test. It records exclusions in data-clean/audit.json. Original files stay unchanged. Review the audit before training. Existing output directories are rejected to protect previous work.

The candidate completes all eight epochs. It selects the best validation micro-F1 epoch at threshold 0.5. Then it tunes thresholds using validation only. Labels need at least 20 positive and negative validation examples for tuning; otherwise, the threshold stays 0.5. This guard can reduce rare-label recall. Use macro_f1 instead of micro_f1 if rare-label performance is the primary objective.

For a controlled baseline, use the same cleaned data, seed, selection metric, and calibration rule. Use lr=2e-5, dropout=0.1, weight_decay=0.01, freeze_lower_layers=0, and no separate head_lr. Compare settings on validation data. The candidate changes several controls; test one factor at a time to identify which helps.

The output directory contains best.pt, history.json, metrics.json, data_audit.json, label_vocab.json, and prediction matrices. Metrics include precision, recall, F1, exact-match accuracy, and Hamming loss. F1 is not accuracy. No gain is guaranteed.

LegalBERT downloads require Hugging Face access. A complete local model directory can be supplied with --encoder_model. The model retains at most 2,032 content tokens with the commands above. Precedent retrieval remains train-only and strictly earlier by date.

Run `OMP_NUM_THREADS=2 python -m pytest -q` from this directory for the local tests. The original TRACE repository was inaccessible to the cloud environment. These files are supplied through the authorized legal-architecture repository; they have not been pushed to the TRACE repository.
