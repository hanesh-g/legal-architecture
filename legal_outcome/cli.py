"""Train, evaluate, predict, or run a download-free architecture smoke check."""
import argparse
import math
import hashlib
import sys
from tqdm.auto import tqdm
from dataclasses import replace
import json
import random
from pathlib import Path
import numpy as np
import torch
from .config import ModelConfig
from .training import CheckpointSelector, make_optimizer, positive_weights, check_split_overlap, save_training_state, load_training_state, linear_schedule
from .data import DocumentProcessor, collate_documents
from .model import LegalOutcomeModel
from .reporting import history_table, write_history, result_table, save_result_files
from .semantic_cache import cache_semantics, model_signature
from .runtime import read_records, prepare_records, batches, evaluate, save_checkpoint, load_checkpoint


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(args):
    from transformers import AutoTokenizer
    from sentence_transformers import SentenceTransformer
    set_seed(args.seed)
    config = ModelConfig(tuple(args.labels), max_tokens=args.max_tokens, max_spans=args.max_spans,
                         task_type=args.task_type, threshold=args.threshold, dropout=args.dropout,
                         freeze_lower_layers=args.freeze_lower_layers,
                         legalbert_model=args.legalbert_model, minilm_model=args.minilm_model,
                         legalbert_revision=args.legalbert_revision, minilm_revision=args.minilm_revision,
                         chunk_tokens=args.chunk_tokens, encoder_chunk_batch_size=args.encoder_chunk_batch_size,
                         gradient_checkpointing=args.gradient_checkpointing)
    if Path(args.output).exists() and not args.resume:
        raise ValueError("Output already exists. Use a new checkpoint directory")
    if args.epochs < 1 or args.learning_rate <= 0 or args.batch_size < 1:
        raise ValueError("Require positive epochs, learning rate, and batch size")
    head_lr = args.head_learning_rate if args.head_learning_rate is not None else args.learning_rate
    if not all(math.isfinite(v) for v in (args.learning_rate, head_lr, args.weight_decay)) or head_lr <= 0 or args.weight_decay < 0:
        raise ValueError("Invalid learning rate or weight decay")
    if args.positive_weight_cap is not None and (config.task_type != "multi_label" or not math.isfinite(args.positive_weight_cap) or args.positive_weight_cap < 1):
        raise ValueError("Positive weight cap requires multi-label mode and a finite value of at least one")
    if args.gradient_accumulation_steps < 1 or not 0 <= args.warmup_ratio < 1:
        raise ValueError("Invalid gradient accumulation or warmup")
    if args.lr_schedule == "constant" and args.warmup_ratio:
        raise ValueError("Warmup requires a linear schedule")
    rows = read_records(args.data, config.labels, config.task_type)
    validation_rows = read_records(args.validation, config.labels, config.task_type) if args.validation else None
    selection = args.selection_metric
    if selection == "auto":
        selection = "macro_f1" if validation_rows else "last"
    if selection != "last" and validation_rows is None:
        raise ValueError("Validation data is required for best-checkpoint selection")
    if ((selection == "accuracy" and config.task_type == "multi_label") or
            (selection == "exact_match_accuracy" and config.task_type == "single_label")):
        raise ValueError("Selection metric is incompatible with task type")
    if validation_rows:
        check_split_overlap(rows, validation_rows)
    print("Preparing tokenizer and documents...", file=sys.stderr, flush=True)
    processor = DocumentProcessor(AutoTokenizer.from_pretrained(config.legalbert_model, revision=config.legalbert_revision), config)
    items = prepare_records(rows, processor)
    validation_items = prepare_records(validation_rows, processor) if validation_rows else None
    print("Loading MiniLM and LegalBERT...", file=sys.stderr, flush=True)
    semantic_encoder = SentenceTransformer(config.minilm_model, revision=config.minilm_revision, device="cpu")
    if args.positive_weight_cap is not None:
        config = replace(config, positive_weights=positive_weights(items, args.positive_weight_cap))
        processor.config = config
    if args.semantic_cache:
        signature = model_signature(semantic_encoder, config.minilm_model, config.minilm_revision)
        cache_semantics(items, semantic_encoder, args.semantic_cache, signature)
        if validation_items:
            cache_semantics(validation_items, semantic_encoder, args.semantic_cache, signature)
    model = LegalOutcomeModel(config).to(args.device)
    optimizer = make_optimizer(model, args.learning_rate, head_lr, args.weight_decay)
    total_batches = math.ceil(len(items) / args.batch_size)
    updates_per_epoch = math.ceil(total_batches / args.gradient_accumulation_steps)
    scheduler = linear_schedule(optimizer, updates_per_epoch * args.epochs, args.warmup_ratio) if args.lr_schedule == "linear" else None
    selector = CheckpointSelector(selection)
    contract = {"config": config.to_dict(), "selection": selection, "seed": args.seed,
                "batch_size": args.batch_size, "encoder_lr": args.learning_rate, "head_lr": head_lr,
                "weight_decay": args.weight_decay, "device": args.device,
                "report_training_metrics": args.report_training_metrics,
                "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                "validation_sha256": hashlib.sha256(Path(args.validation).read_bytes()).hexdigest() if args.validation else None}
    if args.semantic_cache:
        contract["semantic_cache_enabled"] = True
    if args.gradient_accumulation_steps != 1:
        contract["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if scheduler is not None:
        contract["lr_schedule"] = {"type": args.lr_schedule, "warmup_ratio": args.warmup_ratio, "total_epochs": args.epochs}
    history, start_epoch = [], 0
    output_dir = Path(args.output)
    state_path = output_dir / "training-state.pt"
    if args.resume:
        if Path(args.resume).resolve() != state_path.resolve():
            raise ValueError("Resume file must be training-state.pt inside the output directory")
        start_epoch, history = load_training_state(args.resume, model, optimizer, selector, contract, scheduler=scheduler)
        if args.epochs < start_epoch:
            raise ValueError("Total epochs must be at least the completed checkpoint epochs")
        print(f"Resuming after epoch {start_epoch}; next epoch is {start_epoch + 1}", file=sys.stderr, flush=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    if args.display == "table":
        print(f"TRAINING | {len(items)} cases | {len(config.labels)} labels | {args.epochs} total epochs", flush=True)
        print(f"Device: {args.device} | seed: {args.seed} | selection: {selection}", flush=True)
        print(f"Input limit: {config.max_content_tokens} content tokens | decision threshold: {config.threshold}", flush=True)
        print(f"Microbatch: {args.batch_size} | accumulation: {args.gradient_accumulation_steps} | nominal effective batch: {args.batch_size * args.gradient_accumulation_steps}", flush=True)
        print(f"Schedule: {args.lr_schedule} | warmup ratio: {args.warmup_ratio}", flush=True)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total = 0.0
        iterator = batches(items, args.batch_size, semantic_encoder, args.device, shuffle=True)
        progress = tqdm(iterator, total=math.ceil(len(items) / args.batch_size), desc=f"Epoch {epoch + 1}/{args.epochs}")
        optimizer.zero_grad(set_to_none=True)
        optimizer_updates = 0
        for batch_index, batch in enumerate(progress):
            output = model(**batch)
            if not torch.isfinite(output["loss"]):
                raise RuntimeError("Training loss is not finite")
            window_start = (batch_index // args.gradient_accumulation_steps) * args.gradient_accumulation_steps
            window_size = min(args.gradient_accumulation_steps, total_batches - window_start)
            (output["loss"] / window_size).backward()
            if (batch_index + 1) % args.gradient_accumulation_steps == 0 or batch_index + 1 == total_batches:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_updates += 1
            total += output["loss"].item() * len(batch["labels"])
            if args.display == "table":
                progress.set_postfix(loss=f"{output['loss'].item():.4f}", refresh=False)
        report = {"epoch": epoch + 1, "training_loss": total / len(items)}
        if args.gradient_accumulation_steps != 1 or scheduler is not None:
            report.update(optimizer_updates=optimizer_updates, optimizer_group_lrs=[group["lr"] for group in optimizer.param_groups])
        if validation_items:
            print("Evaluating validation set...", file=sys.stderr, flush=True)
            report["validation"] = evaluate(model, validation_items, semantic_encoder, args.batch_size, args.device, progress=args.display == "table")
        if args.report_training_metrics:
            print("Evaluating training set...", file=sys.stderr, flush=True)
            report["training"] = evaluate(model, items, semantic_encoder, args.batch_size, args.device, progress=args.display == "table")
        report["selected"] = selector.update(model, epoch + 1, report.get("validation"))
        history.append(report)
        save_training_state(state_path, model, optimizer, selector, history, contract, scheduler=scheduler)
        write_history(output_dir, history)
        print(f"Checkpoint saved after epoch {epoch + 1}: {state_path}", file=sys.stderr, flush=True)
        print(history_table(history) if args.display == "table" else json.dumps(report), flush=True)
    selector.restore(model)
    save_checkpoint(args.output, model, processor, {"seed": args.seed, "history": history,
                    "selected_epoch": selector.best_epoch, "selection_metric": selection,
                    "selection_score": selector.best_score, "head_learning_rate": head_lr,
                    "weight_decay": args.weight_decay,
                    "data_sha256": contract["data_sha256"], "validation_sha256": contract["validation_sha256"],
                    "epochs": args.epochs, "learning_rate": args.learning_rate,
                    "training_examples": len(items), "batch_size": args.batch_size,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
                    "lr_schedule": args.lr_schedule, "warmup_ratio": args.warmup_ratio}, allow_existing=True)
    if args.display == "table":
        print(f"\nTRAINING COMPLETE | selected epoch: {selector.best_epoch} | {selection}: {selector.best_score:.6f}", flush=True)
        print(f"Prediction model: {output_dir / 'model.pt'}\nRecovery state: {state_path}", flush=True)


def predict(args):
    model, processor, semantic = load_checkpoint(args.checkpoint, args.device)
    item = processor(Path(args.text_file).read_text(encoding="utf-8"))
    batch = {k: v.to(args.device) for k, v in collate_documents([item], semantic).items()}
    with torch.no_grad():
        result = model(**batch)
    report = {"probabilities": dict(zip(model.config.labels, result["probabilities"][0].tolist()))}
    if model.config.task_type == "multi_label":
        selected = torch.where(result["predictions"][0])[0].tolist()
        report.update(predicted_labels=[model.config.labels[i] for i in selected], threshold=model.config.threshold)
    else:
        selected = [result["predictions"].item()]
        report["predicted_label"] = model.config.labels[selected[0]]
    if args.evidence:
        # Attention gives model weights; it does not establish legal causation.
        evidence = {model.config.labels[label]: [{"text": span, "attention": result["span_attention"][0, label, i].item()}
                    for i, span in enumerate(item["spans"])] for label in selected}
        report["span_evidence"] = evidence if model.config.task_type == "multi_label" else evidence[model.config.labels[selected[0]]]
    print(json.dumps(report, indent=2))


def score(args):
    model, processor, semantic = load_checkpoint(args.checkpoint, args.device)
    items = prepare_records(read_records(args.data, model.config.labels, model.config.task_type), processor)
    if args.semantic_cache:
        cache_semantics(items, semantic, args.semantic_cache, model_signature(semantic, model.config.minilm_model, model.config.minilm_revision))
    directory = Path(args.results_dir) if args.results_dir else None
    if directory:
        directory.mkdir(parents=True, exist_ok=True)
    result = evaluate(model, items, semantic, args.batch_size, args.device,
                      predictions_path=directory / "predictions.pt" if directory else None,
                      progress=args.display == "table")
    metadata = json.loads((Path(args.checkpoint) / "metadata.json").read_text(encoding="utf-8"))
    result["selected_epoch"] = metadata.get("training_report", {}).get("selected_epoch")
    result["decision_threshold"] = model.config.threshold if model.config.task_type == "multi_label" else None
    result["max_packed_tokens"] = model.config.max_tokens
    result["max_content_tokens"] = model.config.max_content_tokens
    result["input_variant"] = model.config.input_variant
    result["chunk_tokens"] = model.config.chunk_tokens
    if model.config.chunk_tokens:
        result["input_limit_note"] = (f"Long-document extension: at most {model.config.max_tokens} packed positions and "
                                      f"{model.config.max_content_tokens} content tokens, in {model.config.chunk_tokens}-token independent chunks. "
                                      "CLS context is the mean of valid chunk CLS states. Single-run published-reference comparison.")
    else:
        result["input_limit_note"] = f"Truncated-input variant: at most {model.config.max_content_tokens} content tokens per case; not the paper's 8192-token protocol."
    if directory:
        report = save_result_files(directory, result)
    else:
        report = result_table(result)
    print(report if args.display == "table" else json.dumps(result, indent=2))


def smoke(args):
    from transformers import BertConfig, BertModel
    set_seed(7)
    config = ModelConfig(("example_a", "example_b"), max_tokens=16, dropout=0.0)
    encoder = BertModel(BertConfig(vocab_size=32, hidden_size=16, num_hidden_layers=1,
                                  num_attention_heads=2, intermediate_size=32))
    model = LegalOutcomeModel(config, encoder)
    batch = dict(input_ids=torch.tensor([[1, 4, 5, 6, 2, 0]]), attention_mask=torch.tensor([[1, 1, 1, 1, 1, 0]]),
                 span_ids=torch.tensor([[-1, 0, 0, 1, -1, -1]]), span_mask=torch.tensor([[True, True]]),
                 cue_features=torch.ones(1, 2, 12), semantic_embeddings=torch.randn(1, 2, 8), labels=torch.tensor([1]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    result = model(**batch)
    result["loss"].backward()
    optimizer.step()
    if not torch.isfinite(result["loss"]) or not torch.allclose(result["probabilities"].sum(-1), torch.ones(1)):
        raise RuntimeError("Smoke check failed")
    print(json.dumps({"status": "passed", "loss": result["loss"].item(),
                      "note": "Tiny random BERT; no trained legal predictions or model downloads"}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train")
    p.add_argument("--data", required=True)
    p.add_argument("--validation")
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--resume", help="Path to the output directory training-state.pt; epochs is the total target")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--head-learning-rate", type=float)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--task-type", choices=["single_label", "multi_label"], default="single_label")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--positive-weight-cap", type=float)
    p.add_argument("--selection-metric", choices=["auto", "last", "loss", "micro_f1", "macro_f1", "accuracy", "exact_match_accuracy"], default="auto")
    p.add_argument("--report-training-metrics", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--legalbert-model", default="nlpaueb/legal-bert-base-uncased")
    p.add_argument("--minilm-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--legalbert-revision")
    p.add_argument("--minilm-revision")
    p.add_argument("--semantic-cache")
    p.add_argument("--display", choices=["json", "table"], default="json")
    p.add_argument("--max-tokens", type=int, default=384)
    p.add_argument("--max-spans", type=int, default=8)
    p.add_argument("--chunk-tokens", type=int)
    p.add_argument("--encoder-chunk-batch-size", type=int, default=1)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--lr-schedule", choices=["constant", "linear"], default="constant")
    p.add_argument("--warmup-ratio", type=float, default=0.0)
    p.add_argument("--freeze-lower-layers", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.set_defaults(run=train)
    p = sub.add_parser("predict")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--text-file", required=True)
    p.add_argument("--evidence", action="store_true")
    p.add_argument("--device", default="cpu")
    p.set_defaults(run=predict)
    p = sub.add_parser("evaluate")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--display", choices=["json", "table"], default="json")
    p.add_argument("--results-dir")
    p.add_argument("--semantic-cache")
    p.set_defaults(run=score)
    p = sub.add_parser("smoke")
    p.set_defaults(run=smoke)
    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
