"""Train, evaluate, predict, or run a download-free architecture smoke check."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from .config import ModelConfig
from .data import DocumentProcessor, collate_documents
from .model import LegalOutcomeModel
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
                         freeze_lower_layers=args.freeze_lower_layers,
                         legalbert_model=args.legalbert_model, minilm_model=args.minilm_model)
    if Path(args.output).exists():
        raise ValueError("Output already exists. Use a new checkpoint directory")
    if args.epochs < 1 or args.learning_rate <= 0 or args.batch_size < 1:
        raise ValueError("Require positive epochs, learning rate, and batch size")
    rows = read_records(args.data, config.labels)
    validation_rows = read_records(args.validation, config.labels) if args.validation else None
    if validation_rows and {r['text'] for r in rows} & {r['text'] for r in validation_rows}:
        raise ValueError("Training and validation documents overlap")
    processor = DocumentProcessor(AutoTokenizer.from_pretrained(config.legalbert_model), config)
    items = prepare_records(rows, processor)
    validation_items = prepare_records(validation_rows, processor) if validation_rows else None
    semantic_encoder = SentenceTransformer(config.minilm_model, device="cpu")
    model = LegalOutcomeModel(config).to(args.device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    history = []
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for batch in batches(items, args.batch_size, semantic_encoder, args.device, shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            if not torch.isfinite(output["loss"]):
                raise RuntimeError("Training loss is not finite")
            output["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += output["loss"].item() * len(batch["labels"])
        report = {"epoch": epoch + 1, "training_loss": total / len(items)}
        if validation_items:
            report["validation"] = evaluate(model, validation_items, semantic_encoder, args.batch_size, args.device)
        history.append(report)
        print(json.dumps(report), flush=True)
    save_checkpoint(args.output, model, processor, {"seed": args.seed, "history": history,
                    "epochs": args.epochs, "learning_rate": args.learning_rate,
                    "training_examples": len(items), "batch_size": args.batch_size})


def predict(args):
    model, processor, semantic = load_checkpoint(args.checkpoint, args.device)
    item = processor(Path(args.text_file).read_text(encoding="utf-8"))
    batch = {k: v.to(args.device) for k, v in collate_documents([item], semantic).items()}
    with torch.no_grad():
        result = model(**batch)
    label = result["predictions"].item()
    report = {"predicted_label": model.config.labels[label],
              "probabilities": dict(zip(model.config.labels, result["probabilities"][0].tolist()))}
    if args.evidence:
        # Attention gives model weights; it does not establish legal causation.
        report["span_evidence"] = [{"text": span, "attention": result["span_attention"][0, label, i].item()}
                                   for i, span in enumerate(item["spans"])]
    print(json.dumps(report, indent=2))


def score(args):
    model, processor, semantic = load_checkpoint(args.checkpoint, args.device)
    items = prepare_records(read_records(args.data, model.config.labels), processor)
    print(json.dumps(evaluate(model, items, semantic, args.batch_size, args.device), indent=2))


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train")
    p.add_argument("--data", required=True)
    p.add_argument("--validation")
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--legalbert-model", default="nlpaueb/legal-bert-base-uncased")
    p.add_argument("--minilm-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--max-tokens", type=int, default=384)
    p.add_argument("--max-spans", type=int, default=8)
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
    p.set_defaults(run=score)
    p = sub.add_parser("smoke")
    p.set_defaults(run=smoke)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
