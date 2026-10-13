"""Training data, checkpoint storage, and evaluation helpers."""
import json
from pathlib import Path
import torch
from .config import ModelConfig
from .data import DocumentProcessor, collate_documents
from .model import LegalOutcomeModel
from .cues import CueExtractor


def read_records(path, labels):
    records = []
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("text"), str) or not row["text"].strip():
                raise ValueError(f"Line {line_number}: require nonempty text")
            if row.get("label") not in labels:
                raise ValueError(f"Line {line_number}: unknown or missing label")
            records.append(row)
    if not records:
        raise ValueError("Dataset is empty")
    return records


def prepare_records(records, processor):
    return [{**processor(row["text"]), "label": processor.config.labels.index(row["label"])} for row in records]


def batches(items, size, semantic_encoder, device, shuffle=False):
    if size < 1:
        raise ValueError("Batch size must be positive")
    order = torch.randperm(len(items)).tolist() if shuffle else range(len(items))
    ordered = [items[i] for i in order]
    for offset in range(0, len(ordered), size):
        yield {k: v.to(device) for k, v in collate_documents(ordered[offset:offset + size], semantic_encoder).items()}


def metrics(targets, predictions, label_names):
    matrix = torch.zeros(len(label_names), len(label_names), dtype=torch.long)
    for actual, predicted in zip(targets, predictions):
        matrix[actual, predicted] += 1
    tp = matrix.diag().float()
    denominator = matrix.sum(0) + matrix.sum(1)
    f1 = torch.where(denominator > 0, 2 * tp / denominator, 0)
    return {"examples": len(targets), "accuracy": (tp.sum() / max(1, len(targets))).item(),
            "macro_f1": f1.mean().item(), "labels": list(label_names), "confusion_matrix": matrix.tolist()}


@torch.no_grad()
def evaluate(model, items, semantic_encoder, batch_size, device):
    model.eval()
    targets, predictions, loss_sum = [], [], 0.0
    for batch in batches(items, batch_size, semantic_encoder, device):
        output = model(**batch)
        targets.extend(batch["labels"].tolist())
        predictions.extend(output["predictions"].tolist())
        loss_sum += output["loss"].item() * len(batch["labels"])
    return {**metrics(targets, predictions, model.config.labels), "loss": loss_sum / len(items)}


def save_checkpoint(path, model, processor, report):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    processor.tokenizer.save_pretrained(path / "tokenizer")
    metadata = {"format_version": 1, "model_config": model.config.to_dict(),
                "encoder_config": model.encoder.config.to_dict(),
                "cue_patterns": processor.cues.patterns, "training_report": report}
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), path / "model.pt")


def load_checkpoint(path, device):
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from sentence_transformers import SentenceTransformer
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if metadata["format_version"] != 1:
        raise ValueError("Unsupported checkpoint format")
    config = ModelConfig(**metadata["model_config"])
    encoder_config = dict(metadata["encoder_config"])
    model_type = encoder_config.pop("model_type")
    encoder = AutoModel.from_config(AutoConfig.for_model(model_type, **encoder_config))
    model = LegalOutcomeModel(config, encoder)
    model.load_state_dict(torch.load(path / "model.pt", map_location="cpu", weights_only=True))
    model.to(device).eval()
    processor = DocumentProcessor(AutoTokenizer.from_pretrained(path / "tokenizer", local_files_only=True), config,
                                  CueExtractor(metadata["cue_patterns"]))
    semantic_encoder = SentenceTransformer(config.minilm_model, device="cpu")
    return model, processor, semantic_encoder
