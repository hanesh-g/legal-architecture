"""Training data, checkpoint storage, and evaluation helpers."""
import json
from pathlib import Path
import torch
from .config import ModelConfig
from .data import DocumentProcessor, collate_documents
from .model import LegalOutcomeModel
from .cues import CueExtractor


def read_records(path, labels, task_type="single_label"):
    records = []
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("text"), str) or not row["text"].strip():
                raise ValueError(f"Line {line_number}: require nonempty text")
            if task_type == "multi_label":
                values = row.get("labels")
                if not isinstance(values, list) or any(not isinstance(v, str) or v not in labels for v in values) or len(set(values)) != len(values):
                    raise ValueError(f"Line {line_number}: require a list of distinct known labels")
            elif row.get("label") not in labels:
                raise ValueError(f"Line {line_number}: unknown or missing label")
            records.append(row)
    if not records:
        raise ValueError("Dataset is empty")
    return records


def prepare_records(records, processor):
    if processor.config.task_type == "multi_label":
        return [{**processor(row["text"]), "label": [int(label in row["labels"]) for label in processor.config.labels]} for row in records]
    return [{**processor(row["text"]), "label": processor.config.labels.index(row["label"])} for row in records]


def batches(items, size, semantic_encoder, device, shuffle=False):
    if size < 1:
        raise ValueError("Batch size must be positive")
    order = torch.randperm(len(items)).tolist() if shuffle else range(len(items))
    ordered = [items[i] for i in order]
    for offset in range(0, len(ordered), size):
        yield {k: v.to(device) for k, v in collate_documents(ordered[offset:offset + size], semantic_encoder).items()}


def _divide(numerator, denominator):
    return torch.where(denominator > 0, numerator / denominator.clamp_min(1e-12), 0)


def _class_metrics(tp, fp, fn, support, label_names):
    precision, recall = _divide(tp, tp + fp), _divide(tp, tp + fn)
    f1 = _divide(2 * tp, 2 * tp + fp + fn)
    return {
        "micro_precision": _divide(tp.sum(), (tp + fp).sum()).item(),
        "micro_recall": _divide(tp.sum(), (tp + fn).sum()).item(),
        "micro_f1": _divide(2 * tp.sum(), (2 * tp + fp + fn).sum()).item(),
        "macro_precision": precision.mean().item(), "macro_recall": recall.mean().item(),
        "macro_f1": f1.mean().item(),
        "per_label": {name: {"precision": precision[i].item(), "recall": recall[i].item(),
                            "f1": f1[i].item(), "support": int(support[i])} for i, name in enumerate(label_names)}
    }


def metrics(targets, predictions, label_names, task_type="single_label"):
    if not targets or len(targets) != len(predictions):
        raise ValueError("Metrics require equal nonempty targets and predictions")
    result = {"examples": len(targets), "labels": list(label_names), "task_type": task_type}
    if task_type == "multi_label":
        actual, predicted = torch.as_tensor(targets).bool(), torch.as_tensor(predictions).bool()
        if actual.shape != predicted.shape or actual.shape != (len(targets), len(label_names)):
            raise ValueError("Invalid multi-label metric shape")
        tp = (actual & predicted).sum(0).float()
        fp = (~actual & predicted).sum(0).float()
        fn = (actual & ~predicted).sum(0).float()
        result.update(exact_match_accuracy=(actual == predicted).all(-1).float().mean().item(),
                      hamming_loss=(actual != predicted).float().mean().item())
        return {**result, **_class_metrics(tp, fp, fn, actual.sum(0), label_names)}
    matrix = torch.zeros(len(label_names), len(label_names), dtype=torch.long)
    for actual, predicted in zip(targets, predictions):
        matrix[actual, predicted] += 1
    tp = matrix.diag().float()
    return {**result, "accuracy": (tp.sum() / len(targets)).item(), "confusion_matrix": matrix.tolist(),
            **_class_metrics(tp, matrix.sum(0) - tp, matrix.sum(1) - tp, matrix.sum(1), label_names)}


@torch.no_grad()
def evaluate(model, items, semantic_encoder, batch_size, device):
    model.eval()
    targets, predictions, loss_sum = [], [], 0.0
    for batch in batches(items, batch_size, semantic_encoder, device):
        output = model(**batch)
        targets.extend(batch["labels"].tolist())
        predictions.extend(output["predictions"].tolist())
        loss_sum += output["loss"].item() * len(batch["labels"])
    return {**metrics(targets, predictions, model.config.labels, model.config.task_type), "loss": loss_sum / len(items)}


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
