"""Validation-only checkpoint selection and optimization controls."""
import math
import torch


class CheckpointSelector:
    def __init__(self, metric):
        self.metric = metric
        self.best_score = None
        self.best_epoch = None
        self.state = None

    def update(self, model, epoch, validation=None):
        score = epoch if self.metric == "last" else validation[self.metric]
        if not math.isfinite(score):
            raise ValueError("Checkpoint selection score must be finite")
        improved = self.best_score is None or (score < self.best_score if self.metric == "loss" else score > self.best_score)
        if improved:
            self.best_score, self.best_epoch = score, epoch
            self.state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        return improved

    def restore(self, model):
        if self.state is None:
            raise ValueError("No checkpoint has been selected")
        model.load_state_dict(self.state)


def make_optimizer(model, encoder_lr, head_lr, weight_decay):
    # Do not penalize biases and LayerNorm scales. Small BERT steps preserve
    # pretrained features; new evidence heads can use a separate rate.
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        rate = encoder_lr if name.startswith("encoder.") else head_lr
        decay = weight_decay if parameter.ndim > 1 and not name.endswith("bias") else 0.0
        groups.setdefault((rate, decay), []).append(parameter)
    return torch.optim.AdamW([{"params": values, "lr": rate, "weight_decay": decay}
                             for (rate, decay), values in groups.items()])


def positive_weights(items, cap):
    targets = torch.tensor([x["label"] for x in items], dtype=torch.float32)
    counts = targets.sum(0)
    if (counts == 0).any():
        raise ValueError("Positive weighting requires a training example for every class")
    return tuple(((len(items) - counts) / counts).clamp(min=1, max=cap).tolist())


def check_split_overlap(train, validation):
    train_text = {" ".join(r["text"].split()).casefold() for r in train}
    validation_text = {" ".join(r["text"].split()).casefold() for r in validation}
    train_ids = {str(r["case_id"]) for r in train if r.get("case_id") is not None}
    validation_ids = {str(r["case_id"]) for r in validation if r.get("case_id") is not None}
    if train_text & validation_text or train_ids & validation_ids:
        raise ValueError("Training and validation cases overlap")
