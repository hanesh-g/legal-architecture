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


def atomic_torch_save(payload, path):
    """Keep the previous complete checkpoint until replacement succeeds."""
    from pathlib import Path
    import os
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    try:
        with temporary.open('wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def capture_rng():
    import random
    import numpy as np
    state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [state[0], state[1].tolist(), *state[2:]],
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    import random
    import numpy as np
    random.setstate(state['python'])
    value = state['numpy']
    np.random.set_state((value[0], np.asarray(value[1], dtype=np.uint32), *value[2:]))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        if not torch.cuda.is_available() or len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('Resume requires the original CUDA device count')
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def save_training_state(path, model, optimizer, selector, history, contract):
    atomic_torch_save({'format_version': 1, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                      'best_model': selector.state, 'best_score': selector.best_score,
                      'best_epoch': selector.best_epoch, 'history': history, 'completed_epoch': history[-1]['epoch'],
                      'contract': contract, 'rng': capture_rng()}, path)


def load_training_state(path, model, optimizer, selector, contract):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    from .config import ModelConfig
    saved_contract = dict(payload['contract'])
    saved_contract['config'] = ModelConfig(**saved_contract['config']).to_dict()
    if payload.get('format_version') != 1 or saved_contract != contract:
        raise ValueError('Resume settings or dataset differ from the saved run')
    model.load_state_dict(payload['model'])
    optimizer.load_state_dict(payload['optimizer'])
    selector.state, selector.best_score, selector.best_epoch = payload['best_model'], payload['best_score'], payload['best_epoch']
    restore_rng(payload['rng'])
    return payload['completed_epoch'], payload['history']
