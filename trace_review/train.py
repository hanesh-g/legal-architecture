#!/usr/bin/env python3
"""TRACE-Legal no-LLM/no-RL prototype for user-prepared ECHR2023 JSONL."""
import argparse
import json
import math
import random
from datetime import date
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score, hamming_loss
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path):
    out = []
    with open(path, encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            x = json.loads(line)
            for field in ('id', 'text', 'labels', 'date'):
                if field not in x:
                    raise ValueError(f'{path}:{line_no}: missing {field}')
            if not isinstance(x['labels'], list):
                raise ValueError(f'{path}:{line_no}: labels must be a list')
            x['id'] = str(x['id'])
            x['labels'] = [str(y) for y in x['labels']]
            x['date_ord'] = date.fromisoformat(str(x['date'])).toordinal()
            if not str(x['text']).strip():
                raise ValueError(f'{path}:{line_no}: empty text')
            out.append(x)
    if not out:
        raise ValueError(f'Empty dataset: {path}')
    return out


def validate_splits(splits):
    ids = [x['id'] for rows in splits.values() for x in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate case ID across splits or within a split')
    texts = [' '.join(x['text'].split()).casefold() for rows in splits.values() for x in rows]
    if len(texts) != len(set(texts)):
        raise ValueError('Identical document text detected across or within splits; audit leakage')


class CaseDataset(Dataset):
    def __init__(self, rows, tokenizer, label_to_idx, max_chunks, max_length):
        self.rows = rows
        self.label_to_idx = label_to_idx
        self.tokenizer = tokenizer
        self.max_chunks = max_chunks
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        # Tokenize once without truncation; make non-overlapping chunks of max_length-2 tokens.
        token_ids = self.tokenizer.encode(r['text'], add_special_tokens=False)
        width = self.max_length - 2
        segments = [token_ids[j:j + width] for j in range(0, len(token_ids), width)][:self.max_chunks]
        if not segments:
            segments = [[]]
        ids = []
        masks = []
        for seg in segments:
            seq = self.tokenizer.build_inputs_with_special_tokens(seg)
            seq = seq[:self.max_length]
            ids.append(seq + [self.tokenizer.pad_token_id] * (self.max_length - len(seq)))
            masks.append([1] * len(seq) + [0] * (self.max_length - len(seq)))
        while len(ids) < self.max_chunks:
            ids.append([self.tokenizer.pad_token_id] * self.max_length)
            masks.append([0] * self.max_length)
        y = torch.zeros(len(self.label_to_idx), dtype=torch.float32)
        for label in r['labels']:
            if label not in self.label_to_idx:
                raise ValueError(f'Unknown label {label} for {r["id"]}')
            y[self.label_to_idx[label]] = 1
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(masks, dtype=torch.long),
            'y': y,
            'index': torch.tensor(i, dtype=torch.long)
        }


class HierarchicalEncoder(nn.Module):
    def __init__(self, model_name, dropout):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        h = self.bert.config.hidden_size
        self.chunk_score = nn.Linear(h, 1)
        self.dropout = nn.Dropout(dropout)
        self.dim = h

    def forward(self, ids, mask):
        b, c, l = ids.shape
        flat_ids = ids.reshape(b * c, l)
        flat_mask = mask.reshape(b * c, l)
        valid = flat_mask.any(-1)
        pooled = torch.zeros((b * c, self.dim), device=ids.device)
        if valid.any():
            enc = self.bert(input_ids=flat_ids[valid], attention_mask=flat_mask[valid])
            pooled[valid] = enc.last_hidden_state[:, 0]
        chunks = pooled.reshape(b, c, self.dim)
        present = mask.any(-1)
        score = self.chunk_score(self.dropout(chunks)).squeeze(-1)
        score = score.masked_fill(~present, -1e4)
        weights = score.softmax(-1)
        return (chunks * weights.unsqueeze(-1)).sum(1)


@torch.no_grad()
def cache_precedent_embeddings(encoder, loader, device, n):
    encoder.eval()
    cache = torch.zeros(n, encoder.dim, device='cpu')
    for batch in tqdm(loader, desc='Caching train precedents'):
        rep = encoder(batch['ids'].to(device), batch['mask'].to(device))
        cache[batch['index']] = rep.detach().cpu()
    return F.normalize(cache, dim=-1)


@torch.no_grad()
def cache_query_embeddings(encoder, loader, device, n):
    encoder.eval()
    cache = torch.zeros(n, encoder.dim, device='cpu')
    for batch in tqdm(loader, desc='Caching query retrieval vectors'):
        rep = encoder(batch['ids'].to(device), batch['mask'].to(device))
        cache[batch['index']] = rep.detach().cpu()
    return F.normalize(cache, dim=-1)


def make_neighbors(queries, train, query_vectors, precedent_vectors, k, train_split=False):
    """Strictly earlier, train-only nearest neighbors; zero padding for absent neighbors."""
    if k < 1:
        raise ValueError('neighbors must be >=1')
    similarities = query_vectors @ precedent_vectors.T
    idx = torch.full((len(queries), k), -1, dtype=torch.long)
    sim = torch.zeros((len(queries), k))
    age = torch.zeros((len(queries), k))
    train_dates = np.asarray([r['date_ord'] for r in train])
    for i, q in enumerate(queries):
        allowed = train_dates < q['date_ord']
        if train_split:
            allowed[i] = False
        choices = np.flatnonzero(allowed)
        if len(choices) == 0:
            continue
        scores = similarities[i, torch.as_tensor(choices)]
        top = torch.topk(scores, k=min(k, len(choices)))
        selected = choices[top.indices.numpy()]
        n = len(selected)
        idx[i, :n] = torch.as_tensor(selected)
        sim[i, :n] = top.values
        age[i, :n] = torch.as_tensor((q['date_ord'] - train_dates[selected]) / 365.25, dtype=torch.float32)
    return {'idx': idx, 'sim': sim, 'age': age}


class GraphCrossAttention(nn.Module):
    def __init__(self, h, heads, dropout, disable_graph=False, disable_cross_attention=False):
        super().__init__()
        if h % heads:
            raise ValueError('hidden size must be divisible by attention heads')
        self.disable_graph = disable_graph
        self.disable_cross_attention = disable_cross_attention
        self.edge_proj = nn.Linear(2, h)
        self.graph_query = nn.Linear(h, h)
        self.graph_key = nn.Linear(h, h)
        self.graph_value = nn.Linear(h, h)
        self.cross = nn.MultiheadAttention(h, heads, dropout=dropout, batch_first=True)
        self.gate = nn.Linear(h * 2, h)
        self.norm = nn.LayerNorm(h)
        self.dropout = nn.Dropout(dropout)

    def forward(self, case, precedent_vectors, valid, similarity, age):
        if self.disable_graph or not valid.any():
            return case
        edge = torch.stack([similarity, torch.log1p(age.clamp(min=0))], -1)
        nodes = precedent_vectors + self.edge_proj(edge)
        scores = (self.graph_query(case).unsqueeze(1) * self.graph_key(nodes)).sum(-1) / math.sqrt(case.shape[-1])
        scores = scores.masked_fill(~valid, -1e4)
        weights = scores.softmax(-1) * valid.float()
        weights = weights / weights.sum(-1, keepdim=True).clamp(min=1e-8)
        graph = (weights.unsqueeze(-1) * self.graph_value(nodes)).sum(1)
        if self.disable_cross_attention:
            fused_graph = graph
        else:
            # MHA requires at least one unmasked key in each row.
            safe_valid = valid.clone()
            no_neighbors = ~safe_valid.any(-1)
            safe_valid[no_neighbors, 0] = True
            attended, _ = self.cross(case.unsqueeze(1), nodes, nodes, key_padding_mask=~safe_valid)
            fused_graph = graph + attended.squeeze(1)
        fused_graph = fused_graph * valid.any(-1, keepdim=True).float()
        gate = torch.sigmoid(self.gate(torch.cat([case, fused_graph], -1)))
        fused = self.norm(case + self.dropout(gate * fused_graph))
        return torch.where(valid.any(-1, keepdim=True), fused, case)


class TraceLegal(nn.Module):
    def __init__(self, model_name, n_labels, heads, dropout, disable_graph, disable_cross_attention):
        super().__init__()
        self.encoder = HierarchicalEncoder(model_name, dropout)
        h = self.encoder.dim
        self.graph = GraphCrossAttention(h, heads, dropout, disable_graph, disable_cross_attention)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(h, n_labels))

    def forward(self, ids, mask, vectors, valid, sim, age):
        case = self.encoder(ids, mask)
        fused = self.graph(case, vectors, valid, sim, age)
        return self.classifier(fused)


def gather_graph(batch_idx, neighbor_map, precedent_cache, device):
    indexes = neighbor_map['idx'][batch_idx]
    valid = indexes >= 0
    vectors = precedent_cache[indexes.clamp(min=0)].to(device)
    vectors = vectors * valid.unsqueeze(-1).to(device)
    return (vectors, valid.to(device), neighbor_map['sim'][batch_idx].to(device), neighbor_map['age'][batch_idx].to(device))


@torch.no_grad()
def predict(model, loader, neighbor_map, precedent_cache, device):
    model.eval()
    probs, ys = [], []
    for batch in loader:
        graph = gather_graph(batch['index'], neighbor_map, precedent_cache, device)
        logits = model(batch['ids'].to(device), batch['mask'].to(device), *graph)
        probs.append(logits.sigmoid().cpu().numpy())
        ys.append(batch['y'].numpy())
    return np.concatenate(probs), np.concatenate(ys)


def calibrate_thresholds(probs, truth, min_support=20):
    # Validation only. Rare classes keep 0.5 instead of fitting a tiny sample.
    grid = np.round(np.arange(0.10, 0.901, 0.05), 2)
    thresholds = np.full(truth.shape[1], 0.5)
    for j in range(truth.shape[1]):
        positives = truth[:, j].sum()
        if positives < min_support or len(truth) - positives < min_support:
            continue
        best = f1_score(truth[:, j], probs[:, j] >= 0.5, zero_division=0)
        for t in grid:
            score = f1_score(truth[:, j], probs[:, j] >= t, zero_division=0)
            if score > best + 1e-12 or (abs(score - best) < 1e-12 and abs(t - 0.5) < abs(thresholds[j] - 0.5)):
                best, thresholds[j] = score, float(t)
    return thresholds


def metrics(probs, truth, thresholds, labels):
    pred = (probs >= thresholds[None, :]).astype(int)
    p, r, f, support = precision_recall_fscore_support(truth, pred, average=None, zero_division=0)
    report = {'exact_match_accuracy': float(accuracy_score(truth, pred)),
              'hamming_loss': float(hamming_loss(truth, pred)),
              'per_label': {label: {'precision': float(p[i]), 'recall': float(r[i]), 'f1': float(f[i]), 'support': int(support[i])} for i, label in enumerate(labels)}}
    for average in ('micro', 'macro', 'weighted'):
        precision, recall, f1, _ = precision_recall_fscore_support(truth, pred, average=average, zero_division=0)
        report.update({f'{average}_precision': float(precision), f'{average}_recall': float(recall), f'{average}_f1': float(f1)})
    return report


def probability_loss(probs, truth):
    # Unweighted BCE for a comparable validation diagnostic in every experiment.
    p = np.clip(probs.astype(np.float64), 1e-7, 1 - 1e-7)
    return float(-(truth * np.log(p) + (1 - truth) * np.log1p(-p)).mean())


def optimizer_for(model, lr, head_lr, weight_decay):
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        rate = lr if name.startswith('encoder.bert.') else head_lr
        decay = weight_decay if parameter.ndim > 1 else 0.0
        groups.setdefault((rate, decay), []).append(parameter)
    return torch.optim.AdamW([{'params': values, 'lr': rate, 'weight_decay': decay}
                             for (rate, decay), values in groups.items()])


def selection_improved(score, best, metric):
    if not math.isfinite(score):
        raise ValueError('Nonfinite checkpoint score')
    return best is None or (score < best if metric == 'val_loss' else score > best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--encoder_model', default='nlpaueb/legal-bert-base-uncased')
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch_size', type=int, default=2)
    ap.add_argument('--max_chunks', type=int, default=4)
    ap.add_argument('--max_length', type=int, default=256)
    ap.add_argument('--neighbors', type=int, default=5)
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--weight_decay', type=float, default=0.01)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--disable_graph', action='store_true')
    ap.add_argument('--disable_cross_attention', action='store_true')
    ap.add_argument('--head_lr', type=float)
    ap.add_argument('--encoder_dropout', type=float)
    ap.add_argument('--freeze_lower_layers', type=int, default=0)
    ap.add_argument('--selection_metric', choices=['micro_f1', 'macro_f1', 'val_loss'], default='macro_f1')
    ap.add_argument('--calibration_min_support', type=int, default=20)
    ap.add_argument('--report_train_metrics', action='store_true')
    ap.add_argument('--dry_run', action='store_true')
    ap.add_argument('--device', choices=['cpu', 'cuda'])
    args = ap.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.neighbors < 1 or args.heads < 1:
        raise ValueError('Epochs, batch size, neighbors, and heads must be positive')
    values = [args.lr, args.weight_decay, args.dropout]
    if args.head_lr is not None: values.append(args.head_lr)
    if args.encoder_dropout is not None: values.append(args.encoder_dropout)
    if not all(math.isfinite(v) for v in values) or args.lr <= 0 or args.weight_decay < 0 or not 0 <= args.dropout < 1:
        raise ValueError('Invalid learning rate or regularization setting')
    if args.head_lr is not None and args.head_lr <= 0:
        raise ValueError('head_lr must be positive')
    if args.encoder_dropout is not None and not 0 <= args.encoder_dropout < 1:
        raise ValueError('encoder_dropout must be in [0, 1)')
    if not 0 <= args.freeze_lower_layers <= 12 or args.calibration_min_support < 1:
        raise ValueError('Invalid frozen layer count or calibration support')
    if args.max_length < 8 or args.max_length > 512 or args.max_chunks < 1:
        raise ValueError('Require 8 <= max_length <= 512 and max_chunks >= 1')
    seed_everything(args.seed)
    output = Path(args.output_dir)

    splits = {name: read_jsonl(Path(args.data_dir) / f'{name}.jsonl') for name in ('train', 'val', 'test')}
    validate_splits(splits)
    labels = sorted({label for row in splits['train'] for label in row['labels']})
    if not labels:
        raise ValueError('No labels found')
    unknown = {label for name in ('val', 'test') for row in splits[name] for label in row['labels']} - set(labels)
    if unknown:
        raise ValueError(f'Labels absent from training: {sorted(unknown)}')
    audit = {'split_sizes': {name: len(rows) for name, rows in splits.items()},
             'training_support': {label: sum(label in row['labels'] for row in splits['train']) for label in labels},
             'labels': labels, 'audit': 'IDs and normalized text are distinct; dates parsed'}
    if args.dry_run:
        print(json.dumps(audit, indent=2))
        return
    output.mkdir(parents=True, exist_ok=False)
    (output / 'data_audit.json').write_text(json.dumps(audit, indent=2))
    label_to_idx = {x: i for i, x in enumerate(labels)}
    (output / 'label_vocab.json').write_text(json.dumps(labels, indent=2))
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_model)
    datasets = {name: CaseDataset(rows, tokenizer, label_to_idx, args.max_chunks, args.max_length) for name, rows in splits.items()}
    loaders = {name: DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0) for name, ds in datasets.items()}
    train_loader = DataLoader(datasets['train'], batch_size=args.batch_size, shuffle=True, num_workers=0)
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = TraceLegal(args.encoder_model, len(labels), args.heads, args.dropout, args.disable_graph, args.disable_cross_attention).to(device)

    bert = model.encoder.bert
    if args.encoder_dropout is not None:
        for module in bert.modules():
            if isinstance(module, nn.Dropout):
                module.p = args.encoder_dropout
    if args.freeze_lower_layers:
        if not hasattr(bert, 'encoder') or args.freeze_lower_layers > len(bert.encoder.layer):
            raise ValueError('Freezing requires a compatible BERT encoder and valid layer count')
        for module in [bert.embeddings, *bert.encoder.layer[:args.freeze_lower_layers]]:
            for parameter in module.parameters():
                parameter.requires_grad = False

    # Cache vectors once from the pretrained encoder to avoid future leakage and
    # changing neighbor lists during training. This is a fixed-retrieval experiment.
    with torch.no_grad():
        precedent_cache = cache_precedent_embeddings(model.encoder, loaders['train'], device, len(splits['train']))
        retrieval_vectors = {'train': precedent_cache}
        for name in ('val', 'test'):
            retrieval_vectors[name] = cache_query_embeddings(model.encoder, loaders[name], device, len(splits[name]))
    maps = {name: make_neighbors(splits[name], splits['train'], retrieval_vectors[name], precedent_cache, args.neighbors, name == 'train') for name in splits}
    print('Available train precedents per case:', {name: float(m['idx'].ge(0).sum(1).float().mean()) for name, m in maps.items()})

    optimizer = optimizer_for(model, args.lr, args.head_lr or args.lr, args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(0, int(0.1 * total_steps)), num_training_steps=total_steps)
    best_score = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}'):
            graph = gather_graph(batch['index'], maps['train'], precedent_cache, device)
            logits = model(batch['ids'].to(device), batch['mask'].to(device), *graph)
            loss = F.binary_cross_entropy_with_logits(logits, batch['y'].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
        val_probs, val_truth = predict(model, loaders['val'], maps['val'], precedent_cache, device)
        thresholds = np.full(len(labels), 0.5)
        val_metrics = metrics(val_probs, val_truth, thresholds, labels)
        val_loss = probability_loss(val_probs, val_truth)
        history.append({'epoch': epoch, 'loss': float(np.mean(losses)), 'val_micro_f1': val_metrics['micro_f1'], 'val_macro_f1': val_metrics['macro_f1']})
        history[-1].update(val_loss=val_loss, validation_fixed_threshold=val_metrics)
        if args.report_train_metrics:
            train_probs, train_truth = predict(model, loaders['train'], maps['train'], precedent_cache, device)
            history[-1]['training_fixed_threshold'] = metrics(train_probs, train_truth, thresholds, labels)
            history[-1]['train_eval_loss'] = probability_loss(train_probs, train_truth)
        score = val_loss if args.selection_metric == 'val_loss' else val_metrics[args.selection_metric]
        print(json.dumps(history[-1], indent=2))
        (output / 'history.json').write_text(json.dumps(history, indent=2))
        if selection_improved(score, best_score, args.selection_metric):
            best_score = score
            torch.save({'model': model.state_dict(), 'thresholds': thresholds.tolist(), 'epoch': epoch, 'labels': labels, 'args': vars(args)}, output / 'best.pt')
    (output / 'history.json').write_text(json.dumps(history, indent=2))
    checkpoint = torch.load(output / 'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    # Fit thresholds once, after checkpoint selection, using validation only.
    selected_val_probs, selected_val_truth = predict(model, loaders['val'], maps['val'], precedent_cache, device)
    thresholds = calibrate_thresholds(selected_val_probs, selected_val_truth, args.calibration_min_support)
    checkpoint['thresholds'] = thresholds.tolist()
    torch.save(checkpoint, output / 'best.pt')
    np.savez_compressed(output / 'validation_predictions.npz', probabilities=selected_val_probs, truth=selected_val_truth)
    test_probs, test_truth = predict(model, loaders['test'], maps['test'], precedent_cache, device)
    report = {'selected_epoch': checkpoint['epoch'], 'selection_metric': args.selection_metric, 'selection_score': best_score, 'validation': metrics(selected_val_probs, selected_val_truth, thresholds, labels), 'validation_fixed_threshold': metrics(selected_val_probs, selected_val_truth, np.full(len(labels), 0.5), labels), 'thresholds': dict(zip(labels, thresholds.tolist())), 'test': metrics(test_probs, test_truth, thresholds, labels), 'config': vars(args)}
    np.savez_compressed(output / 'test_predictions.npz', probabilities=test_probs, truth=test_truth)
    (output / 'metrics.json').write_text(json.dumps(report, indent=2))
    print('TEST', json.dumps(report['test'], indent=2))


if __name__ == '__main__':
    main()
