"""Frozen ECtHR-A document retrieval with validation-only model selection."""
import argparse
import hashlib
import json
import math
from importlib.metadata import version
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

LABELS = ['2', '3', '5', '6', '8', '9', '10', '11', '14', 'P1-1']
DATASET = 'coastalcph/lex_glue'
FORMAT_VERSION = 1


def atomic_save(value, path, tensor=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    try:
        with temporary.open('wb') as stream:
            if tensor:
                torch.save(value, stream)
            else:
                stream.write((json.dumps(value, indent=2, allow_nan=False) + '\n').encode())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def prepare_records(records):
    texts, labels = [], []
    for i, row in enumerate(records):
        text = row['text']
        if isinstance(text, list):
            if any(not isinstance(paragraph, str) for paragraph in text):
                raise ValueError(f'Row {i}: text paragraphs must be strings')
            text = ' '.join(text)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f'Row {i}: empty or invalid text')
        ids = row['labels']
        if not isinstance(ids, list) or any(type(label) is not int or not 0 <= label < 10 for label in ids) or len(ids) != len(set(ids)):
            raise ValueError(f'Row {i}: labels must be distinct integer indices from 0 to 9')
        target = [0.0] * len(LABELS)
        for label in ids:
            target[label] = 1.0
        texts.append(text)
        labels.append(target)
    if not texts:
        raise ValueError('Empty split')
    return texts, torch.tensor(labels, dtype=torch.float32)


def split_fingerprint(texts, labels):
    digest = hashlib.sha256()
    for text, target in zip(texts, labels.tolist()):
        digest.update(json.dumps([text, target], ensure_ascii=False).encode())
        digest.update(b'\n')
    return digest.hexdigest()


def load_inputs(local_data=None, revision='main', smoke=False):
    if local_data:
        records = {}
        for split in ['train', 'validation', 'test']:
            path = Path(local_data) / f'{split}.jsonl'
            records[split] = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        provenance = {'name': str(Path(local_data).resolve()), 'mode': 'local_fixture_or_custom_data', 'revision': None}
    else:
        from datasets import load_dataset
        from huggingface_hub import HfApi
        resolved = HfApi().dataset_info(DATASET, revision=revision).sha
        dataset = load_dataset(DATASET, 'ecthr_a', revision=resolved)
        names = list(dataset['train'].features['labels'].feature.names)
        if names != LABELS:
            raise ValueError(f'Unexpected source label mapping: {names}')
        records = {split: dataset[split] for split in ['train', 'validation', 'test']}
        if {s: len(v) for s, v in records.items()} != {'train': 9000, 'validation': 1000, 'test': 1000}:
            raise ValueError('Official split sizes differ from the expected 9000/1000/1000; inspect before running')
        provenance = {'name': DATASET, 'config': 'ecthr_a', 'revision': resolved, 'mode': 'official'}
    prepared = {}
    for split, values in records.items():
        if smoke:
            limit = 20 if split == 'train' else 5
            values = values.select(range(min(limit, len(values)))) if hasattr(values, 'select') else list(values)[:limit]
        prepared[split] = prepare_records(values)
    provenance['smoke'] = smoke
    hashes = {s: split_fingerprint(*value) for s, value in prepared.items()}
    # Report text overlaps. Keep official splits intact; do not silently delete cases.
    text_keys = {s: {hashlib.sha256(' '.join(t.split()).casefold().encode()).hexdigest() for t in value[0]}
                 for s, value in prepared.items()}
    provenance['normalized_text_overlap'] = {f'{a}/{b}': len(text_keys[a] & text_keys[b])
                                             for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]}
    return prepared, provenance, hashes


def last_token_pool(hidden, mask):
    if hidden.ndim != 3 or mask.shape != hidden.shape[:2] or not mask.bool().any(-1).all():
        raise ValueError('Pooling requires visible tokens in every row')
    indices = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    positions = indices.masked_fill(~mask.bool(), -1).max(-1).values
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), positions]


def encode_split(texts, targets, split, model, tokenizer, output, batch_size, max_length, shard_size, device):
    directory = output / 'cache' / split
    directory.mkdir(parents=True, exist_ok=True)
    pieces, reused, encoded = [], 0, 0
    for start in tqdm(range(0, len(texts), shard_size), desc=f'{split}: embedding shards'):
        end = min(len(texts), start + shard_size)
        path = directory / f'{start:06d}.pt'
        if path.exists():
            payload = torch.load(path, map_location='cpu', weights_only=True)
            if payload['start'] != start or payload['end'] != end or not torch.equal(payload['labels'], targets[start:end]):
                raise ValueError(f'Invalid shard metadata: {path}')
            vectors = payload['embeddings']
            reused += end - start
        else:
            parts = []
            for offset in tqdm(range(start, end, batch_size), desc=f'{split} documents {start + 1}-{end}', leave=False):
                batch_texts = texts[offset:min(offset + batch_size, end)]
                tokens = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
                tokens = {k: v.to(device) for k, v in tokens.items() if k in ('input_ids', 'attention_mask')}
                with torch.inference_mode():
                    result = model(**tokens, use_cache=False)
                    embedding = F.normalize(last_token_pool(result.last_hidden_state, tokens['attention_mask']).float(), dim=-1)
                if not torch.isfinite(embedding).all() or (embedding.norm(dim=-1) < .99).any():
                    raise RuntimeError('Invalid embeddings; inspect model precision before continuing')
                parts.append(embedding.cpu())
            vectors = torch.cat(parts)
            atomic_save({'start': start, 'end': end, 'embeddings': vectors, 'labels': targets[start:end]}, path, tensor=True)
            encoded += end - start
            tqdm.write(f'Saved {split} documents {start + 1}-{end}: {path}')
        if vectors.ndim != 2 or vectors.shape[0] != end - start or not torch.isfinite(vectors).all():
            raise ValueError(f'Invalid embedding shard: {path}')
        pieces.append(vectors)
    return torch.cat(pieces), {'new_documents': encoded, 'reused_documents': reused}


def top_neighbors(index, queries, max_k, batch_size=64):
    if index.ndim != 2 or queries.ndim != 2 or index.shape[1] != queries.shape[1] or not 1 <= max_k <= len(index):
        raise ValueError('Invalid retrieval dimensions or neighbor count')
    indices, similarities = [], []
    for start in range(0, len(queries), batch_size):
        values, positions = torch.topk(queries[start:start + batch_size].float() @ index.float().T, max_k, dim=-1)
        indices.append(positions)
        similarities.append(values)
    return torch.cat(indices), torch.cat(similarities)


def vote(indices, similarities, train_labels, k):
    if not 1 <= k <= indices.shape[1]:
        raise ValueError('Invalid k')
    weights = similarities[:, :k].clamp_min(0).unsqueeze(-1)
    targets = train_labels[indices[:, :k]].float()
    return ((targets * weights).sum(1) / weights.sum(1).clamp_min(1e-9)).numpy()


def divide(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.divide(a, b, out=np.zeros(np.broadcast_shapes(a.shape, b.shape)), where=b != 0)


def metrics(truth, probabilities, threshold):
    truth, probabilities = np.asarray(truth), np.asarray(probabilities)
    if truth.shape != probabilities.shape or truth.ndim != 2 or not truth.size or not np.isin(truth, [0, 1]).all():
        raise ValueError('Invalid metric targets or probability shapes')
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1 + 1e-6)).any():
        raise ValueError('Invalid probabilities')
    actual, predicted = truth.astype(bool), probabilities >= threshold
    tp = (actual & predicted).sum(0)
    fp = (~actual & predicted).sum(0)
    fn = (actual & ~predicted).sum(0)
    p, r, f = divide(tp, tp + fp), divide(tp, tp + fn), divide(2 * tp, 2 * tp + fp + fn)
    return {'examples': len(truth), 'micro_precision': float(divide(tp.sum(), (tp + fp).sum())),
            'micro_recall': float(divide(tp.sum(), (tp + fn).sum())),
            'micro_f1': float(divide(2 * tp.sum(), (2 * tp + fp + fn).sum())),
            'macro_precision': float(p.mean()), 'macro_recall': float(r.mean()), 'macro_f1': float(f.mean()),
            'exact_match_accuracy': float((actual == predicted).all(-1).mean()),
            'hamming_loss': float((actual != predicted).mean()),
            'per_label': {label: {'precision': float(p[i]), 'recall': float(r[i]), 'f1': float(f[i]), 'support': int(actual[:, i].sum())}
                          for i, label in enumerate(LABELS[:truth.shape[1]])}}


def select_parameters(index, train_labels, validation, validation_labels, k_values, thresholds):
    # This function has no test vectors or test targets. Selection uses validation only.
    indices, values = top_neighbors(index, validation, max(k_values))
    rows, best, best_key = [], None, None
    for k in sorted(k_values):
        probabilities = vote(indices, values, train_labels, k)
        for threshold in sorted(thresholds):
            scores = metrics(validation_labels.numpy(), probabilities, threshold)
            row = {'k': k, 'threshold': threshold, 'micro_f1': scores['micro_f1'], 'macro_f1': scores['macro_f1']}
            rows.append(row)
            key = (scores['micro_f1'], scores['macro_f1'], -k, -round(abs(threshold - .5), 12))
            if best_key is None or key > best_key:
                best, best_key = row, key
    return best, rows


def resolve_model(model_name, revision):
    if Path(model_name).is_dir():
        digest = hashlib.sha256()
        # Hash local weights and tokenizer files to prevent stale local cache reuse.
        for path in sorted(Path(model_name).rglob('*')):
            if path.is_file():
                digest.update(str(path.relative_to(model_name)).encode())
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        digest.update(block)
        return str(Path(model_name).resolve()), None, digest.hexdigest()
    from huggingface_hub import HfApi
    resolved = HfApi().model_info(model_name, revision=revision).sha
    return model_name, resolved, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', default='Qwen/Qwen3-Embedding-0.6B')
    parser.add_argument('--revision', default='main')
    parser.add_argument('--dataset-revision', default='main')
    parser.add_argument('--local-data', help='Optional JSONL fixtures/custom data, not an official benchmark run')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--max-length', type=int, default=8192)
    parser.add_argument('--shard-size', type=int, default=64)
    parser.add_argument('--k-values', type=int, nargs='+', default=[5, 10, 15, 20])
    parser.add_argument('--thresholds', type=float, nargs='+', default=[.2, .3, .4, .5, .6])
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1 or args.shard_size < 1 or not 1 <= args.max_length <= 8192:
        raise ValueError('Invalid batch, shard, or context size')
    if any(k < 1 for k in args.k_values) or any(not math.isfinite(t) or not 0 < t < 1 for t in args.thresholds):
        raise ValueError('Invalid k or threshold grid')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Install a CUDA PyTorch build or use --device cpu')
    started = time.perf_counter()
    output = Path(args.output)
    if output.exists() and not (output / 'run.json').exists():
        raise ValueError('Output exists without a run manifest. Use a new output directory')
    previous = read_json(output / 'run.json') if (output / 'run.json').exists() else None
    # A repeated invocation pins the same remote revisions used by its first invocation.
    dataset_revision = previous['dataset'].get('revision') if previous and not args.local_data else args.dataset_revision
    model_revision = previous['model'].get('revision') if previous and not Path(args.model).is_dir() else args.revision
    print('Loading official ECtHR-A inputs (or explicitly requested local fixtures)...', flush=True)
    inputs, provenance, fingerprints = load_inputs(args.local_data, dataset_revision, args.smoke)
    if max(args.k_values) > len(inputs['train'][0]):
        raise ValueError('A neighbor count exceeds the training index size')
    model_name, resolved_revision, local_hash = resolve_model(args.model, model_revision)
    dtype = torch.float32 if args.device == 'cpu' else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    contract = {'format_version': FORMAT_VERSION, 'dataset': provenance, 'fingerprints': fingerprints,
                'split_sizes': {s: len(t[0]) for s, t in inputs.items()}, 'labels': LABELS,
                'model': {'name': model_name, 'revision': resolved_revision, 'local_files_sha256': local_hash},
                'device': args.device, 'dtype': str(dtype), 'max_length': args.max_length,
                'batch_size': args.batch_size, 'shard_size': args.shard_size,
                'k_values': sorted(set(args.k_values)), 'thresholds': sorted(set(args.thresholds)),
                'pooling': 'last non-padding token; L2 normalized', 'prompt': 'none',
                'seed': 42, 'software': {'torch': str(torch.__version__), 'transformers': version('transformers'),
                                          'numpy': np.__version__}}
    if previous and previous != contract:
        raise ValueError('Run settings or data changed. Use a new output directory to avoid reusing incompatible caches')
    if not previous:
        atomic_save(contract, output / 'run.json')
    if (output / 'results.json').exists():
        print('Run already completed. Existing results follow; no new test evaluation was performed.')
        print(json.dumps(read_json(output / 'results.json'), indent=2))
        return
    from transformers import AutoModel, AutoTokenizer, set_seed
    set_seed(42)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    print(f'Loading frozen model: {model_name}; dtype={dtype}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=resolved_revision, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError('Tokenizer needs a padding or EOS token')
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(model_name, revision=resolved_revision, torch_dtype=dtype,
                                      trust_remote_code=False, attn_implementation='sdpa').to(args.device).eval()
    model.requires_grad_(False)
    encoding = {}
    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    train_vectors, encoding['train'] = encode_split(*inputs['train'], 'train', model, tokenizer, output,
                                                    args.batch_size, args.max_length, args.shard_size, args.device)
    validation_vectors, encoding['validation'] = encode_split(*inputs['validation'], 'validation', model, tokenizer, output,
                                                              args.batch_size, args.max_length, args.shard_size, args.device)
    selection_file = output / 'selection.json'
    if selection_file.exists():
        selection = read_json(selection_file)
        print('Reusing saved validation selection:', selection['best'], flush=True)
    else:
        print('Selecting k and threshold on validation only...', flush=True)
        best, rows = select_parameters(train_vectors, inputs['train'][1], validation_vectors, inputs['validation'][1],
                                       contract['k_values'], contract['thresholds'])
        selection = {'criterion': 'validation_micro_f1', 'tie_rule': 'macro F1, smaller k, threshold closest to 0.5',
                     'best': best, 'grid': rows, 'source_split': 'validation'}
        atomic_save(selection, selection_file)
    print('Validation selection fixed. Starting test embedding and final evaluation.', flush=True)
    test_vectors, encoding['test'] = encode_split(*inputs['test'], 'test', model, tokenizer, output,
                                                 args.batch_size, args.max_length, args.shard_size, args.device)
    peak = torch.cuda.max_memory_allocated() / 1024**3 if args.device == 'cuda' else None
    del model
    if args.device == 'cuda':
        torch.cuda.empty_cache()
    best = selection['best']
    indices, values = top_neighbors(train_vectors, test_vectors, best['k'])
    probabilities = vote(indices, values, inputs['train'][1], best['k'])
    scores = metrics(inputs['test'][1].numpy(), probabilities, best['threshold'])
    atomic_save({'probabilities': torch.from_numpy(probabilities), 'targets': inputs['test'][1], 'labels': LABELS},
                output / 'test-predictions.pt', tensor=True)
    report = {'status': 'completed', 'smoke': args.smoke, 'dataset_mode': provenance['mode'],
              'method': 'frozen training-document k-NN with cosine-weighted multi-label voting',
              'selection': best, 'selection_split': 'validation', 'test': scores, 'run': contract,
              'encoding_this_invocation': encoding, 'elapsed_seconds_this_invocation': time.perf_counter() - started,
              'peak_gpu_allocated_gib_this_invocation': peak,
              'timing_note': 'Current invocation only; reused shards omit earlier encoding time',
              'scope_note': 'A practical released-code-style baseline, not a complete reproduction of all paper experiments'}
    atomic_save(report, output / 'results.json')
    print(json.dumps(report, indent=2))
    print(f'Completed. Results saved to {output / "results.json"}', flush=True)


if __name__ == '__main__':
    main()
