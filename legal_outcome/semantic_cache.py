"""Cache frozen MiniLM span embeddings; do not cache trainable LegalBERT states."""
import hashlib
import json
from pathlib import Path
import torch
from tqdm.auto import tqdm
from .training import atomic_torch_save, capture_rng, restore_rng


def model_signature(encoder, model_name, revision=None):
    path = Path(model_name)
    if path.is_dir():
        digest = hashlib.sha256()
        for file in sorted(path.rglob('*')):
            if file.is_file():
                digest.update(str(file.relative_to(path)).encode())
                with file.open('rb') as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        digest.update(block)
        return {'path': str(path.resolve()), 'sha256': digest.hexdigest()}
    if revision and len(revision) == 40 and all(c in '0123456789abcdef' for c in revision.lower()):
        return {'model': model_name, 'revision': revision}
    first = encoder._first_module()
    revision = getattr(first.auto_model.config, '_commit_hash', None)
    if not revision:
        raise ValueError('Cannot identify the MiniLM revision for a reusable cache')
    return {'model': model_name, 'revision': revision}


def _cache_semantics(items, encoder, cache_dir, signature, batch_size=32):
    spans = [span for item in items for span in item['spans']]
    fingerprint = hashlib.sha256(json.dumps({'signature': signature, 'spans': spans}, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (fingerprint + '.pt')
    if path.exists():
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload.get('fingerprint') != fingerprint:
            raise ValueError('Semantic cache identity differs')
        embeddings = payload['embeddings']
        print(f'Reusing frozen MiniLM cache: {len(spans)} spans', flush=True)
    else:
        values = []
        for start in tqdm(range(0, len(spans), batch_size), desc='Caching frozen MiniLM spans'):
            with torch.no_grad():
                encoded = encoder.encode(spans[start:start + batch_size], convert_to_tensor=True,
                                         normalize_embeddings=True, show_progress_bar=False)
            values.append(torch.as_tensor(encoded).detach().cpu().float())
        embeddings = torch.cat(values)
        atomic_torch_save({'fingerprint': fingerprint, 'embeddings': embeddings}, path)
    if embeddings.ndim != 2 or len(embeddings) != len(spans) or not torch.isfinite(embeddings).all():
        raise ValueError('Invalid frozen semantic cache')
    offset = 0
    for item in items:
        count = len(item['spans'])
        item['semantic_embeddings'] = embeddings[offset:offset + count]
        offset += count
    return path


def cache_semantics(items, encoder, cache_dir, signature, batch_size=32):
    # Cache hits and misses must not change subsequent initialization or dropout.
    state = capture_rng()
    try:
        return _cache_semantics(items, encoder, cache_dir, signature, batch_size)
    finally:
        restore_rng(state)
