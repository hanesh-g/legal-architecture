"""Prepare, train, and test the proposed architecture on official ECtHR-A."""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import torch
from tqdm.auto import tqdm
from .config import ModelConfig
from .data import DocumentProcessor
from .reporting import Tee, atomic_text, csv_file, json_file, table, history_table

LABELS = ['2', '3', '5', '6', '8', '9', '10', '11', '14', 'P1-1']
SOURCE = 'coastalcph/lex_glue'


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def local_model_hash(path):
    digest = hashlib.sha256()
    for file in sorted(Path(path).rglob('*')):
        if file.is_file():
            digest.update(str(file.relative_to(path)).encode())
            digest.update(digest_file(file).encode())
    return digest.hexdigest()


def model_reference(name, revision):
    if Path(name).is_dir():
        return {'name': str(Path(name).resolve()), 'revision': None, 'local_sha256': local_model_hash(name)}
    from huggingface_hub import HfApi
    resolved = HfApi().model_info(name, revision=revision).sha
    return {'name': name, 'revision': resolved, 'local_sha256': None}


def convert_rows(rows, split):
    converted = []
    for index, row in enumerate(rows):
        paragraphs = row['text']
        if isinstance(paragraphs, list):
            if any(not isinstance(text, str) for text in paragraphs):
                raise ValueError(f'{split} row {index}: non-string paragraph')
            text = '\n\n'.join(paragraphs)
        else:
            text = paragraphs
        ids = row['labels']
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f'{split} row {index}: empty or invalid text')
        if not isinstance(ids, list) or any(type(i) is not int or not 0 <= i < len(LABELS) for i in ids) or len(ids) != len(set(ids)):
            raise ValueError(f'{split} row {index}: invalid class indices')
        converted.append({'case_id': f'ecthr-a:{split}:{index}', 'source_row': index,
                          'text': text, 'labels': [LABELS[i] for i in ids]})
    return converted


def verify_prepared(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'dataset-manifest.json').read_text(encoding='utf-8'))
    if manifest.get('labels') != LABELS:
        raise ValueError('Wrong article label mapping in prepared data')
    for split in ['train', 'val', 'test']:
        if digest_file(directory / f'{split}.jsonl') != manifest['files'][split]['sha256']:
            raise ValueError(f'Prepared {split} data changed; do not mix data revisions')
    if not manifest['ready_for_training']:
        raise ValueError('Dataset audit found train/validation text overlap; inspect dataset-manifest.json before training')
    return manifest


def prepare(args):
    directory = Path(args.data)
    if directory.exists():
        if not (directory / 'dataset-manifest.json').exists():
            raise ValueError('Data directory exists without a complete manifest. Preserve it and use a new --data directory')
        manifest = verify_prepared(directory)
        print('Prepared files already exist and their hashes match. No files were overwritten.')
        print(dataset_table(manifest))
        return
    if args.local_data:
        rows = {split: [json.loads(line) for line in (Path(args.local_data) / f'{split}.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
                for split in ['train', 'validation', 'test']}
        source = {'name': str(Path(args.local_data).resolve()), 'mode': 'local_fixture_or_custom_data', 'revision': None}
    else:
        from datasets import load_dataset
        from huggingface_hub import HfApi
        revision = HfApi().dataset_info(SOURCE, revision=args.dataset_revision).sha
        print('Downloading/loading official ECtHR-A. The first invocation needs Internet access.', flush=True)
        dataset = load_dataset(SOURCE, 'ecthr_a', revision=revision)
        if list(dataset['train'].features['labels'].feature.names) != LABELS:
            raise ValueError('Source label mapping changed')
        if {s: len(dataset[s]) for s in ['train', 'validation', 'test']} != {'train': 9000, 'validation': 1000, 'test': 1000}:
            raise ValueError('Official split sizes changed; inspect before training')
        rows = {s: dataset[s] for s in ['train', 'validation', 'test']}
        source = {'name': SOURCE, 'config': 'ecthr_a', 'mode': 'official', 'revision': revision}
    references = {'legalbert': model_reference(args.legalbert_model, args.model_revision),
                  'minilm': model_reference(args.minilm_model, args.model_revision)}
    from transformers import AutoTokenizer
    reference = references['legalbert']
    tokenizer = AutoTokenizer.from_pretrained(reference['name'], revision=reference['revision'])
    config = ModelConfig(tuple(LABELS), max_tokens=args.max_tokens, task_type='multi_label',
                         chunk_tokens=args.chunk_tokens, encoder_chunk_batch_size=args.encoder_chunk_batch_size,
                         gradient_checkpointing=args.gradient_checkpointing)
    processor = DocumentProcessor(tokenizer, config)
    prepared = {('val' if split == 'validation' else split): convert_rows(records, split) for split, records in rows.items()}
    directory.mkdir(parents=True, exist_ok=False)
    hashes, summary, coverage_rows, text_sets = {}, {}, [], {}
    for split, records in prepared.items():
        text_sets[split] = {hashlib.sha256(' '.join(r['text'].split()).casefold().encode()).hexdigest() for r in records}
        encoded = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records)
        atomic_text(directory / f'{split}.jsonl', encoded)
        hashes[split] = {'sha256': digest_file(directory / f'{split}.jsonl'), 'examples': len(records)}
        token_counts, kept_counts, retained_spans = [], [], []
        for row in tqdm(records, desc=f'{split}: token coverage audit'):
            full = len(tokenizer(row['text'], add_special_tokens=False, truncation=False, verbose=False)['input_ids'])
            item = processor(row['text'])
            kept = int((item['span_ids'] >= 0).sum())
            count = len(item['spans'])
            token_counts.append(full); kept_counts.append(kept); retained_spans.append(count)
            coverage_rows.append([split, row['case_id'], full, kept, count, full > kept])
        summary[split] = {'examples': len(records), 'empty_label_cases': sum(not r['labels'] for r in records),
                          'article_support': {label: sum(label in r['labels'] for r in records) for label in LABELS},
                          'truncated_cases': sum(a > b for a, b in zip(token_counts, kept_counts)),
                          'retained_content_token_fraction': sum(kept_counts) / max(1, sum(token_counts)),
                          'mean_retained_spans': sum(retained_spans) / max(1, len(records))}
    overlap = {f'{a}/{b}': len(text_sets[a] & text_sets[b]) for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]}
    manifest = {'format_version': 1, 'source': source, 'labels': LABELS, 'models': references,
                'files': hashes, 'audit': summary, 'normalized_text_overlap': overlap,
                'max_packed_tokens': args.max_tokens, 'max_content_tokens': config.max_content_tokens,
                'chunk_tokens': config.chunk_tokens, 'encoder_chunk_batch_size': config.encoder_chunk_batch_size,
                'gradient_checkpointing': config.gradient_checkpointing, 'variant': config.input_variant, 'ready_for_training': overlap['train/val'] == 0,
                'case_id_note': 'Generated source-row references, not verified court case identifiers',
                'test_policy': 'Test labels are not used for training or checkpoint selection'}
    csv_file(directory / 'token-coverage.csv', ['split', 'row_reference', 'full_content_tokens', 'retained_content_tokens', 'retained_spans', 'truncated'], coverage_rows)
    json_file(directory / 'dataset-manifest.json', manifest)
    print(dataset_table(manifest))
    print(f'Input variant: {config.input_variant}; at most {config.max_content_tokens} content tokens in {config.max_tokens} packed positions.')
    print(f'Data and coverage audit saved to {directory}', flush=True)
    if not manifest['ready_for_training']:
        raise ValueError('Official train/validation text overlap found. Files preserved for review; do not train before inspection')


def dataset_table(manifest):
    return table(['Split', 'Cases', 'Empty labels', 'Truncated', 'Tokens retained', 'Mean spans'],
                 [[split, values['examples'], values['empty_label_cases'], values['truncated_cases'],
                   f"{100 * values['retained_content_token_fraction']:.2f}%", f"{values['mean_retained_spans']:.2f}"]
                  for split, values in manifest['audit'].items()])


@contextmanager
def logged(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stdout, stderr = sys.stdout, sys.stderr
    with path.open('a', encoding='utf-8') as stream:
        sys.stdout, sys.stderr = Tee(stdout, stream), Tee(stderr, stream)
        try:
            yield
        finally:
            sys.stdout, sys.stderr = stdout, stderr


def check_device(device):
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable. Use a CUDA-enabled PyTorch build before GPU training')
        value = torch.ones(1, device='cuda').sum().item()
        print(f'GPU ready: {torch.cuda.get_device_name(0)} | compute check: {value}', flush=True)


def train(args):
    from .cli import main as architecture_cli
    manifest = verify_prepared(args.data)
    check_device(args.device)
    output = Path(args.output)
    if not args.resume and output.exists():
        raise ValueError('Training output exists. Continue with the same command plus --resume; do not delete it')
    if args.resume and not (output / 'training-state.pt').exists():
        raise ValueError('No recovery checkpoint found. Do not use --resume for a new run')
    model, semantic = manifest['models']['legalbert'], manifest['models']['minilm']
    for ref in [model, semantic]:
        if ref['local_sha256'] and local_model_hash(ref['name']) != ref['local_sha256']:
            raise ValueError('A local model changed after preparation')
    command = ['train', '--data', str(Path(args.data) / 'train.jsonl'), '--validation', str(Path(args.data) / 'val.jsonl'),
               '--labels', *LABELS, '--task-type', 'multi_label', '--output', str(output), '--epochs', str(args.epochs),
               '--batch-size', str(args.batch_size), '--learning-rate', str(args.learning_rate), '--dropout', str(args.dropout),
               '--weight-decay', '.01', '--freeze-lower-layers', str(args.freeze_lower_layers), '--selection-metric', 'micro_f1',
               '--threshold', '.5', '--seed', str(args.seed), '--max-tokens', str(manifest['max_packed_tokens']),
               '--legalbert-model', model['name'], '--minilm-model', semantic['name'], '--device', args.device,
               '--display', 'table', '--report-training-metrics', '--semantic-cache', str(args.semantic_cache)]
    if manifest.get('chunk_tokens'):
        command += ['--chunk-tokens', str(manifest['chunk_tokens']), '--encoder-chunk-batch-size', str(manifest['encoder_chunk_batch_size'])]
    if manifest.get('gradient_checkpointing'):
        command += ['--gradient-checkpointing']
    command += ['--gradient-accumulation-steps', str(args.gradient_accumulation_steps),
                '--lr-schedule', args.lr_schedule, '--warmup-ratio', str(args.warmup_ratio)]
    if args.head_learning_rate is not None:
        command += ['--head-learning-rate', str(args.head_learning_rate)]
    if model['revision']:
        command += ['--legalbert-revision', model['revision']]
    if semantic['revision']:
        command += ['--minilm-revision', semantic['revision']]
    if args.resume:
        command += ['--resume', str(output / 'training-state.pt')]
    recipe = {'dataset_manifest': manifest, 'command': command, 'reporting': 'console percentages; JSON and CSV fractions',
              'primary_selection_metric': 'validation_micro_f1', 'requested_total_epochs': args.epochs}
    json_file(Path(args.logs) / (output.name + '-recipe.json'), recipe)
    with logged(Path(args.logs) / (output.name + '-training.log')):
        print('PROPOSED ARCHITECTURE | ECtHR-A | TRAINING ONLY', flush=True)
        print('LegalBERT + Dual Token-LCA + QRASR + span graph + Span-LCA + gated fusion', flush=True)
        print('Test data is not passed to the trainer. No frozen Qwen retrieval is run.', flush=True)
        architecture_cli(command)
    print(f'Training log: {Path(args.logs) / (output.name + "-training.log")}')
    print('Next: review validation history, then run the explicit test command for your final selected configuration.')


def test(args):
    from .cli import main as architecture_cli
    from .reporting import result_table
    manifest = verify_prepared(args.data)
    output = Path(args.output)
    result_dir = Path(args.results)
    metadata = json.loads((output / 'metadata.json').read_text(encoding='utf-8'))
    config = metadata['model_config']
    if config['labels'] != LABELS or config['task_type'] != 'multi_label':
        raise ValueError('Checkpoint is not the expected ten-label ECtHR-A model')
    if config['max_tokens'] != manifest['max_packed_tokens'] or config.get('chunk_tokens') != manifest.get('chunk_tokens'):
        raise ValueError('Input limit differs from the prepared experiment')
    report = metadata['training_report']
    if report.get('data_sha256') != manifest['files']['train']['sha256'] or report.get('validation_sha256') != manifest['files']['val']['sha256']:
        raise ValueError('Checkpoint training/validation fingerprints differ from the prepared experiment')
    completed = report['history'][-1]['epoch']
    if completed < args.required_epochs:
        raise ValueError(f'Only {completed} epochs completed; expected at least {args.required_epochs}. Resume training first')
    identity = {'model_sha256': digest_file(output / 'model.pt'), 'metadata_sha256': digest_file(output / 'metadata.json'),
                'test_sha256': manifest['files']['test']['sha256']}
    if (result_dir / 'evaluation-identity.json').exists():
        if json.loads((result_dir / 'evaluation-identity.json').read_text()) != identity:
            raise ValueError('Result directory belongs to a different model or data; preserve it and use a new --results directory')
        if (result_dir / 'test-results.json').exists():
            result = json.loads((result_dir / 'test-results.json').read_text())
            print('Existing test report reused. No new test evaluation performed.')
            print(result_table(result))
            return
    elif result_dir.exists():
        raise ValueError('Results directory exists without an evaluation identity. Use a new --results directory')
    check_device(args.device)
    json_file(result_dir / 'evaluation-identity.json', identity)
    with logged(Path(args.logs) / (output.name + '-testing.log')):
        print(f'FINAL TEST | selected epoch: {report["selected_epoch"]} | completed epochs: {completed}', flush=True)
        architecture_cli(['evaluate', '--checkpoint', str(output), '--data', str(Path(args.data) / 'test.jsonl'),
                          '--device', args.device, '--display', 'table', '--results-dir', str(result_dir),
                          '--semantic-cache', str(args.semantic_cache), '--batch-size', str(args.batch_size)])
    result = json.loads((result_dir / 'test-results.json').read_text())
    result['benchmark'] = manifest['source']
    result['training_epochs_completed'] = completed
    result['variant'] = manifest['variant']
    from .reporting import save_result_files
    save_result_files(result_dir, result)
    comparison = [
        ['Qwen-0.6B retrieval', 'Published Haneesh3 paper', 8192, .6747, .5902],
        ['Legal-BERT', 'Published Haneesh3 paper', 8192, .7625, .6562],
        ['Proposed architecture', 'Measured local test run', config['max_tokens'], result['micro_f1'], result['macro_f1']],
    ]
    csv_file(result_dir / 'published-reference-comparison.csv',
             ['method', 'source', 'maximum_packed_tokens', 'micro_f1', 'macro_f1'], comparison)
    atomic_text(result_dir / 'comparison-notes.txt',
                'Published reference values come from Haneesh3 Table 3. They are not locally reproduced results.\n'
                f'The proposed run uses {config["max_tokens"]} packed positions; the paper context cap is 8192. Model aggregation and other settings can still differ.\n'
                'A numerical difference alone does not establish a controlled improvement. Label/scoring definitions, context, seeds, and selection must be reported.\n'
                'No 2-3 percentage-point gain is assumed or guaranteed.\n')
    print(f'Final reports saved to: {result_dir}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='phase', required=True)
    p = sub.add_parser('prepare', help='Download official data, map labels, audit splits and truncation')
    p.add_argument('--data', default='data/ecthr_a')
    p.add_argument('--dataset-revision', default='main')
    p.add_argument('--model-revision', default='main')
    p.add_argument('--legalbert-model', default='nlpaueb/legal-bert-base-uncased')
    p.add_argument('--minilm-model', default='sentence-transformers/all-MiniLM-L6-v2')
    p.add_argument('--max-tokens', type=int, default=384)
    p.add_argument('--chunk-tokens', type=int)
    p.add_argument('--encoder-chunk-batch-size', type=int, default=1)
    p.add_argument('--gradient-checkpointing', action='store_true')
    p.add_argument('--local-data', help='JSONL fixtures/custom data only; the result is marked non-official')
    p.set_defaults(run=prepare)
    p = sub.add_parser('train', help='Train the proposed architecture, with epoch checkpoints')
    p.add_argument('--data', default='data/ecthr_a')
    p.add_argument('--output', default='checkpoints/ecthr_a_proposed')
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--learning-rate', type=float, default=2e-5)
    p.add_argument('--head-learning-rate', type=float)
    p.add_argument('--dropout', type=float, default=.1)
    p.add_argument('--freeze-lower-layers', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--gradient-accumulation-steps', type=int, default=1)
    p.add_argument('--lr-schedule', choices=['constant', 'linear'], default='constant')
    p.add_argument('--warmup-ratio', type=float, default=0.0)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--semantic-cache', default='cache/ecthr_a_semantics')
    p.add_argument('--logs', default='logs')
    p.add_argument('--resume', action='store_true')
    p.set_defaults(run=train)
    p = sub.add_parser('test', help='Explicit final evaluation of the selected saved model')
    p.add_argument('--data', default='data/ecthr_a')
    p.add_argument('--output', default='checkpoints/ecthr_a_proposed')
    p.add_argument('--results', default='results/ecthr_a_proposed')
    p.add_argument('--required-epochs', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--semantic-cache', default='cache/ecthr_a_semantics')
    p.add_argument('--logs', default='logs')
    p.set_defaults(run=test)
    args = parser.parse_args(argv)
    args.run(args)


if __name__ == '__main__':
    main()
