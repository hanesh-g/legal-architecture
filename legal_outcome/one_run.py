"""One predeclared long-document ECtHR-A experiment; no hyperparameter search."""
import argparse
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
from types import SimpleNamespace
import torch
from . import ecthr_a
from .config import ModelConfig
from .data import DocumentProcessor, collate_documents
from .model import LegalOutcomeModel
from .reporting import atomic_text, json_file, table

PROFILE = {
    'name': 'ecthr_a_long_document_single_run_v1', 'epochs': 8, 'seed': 42,
    'max_packed_tokens': 8192, 'chunk_tokens': 128, 'chunks': 64, 'max_content_tokens': 8064,
    'global_spans': 8, 'encoder_chunk_batch_size': 4, 'gradient_checkpointing': True,
    'micro_batch_size': 1, 'gradient_accumulation_steps': 8, 'effective_batch_size': 8,
    'encoder_learning_rate': 1e-5, 'head_learning_rate': 1e-4,
    'dropout': .2, 'weight_decay': .01, 'freeze_lower_layers': 4,
    'lr_schedule': 'linear', 'warmup_ratio': .1, 'decision_threshold': .5,
    'selection': 'validation_micro_f1', 'threshold_search': False,
    'primary_reference': {'method': 'GPT-5.2', 'source': 'Haneesh3 Table 3 (published; best ECtHR-A row)',
                          'micro_f1': .7933, 'macro_f1': .7631},
    'encoder_reference': {'method': 'Legal-BERT', 'micro_f1': .7625, 'macro_f1': .6562},
    'secondary_reference': {'method': 'Qwen-0.6B retrieval', 'micro_f1': .6747, 'macro_f1': .5902},
    'primary_improvement_target_percentage_points': [2, 3],
    'claim_scope': 'One observed test result against published references, not a controlled multi-seed improvement',
}


def options(args):
    return SimpleNamespace(data=args.data, output=args.output, results=args.results,
                           logs=args.logs, device=args.device, semantic_cache=args.semantic_cache)


def profile_manifest(args):
    manifest = ecthr_a.verify_prepared(args.data)
    if (manifest['max_packed_tokens'] != 8192 or manifest.get('chunk_tokens') != 128
            or not manifest.get('gradient_checkpointing') or manifest.get('encoder_chunk_batch_size') != 4):
        raise ValueError('This fixed experiment requires the new long-document preparation. Do not use the old 382-token dataset manifest')
    return manifest


def freeze_plan(args, manifest):
    root = Path(__file__).parent
    code = {file.name: ecthr_a.digest_file(file) for file in sorted(root.glob('*.py'))}
    packages = {}
    for name in ['torch', 'transformers', 'sentence-transformers', 'numpy', 'datasets']:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = 'not installed'
    plan = {'profile': PROFILE, 'dataset': manifest['source'], 'data_files': manifest['files'],
            'models': manifest['models'], 'software': packages, 'code_sha256': code, 'device': args.device}
    path = Path(args.logs) / 'ecthr-a-one-run-plan.json'
    if path.exists():
        saved = json.loads(path.read_text(encoding='utf-8'))
        if saved != plan:
            raise ValueError('The predeclared plan, data, code, environment, or device changed. Preserve this experiment; do not silently change a single-run comparison')
    else:
        json_file(path, plan)
    return plan


def prepare(args):
    base = SimpleNamespace(data=args.data, dataset_revision='main', model_revision='main',
                           legalbert_model=args.legalbert_model, minilm_model=args.minilm_model,
                           local_data=args.local_data, max_tokens=8192, chunk_tokens=128,
                           encoder_chunk_batch_size=4, gradient_checkpointing=True)
    ecthr_a.prepare(base)
    manifest = profile_manifest(args)
    freeze_plan(args, manifest)
    print('\nFixed plan saved. Primary target: improve on the published GPT-5.2 micro F1 of 79.33%. Also report the Legal-BERT reference of 76.25%.')
    print('A +2 to +3 percentage-point target means 81.33% to 82.33%; this is a goal, not an expected result.')
    print('Next: run the full-context hardware preflight. No research training run has started.')


def preflight(args):
    manifest = profile_manifest(args)
    freeze_plan(args, manifest)
    ecthr_a.check_device(args.device)
    from transformers import AutoTokenizer
    reference = manifest['models']['legalbert']
    config = ModelConfig(tuple(ecthr_a.LABELS), legalbert_model=reference['name'], legalbert_revision=reference['revision'],
                         task_type='multi_label', max_tokens=8192, chunk_tokens=128,
                         gradient_checkpointing=True, encoder_chunk_batch_size=4, dropout=.2, freeze_lower_layers=4)
    tokenizer = AutoTokenizer.from_pretrained(reference['name'], revision=reference['revision'])
    item = DocumentProcessor(tokenizer, config)('the evidence ' * 8064)
    class FixedSemanticFeatures:
        def encode(self, texts, **kwargs):
            # Synthetic graph features for a memory/gradient test, not label evaluation.
            return torch.eye(len(texts), 384)
    batch = collate_documents([item], FixedSemanticFeatures())
    batch['labels'] = torch.zeros(1, len(ecthr_a.LABELS))
    batch = {key: value.to(args.device) for key, value in batch.items()}
    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    model = LegalOutcomeModel(config).to(args.device).train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Reserve the approximate space of two fp32 Adam moment buffers. No step is taken.
    moment_reservation = torch.empty(2 * trainable, device=args.device, dtype=torch.float32)
    output = model(**batch)
    output['loss'].backward()
    if not torch.isfinite(output['loss']) or model.head_weight.grad is None:
        raise RuntimeError('Full-context forward/backward preflight failed')
    result = {'status': 'passed', 'synthetic_only': True, 'optimizer_step_performed': False,
              'research_training_started': False, 'packed_tokens': batch['input_ids'].shape[1],
              'content_tokens': int((batch['span_ids'] >= 0).sum()), 'trainable_parameters': trainable,
              'device': args.device, 'estimated_adam_moment_bytes': moment_reservation.numel() * 4,
              'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3 if args.device == 'cuda' else None,
              'note': 'A synthetic memory/gradient check with reserved moment buffers, not a guarantee for every runtime condition'}
    json_file(Path(args.logs) / 'ecthr-a-one-run-preflight.json', result)
    print(table(['Check', 'Result'], [[k, v] for k, v in result.items() if not isinstance(v, dict)]))
    print('No dataset case or test label was used. No parameter update was performed. Next: train once.')


def train(args):
    manifest = profile_manifest(args)
    freeze_plan(args, manifest)
    report_file = Path(args.logs) / 'ecthr-a-one-run-preflight.json'
    if not report_file.exists() or json.loads(report_file.read_text())['status'] != 'passed':
        raise ValueError('Run preflight successfully on this system before the one training run')
    base = options(args)
    base.epochs = 8; base.batch_size = 1; base.learning_rate = 1e-5; base.head_learning_rate = 1e-4
    base.dropout = .2; base.freeze_lower_layers = 4; base.seed = 42; base.resume = args.resume
    base.gradient_accumulation_steps = 8; base.lr_schedule = 'linear'; base.warmup_ratio = .1
    print('ONE RESEARCH TRAINING RUN | eight epochs | effective batch size eight | fixed threshold 0.5', flush=True)
    ecthr_a.train(base)


def test(args):
    manifest = profile_manifest(args)
    freeze_plan(args, manifest)
    base = options(args)
    base.required_epochs = 8; base.batch_size = 1
    metadata = json.loads((Path(args.output) / 'metadata.json').read_text(encoding='utf-8'))
    cfg, report = metadata['model_config'], metadata['training_report']
    if (cfg.get('chunk_tokens') != 128 or cfg.get('max_tokens') != 8192 or cfg['dropout'] != .2
            or cfg['freeze_lower_layers'] != 4 or report.get('epochs') != 8
            or report.get('gradient_accumulation_steps') != 8 or report.get('lr_schedule') != 'linear'
            or report.get('learning_rate') != 1e-5 or report.get('head_learning_rate') != 1e-4
            or report.get('seed') != 42 or report.get('selection_metric') != 'micro_f1'
            or report.get('warmup_ratio') != .1 or report.get('weight_decay') != .01
            or report.get('batch_size') != 1 or report.get('effective_batch_size') != 8
            or cfg.get('threshold') != .5 or not cfg.get('gradient_checkpointing')
            or cfg.get('encoder_chunk_batch_size') != 4 or cfg.get('max_spans') != 8):
        raise ValueError('Saved model does not match the fixed one-run profile')
    ecthr_a.test(base)
    directory = Path(args.results)
    result = json.loads((directory / 'test-results.json').read_text(encoding='utf-8'))
    delta_micro = 100 * (result['micro_f1'] - PROFILE['primary_reference']['micro_f1'])
    delta_macro = 100 * (result['macro_f1'] - PROFILE['primary_reference']['macro_f1'])
    eligible = manifest['source']['mode'] == 'official' and result['examples'] == 1000
    comparison = {'primary_reference': PROFILE['primary_reference'],
                  'dataset_mode': manifest['source']['mode'],
                  'official_benchmark_result': eligible,
                  'encoder_reference': PROFILE['encoder_reference'],
                  'observed_encoder_micro_f1_percentage_points': 100 * (result['micro_f1'] - .7625),
                  'observed_encoder_macro_f1_percentage_points': 100 * (result['macro_f1'] - .6562),
                  'observed_micro_f1_percentage_points': delta_micro,
                  'observed_macro_f1_percentage_points': delta_macro,
                  'at_least_two_point_primary_gain': delta_micro >= 2 if eligible else None,
                  'at_least_three_point_primary_gain': delta_micro >= 3 if eligible else None,
                  'macro_f1_not_below_published_reference': delta_macro >= 0,
                  'statistical_significance_established': False,
                  'interpretation': 'One observed comparison to a published baseline. Matching the nominal context cap does not match every protocol detail.'}
    json_file(directory / 'one-run-comparison.json', comparison)
    rows = [['Micro F1', f"{100 * result['micro_f1']:.2f}%", '79.33%', f'{delta_micro:+.2f} pp'],
            ['Macro F1', f"{100 * result['macro_f1']:.2f}%", '76.31%', f'{delta_macro:+.2f} pp']]
    text = table(['Measure', 'This test', 'Published GPT-5.2', 'Observed difference'], rows)
    text += f"\n\nDifference from Legal-BERT: micro F1 {100 * (result['micro_f1'] - .7625):+.2f} pp; macro F1 {100 * (result['macro_f1'] - .6562):+.2f} pp.\n"
    if not eligible:
        text += '\nLOCAL FIXTURE/CUSTOM DATA: these numbers are not official ECtHR-A benchmark results.\n'
    text += '\nSingle run. No significance claim. No guaranteed improvement.\n'
    if delta_micro > 0 and delta_macro < 0:
        text += 'Tradeoff: micro F1 increased while macro F1 decreased; do not claim an unqualified overall improvement.\n'
    atomic_text(directory / 'one-run-comparison.txt', text)
    print('\n' + text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='phase', required=True)
    for phase, function in [('prepare', prepare), ('preflight', preflight), ('train', train), ('test', test)]:
        p = sub.add_parser(phase)
        p.add_argument('--data', default='data/ecthr_a_one_run')
        p.add_argument('--output', default='checkpoints/ecthr_a_one_run')
        p.add_argument('--results', default='results/ecthr_a_one_run')
        p.add_argument('--logs', default='logs/ecthr_a_one_run')
        p.add_argument('--semantic-cache', default='cache/ecthr_a_one_run')
        p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
        if phase == 'prepare':
            p.add_argument('--legalbert-model', default='nlpaueb/legal-bert-base-uncased')
            p.add_argument('--minilm-model', default='sentence-transformers/all-MiniLM-L6-v2')
            p.add_argument('--local-data', help='Local JSONL fixtures only; results will not be marked official')
        if phase == 'train':
            p.add_argument('--resume', action='store_true')
        p.set_defaults(run=function)
    args = parser.parse_args(argv)
    args.run(args)


if __name__ == '__main__':
    main()
