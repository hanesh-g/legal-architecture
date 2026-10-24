"""Inspect and evaluate the completed first-dataset model, without retraining."""
import argparse
import json
from pathlib import Path
import torch
from .config import ModelConfig
from .ecthr_a import digest_file, logged, check_device
from .reporting import json_file, table, result_table, save_result_files, csv_file, atomic_text

LABELS = {'10', '11', '13', '14', '18', '2', '3', '34', '38', '5', '6', '7', '8', '9', 'P1-1', 'P4-7'}
PAPER_REFERENCE = {'micro_f1': .715, 'micro_jaccard': .557, 'micro_pr_auc': .543, 'micro_roc_auc': .831}


def matches(config):
    return config.get('task_type') == 'multi_label' and set(config.get('labels', [])) == LABELS and len(config.get('labels', [])) == 16


def read_state(path):
    # Map tensors lazily; inspecting a multi-GB recovery file does not copy its optimizer to the GPU.
    return torch.load(path, map_location='cpu', weights_only=True, mmap=True)


def inspect_folder(directory):
    directory = Path(directory)
    meta = json.loads((directory / 'metadata.json').read_text(encoding='utf-8')) if (directory / 'metadata.json').exists() else None
    state = read_state(directory / 'training-state.pt') if (directory / 'training-state.pt').exists() else None
    if meta:
        config = meta['model_config']
    elif state and 'contract' in state:
        config = state['contract']['config']
    else:
        raise ValueError('No supported prediction or recovery checkpoint found')
    if not matches(config):
        raise ValueError('This is not a 16-label multi-label ECHR2023-compatible model')
    report = meta.get('training_report', {}) if meta else {}
    history = report.get('history', [])
    exported_epoch = history[-1]['epoch'] if history else 0
    recovered_epoch = state.get('completed_epoch', 0) if state else 0
    if state and not matches(state.get('contract', {}).get('config', {})):
        raise ValueError('Prediction metadata and recovery checkpoint have different tasks')
    if meta and state and ModelConfig(**config) != ModelConfig(**state['contract']['config']):
        raise ValueError('Prediction metadata and recovery checkpoint have different model settings')
    ready = bool(meta and (directory / 'model.pt').exists() and (directory / 'tokenizer').is_dir())
    summary = {'checkpoint': str(directory.resolve()), 'completed_epochs': max(exported_epoch, recovered_epoch),
               'exported_epochs': exported_epoch, 'recovery_epoch': recovered_epoch,
               'selected_epoch': report.get('selected_epoch') if ready and exported_epoch >= recovered_epoch else state.get('best_epoch') if state else None,
               'prediction_files_present': ready, 'recovery_present': state is not None,
               'labels': list(config['labels']), 'task_type': config['task_type']}
    return summary, meta, state


def inspect(args):
    root = Path(args.root)
    if not root.is_dir():
        raise FileNotFoundError(f'Checkpoint search folder does not exist: {root}')
    folders = {path.parent for filename in ['metadata.json', 'training-state.pt'] for path in root.rglob(filename)}
    rows, found = [], []
    for folder in sorted(folders):
        try:
            summary, _, _ = inspect_folder(folder)
        except ValueError as error:
            # The scan can include Realistic LJP, IJJD, and ECtHR-A folders. List only the first-dataset task.
            if 'not a 16-label' in str(error) or 'No supported' in str(error):
                continue
            raise
        found.append(summary)
        rows.append([summary['checkpoint'], summary['completed_epochs'], summary['selected_epoch'],
                     'YES' if summary['prediction_files_present'] else 'NO',
                     'YES' if summary['recovery_present'] else 'NO'])
    if not found:
        print('No compatible first-dataset checkpoint found. Inspect the original ECHR project, not the new ECtHR-A folder.')
        return
    print(table(['Checkpoint folder', 'Epochs', 'Selected', 'Prediction files', 'Recovery'], rows))
    print('Choose the already selected ECHR model using its original validation rule. Do not choose a run by test results.')


def export_best(source, destination, metadata, state):
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from .model import LegalOutcomeModel
    from .data import DocumentProcessor
    from .cues import CueExtractor
    from .runtime import save_checkpoint
    if state is None or state.get('best_model') is None or state.get('best_epoch') is None:
        raise ValueError('Recovery state has no saved best model. Do not substitute arbitrary weights')
    destination = Path(destination)
    identity = {'recovery_sha256': digest_file(Path(source) / 'training-state.pt'), 'selected_epoch': state['best_epoch']}
    if destination.exists():
        marker = destination / 'recovery-export.json'
        if not marker.exists() or json.loads(marker.read_text()) != identity:
            raise ValueError('Recovery export folder exists with a different identity. Preserve it and use a new --results folder')
        if (destination / 'metadata.json').exists() and (destination / 'model.pt').exists():
            return destination
        raise ValueError('Recovery export is incomplete. Preserve it and use a new --results folder')
    contract = state['contract']
    config = ModelConfig(**contract['config'])
    if metadata:
        encoder_config = dict(metadata['encoder_config'])
        model_type = encoder_config.pop('model_type')
        encoder = AutoModel.from_config(AutoConfig.for_model(model_type, **encoder_config))
    else:
        encoder_config = AutoConfig.from_pretrained(config.legalbert_model, revision=config.legalbert_revision)
        encoder = AutoModel.from_config(encoder_config)
    model = LegalOutcomeModel(config, encoder)
    model.load_state_dict(state['best_model'], strict=True)
    tokenizer = AutoTokenizer.from_pretrained(config.legalbert_model, revision=config.legalbert_revision)
    cues = CueExtractor(metadata['cue_patterns']) if metadata and 'cue_patterns' in metadata else CueExtractor()
    processor = DocumentProcessor(tokenizer, config, cues)
    history = state['history']
    training_report = {'history': history, 'epochs': state['completed_epoch'], 'selected_epoch': state['best_epoch'],
                       'selection_metric': contract.get('selection'), 'selection_score': state.get('best_score'),
                       'data_sha256': contract.get('data_sha256'), 'validation_sha256': contract.get('validation_sha256'),
                       'seed': contract.get('seed'), 'recovered_without_retraining': True,
                       'cue_patterns_source': 'existing metadata' if metadata else 'repository default patterns used by the original CLI'}
    save_checkpoint(destination, model, processor, training_report)
    json_file(destination / 'recovery-export.json', identity)
    print(f'Exported saved epoch {state["best_epoch"]} from recovery state. No optimizer step or training was run.')
    return destination


def finish_reports(result_dir, result, summary, count):
    """Also repair an interruption after predictions were saved, without evaluating again."""
    result.update(dataset_role='user-supplied processed ECHR2023/PILOT test file',
                  paper_test_examples=3000, actual_test_examples=count, training_epochs_completed=summary['completed_epochs'],
                  model_note='Existing proposed-architecture checkpoint; not a PILOT model reproduction')
    save_result_files(result_dir, result)
    required = [('micro_f1', 'Micro F1'), ('micro_jaccard', 'Micro Jaccard'),
                ('micro_pr_auc', 'Micro PR-AUC (trapezoidal)'), ('micro_roc_auc', 'Micro ROC-AUC')]
    rows = [[name, 'undefined' if result[key] is None else f'{100 * result[key]:.2f}%', result[key]] for key, name in required]
    text = table(['Required measure', 'Percent scale', 'Raw value'], rows)
    text += ('\n\nAverage precision is also saved separately; confirm the paper implementation before treating it as trapezoidal PR-AUC.\n'
             f'The paper reports 3000 test cases; this file has {count}. State any preprocessing difference.\n'
             'This is a saved-model test evaluation, not a training or threshold-tuning run.\n'
             'Published PILOT values average five seeds. Observed differences from one run do not establish a repeatable improvement.\n')
    atomic_text(result_dir / 'paper-metrics.txt', text)
    csv_file(result_dir / 'paper-metrics.csv', ['measure', 'value'], [[key, result[key]] for key, _ in required])
    csv_file(result_dir / 'published-reference-comparison.csv',
             ['measure', 'published_PILOT_five_seed_mean', 'this_test_run', 'observed_difference_percentage_points', 'comparison_note'],
             [[key, PAPER_REFERENCE[key], result[key], None if result[key] is None else 100 * (result[key] - PAPER_REFERENCE[key]),
               'Descriptive reference comparison; verify split and metric definition equivalence'] for key, _ in required])
    print('\n' + text)
    print(f'All reports saved to {result_dir.resolve()}')


def evaluate(args):
    from .cli import main as architecture_cli
    from .runtime import read_records
    summary, metadata, state = inspect_folder(args.checkpoint)
    if summary['completed_epochs'] < args.required_epochs:
        raise ValueError(f'Only {summary["completed_epochs"]} epochs are saved. Complete the original run before final testing; no new training is started by this helper')
    # Verify the original processed schema before loading a model. Raw PILOT facts/violated_articles files need the original preprocessing.
    try:
        records = read_records(args.test_data, summary['labels'], 'multi_label')
    except ValueError as error:
        raise ValueError(f'{error}. Use the original processed JSONL with text and labels fields; raw PILOT facts/violated_articles files are not direct inputs') from error
    test_hash = digest_file(args.test_data)
    provenance = metadata.get('training_report', {}) if metadata else {}
    if state:
        provenance = {**provenance, **state['contract']}
    if test_hash in [provenance.get('data_sha256'), provenance.get('validation_sha256')]:
        raise ValueError('The test file is identical to the saved training or validation input. Use the original held-out test split')
    source_identity = {'checkpoint': str(Path(args.checkpoint).resolve()), 'test_sha256': test_hash,
                       'test_examples': len(records), 'required_epochs': args.required_epochs}
    final_ready = summary['prediction_files_present'] and summary['exported_epochs'] >= summary['completed_epochs']
    source_identity['source_sha256'] = digest_file(Path(args.checkpoint) / ('model.pt' if final_ready else 'training-state.pt'))
    if final_ready:
        source_identity['metadata_sha256'] = digest_file(Path(args.checkpoint) / 'metadata.json')
    result_dir = Path(args.results)
    marker = result_dir / 'first-dataset-identity.json'
    if marker.exists():
        if json.loads(marker.read_text()) != source_identity:
            raise ValueError('Result directory belongs to another model or test file. Use a new --results directory')
        if (result_dir / 'test-results.json').exists():
            result = json.loads((result_dir / 'test-results.json').read_text())
            print('Existing report reused. No new test evaluation was performed.')
            print(result_table(result))
            finish_reports(result_dir, result, summary, len(records))
            return
    elif result_dir.exists():
        raise ValueError('Result folder already exists without an evaluation identity. Use a new --results folder')
    check_device(args.device)
    json_file(marker, source_identity)
    with logged(result_dir / 'evaluation.log'):
        checkpoint = Path(args.checkpoint) if final_ready else export_best(args.checkpoint, result_dir / 'recovered-selected-model', metadata, state)
        print(f'FIRST DATASET: ECHR2023/PILOT-compatible model | {summary["completed_epochs"]} saved epochs', flush=True)
        print(f'Test cases: {len(records)} | articles: 16 | saved selected epoch: {summary["selected_epoch"]}', flush=True)
        architecture_cli(['evaluate', '--checkpoint', str(checkpoint), '--data', args.test_data, '--device', args.device,
                          '--batch-size', str(args.batch_size), '--display', 'table', '--results-dir', str(result_dir),
                          '--semantic-cache', str(Path(args.cache))])
        result = json.loads((result_dir / 'test-results.json').read_text())
        finish_reports(result_dir, result, summary, len(records))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='phase', required=True)
    p = sub.add_parser('inspect')
    p.add_argument('--root', required=True, help='Original project or checkpoint directory')
    p.set_defaults(run=inspect)
    p = sub.add_parser('evaluate')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--test-data', required=True, help='Original processed JSONL with text and string article labels')
    p.add_argument('--results', default='results/echr2023_final')
    p.add_argument('--cache', default='cache/echr2023_semantics')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--required-epochs', type=int, default=8)
    p.set_defaults(run=evaluate)
    args = parser.parse_args(argv)
    args.run(args)


if __name__ == '__main__':
    main()
