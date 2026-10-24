"""First-dataset evaluation keeps the selected model and never retrains it."""
import json
from types import SimpleNamespace

import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from legal_outcome import ModelConfig, LegalOutcomeModel
from legal_outcome.data import DocumentProcessor
from legal_outcome.echr2023 import LABELS, inspect_folder, inspect, evaluate
from legal_outcome.ecthr_a import digest_file
from legal_outcome.runtime import save_checkpoint
from legal_outcome.training import CheckpointSelector, make_optimizer, save_training_state


@pytest.fixture
def saved_run(tmp_path, monkeypatch):
    model_dir = tmp_path / 'original-encoder'
    vocab = tmp_path / 'vocab.txt'
    vocab.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ncase\nappeal\nallowed\n')
    encoder = BertModel(BertConfig(vocab_size=8, hidden_size=16, num_hidden_layers=1,
                                  num_attention_heads=2, intermediate_size=32))
    encoder.save_pretrained(model_dir)
    tokenizer = BertTokenizerFast(vocab_file=str(vocab))
    tokenizer.save_pretrained(model_dir)
    config = ModelConfig(tuple(sorted(LABELS)), task_type='multi_label', max_tokens=16, legalbert_model=str(model_dir),
                         minilm_model=str(model_dir), dropout=0.0)
    model = LegalOutcomeModel(config, encoder).eval()
    processor = DocumentProcessor(tokenizer, config)
    history = [{'epoch': i, 'training_loss': .1} for i in range(1, 9)]
    selector = CheckpointSelector('micro_f1')
    selector.update(model, 3, {'micro_f1': .7})
    # Recovery includes both latest and best weights. Make the latest distinct.
    with torch.no_grad():
        next(model.parameters()).add_(.25)
    checkpoint = tmp_path / 'baseline_safe'
    checkpoint.mkdir()
    contract = {'config': config.to_dict(), 'selection': 'micro_f1', 'seed': 42,
                'data_sha256': 'training-file', 'validation_sha256': 'validation-file'}
    save_training_state(checkpoint / 'training-state.pt', model, make_optimizer(model, 1e-5, 1e-4, .01),
                        selector, history, contract)
    class Semantic:
        def __init__(self, *args, **kwargs):
            pass
        def encode(self, texts, **kwargs):
            return torch.ones(len(texts), 8)
    import sentence_transformers
    monkeypatch.setattr(sentence_transformers, 'SentenceTransformer', Semantic)
    data = tmp_path / 'test.jsonl'
    data.write_text(json.dumps({'text': 'case appeal allowed', 'labels': ['2', '6']}) + '\n'
                    + json.dumps({'text': 'case appeal', 'labels': []}) + '\n')
    return SimpleNamespace(checkpoint=checkpoint, model=model, processor=processor, selector=selector,
                           history=history, data=data, best=selector.state)


def args(run, tmp_path):
    return SimpleNamespace(checkpoint=str(run.checkpoint), test_data=str(run.data),
                           results=str(tmp_path / 'results'), cache=str(tmp_path / 'cache'),
                           device='cpu', batch_size=2, required_epochs=8)


def test_recovery_evaluation_exports_best_without_changing_source(saved_run, tmp_path, capsys):
    run = saved_run
    original = digest_file(run.checkpoint / 'training-state.pt')
    summary, _, _ = inspect_folder(run.checkpoint)
    assert summary['completed_epochs'] == 8 and summary['selected_epoch'] == 3
    assert not summary['prediction_files_present']
    evaluate(args(run, tmp_path))
    result_dir = tmp_path / 'results'
    export = result_dir / 'recovered-selected-model'
    weights = torch.load(export / 'model.pt', weights_only=True)
    assert all(torch.equal(weights[k], v) for k, v in run.best.items())
    assert digest_file(run.checkpoint / 'training-state.pt') == original
    assert not (run.checkpoint / 'model.pt').exists()
    result = json.loads((result_dir / 'test-results.json').read_text())
    assert result['selected_epoch'] == 3 and result['training_epochs_completed'] == 8
    assert result['examples'] == 2 and result['actual_test_examples'] == 2
    assert set(['micro_f1', 'micro_jaccard', 'micro_pr_auc', 'micro_roc_auc']) <= result.keys()
    assert result['decision_threshold'] == .5 and result['max_content_tokens'] == 14
    for name in ['paper-metrics.txt', 'paper-metrics.csv', 'published-reference-comparison.csv',
                 'test-report.txt', 'test-summary.csv', 'per-article-results.csv', 'predictions.pt', 'evaluation.log']:
        assert (result_dir / name).exists()
    assert 'No optimizer step' in capsys.readouterr().out
    predictions_hash = digest_file(result_dir / 'predictions.pt')
    # Repair reports after an interrupted final write without another evaluation.
    (result_dir / 'paper-metrics.txt').unlink()
    evaluate(args(run, tmp_path))
    assert 'No new test evaluation' in capsys.readouterr().out
    assert (result_dir / 'paper-metrics.txt').exists()
    assert digest_file(result_dir / 'predictions.pt') == predictions_hash


def test_completed_inference_model_uses_original_export(saved_run, tmp_path):
    run = saved_run
    run.model.load_state_dict(run.best)
    save_checkpoint(run.checkpoint, run.model, run.processor,
                    {'history': run.history, 'selected_epoch': 3}, allow_existing=True)
    # Early completed runs exported inference files without an epoch recovery file.
    (run.checkpoint / 'training-state.pt').unlink()
    evaluate(args(run, tmp_path))
    assert not (tmp_path / 'results' / 'recovered-selected-model').exists()
    assert json.loads((tmp_path / 'results' / 'test-results.json').read_text())['selected_epoch'] == 3


def test_stale_export_does_not_hide_later_recovery_epochs(saved_run, tmp_path):
    run = saved_run
    save_checkpoint(run.checkpoint, run.model, run.processor,
                    {'history': run.history[:4], 'selected_epoch': 1}, allow_existing=True)
    summary, _, _ = inspect_folder(run.checkpoint)
    assert summary['exported_epochs'] == 4 and summary['completed_epochs'] == 8
    assert summary['selected_epoch'] == 3
    evaluate(args(run, tmp_path))
    export = tmp_path / 'results' / 'recovered-selected-model'
    weights = torch.load(export / 'model.pt', weights_only=True)
    assert all(torch.equal(weights[k], v) for k, v in run.best.items())


def test_inspection_lists_echr_and_rejects_incomplete_or_other_task(saved_run, tmp_path, capsys):
    run = saved_run
    other = tmp_path / 'realistic'; other.mkdir()
    (other / 'metadata.json').write_text(json.dumps({'model_config': {'task_type': 'single_label', 'labels': ['0', '1']}}))
    inspect(SimpleNamespace(root=str(tmp_path)))
    output = capsys.readouterr().out
    assert 'baseline_safe' in output and 'realistic' not in output
    payload = torch.load(run.checkpoint / 'training-state.pt', weights_only=True)
    payload['completed_epoch'] = 4
    torch.save(payload, run.checkpoint / 'training-state.pt')
    with pytest.raises(ValueError, match='Only 4 epochs'):
        evaluate(args(run, tmp_path))
    assert not (tmp_path / 'results').exists()


def test_raw_or_validation_data_is_not_accepted_as_test(saved_run, tmp_path):
    run = saved_run
    run.data.write_text(json.dumps({'facts': 'case', 'violated_articles': ['2']}))
    with pytest.raises(ValueError, match='original processed JSONL'):
        evaluate(args(run, tmp_path))
    run.data.write_text(json.dumps({'text': 'case', 'labels': ['2']}))
    payload = torch.load(run.checkpoint / 'training-state.pt', weights_only=True)
    payload['contract']['validation_sha256'] = digest_file(run.data)
    torch.save(payload, run.checkpoint / 'training-state.pt')
    with pytest.raises(ValueError, match='identical to the saved training or validation'):
        evaluate(args(run, tmp_path))
    assert not (tmp_path / 'results').exists()
