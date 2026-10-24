import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from legal_outcome.ecthr_a import LABELS, convert_rows, verify_prepared
from legal_outcome.reporting import history_table, result_table, save_result_files
from legal_outcome.semantic_cache import cache_semantics
from legal_outcome.data import collate_documents
from test_architecture import Tokenizer, SemanticEncoder
from legal_outcome import ModelConfig
from legal_outcome.data import DocumentProcessor


def command(*args):
    result = subprocess.run([sys.executable, '-m', 'legal_outcome.ecthr_a', *args], capture_output=True, text=True,
                            env={**os.environ, 'HF_HUB_OFFLINE': '1', 'OMP_NUM_THREADS': '2'})
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_mapping_and_empty_labels_preserved():
    rows = convert_rows([{'text': ['one', 'two'], 'labels': [0, 9]}, {'text': 'none', 'labels': []}], 'train')
    assert rows[0]['text'] == 'one\n\ntwo'
    assert rows[0]['labels'] == ['2', 'P1-1']
    assert rows[1]['labels'] == []
    with pytest.raises(ValueError):
        convert_rows([{'text': 'case', 'labels': [10]}], 'train')


def test_readable_tables_and_saved_reports(tmp_path):
    values = {'examples': 3, 'micro_f1': .7, 'macro_f1': .5, 'selected_epoch': 2,
              'per_label': {'2': {'precision': .75, 'recall': .5, 'f1': .6, 'support': 2}}}
    text = save_result_files(tmp_path, values)
    assert '70.00%' in text and 'selected epoch: 2' in text
    assert 'Article' in text and 'Support' in text
    assert json.loads((tmp_path / 'test-results.json').read_text())['micro_f1'] == .7
    assert (tmp_path / 'per-article-results.csv').exists()
    assert 'Epoch' in history_table([{'epoch': 1, 'training_loss': .6, 'selected': True}])


def test_cached_frozen_features_are_reused(tmp_path):
    processor = DocumentProcessor(Tokenizer(), ModelConfig(('a', 'b'), max_tokens=16))
    items = [processor('one two three'), processor('four five')]
    encoder = SemanticEncoder()
    class Compatible:
        def encode(self, texts, **kwargs):
            return encoder.encode(texts)
    path = cache_semantics(items, Compatible(), tmp_path, {'model': 'fixture'})
    first = collate_documents(items, None)
    fresh = [processor('one two three'), processor('four five')]
    class Fails:
        def encode(self, *args, **kwargs):
            raise AssertionError('Existing cache must avoid encode')
    assert cache_semantics(fresh, Fails(), tmp_path, {'model': 'fixture'}) == path
    assert torch.equal(first['semantic_embeddings'], collate_documents(fresh, None)['semantic_embeddings'])


def test_complete_eight_epoch_proposed_workflow_and_resume(tmp_path):
    from transformers import BertConfig, BertModel, BertTokenizerFast
    from sentence_transformers import SentenceTransformer, models
    model_path = tmp_path / 'encoder'
    vocab = tmp_path / 'vocab.txt'
    vocab.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ncase\nevidence\nappeal\nallowed\ndenied\n')
    BertModel(BertConfig(vocab_size=10, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                         intermediate_size=32)).save_pretrained(model_path)
    BertTokenizerFast(vocab_file=str(vocab)).save_pretrained(model_path)
    semantic_path = tmp_path / 'semantic'
    SentenceTransformer(modules=[models.Transformer(str(model_path), max_seq_length=16), models.Pooling(16)]).save(str(semantic_path))
    source = tmp_path / 'source'; source.mkdir()
    for split, count in [('train', 4), ('validation', 2), ('test', 2)]:
        rows = [{'text': [f'case {split} {i}', 'evidence appeal'], 'labels': [] if i == 0 else [i % 2]} for i in range(count)]
        (source / f'{split}.jsonl').write_text('\n'.join(json.dumps(x) for x in rows))
    data = tmp_path / 'prepared'
    command('prepare', '--data', str(data), '--local-data', str(source), '--legalbert-model', str(model_path),
            '--minilm-model', str(semantic_path), '--max-tokens', '16')
    manifest = verify_prepared(data)
    assert manifest['labels'] == LABELS
    assert manifest['source']['mode'] == 'local_fixture_or_custom_data'
    assert manifest['audit']['train']['empty_label_cases'] == 1
    assert (data / 'token-coverage.csv').exists()
    output = tmp_path / 'checkpoint'
    cache = tmp_path / 'semantic-cache'
    logs = tmp_path / 'logs'
    common = ['train', '--data', str(data), '--output', str(output), '--device', 'cpu', '--semantic-cache', str(cache), '--logs', str(logs)]
    # Stop deliberately after epoch 3, then continue to the total target of 8.
    started = command(*common, '--epochs', '3')
    assert 'Val micro F1' in started.stdout and 'TRAINING COMPLETE' in started.stdout
    resumed = command(*common, '--epochs', '8', '--resume')
    assert 'Epoch' in resumed.stdout
    history = json.loads((output / 'history.json').read_text())
    assert [r['epoch'] for r in history] == list(range(1, 9))
    assert (output / 'history.csv').exists() and (output / 'training-summary.txt').exists()
    assert (logs / 'checkpoint-training.log').exists()
    assert torch.load(output / 'training-state.pt', weights_only=True)['completed_epoch'] == 8
    # Recover a stop between the final epoch save and inference export.
    (output / 'metadata.json').unlink(); (output / 'model.pt').unlink()
    command(*common, '--epochs', '8', '--resume')
    assert (output / 'metadata.json').exists() and (output / 'model.pt').exists()
    results = tmp_path / 'results'
    scored = command('test', '--data', str(data), '--output', str(output), '--results', str(results), '--device', 'cpu',
                     '--semantic-cache', str(cache), '--logs', str(logs))
    assert 'TEST RESULTS' in scored.stdout and 'Micro F1' in scored.stdout
    result = json.loads((results / 'test-results.json').read_text())
    assert result['examples'] == 2 and result['training_epochs_completed'] == 8
    assert result['max_content_tokens'] == 14
    assert result['benchmark']['mode'] == 'local_fixture_or_custom_data'
    for name in ['test-report.txt', 'test-summary.csv', 'per-article-results.csv', 'predictions.pt', 'published-reference-comparison.csv']:
        assert (results / name).exists()
    previous = (results / 'test-results.json').read_bytes()
    again = command('test', '--data', str(data), '--output', str(output), '--results', str(results), '--device', 'cpu')
    assert 'No new test evaluation' in again.stdout
    assert (results / 'test-results.json').read_bytes() == previous
    with (data / 'train.jsonl').open('a') as stream:
        stream.write('\n')
    with pytest.raises(ValueError, match='changed'):
        verify_prepared(data)


def test_semantic_cache_does_not_change_training_random_state(tmp_path):
    import random
    import numpy as np
    from legal_outcome.training import capture_rng
    class RandomEncoder:
        def encode(self, texts, **kwargs):
            random.random()
            np.random.random()
            return torch.rand(len(texts), 3)
    items = [{'spans': ['one', 'two']}]
    before = capture_rng()
    cache_semantics(items, RandomEncoder(), tmp_path, {'model': 'fixture'})
    after = capture_rng()
    assert torch.equal(before['torch'], after['torch'])
    assert before['python'] == after['python']
    assert before['numpy'] == after['numpy']


def test_pinned_remote_revision_is_a_complete_cache_identity():
    from legal_outcome.semantic_cache import model_signature
    assert model_signature(None, 'public/model', 'a' * 40) == {'model': 'public/model', 'revision': 'a' * 40}
