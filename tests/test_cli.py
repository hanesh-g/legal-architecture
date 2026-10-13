"""Actual command workflow with local Transformer and SentenceTransformer models."""
import json
import os
import subprocess
import sys
from transformers import BertConfig, BertModel, BertTokenizerFast
from sentence_transformers import SentenceTransformer, models


def run(*args):
    result = subprocess.run([sys.executable, '-m', 'legal_outcome.cli', *args], capture_output=True,
                            text=True, env={**os.environ, 'HF_HUB_OFFLINE': '1', 'OMP_NUM_THREADS': '2'})
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_train_evaluate_predict_commands(tmp_path):
    vocab = tmp_path / 'vocab.txt'
    vocab.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nappeal\nis\nallowed\ndenied\nevidence\nsection\n')
    encoder_path = tmp_path / 'encoder'
    BertModel(BertConfig(vocab_size=11, hidden_size=16, num_hidden_layers=1,
                         num_attention_heads=2, intermediate_size=32)).save_pretrained(encoder_path)
    BertTokenizerFast(vocab_file=str(vocab)).save_pretrained(encoder_path)
    semantic_path = tmp_path / 'semantic'
    transformer = models.Transformer(str(encoder_path), max_seq_length=16)
    SentenceTransformer(modules=[transformer, models.Pooling(16)]).save(str(semantic_path))
    rows = [{'text': 'appeal is allowed evidence', 'label': 'allow'},
            {'text': 'appeal is denied section', 'label': 'deny'}]
    train_file = tmp_path / 'train.jsonl'
    train_file.write_text('\n'.join(json.dumps(x) for x in rows))
    validation = tmp_path / 'validation.jsonl'
    validation.write_text(json.dumps({'text': 'evidence appeal allowed', 'label': 'allow'}))
    checkpoint = tmp_path / 'output'
    history = run('train', '--data', str(train_file), '--validation', str(validation), '--labels', 'allow', 'deny',
                  '--output', str(checkpoint), '--legalbert-model', str(encoder_path),
                  '--minilm-model', str(semantic_path), '--epochs', '2', '--max-tokens', '16')
    reports = [json.loads(line) for line in history.splitlines()]
    assert len(reports) == 2
    assert reports[-1]['validation']['examples'] == 1
    score = json.loads(run('evaluate', '--checkpoint', str(checkpoint), '--data', str(train_file)))
    assert score['examples'] == 2
    assert score['labels'] == ['allow', 'deny']
    text = tmp_path / 'case.txt'
    text.write_text('appeal is allowed')
    result = json.loads(run('predict', '--checkpoint', str(checkpoint), '--text-file', str(text), '--evidence'))
    assert result['predicted_label'] in ['allow', 'deny']
    assert abs(sum(result['probabilities'].values()) - 1) < 1e-6
    assert len(result['span_evidence']) == 3
    assert abs(sum(s['attention'] for s in result['span_evidence']) - 1) < 1e-6
    # Existing output must not be overwritten.
    rejected = subprocess.run([sys.executable, '-m', 'legal_outcome.cli', 'train', '--data', str(train_file),
                               '--labels', 'allow', 'deny', '--output', str(checkpoint)], capture_output=True, text=True)
    assert rejected.returncode != 0
    assert 'Output already exists' in rejected.stderr
