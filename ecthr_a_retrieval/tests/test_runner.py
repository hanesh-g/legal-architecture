import importlib.util
from pathlib import Path
import sys
import json
import os
import subprocess
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ecthr_runner', ROOT / 'run_ecthr_a.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_label_mapping_empty_labels_and_paragraph_order():
    text, labels = runner.prepare_records([{'text': ['first', 'second'], 'labels': [0, 9]}, {'text': 'none', 'labels': []}])
    assert text == ['first second', 'none']
    assert runner.LABELS[0] == '2' and runner.LABELS[9] == 'P1-1'
    assert labels[0, 0] == labels[0, 9] == 1
    assert labels[1].sum() == 0


@pytest.mark.parametrize('labels', [[10], [-1], [True], [0, 0]])
def test_invalid_labels_rejected(labels):
    with pytest.raises(ValueError):
        runner.prepare_records([{'text': 'case', 'labels': labels}])


def test_last_visible_token_pool_for_both_padding_sides():
    hidden = torch.arange(24).reshape(2, 4, 3)
    mask = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]])
    actual = runner.last_token_pool(hidden, mask)
    assert torch.equal(actual[0], hidden[0, 1])
    assert torch.equal(actual[1], hidden[1, 3])


def test_known_weighted_vote_and_metrics():
    scores = runner.vote(torch.tensor([[0, 1]]), torch.tensor([[.75, .25]]), torch.tensor([[1., 0.], [0., 1.]]), 2)
    assert np.allclose(scores, [[.75, .25]])
    result = runner.metrics([[1, 1], [0, 1], [0, 0]], [[.8, .1], [.8, .8], [.1, .1]], .5)
    assert result['micro_f1'] == pytest.approx(2/3)
    assert result['macro_f1'] == pytest.approx(2/3)
    assert result['exact_match_accuracy'] == pytest.approx(1/3)
    assert result['hamming_loss'] == pytest.approx(1/3)


def test_zero_similarity_vote_is_finite_and_empty():
    probabilities = runner.vote(torch.tensor([[0, 1]]), torch.tensor([[-.5, -.1]]), torch.ones(2, 10), 2)
    assert np.isfinite(probabilities).all() and not probabilities.any()


def test_validation_selection_and_deterministic_ties():
    index = torch.tensor([[1., 0.], [0., 1.]])
    labels = torch.zeros(2, 10); labels[0, 0] = 1; labels[1, 1] = 1
    best, grid = runner.select_parameters(index, labels, index, labels, [1, 2], [.3, .5])
    assert best['k'] == 1 and best['threshold'] == .5
    assert best['micro_f1'] == 1 and len(grid) == 4


def test_completed_shards_are_reused(tmp_path):
    path = tmp_path / 'cache/train/000000.pt'
    path.parent.mkdir(parents=True)
    vectors = torch.tensor([[1., 0.], [0., 1.]])
    labels = torch.zeros(2, 10)
    runner.atomic_save({'start': 0, 'end': 2, 'embeddings': vectors, 'labels': labels}, path, tensor=True)
    actual, report = runner.encode_split(['a', 'b'], labels, 'train', None, None, tmp_path, 1, 16, 2, 'cpu')
    assert torch.equal(actual, vectors)
    assert report == {'new_documents': 0, 'reused_documents': 2}


def test_full_pipeline_local_qwen_and_restart(tmp_path):
    from transformers import Qwen3Config, Qwen3Model, BertTokenizerFast
    model_path = tmp_path / 'tiny-qwen'
    model_path.mkdir()
    model = Qwen3Model(Qwen3Config(vocab_size=12, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                                  num_key_value_heads=2, head_dim=8, intermediate_size=32, max_position_embeddings=64))
    model.save_pretrained(model_path)
    vocab = tmp_path / 'vocab.txt'
    vocab.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ncase\none\ntwo\nthree\nfour\nnone\nfacts\n')
    BertTokenizerFast(vocab_file=str(vocab)).save_pretrained(model_path)
    inputs = tmp_path / 'inputs'; inputs.mkdir()
    for split, count in [('train', 4), ('validation', 2), ('test', 2)]:
        rows = [{'text': ['case facts', str(i)], 'labels': [] if i == 0 else [i % 2]} for i in range(count)]
        (inputs / f'{split}.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    output = tmp_path / 'output'
    args = [sys.executable, str(ROOT / 'run_ecthr_a.py'), '--output', str(output), '--model', str(model_path),
            '--local-data', str(inputs), '--device', 'cpu', '--max-length', '16', '--shard-size', '2',
            '--k-values', '1', '2', '--thresholds', '.3', '.5']
    def run(arguments):
        return subprocess.run(arguments, capture_output=True, text=True,
                              env={**os.environ, 'HF_HUB_OFFLINE': '1', 'OMP_NUM_THREADS': '2'})
    first = run(args)
    assert first.returncode == 0, first.stderr
    result = runner.read_json(output / 'results.json')
    assert result['selection_split'] == 'validation'
    assert result['test']['examples'] == 2
    assert result['dataset_mode'] == 'local_fixture_or_custom_data'
    assert result['encoding_this_invocation']['train']['new_documents'] == 4
    assert runner.read_json(output / 'selection.json')['source_split'] == 'validation'
    assert (output / 'test-predictions.pt').exists()
    original_bytes = (output / 'results.json').read_bytes()
    second = run(args)
    assert second.returncode == 0 and 'already completed' in second.stdout
    assert (output / 'results.json').read_bytes() == original_bytes
    # Simulate an interruption after selecting parameters, before final results.
    (output / 'results.json').unlink()
    resumed = run(args)
    assert resumed.returncode == 0, resumed.stderr
    restored = runner.read_json(output / 'results.json')
    assert restored['test'] == result['test']
    assert all(v['new_documents'] == 0 for v in restored['encoding_this_invocation'].values())
    changed = run(args + ['--max-length', '24'])
    assert changed.returncode != 0 and 'settings or data changed' in changed.stderr.lower()
