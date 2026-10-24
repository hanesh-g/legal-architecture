import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from legal_outcome.one_run import PROFILE
from legal_outcome.training import linear_schedule


def run(*args):
    process = subprocess.run([sys.executable, '-m', 'legal_outcome.one_run', *args], capture_output=True, text=True,
                             env={**os.environ, 'HF_HUB_OFFLINE': '1', 'OMP_NUM_THREADS': '2'})
    assert process.returncode == 0, process.stdout + process.stderr
    return process


def test_schedule_warmup_then_decay():
    weight = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([weight], lr=.1)
    scheduler = linear_schedule(optimizer, 10, .2)
    assert optimizer.param_groups[0]['lr'] == 0
    values = []
    for _ in range(10):
        weight.grad = torch.ones_like(weight)
        optimizer.step(); scheduler.step()
        values.append(optimizer.param_groups[0]['lr'])
    assert values[0] == pytest.approx(.05)
    assert values[1] == pytest.approx(.1)
    assert values[-1] == 0


def test_profile_names_strongest_and_encoder_references():
    assert PROFILE['primary_reference']['method'] == 'GPT-5.2'
    assert PROFILE['encoder_reference']['micro_f1'] == .7625
    assert PROFILE['max_content_tokens'] == 8064
    assert PROFILE['micro_batch_size'] * PROFILE['gradient_accumulation_steps'] == 8


def test_one_run_fixed_plan_preflight_interruption_resume_and_test(tmp_path):
    from transformers import BertConfig, BertModel, BertTokenizerFast
    from sentence_transformers import SentenceTransformer, models
    model = tmp_path / 'encoder'
    vocab = tmp_path / 'vocab.txt'
    vocab.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nthe\nevidence\ncase\nallow\ndeny\n')
    BertModel(BertConfig(vocab_size=10, hidden_size=16, num_hidden_layers=5, num_attention_heads=2,
                         intermediate_size=32, max_position_embeddings=128)).save_pretrained(model)
    BertTokenizerFast(vocab_file=str(vocab)).save_pretrained(model)
    semantic = tmp_path / 'semantic'
    SentenceTransformer(modules=[models.Transformer(str(model), max_seq_length=16), models.Pooling(16)]).save(str(semantic))
    source = tmp_path / 'source'; source.mkdir()
    for split, count in [('train', 4), ('validation', 2), ('test', 2)]:
        rows = [{'text': [f'the case {split} {i}', 'the evidence'], 'labels': [] if i == 0 else [i % 2]} for i in range(count)]
        (source / f'{split}.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    data, logs, output, results, cache = [tmp_path / x for x in ['data', 'logs', 'output', 'results', 'cache']]
    common = ['--data', str(data), '--logs', str(logs), '--output', str(output), '--results', str(results),
              '--semantic-cache', str(cache), '--device', 'cpu']
    run('prepare', *common, '--local-data', str(source), '--legalbert-model', str(model), '--minilm-model', str(semantic))
    manifest = json.loads((data / 'dataset-manifest.json').read_text())
    assert manifest['max_packed_tokens'] == 8192 and manifest['chunk_tokens'] == 128
    assert manifest['max_content_tokens'] == 8064
    run('preflight', *common)
    check = json.loads((logs / 'ecthr-a-one-run-preflight.json').read_text())
    assert check['content_tokens'] == 8064 and check['optimizer_step_performed'] is False
    # Terminate a real process after its epoch-3 save, keeping the fixed eight-epoch schedule.
    args = [sys.executable, '-u', '-m', 'legal_outcome.one_run', 'train', *common]
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                               env={**os.environ, 'HF_HUB_OFFLINE': '1', 'OMP_NUM_THREADS': '2'})
    output_lines = []
    for line in process.stdout:
        output_lines.append(line)
        if 'Checkpoint saved after epoch 3:' in line:
            process.terminate()
            break
    process.wait(timeout=30)
    assert any('Checkpoint saved after epoch 3:' in line for line in output_lines), ''.join(output_lines)
    saved = torch.load(output / 'training-state.pt', weights_only=True)
    assert saved['completed_epoch'] == 3
    assert saved['scheduler'] is not None
    run('train', *common, '--resume')
    metadata = json.loads((output / 'metadata.json').read_text())
    history = metadata['training_report']['history']
    assert len(history) == 8 and [r['epoch'] for r in history] == list(range(1, 9))
    assert all(r['optimizer_updates'] == 1 for r in history)
    assert metadata['training_report']['effective_batch_size'] == 8
    assert metadata['training_report']['lr_schedule'] == 'linear'
    assert history[-1]['optimizer_group_lrs'] == [0.0] * len(history[-1]['optimizer_group_lrs'])
    # Scheduler, accumulation, dropout, and best weights match an uninterrupted run.
    uninterrupted = tmp_path / 'uninterrupted'
    run('train', *common, '--output', str(uninterrupted))
    baseline_state = torch.load(uninterrupted / 'training-state.pt', weights_only=True)
    resumed_state = torch.load(output / 'training-state.pt', weights_only=True)
    assert baseline_state['history'] == resumed_state['history']
    assert baseline_state['scheduler'] == resumed_state['scheduler']
    for name, value in baseline_state['model'].items():
        assert torch.equal(value, resumed_state['model'][name]), name
    run('test', *common)
    result = json.loads((results / 'test-results.json').read_text())
    assert result['max_content_tokens'] == 8064 and result['chunk_tokens'] == 128
    assert result['input_variant'] == 'long_document_chunked_extension'
    comparison = json.loads((results / 'one-run-comparison.json').read_text())
    assert comparison['primary_reference']['method'] == 'GPT-5.2'
    assert not comparison['statistical_significance_established']
    previous = (results / 'test-results.json').read_bytes()
    again = run('test', *common)
    assert 'No new test evaluation' in again.stdout
    assert (results / 'test-results.json').read_bytes() == previous
