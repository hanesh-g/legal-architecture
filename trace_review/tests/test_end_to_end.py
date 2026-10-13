import json
import os
import subprocess
import sys
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast


def test_eight_epoch_training_with_local_encoder(tmp_path):
    encoder = tmp_path / 'encoder'
    vocab = tmp_path / 'vocab.txt'
    vocab.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ncase\nappeal\nevidence\nallowed\ndenied\n')
    BertModel(BertConfig(vocab_size=10, hidden_size=16, num_hidden_layers=1,
                         num_attention_heads=2, intermediate_size=32)).save_pretrained(encoder)
    BertTokenizerFast(vocab_file=str(vocab)).save_pretrained(encoder)
    data = tmp_path / 'data'; data.mkdir()
    for name, year, count in [('train', 2000, 4), ('val', 2001, 2), ('test', 2002, 2)]:
        records = [{'id': f'{name}-{i}', 'date': f'{year}-01-0{i+1}',
                    'text': f'case {name} {i} evidence '+('allowed' if i % 2 else 'denied'),
                    'labels': ['6'] if i % 2 else ['3']} for i in range(count)]
        (data / f'{name}.jsonl').write_text('\n'.join(json.dumps(x) for x in records))
    output = tmp_path / 'run'
    process = subprocess.run([sys.executable, 'train.py', '--data_dir', str(data), '--output_dir', str(output),
                              '--encoder_model', str(encoder), '--epochs', '8', '--max_chunks', '2', '--max_length', '16',
                              '--batch_size', '2', '--heads', '2', '--neighbors', '2', '--report_train_metrics',
                              '--selection_metric', 'micro_f1', '--encoder_dropout', '.2', '--freeze_lower_layers', '1',
                              '--head_lr', '.001', '--calibration_min_support', '2'], capture_output=True, text=True,
                             env={**os.environ, 'HF_HUB_OFFLINE': '1', 'OMP_NUM_THREADS': '2'})
    assert process.returncode == 0, process.stderr
    history = json.loads((output / 'history.json').read_text())
    assert len(history) == 8
    assert all('val_loss' in r and 'train_eval_loss' in r for r in history)
    expected = max(history, key=lambda r: r['val_micro_f1'])['epoch']
    checkpoint = torch.load(output / 'best.pt', weights_only=True)
    report = json.loads((output / 'metrics.json').read_text())
    assert checkpoint['epoch'] == report['selected_epoch'] == expected
    assert report['selection_metric'] == 'micro_f1'
    assert set(['micro_precision', 'micro_recall', 'micro_f1', 'macro_f1', 'exact_match_accuracy']) <= set(report['test'])
    assert (output / 'validation_predictions.npz').exists()
    assert (output / 'test_predictions.npz').exists()
