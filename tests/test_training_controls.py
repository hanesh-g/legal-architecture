import pytest
import torch
from torch import nn
from dataclasses import replace
from legal_outcome import ModelConfig, LegalOutcomeModel
from legal_outcome.data import DocumentProcessor, collate_documents
from legal_outcome.runtime import metrics, read_records, prepare_records
from legal_outcome.training import CheckpointSelector, make_optimizer, positive_weights, check_split_overlap
from test_architecture import Encoder, Tokenizer, SemanticEncoder


def test_multilabel_sigmoid_loss_and_gradients():
    config = ModelConfig(('2', '3', '6'), task_type='multi_label', dropout=0)
    model = LegalOutcomeModel(config, Encoder())
    processor = DocumentProcessor(Tokenizer(), config)
    records = [{'text': 'appeal evidence', 'labels': ['2', '6']}, {'text': 'dismiss', 'labels': []}]
    batch = collate_documents(prepare_records(records, processor), SemanticEncoder())
    assert batch['labels'].tolist() == [[1., 0., 1.], [0., 0., 0.]]
    output = model(**batch)
    assert torch.allclose(output['probabilities'], output['logits'].sigmoid())
    assert torch.equal(output['predictions'], output['probabilities'] >= 0.5)
    assert torch.allclose(output['loss'], nn.functional.binary_cross_entropy_with_logits(output['logits'], batch['labels']))
    output['loss'].backward()
    assert model.head_weight.grad.abs().sum() > 0
    batch['labels'][0, 0] = 2
    with pytest.raises(ValueError, match='binary'):
        model(**batch)


def test_multilabel_known_metrics():
    result = metrics([[1, 1], [0, 1], [0, 0]], [[1, 0], [1, 1], [0, 0]], ['a', 'b'], 'multi_label')
    assert result['micro_precision'] == pytest.approx(2 / 3)
    assert result['micro_recall'] == pytest.approx(2 / 3)
    assert result['macro_f1'] == pytest.approx(2 / 3)
    assert result['exact_match_accuracy'] == pytest.approx(1 / 3)
    assert result['hamming_loss'] == pytest.approx(2 / 6)
    assert result['per_label']['a']['precision'] == 0.5
    assert 'accuracy' not in result


def test_empty_predicted_positive_metrics_are_finite():
    result = metrics([[1, 0]], [[0, 0]], ['a', 'b'], 'multi_label')
    assert result['micro_precision'] == result['micro_recall'] == result['macro_f1'] == 0


@pytest.mark.parametrize('metric,values,selected', [('macro_f1', [.6, .8, .7], 2), ('loss', [.7, .3, .4], 2), ('last', [0, 0, 0], 3)])
def test_selection_restores_best_not_last(metric, values, selected):
    model = nn.Linear(1, 1, bias=False)
    selector = CheckpointSelector(metric)
    for epoch, value in enumerate(values, 1):
        with torch.no_grad():
            model.weight.fill_(epoch)
        selector.update(model, epoch, {metric: value})
    assert selector.best_epoch == selected
    selector.restore(model)
    assert model.weight.item() == selected


def test_optimizer_rates_and_bias_decay():
    model = LegalOutcomeModel(ModelConfig(('a', 'b')), Encoder())
    optimizer = make_optimizer(model, 1e-5, 1e-4, .02)
    for group in optimizer.param_groups:
        for parameter in group['params']:
            name = next(n for n, p in model.named_parameters() if p is parameter)
            assert group['lr'] == (1e-5 if name.startswith('encoder.') else 1e-4)
            if parameter.ndim == 1:
                assert group['weight_decay'] == 0


def test_capped_weights_from_training_only():
    weights = positive_weights([{'label': [1, 1]}] + [{'label': [1, 0]}] * 9, cap=3)
    assert weights == (1, 3)
    with pytest.raises(ValueError, match='every class'):
        positive_weights([{'label': [1, 0]}], cap=3)


def test_split_overlap_checks_case_ids_and_normalized_text():
    with pytest.raises(ValueError, match='overlap'):
        check_split_overlap([{'text': 'Case  facts'}], [{'text': 'case facts'}])
    with pytest.raises(ValueError, match='overlap'):
        check_split_overlap([{'text': 'a', 'case_id': '12'}], [{'text': 'b', 'case_id': '12'}])
    check_split_overlap([{'text': 'a', 'case_id': '12'}], [{'text': 'b', 'case_id': '13'}])


def test_multilabel_reader(tmp_path):
    path = tmp_path / 'cases.jsonl'
    path.write_text('{"text":"case", "labels":["2", "6"]}')
    assert len(read_records(path, ['2', '6'], 'multi_label')) == 1
    path.write_text('{"text":"case", "labels":["2", "2"]}')
    with pytest.raises(ValueError, match='distinct'):
        read_records(path, ['2', '6'], 'multi_label')


def test_config_rejects_invalid_weights_and_threshold():
    with pytest.raises(ValueError):
        ModelConfig(('a', 'b'), task_type='multi_label', positive_weights=(1, float('nan')))
    with pytest.raises(ValueError):
        ModelConfig(('a', 'b'), threshold=1)
