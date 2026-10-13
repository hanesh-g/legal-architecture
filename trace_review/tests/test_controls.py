import numpy as np
import pytest
import torch
from torch import nn
from train import calibrate_thresholds, metrics, probability_loss, selection_improved, validate_splits, make_neighbors, GraphCrossAttention, optimizer_for


def test_rare_thresholds_keep_default():
    truth = np.zeros((30, 2)); truth[0, 0] = 1; truth[:20, 1] = 1
    probs = np.full((30, 2), .2)
    probs[:20, 1] = .4
    result = calibrate_thresholds(probs, truth, min_support=5)
    assert result[0] == .5
    assert result[1] < .5


def test_threshold_tie_prefers_half():
    assert calibrate_thresholds(np.array([[.9], [.1]]), np.array([[1], [0]]), 1)[0] == .5


def test_known_metrics_and_loss():
    truth = np.array([[1, 1], [0, 1], [0, 0]])
    probs = np.array([[.8, .2], [.8, .8], [.1, .1]])
    result = metrics(probs, truth, np.array([.5, .5]), ['a', 'b'])
    assert result['micro_precision'] == pytest.approx(2/3)
    assert result['micro_recall'] == pytest.approx(2/3)
    assert result['exact_match_accuracy'] == pytest.approx(1/3)
    assert result['hamming_loss'] == pytest.approx(1/3)
    assert np.isfinite(probability_loss(probs, truth))
    assert np.isfinite(probability_loss(np.array([[0, 1]]), np.array([[1, 0]])))


def test_selection_metric_direction():
    assert selection_improved(.7, .6, 'micro_f1')
    assert not selection_improved(.5, .6, 'micro_f1')
    assert selection_improved(.2, .3, 'val_loss')
    assert not selection_improved(.3, .3, 'val_loss')


def test_normalized_split_overlap():
    with pytest.raises(ValueError, match='Identical'):
        validate_splits({'train': [{'id': '1', 'text': 'A  CASE'}], 'test': [{'id': '2', 'text': 'a case'}]})


def test_temporal_graph_excludes_future_and_self():
    rows = [{'date_ord': 10}, {'date_ord': 20}, {'date_ord': 30}]
    vectors = torch.eye(3)
    result = make_neighbors(rows, rows, vectors, vectors, 2, True)
    assert result['idx'][0].tolist() == [-1, -1]
    assert result['idx'][1].tolist() == [0, -1]
    assert set(result['idx'][2].tolist()) == {0, 1}


def test_no_neighbor_case_unchanged_in_mixed_batch():
    layer = GraphCrossAttention(8, 2, 0).eval()
    case = torch.randn(2, 8)
    result = layer(case, torch.randn(2, 2, 8), torch.tensor([[False, False], [True, False]]),
                   torch.zeros(2, 2), torch.zeros(2, 2))
    assert torch.equal(result[0], case[0])
    assert torch.isfinite(result).all()


def test_optimizer_groups():
    model = nn.Module()
    model.encoder = nn.Module(); model.encoder.bert = nn.Linear(3, 3)
    model.classifier = nn.Linear(3, 2)
    optimizer = optimizer_for(model, 1e-5, 1e-4, .01)
    for group in optimizer.param_groups:
        for parameter in group['params']:
            name = next(n for n, p in model.named_parameters() if p is parameter)
            assert group['lr'] == (1e-5 if name.startswith('encoder.bert.') else 1e-4)
            if parameter.ndim == 1: assert group['weight_decay'] == 0
