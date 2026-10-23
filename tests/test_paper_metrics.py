import numpy as np
import pytest
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, average_precision_score
from legal_outcome.paper_metrics import ranking_metrics
from legal_outcome.runtime import metrics


@pytest.mark.parametrize('seed', [1, 42, 99])
def test_micro_auc_matches_reference_with_ties(seed):
    rng = np.random.default_rng(seed)
    targets = rng.integers(0, 2, size=(7, 4))
    scores = rng.choice([0.1, 0.4, 0.5, 0.8], size=(7, 4))
    actual = ranking_metrics(targets, scores)
    p, r, _ = precision_recall_curve(targets.ravel(), scores.ravel())
    assert actual['micro_pr_auc'] == pytest.approx(auc(r, p))
    assert actual['micro_roc_auc'] == pytest.approx(roc_auc_score(targets.ravel(), scores.ravel()))
    assert actual['micro_average_precision'] == pytest.approx(average_precision_score(targets.ravel(), scores.ravel()))


def test_micro_jaccard_and_f1_relation():
    result = metrics([[1, 1], [0, 1]], [[1, 0], [1, 1]], ['a', 'b'], 'multi_label')
    assert result['micro_jaccard'] == pytest.approx(.5)
    assert result['micro_jaccard'] == pytest.approx(result['micro_f1'] / (2 - result['micro_f1']))


def test_probability_metrics_do_not_depend_on_threshold():
    result = ranking_metrics([[1, 0]], [[0.6, 0.4]])
    assert result['micro_roc_auc'] == result['micro_pr_auc'] == 1


def test_constant_scores_and_degenerate_targets():
    result = ranking_metrics([[1, 0]], [[.5, .5]])
    assert result['micro_roc_auc'] == .5
    assert result['micro_average_precision'] == .5
    assert result['micro_pr_auc'] == .75
    result = ranking_metrics([[0, 0]], [[.1, .2]])
    assert result['micro_roc_auc'] is None and result['micro_pr_auc'] is None
    assert 'auc_notes' in result
    assert ranking_metrics([[1, 1]], [[.1, .2]])['micro_roc_auc'] is None


@pytest.mark.parametrize('truth,score', [([[1, 0]], [[.3]]), ([[1, 2]], [[.3,.4]]), ([[1,0]], [[float('nan'),.4]]), ([[1,0]], [[1.1,.4]])])
def test_invalid_score_inputs(truth, score):
    with pytest.raises(ValueError):
        ranking_metrics(truth, score)
