"""Probability-based micro metrics for the PILOT evaluation protocol."""
import numpy as np


def ranking_metrics(targets, probabilities):
    """Pool all case/article decisions, including score ties.

    PR-AUC uses trapezoidal integration. Average precision is also returned,
    because these two definitions are distinct and papers may use either.
    """
    truth = np.asarray(targets)
    scores = np.asarray(probabilities, dtype=np.float64)
    if truth.ndim != 2 or scores.shape != truth.shape or truth.size == 0:
        raise ValueError('Ranking metrics require matching nonempty [cases, labels] arrays')
    if not np.isin(truth, [0, 1]).all() or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError('Require binary targets and finite probabilities between zero and one')
    truth, scores = truth.ravel().astype(np.int64), scores.ravel()
    order = np.argsort(-scores, kind='stable')
    truth, scores = truth[order], scores[order]
    ends = np.r_[np.flatnonzero(np.diff(scores)), len(scores) - 1]
    tp = np.cumsum(truth)[ends].astype(np.float64)
    fp = (ends + 1).astype(np.float64) - tp
    positives, negatives = int(truth.sum()), len(truth) - int(truth.sum())
    result = {'micro_pr_auc': None, 'micro_average_precision': None, 'micro_roc_auc': None,
              'micro_pr_auc_definition': 'Trapezoidal area under the pooled precision-recall curve',
              'micro_average_precision_definition': 'Recall-increment-weighted precision; distinct from trapezoidal PR-AUC'}
    notes = []
    if positives:
        recall, precision = tp / positives, tp / (tp + fp)
        recall, precision = np.r_[0.0, recall], np.r_[1.0, precision]
        result['micro_pr_auc'] = float(np.sum(np.diff(recall) * (precision[:-1] + precision[1:]) / 2))
        result['micro_average_precision'] = float(np.sum(np.diff(recall) * precision[1:]))
    else:
        notes.append('PR-AUC and average precision are undefined: no positive targets')
    if positives and negatives:
        tpr, fpr = np.r_[0.0, tp / positives], np.r_[0.0, fp / negatives]
        result['micro_roc_auc'] = float(np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) / 2))
    else:
        notes.append('ROC-AUC is undefined: both positive and negative targets are required')
    if notes:
        result['auc_notes'] = notes
    return result
