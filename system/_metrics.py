from __future__ import annotations

import math
from typing import Sequence

import torchmetrics

__all__ = ["Accuracy", "Precision", "Recall", "F1Score", "std", "avg", "qoi", "entropy", "jain_index",
           "MetricCollection"]

# performance metrics evaluated at client
Accuracy = torchmetrics.Accuracy
Precision = torchmetrics.Precision
Recall = torchmetrics.Recall
F1Score = torchmetrics.F1Score
MeanMetric = torchmetrics.MeanMetric
MetricCollection = torchmetrics.MetricCollection


# common math function
def std(values: Sequence):
    n = len(values)
    avg_ = sum(values) / n
    return math.sqrt(sum([(value - avg_) ** 2 for value in values]) / n)


def avg(values: Sequence):
    return sum(values) / len(values)


# fairness metrics evaluated at _server
def qoi(personal_perf: float, global_perf: float, local_perf: float | None, increment: bool) -> float:
    """
    Quantum of Improvement

    Reference: S Divi, Y S Lin, H Farrukh, et al. New Metrics to Evaluate the Performance and Fairness of Personalized
    Federated Learning[C]. International Workshop on Federated Learning for User Privacy and Data Confidentiality
    in Conjunction with ICML, 2021.
    """
    if local_perf is None:
        local_perf = global_perf

    if increment:
        return personal_perf - max(global_perf, local_perf)
    else:
        return - (personal_perf - min(global_perf, local_perf))


def entropy(qois: Sequence[float]) -> float:
    """
    Reference: S Divi, Y S Lin, H Farrukh, et al. New Metrics to Evaluate the Performance and Fairness of Personalized
    Federated Learning[C]. International Workshop on Federated Learning for User Privacy and Data Confidentiality
    in Conjunction with ICML, 2021.
    """
    sum_qois = sum(qois)
    _entropy = - sum([(_qoi / sum_qois) * math.log(_qoi / sum_qois) for _qoi in qois])
    return _entropy


def jain_index(qois: Sequence[float]) -> float:
    """
    Jain's Index of QoI,  ``1 / K <= JI <= 1``

    Reference: S Divi, Y S Lin, H Farrukh, et al. New Metrics to Evaluate the Performance and Fairness of Personalized
    Federated Learning[C]. International Workshop on Federated Learning for User Privacy and Data Confidentiality
    in Conjunction with ICML, 2021.
    """
    if not qois:
        return 0.0
    k = len(qois)
    ji = (sum(qois) ** 2) / (k * sum([_qoi ** 2 for _qoi in qois]))
    return ji
