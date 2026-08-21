"""Clinical diagnostic metrics. Ref: Sec. III-I and Table 4."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def confusion_matrix(labels: Sequence[int], predictions: Sequence[int], classes: int) -> np.ndarray:
    if len(labels) != len(predictions) or not labels:
        raise ValueError("labels and predictions must be nonempty and equal")
    if classes <= 1:
        raise ValueError("class count must exceed one")
    matrix = np.zeros((classes, classes), dtype=np.int64)
    for label, prediction in zip(labels, predictions, strict=False):
        if not 0 <= label < classes or not 0 <= prediction < classes:
            raise ValueError("class identifier is out of range")
        matrix[label, prediction] += 1
    return matrix


def sensitivity_specificity(
    labels: Sequence[bool], predictions: Sequence[bool]
) -> tuple[float, float]:
    if len(labels) != len(predictions) or not labels:
        raise ValueError("binary labels and predictions must be nonempty and equal")
    true_positive = sum(
        label and prediction for label, prediction in zip(labels, predictions, strict=False)
    )
    false_negative = sum(
        label and not prediction for label, prediction in zip(labels, predictions, strict=False)
    )
    true_negative = sum(
        not label and not prediction for label, prediction in zip(labels, predictions, strict=False)
    )
    false_positive = sum(
        not label and prediction for label, prediction in zip(labels, predictions, strict=False)
    )
    sensitivity = (
        true_positive / (true_positive + false_negative) if true_positive + false_negative else 1.0
    )
    specificity = (
        true_negative / (true_negative + false_positive) if true_negative + false_positive else 1.0
    )
    return sensitivity, specificity


def binary_auc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    if len(labels) != len(scores) or not labels:
        raise ValueError("binary labels and scores must be nonempty and equal")
    positives = [score for label, score in zip(labels, scores, strict=False) if label]
    negatives = [score for label, score in zip(labels, scores, strict=False) if not label]
    if not positives or not negatives:
        raise ValueError("AUC requires both classes")
    favorable = 0.0
    for positive in positives:
        for negative in negatives:
            favorable += float(positive > negative) + 0.5 * float(positive == negative)
    return favorable / (len(positives) * len(negatives))


def macro_f1(labels: Sequence[int], predictions: Sequence[int], classes: int) -> float:
    matrix = confusion_matrix(labels, predictions, classes)
    scores: list[float] = []
    for index in range(classes):
        true_positive = matrix[index, index]
        false_positive = matrix[:, index].sum() - true_positive
        false_negative = matrix[index, :].sum() - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(float(2 * true_positive / denominator) if denominator else 0.0)
    return float(np.mean(scores))


def quadratic_weighted_kappa(
    labels: Sequence[int], predictions: Sequence[int], classes: int
) -> float:
    observed = confusion_matrix(labels, predictions, classes).astype(np.float64)
    observed /= observed.sum()
    label_histogram = observed.sum(axis=1)
    prediction_histogram = observed.sum(axis=0)
    expected = np.outer(label_histogram, prediction_histogram)
    indices = np.arange(classes, dtype=np.float64)
    weights = ((indices[:, None] - indices[None, :]) / (classes - 1)) ** 2
    observed_disagreement = float((weights * observed).sum())
    expected_disagreement = float((weights * expected).sum())
    return 1.0 - observed_disagreement / expected_disagreement if expected_disagreement else 1.0


def classification_metrics(
    labels: Sequence[int],
    predictions: Sequence[int],
    classes: int,
    positive_threshold: int | None = None,
) -> dict[str, float]:
    matrix = confusion_matrix(labels, predictions, classes)
    output = {
        "accuracy": float(np.trace(matrix) / matrix.sum()),
        "macro_f1": macro_f1(labels, predictions, classes),
        "quadratic_weighted_kappa": quadratic_weighted_kappa(labels, predictions, classes),
    }
    if positive_threshold is not None:
        binary_labels = [label >= positive_threshold for label in labels]
        binary_predictions = [prediction >= positive_threshold for prediction in predictions]
        sensitivity, specificity = sensitivity_specificity(binary_labels, binary_predictions)
        output["sensitivity"] = sensitivity
        output["specificity"] = specificity
    return output


def multiclass_one_vs_rest_auc(labels: Sequence[int], probabilities: np.ndarray) -> float:
    if probabilities.ndim != 2 or probabilities.shape[0] != len(labels):
        raise ValueError("probabilities must have shape samples x classes")
    classes = probabilities.shape[1]
    auc_values = [
        binary_auc([label == index for label in labels], probabilities[:, index].tolist())
        for index in range(classes)
    ]
    return float(np.mean(auc_values))


def positive_negative_predictive_value(
    labels: Sequence[bool], predictions: Sequence[bool]
) -> tuple[float, float]:
    if len(labels) != len(predictions) or not labels:
        raise ValueError("binary labels and predictions must be nonempty and equal")
    true_positive = sum(
        label and prediction for label, prediction in zip(labels, predictions, strict=False)
    )
    false_positive = sum(
        not label and prediction for label, prediction in zip(labels, predictions, strict=False)
    )
    true_negative = sum(
        not label and not prediction for label, prediction in zip(labels, predictions, strict=False)
    )
    false_negative = sum(
        label and not prediction for label, prediction in zip(labels, predictions, strict=False)
    )
    ppv = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else 1.0
    )
    npv = (
        true_negative / (true_negative + false_negative) if true_negative + false_negative else 1.0
    )
    return ppv, npv
