"""Patient-clustered logistic GEE. Ref: Sec. III-G."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

G = TypeVar("G", bound=Hashable)


@dataclass(frozen=True)
class GEEResult:
    coefficients: np.ndarray
    covariance: np.ndarray
    standard_errors: np.ndarray
    correlation: float
    iterations: int
    converged: bool

    def __post_init__(self) -> None:
        dimension = len(self.coefficients)
        if self.covariance.shape != (dimension, dimension):
            raise ValueError("GEE covariance shape mismatch")
        if self.standard_errors.shape != (dimension,):
            raise ValueError("GEE standard-error shape mismatch")
        if not -1 <= self.correlation <= 1:
            raise ValueError("working correlation must be in [-1, 1]")

    def confidence_intervals(self, z_value: float = 1.959963984540054) -> np.ndarray:
        lower = self.coefficients - z_value * self.standard_errors
        upper = self.coefficients + z_value * self.standard_errors
        return np.stack((lower, upper), axis=-1)

    def odds_ratios(self) -> np.ndarray:
        return np.exp(self.coefficients)


def logistic(values: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    negative_exp = np.exp(values[~positive])
    output[~positive] = negative_exp / (1.0 + negative_exp)
    return output


def group_rows(groups: Sequence[G]) -> tuple[np.ndarray, ...]:
    mapping: dict[G, list[int]] = {}
    for index, group in enumerate(groups):
        mapping.setdefault(group, []).append(index)
    return tuple(np.asarray(indices, dtype=np.int64) for indices in mapping.values())


def exchangeable_correlation(size: int, correlation: float) -> np.ndarray:
    if size <= 0:
        raise ValueError("cluster size must be positive")
    lower_bound = -1.0 / max(1, size - 1)
    if not lower_bound < correlation < 1:
        raise ValueError("correlation is outside the positive-definite range")
    matrix = np.full((size, size), correlation, dtype=np.float64)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def estimate_exchangeable_correlation(
    labels: np.ndarray,
    means: np.ndarray,
    clusters: Sequence[np.ndarray],
    parameter_count: int,
) -> float:
    variance = np.clip(means * (1.0 - means), 1e-8, None)
    residuals = (labels - means) / np.sqrt(variance)
    numerator = 0.0
    pairs = 0
    for rows in clusters:
        cluster_residuals = residuals[rows]
        if len(rows) < 2:
            continue
        outer = np.outer(cluster_residuals, cluster_residuals)
        numerator += float(np.triu(outer, k=1).sum())
        pairs += len(rows) * (len(rows) - 1) // 2
    denominator = max(1, pairs - parameter_count)
    estimate = numerator / denominator
    maximum_cluster = max(len(rows) for rows in clusters)
    lower = -1.0 / max(1, maximum_cluster - 1) + 1e-6
    return float(np.clip(estimate, lower, 0.95))


def working_covariance(means: np.ndarray, correlation: float) -> np.ndarray:
    standard = np.sqrt(np.clip(means * (1.0 - means), 1e-8, None))
    correlation_matrix = exchangeable_correlation(len(means), correlation)
    return standard[:, None] * correlation_matrix * standard[None, :]


def estimating_equation(
    design: np.ndarray,
    labels: np.ndarray,
    coefficients: np.ndarray,
    clusters: Sequence[np.ndarray],
    correlation: float,
) -> tuple[np.ndarray, np.ndarray]:
    score = np.zeros(design.shape[1], dtype=np.float64)
    information = np.zeros((design.shape[1], design.shape[1]), dtype=np.float64)
    means = logistic(design @ coefficients)
    for rows in clusters:
        cluster_design = design[rows]
        cluster_means = means[rows]
        derivatives = cluster_means * (1.0 - cluster_means)
        derivative_matrix = cluster_design * derivatives[:, None]
        covariance = working_covariance(cluster_means, correlation)
        inverse = np.linalg.pinv(covariance)
        residual = labels[rows] - cluster_means
        score += derivative_matrix.T @ inverse @ residual
        information += derivative_matrix.T @ inverse @ derivative_matrix
    return score, information


def sandwich_covariance(
    design: np.ndarray,
    labels: np.ndarray,
    coefficients: np.ndarray,
    clusters: Sequence[np.ndarray],
    correlation: float,
) -> np.ndarray:
    _, information = estimating_equation(design, labels, coefficients, clusters, correlation)
    bread = np.linalg.pinv(information)
    meat = np.zeros_like(information)
    means = logistic(design @ coefficients)
    for rows in clusters:
        cluster_design = design[rows]
        cluster_means = means[rows]
        derivatives = cluster_means * (1.0 - cluster_means)
        derivative_matrix = cluster_design * derivatives[:, None]
        covariance = working_covariance(cluster_means, correlation)
        inverse = np.linalg.pinv(covariance)
        residual = labels[rows] - cluster_means
        cluster_score = derivative_matrix.T @ inverse @ residual
        meat += np.outer(cluster_score, cluster_score)
    return bread @ meat @ bread


def fit_logistic_gee(
    design: np.ndarray,
    labels: np.ndarray,
    groups: Sequence[G],
    maximum_iterations: int = 100,
    tolerance: float = 1e-8,
) -> GEEResult:
    design = np.asarray(design, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if design.ndim != 2 or labels.shape != design.shape[:1] or len(groups) != len(labels):
        raise ValueError("GEE inputs have incompatible shapes")
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError("logistic GEE labels must be binary")
    if maximum_iterations <= 0 or tolerance <= 0:
        raise ValueError("GEE optimization controls must be positive")
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("GEE design matrix is rank deficient")
    clusters = group_rows(groups)
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    correlation = 0.0
    converged = False
    iteration = 0
    for current_iteration in range(1, maximum_iterations + 1):
        iteration = current_iteration
        score, information = estimating_equation(
            design, labels, coefficients, clusters, correlation
        )
        update = np.linalg.pinv(information) @ score
        coefficients += update
        means = logistic(design @ coefficients)
        correlation = estimate_exchangeable_correlation(labels, means, clusters, design.shape[1])
        if np.max(np.abs(update)) < tolerance:
            converged = True
            break
    covariance = sandwich_covariance(design, labels, coefficients, clusters, correlation)
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    return GEEResult(coefficients, covariance, standard_errors, correlation, iteration, converged)


def intercept_and_treatment(treatment: Sequence[bool]) -> np.ndarray:
    values = np.asarray(treatment, dtype=np.float64)
    return np.column_stack((np.ones(len(values), dtype=np.float64), values))


def predict_probability(result: GEEResult, design: np.ndarray) -> np.ndarray:
    design = np.asarray(design, dtype=np.float64)
    if design.ndim != 2 or design.shape[1] != len(result.coefficients):
        raise ValueError("prediction design shape mismatch")
    return logistic(design @ result.coefficients)
