"""
Evaluation Metrics (Phase 5)
============================

Standard metrics for evaluating simulation accuracy, drug predictions,
and model performance.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union
import numpy as np


def mean_absolute_error(
    y_true: Union[List[float], np.ndarray],
    y_pred: Union[List[float], np.ndarray]
) -> float:
    """Calculate Mean Absolute Error (MAE).

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        MAE score (lower is better)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mean_squared_error(
    y_true: Union[List[float], np.ndarray],
    y_pred: Union[List[float], np.ndarray],
    squared: bool = True
) -> float:
    """Calculate Mean Squared Error (MSE) or Root MSE.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        squared: If True, return MSE; if False, return RMSE

    Returns:
        MSE or RMSE score (lower is better)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mse = float(np.mean((y_true - y_pred) ** 2))
    return mse if squared else math.sqrt(mse)


def correlation_coefficient(
    y_true: Union[List[float], np.ndarray],
    y_pred: Union[List[float], np.ndarray]
) -> float:
    """Calculate Pearson correlation coefficient.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        Correlation coefficient (-1 to 1, higher is better)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) < 2:
        return 0.0

    return float(np.corrcoef(y_true, y_pred)[0, 1])


def spearman_correlation(
    y_true: Union[List[float], np.ndarray],
    y_pred: Union[List[float], np.ndarray]
) -> float:
    """Calculate Spearman rank correlation coefficient.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        Spearman correlation (-1 to 1, higher is better)
    """
    from scipy import stats
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(stats.spearmanr(y_true, y_pred).correlation)


def concordance_index(
    y_true: Union[List[float], np.ndarray],
    y_pred: Union[List[float], np.ndarray]
) -> float:
    """Calculate concordance index (C-index) for survival/ranking.

    The C-index measures how well the predicted scores rank the samples
    compared to the true outcomes.

    Args:
        y_true: Ground truth values (e.g., survival times, IC50)
        y_pred: Predicted values/scores

    Returns:
        C-index (0.5 = random, 1.0 = perfect concordance)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n = len(y_true)
    if n < 2:
        return 0.5

    concordant = 0
    discordant = 0
    tied = 0

    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] != y_true[j]:
                if (y_pred[i] > y_pred[j]) == (y_true[i] > y_true[j]):
                    concordant += 1
                elif (y_pred[i] < y_pred[j]) == (y_true[i] < y_true[j]):
                    concordant += 1
                else:
                    discordant += 1
            else:
                tied += 1

    total = concordant + discordant
    if total == 0:
        return 0.5

    return concordant / total


def ndcg_score(
    y_true: Union[List[float], np.ndarray],
    y_pred: Union[List[float], np.ndarray],
    k: Optional[int] = None
) -> float:
    """Calculate Normalized Discounted Cumulative Gain (NDCG).

    NDCG measures ranking quality, giving higher weight to top positions.

    Args:
        y_true: Ground truth relevance scores
        y_pred: Predicted scores (used for ranking)
        k: Number of top positions to consider (None = all)

    Returns:
        NDCG score (0 to 1, higher is better)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Get ranking by predicted scores (descending)
    pred_order = np.argsort(-y_pred)
    true_at_pred_order = y_true[pred_order]

    # Get ideal ranking
    ideal_order = np.argsort(-y_true)
    true_at_ideal_order = y_true[ideal_order]

    if k is not None:
        true_at_pred_order = true_at_pred_order[:k]
        true_at_ideal_order = true_at_ideal_order[:k]

    # Calculate DCG
    def dcg(relevances):
        positions = np.arange(1, len(relevances) + 1)
        return np.sum(relevances / np.log2(positions + 1))

    dcg_pred = dcg(true_at_pred_order)
    dcg_ideal = dcg(true_at_ideal_order)

    if dcg_ideal == 0:
        return 0.0

    return float(dcg_pred / dcg_ideal)


def precision_at_k(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[float], np.ndarray],
    k: int = 10,
    threshold: float = 0.5
) -> float:
    """Calculate Precision@K for binary relevance.

    Args:
        y_true: Binary relevance labels (0 or 1)
        y_pred: Predicted scores (used for ranking)
        k: Number of top positions to consider
        threshold: Threshold for converting y_true to binary if needed

    Returns:
        Precision@K (0 to 1, higher is better)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Convert to binary if needed
    if y_true.dtype != bool and not np.all((y_true == 0) | (y_true == 1)):
        y_true = (y_true >= threshold).astype(int)

    # Get top-k by predicted score
    top_k_indices = np.argsort(-y_pred)[:k]
    relevant_in_top_k = y_true[top_k_indices].sum()

    return float(relevant_in_top_k / k)


def recall_at_k(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[float], np.ndarray],
    k: int = 10,
    threshold: float = 0.5
) -> float:
    """Calculate Recall@K for binary relevance.

    Args:
        y_true: Binary relevance labels (0 or 1)
        y_pred: Predicted scores (used for ranking)
        k: Number of top positions to consider
        threshold: Threshold for converting y_true to binary if needed

    Returns:
        Recall@K (0 to 1, higher is better)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Convert to binary if needed
    if y_true.dtype != bool and not np.all((y_true == 0) | (y_true == 1)):
        y_true = (y_true >= threshold).astype(int)

    total_relevant = y_true.sum()
    if total_relevant == 0:
        return 0.0

    # Get top-k by predicted score
    top_k_indices = np.argsort(-y_pred)[:k]
    relevant_in_top_k = y_true[top_k_indices].sum()

    return float(relevant_in_top_k / total_relevant)


def mean_average_precision(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[float], np.ndarray]
) -> float:
    """Calculate Mean Average Precision (MAP).

    Args:
        y_true: Binary relevance labels (0 or 1)
        y_pred: Predicted scores (used for ranking)

    Returns:
        MAP score (0 to 1, higher is better)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Sort by predicted score
    order = np.argsort(-y_pred)
    y_true_sorted = y_true[order]

    # Calculate precision at each relevant item
    precisions = []
    relevant_count = 0

    for i, rel in enumerate(y_true_sorted):
        if rel == 1:
            relevant_count += 1
            precisions.append(relevant_count / (i + 1))

    if not precisions:
        return 0.0

    return float(np.mean(precisions))


def r_squared(
    y_true: Union[List[float], np.ndarray],
    y_pred: Union[List[float], np.ndarray]
) -> float:
    """Calculate R-squared (coefficient of determination).

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        R-squared (can be negative, 1.0 is perfect)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0

    return float(1 - (ss_res / ss_tot))


def percent_within_fold(
    y_true: Union[List[float], np.ndarray],
    y_pred: Union[List[float], np.ndarray],
    fold: float = 2.0
) -> float:
    """Calculate percentage of predictions within X-fold of true value.

    Common in drug response prediction (e.g., within 2-fold of IC50).

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        fold: Fold threshold (e.g., 2.0 for 2-fold)

    Returns:
        Percentage within fold (0 to 100)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Handle zeros
    y_true = np.maximum(y_true, 1e-10)
    y_pred = np.maximum(y_pred, 1e-10)

    # Calculate fold difference
    fold_diff = np.maximum(y_pred / y_true, y_true / y_pred)
    within_fold = fold_diff <= fold

    return float(np.mean(within_fold) * 100)


def enrichment_factor(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[float], np.ndarray],
    fraction: float = 0.01
) -> float:
    """Calculate Enrichment Factor for virtual screening.

    EF measures how many more actives are found in the top fraction
    compared to random selection.

    Args:
        y_true: Binary activity labels (0 or 1)
        y_pred: Predicted scores (higher = more likely active)
        fraction: Top fraction to consider (e.g., 0.01 = top 1%)

    Returns:
        Enrichment factor (1.0 = random, higher is better)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n_total = len(y_true)
    n_actives = y_true.sum()

    if n_actives == 0:
        return 0.0

    # Get top fraction
    n_top = max(1, int(n_total * fraction))
    top_indices = np.argsort(-y_pred)[:n_top]
    actives_in_top = y_true[top_indices].sum()

    # Expected actives in random selection
    expected = n_actives * fraction

    if expected == 0:
        return 0.0

    return float(actives_in_top / expected)


def auc_roc(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[float], np.ndarray]
) -> float:
    """Calculate Area Under ROC Curve.

    Args:
        y_true: Binary labels (0 or 1)
        y_pred: Predicted probabilities/scores

    Returns:
        AUC-ROC (0.5 = random, 1.0 = perfect)
    """
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_true, y_pred))


def auc_prc(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[float], np.ndarray]
) -> float:
    """Calculate Area Under Precision-Recall Curve.

    Better than AUC-ROC for imbalanced datasets.

    Args:
        y_true: Binary labels (0 or 1)
        y_pred: Predicted probabilities/scores

    Returns:
        AUC-PRC (higher is better)
    """
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y_true, y_pred))


def cell_count_accuracy(
    observed: Union[List[int], np.ndarray],
    simulated: Union[List[int], np.ndarray],
    time_points: Optional[Union[List[float], np.ndarray]] = None
) -> dict:
    """Calculate cell count accuracy metrics over time.

    Args:
        observed: Observed cell counts
        simulated: Simulated cell counts
        time_points: Time points for each measurement

    Returns:
        Dictionary of accuracy metrics
    """
    observed = np.asarray(observed)
    simulated = np.asarray(simulated)

    # Avoid division by zero
    observed_safe = np.maximum(observed, 1)

    # Percent error at each point
    percent_errors = np.abs(simulated - observed) / observed_safe * 100

    return {
        "mae": mean_absolute_error(observed, simulated),
        "rmse": mean_squared_error(observed, simulated, squared=False),
        "correlation": correlation_coefficient(observed, simulated),
        "r_squared": r_squared(observed, simulated),
        "mean_percent_error": float(np.mean(percent_errors)),
        "max_percent_error": float(np.max(percent_errors)),
        "within_10pct": float(np.mean(percent_errors <= 10) * 100),
        "within_20pct": float(np.mean(percent_errors <= 20) * 100),
    }
