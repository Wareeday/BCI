"""
ml/evaluate.py
===============
Model evaluation utilities — confusion matrix, per-class accuracy,
latency profiling, and BCI Competition IV benchmark reporting.

Used by:
  - ml/train.py (post-training evaluation)
  - scripts/validate_iso14155.py (Phase 3 pilot results)
"""
import time
import numpy as np
from typing import Optional
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, roc_auc_score,
)
from loguru import logger


MI_CLASS_NAMES = ["left", "right", "feet", "rest"]
P300_CLASS_NAMES = ["non-target", "target"]


def evaluate_classifier(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    paradigm: str = "motor_imagery",
    n_inference_trials: int = 100,
) -> dict:
    """
    Full evaluation of a BCI classifier.

    Args:
        model:    Any model with a .predict() method
        X_test:   (n_trials, n_features) or (n_trials, n_ch, n_samples)
        y_test:   (n_trials,) integer labels
        paradigm: 'motor_imagery' or 'p300'

    Returns:
        dict with accuracy, confusion matrix, per-class metrics, latency
    """
    class_names = MI_CLASS_NAMES if paradigm == "motor_imagery" else P300_CLASS_NAMES

    # ── Predictions ───────────────────────────────────────────────
    y_pred = []
    y_probs = []
    for i in range(len(X_test)):
        pred, probs, _ = model.predict(X_test[i])
        y_pred.append(pred)
        y_probs.append(probs)

    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=class_names[:len(np.unique(y_test))],
        output_dict=True,
    )

    # ── Inference latency ─────────────────────────────────────────
    latencies = []
    sample = X_test[0]
    for _ in range(n_inference_trials):
        t0 = time.perf_counter()
        model.predict(sample)
        latencies.append((time.perf_counter() - t0) * 1000)

    mean_lat = float(np.mean(latencies))
    p95_lat  = float(np.percentile(latencies, 95))

    logger.info(
        f"Evaluation: acc={acc:.3f}, "
        f"inference mean={mean_lat:.2f}ms, p95={p95_lat:.2f}ms"
    )

    result = {
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "inference_mean_ms": mean_lat,
        "inference_p95_ms": p95_lat,
        "latency_budget_met": mean_lat < 10.0,
        "n_test_samples": len(X_test),
        "paradigm": paradigm,
    }

    # ── Print summary table ───────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  Accuracy:       {acc*100:.1f}%")
    print(f"  Inference:      {mean_lat:.1f}ms mean, {p95_lat:.1f}ms p95")
    print(f"  Budget (<10ms): {'✓' if mean_lat < 10 else '✗'}")
    print(f"{'='*50}\n")
    print(classification_report(y_test, y_pred,
                                  target_names=class_names[:len(np.unique(y_test))]))
    return result


def false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute false positive rate (FPR) — critical for wheelchair safety.
    FPR = unintended commands / total commands.
    Target: <5% (ISO 14155 Phase 2 criterion).
    """
    # In multi-class: FPR = mean per-class FPR
    fprs = []
    for cls in np.unique(y_true):
        tn = np.sum((y_true != cls) & (y_pred != cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fprs.append(fp / (fp + tn + 1e-10))
    return float(np.mean(fprs))


def false_negative_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute false negative rate — missed commands.
    FNR = missed commands / total intended commands.
    """
    fnrs = []
    for cls in np.unique(y_true):
        fn = np.sum((y_true == cls) & (y_pred != cls))
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fnrs.append(fn / (fn + tp + 1e-10))
    return float(np.mean(fnrs))