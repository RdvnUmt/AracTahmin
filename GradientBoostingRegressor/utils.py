from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import inspect



def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_load_npy(path: str | Path) -> np.ndarray:
    """Loads .npy safely; falls back to allow_pickle=True only if needed."""
    path = Path(path)
    try:
        return np.load(path, allow_pickle=False)
    except ValueError:
        # Only enable pickle if the file is trusted
        return np.load(path, allow_pickle=True)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    mae = float(mean_absolute_error(y_true, y_pred))

    # RMSE: sklearn versiyon farklarına dayanıklı hesap
    sig = inspect.signature(mean_squared_error)
    if "squared" in sig.parameters:
        rmse = float(mean_squared_error(y_true, y_pred, squared=False))
        mse = float(mean_squared_error(y_true, y_pred, squared=True))
    else:
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))

    r2 = float(r2_score(y_true, y_pred))

    eps = 1e-9
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0)

    return {"mae": mae, "rmse": rmse, "mse": mse, "r2": r2, "mape_%": mape}



def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str | Path,
    title: str = "Predicted vs Actual",
) -> None:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    ensure_dir(Path(out_path).parent)

    plt.figure(figsize=(7, 7))
    sns.scatterplot(x=y_true, y=y_pred, s=18, alpha=0.6)

    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)

    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path_prefix: str | Path,
    title_prefix: str = "Residuals",
) -> None:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    residuals = y_true - y_pred

    out_path_prefix = Path(out_path_prefix)
    ensure_dir(out_path_prefix.parent)

    plt.figure(figsize=(8, 4.5))
    sns.histplot(residuals, bins=50, kde=True)
    plt.title(f"{title_prefix} distribution")
    plt.xlabel("Residual (y_true - y_pred)")
    plt.tight_layout()
    plt.savefig(str(out_path_prefix) + "_hist.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    sns.scatterplot(x=y_pred, y=residuals, s=18, alpha=0.6)
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.title(f"{title_prefix} vs Predicted")
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(str(out_path_prefix) + "_vs_pred.png", dpi=160)
    plt.close()
