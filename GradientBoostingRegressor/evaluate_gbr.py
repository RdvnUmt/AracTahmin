from __future__ import annotations

import argparse
from pathlib import Path

from joblib import load

from utils import safe_load_npy, regression_metrics, plot_predictions, plot_residuals, ensure_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=".", help="npy dosyalarının bulunduğu klasör")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="modelin/çıktıların bulunduğu klasör")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    artifacts_dir = ensure_dir(args.artifacts_dir)

    model_path = artifacts_dir / "gbr_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model bulunamadı: {model_path}. Önce train_gbr.py çalıştır.")

    model = load(model_path)

    X_test = safe_load_npy(data_dir / "X_test_processed.npy")
    y_test = safe_load_npy(data_dir / "y_test.npy").reshape(-1)

    y_pred = model.predict(X_test)
    metrics = regression_metrics(y_test, y_pred)

    print("Test metrics:", metrics)

    plot_predictions(y_test, y_pred, artifacts_dir / "pred_vs_actual_test_eval.png", title="Test: Predicted vs Actual (eval)")
    plot_residuals(y_test, y_pred, artifacts_dir / "residuals_test_eval", title_prefix="Test residuals (eval)")

    print(f"Saved eval plots to: {artifacts_dir.resolve()}")


if __name__ == "__main__":
    main()
