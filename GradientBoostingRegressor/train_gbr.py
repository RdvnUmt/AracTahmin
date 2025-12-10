from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.compose import TransformedTargetRegressor

from utils import (
    safe_load_npy,
    regression_metrics,
    save_json,
    plot_predictions,
    plot_residuals,
    ensure_dir,
)


def maybe_wrap_log_target(regressor, y_train: np.ndarray):
    """
    If y is suitable (min > -1), wrap model with log1p target transform.
    This often helps for heavy-tailed price targets.
    """
    y_train = np.asarray(y_train).reshape(-1)
    if np.nanmin(y_train) > -1.0:
        return (
            TransformedTargetRegressor(
                regressor=regressor,
                func=np.log1p,
                inverse_func=np.expm1,
                check_inverse=True,
            ),
            True,
        )
    return regressor, False


def get_feature_names(preprocessor_path: Path) -> list[str] | None:
    """Best-effort feature name extraction for feature importance."""
    if not preprocessor_path.exists():
        return None
    try:
        pre = load(preprocessor_path)
        if hasattr(pre, "get_feature_names_out"):
            names = pre.get_feature_names_out()
            return [str(x) for x in names]
    except Exception:
        return None
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=".", help="npy/joblib dosyalarının bulunduğu klasör")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="çıktıların kaydedileceği klasör")
    parser.add_argument("--tune", action="store_true", help="RandomizedSearchCV ile hiperparametre araması yap")
    parser.add_argument("--cv", type=int, default=3, help="tuning için CV fold sayısı")
    parser.add_argument("--n-iter", type=int, default=30, help="RandomizedSearch iter sayısı")
    parser.add_argument("--seed", type=int, default=42, help="random_state")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    artifacts_dir = ensure_dir(args.artifacts_dir)

    # Load data
    X_train = safe_load_npy(data_dir / "X_train_processed.npy")
    X_test = safe_load_npy(data_dir / "X_test_processed.npy")
    y_train = safe_load_npy(data_dir / "y_train.npy").reshape(-1)
    y_test = safe_load_npy(data_dir / "y_test.npy").reshape(-1)

    print(f"X_train: {X_train.shape}  X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}  y_test: {y_test.shape}")

    # Base model
    base = GradientBoostingRegressor(random_state=args.seed)
    model, used_log = maybe_wrap_log_target(base, y_train)

    best_estimator = model
    best_params = getattr(base, "get_params", lambda: {})()

    if args.tune:
        # NOTE: With TransformedTargetRegressor, inner regressor params are prefixed with 'regressor__'
        param_dist = {
            "regressor__n_estimators": [200, 400, 600, 800, 1000],
            "regressor__learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
            "regressor__max_depth": [2, 3, 4, 5],
            "regressor__subsample": [0.6, 0.8, 1.0],
            "regressor__min_samples_split": [2, 5, 10, 20],
            "regressor__min_samples_leaf": [1, 2, 4, 8],
            "regressor__max_features": [None, "sqrt", "log2"],
        }

        cv = KFold(n_splits=args.cv, shuffle=True, random_state=args.seed)

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=args.n_iter,
            scoring="neg_mean_absolute_error",
            cv=cv,
            random_state=args.seed,
            n_jobs=-1,
            verbose=1,
        )
        search.fit(X_train, y_train)
        best_estimator = search.best_estimator_
        best_params = search.best_params_
        print("Best params:", best_params)
    else:
        best_estimator.fit(X_train, y_train)

    # Evaluate
    y_pred_train = best_estimator.predict(X_train)
    y_pred_test = best_estimator.predict(X_test)

    train_metrics = regression_metrics(y_train, y_pred_train)
    test_metrics = regression_metrics(y_test, y_pred_test)

    print("Train metrics:", train_metrics)
    print("Test  metrics:", test_metrics)

    # Save model + metadata
    dump(best_estimator, artifacts_dir / "gbr_model.joblib")

    meta = {
        "model": "GradientBoostingRegressor",
        "used_log_target": bool(used_log),
        "best_params": best_params,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "shapes": {
            "X_train": list(X_train.shape),
            "X_test": list(X_test.shape),
            "y_train": list(y_train.shape),
            "y_test": list(y_test.shape),
        },
    }
    save_json(meta, artifacts_dir / "metrics_and_params.json")

    # Plots
    plot_predictions(y_test, y_pred_test, artifacts_dir / "pred_vs_actual_test.png", title="Test: Predicted vs Actual")
    plot_residuals(y_test, y_pred_test, artifacts_dir / "residuals_test", title_prefix="Test residuals")

    # Feature importances (best effort)
    inner = getattr(best_estimator, "regressor_", best_estimator)
    if hasattr(inner, "feature_importances_"):
        importances = np.asarray(inner.feature_importances_, dtype=float).reshape(-1)
        feat_names = get_feature_names(data_dir / "preprocessor.joblib")
        if feat_names is None or len(feat_names) != len(importances):
            feat_names = [f"f_{i}" for i in range(len(importances))]

        df_imp = pd.DataFrame({"feature": feat_names, "importance": importances})
        df_imp = df_imp.sort_values("importance", ascending=False).reset_index(drop=True)
        df_imp.to_csv(artifacts_dir / "feature_importance.csv", index=False)

        import matplotlib.pyplot as plt
        import seaborn as sns

        topk = df_imp.head(30).iloc[::-1]
        plt.figure(figsize=(10, 7))
        sns.barplot(data=topk, x="importance", y="feature")
        plt.title("Top 30 Feature Importances (GBR)")
        plt.tight_layout()
        plt.savefig(artifacts_dir / "feature_importance_top30.png", dpi=160)
        plt.close()

    print(f"Saved artifacts to: {artifacts_dir.resolve()}")


if __name__ == "__main__":
    main()
