from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.base import clone
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
    Eğer hedef değişken bunun için uygunsa (min > -1),
    modeli log1p hedef dönüşümü ile wrap eder.
    Özellikle ağır kuyruklu fiyat dağılımlarında işe yarar.
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
    """Feature importance için, mümkünse preprocessor içinden feature isimlerini çek."""
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


def get_inner_estimator(estimator):
    """
    TransformedTargetRegressor kullanıldığında içteki asıl regresörü döndür.
    Aksi halde estimator'ın kendisini döndür.
    """
    return getattr(estimator, "regressor_", estimator)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=".", help="npy/joblib dosyalarının bulunduğu klasör")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="çıktıların kaydedileceği klasör")
    parser.add_argument("--tune", action="store_true", help="RandomizedSearchCV ile hiperparametre araması yap")
    parser.add_argument("--cv", type=int, default=3, help="tuning için KFold fold sayısı")
    parser.add_argument("--n-iter", type=int, default=30, help="RandomizedSearch iter sayısı")
    parser.add_argument("--seed", type=int, default=42, help="random_state")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    artifacts_dir = ensure_dir(args.artifacts_dir)

    # -------------------------------------------------------------------------
    # 1) Veriyi yükle (train / test olarak ikiye bölünmüş durumda)
    # -------------------------------------------------------------------------
    X_train = safe_load_npy(data_dir / "X_train_processed.npy")
    X_test = safe_load_npy(data_dir / "X_test_processed.npy")
    y_train = safe_load_npy(data_dir / "y_train.npy").reshape(-1)
    y_test = safe_load_npy(data_dir / "y_test.npy").reshape(-1)

    print(f"X_train: {X_train.shape}  X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}  y_test: {y_test.shape}")

    # -------------------------------------------------------------------------
    # 2) Baz model şablonu + gerekiyorsa log-target wrap
    # -------------------------------------------------------------------------
    base = GradientBoostingRegressor(random_state=args.seed)
    model_template, used_log = maybe_wrap_log_target(base, y_train)

    # -------------------------------------------------------------------------
    # 3) Baseline model (hiç tuning yok, default hiperparametreler)
    # -------------------------------------------------------------------------
    baseline_estimator = clone(model_template)
    baseline_estimator.fit(X_train, y_train)

    y_pred_train_baseline = baseline_estimator.predict(X_train)
    y_pred_test_baseline = baseline_estimator.predict(X_test)

    baseline_train_metrics = regression_metrics(y_train, y_pred_train_baseline)
    baseline_test_metrics = regression_metrics(y_test, y_pred_test_baseline)

    print("\n=== Baseline model (default GBR) ===")
    print("Train metrics:", baseline_train_metrics)
    print("Test  metrics:", baseline_test_metrics)

    baseline_inner = get_inner_estimator(baseline_estimator)
    baseline_params = baseline_inner.get_params()

    # -------------------------------------------------------------------------
    # 4) Hiperparametre optimizasyonu (RandomizedSearchCV + KFold)
    #    Yalnızca --tune verilmişse çalışır
    #    -> ARTIK MSE'yi (mean_squared_error) minimize ediyoruz
    # -------------------------------------------------------------------------
    tuned_estimator = None
    tuned_train_metrics = None
    tuned_test_metrics = None
    best_params = None
    cv_info = None

    if args.tune:
        # TransformedTargetRegressor kullanıldığında iç regresör parametreleri
        # 'regressor__' prefix'i ile verilmeli
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
            estimator=clone(model_template),
            param_distributions=param_dist,
            n_iter=args.n_iter,
            scoring="neg_mean_squared_error",  # MSE'yi minimize ediyoruz
            cv=cv,
            random_state=args.seed,
            n_jobs=-1,
            verbose=1,
        )

        print("\n=== RandomizedSearchCV ile hiperparametre optimizasyonu başlıyor (scoring = neg_mean_squared_error) ===")
        search.fit(X_train, y_train)

        tuned_estimator = search.best_estimator_
        best_params = search.best_params_

        # CV sonuçlarını CSV'ye kaydet (MSE bazlı)
        cv_results = pd.DataFrame(search.cv_results_)
        if "mean_test_score" in cv_results.columns:
            # skor: neg_mean_squared_error; pozitif MSE için işaret çeviriyoruz
            cv_results["mean_val_mse"] = -cv_results["mean_test_score"]
        cv_results.to_csv(artifacts_dir / "gbr_cv_results.csv", index=False)

        # CV özeti
        cv_info = {
            "cv_folds": int(args.cv),
            "n_iter": int(args.n_iter),
            "scoring": "neg_mean_squared_error",
            "best_cv_score_neg_mse": float(search.best_score_),
            "best_cv_mse": float(-search.best_score_),
        }

        # Tuned model metrikleri (MAE, RMSE, MSE, R2)
        y_pred_train_tuned = tuned_estimator.predict(X_train)
        y_pred_test_tuned = tuned_estimator.predict(X_test)

        tuned_train_metrics = regression_metrics(y_train, y_pred_train_tuned)
        tuned_test_metrics = regression_metrics(y_test, y_pred_test_tuned)

        print("\n=== Tuned model (RandomizedSearchCV sonucu, MSE optimizasyonu) ===")
        print("Best params:", best_params)
        print("Train metrics:", tuned_train_metrics)
        print("Test  metrics:", tuned_test_metrics)

    # -------------------------------------------------------------------------
    # 5) Nihai seçilen model (tune varsa tuned, yoksa baseline)
    # -------------------------------------------------------------------------
    if tuned_estimator is not None:
        final_estimator = tuned_estimator
        final_train_metrics = tuned_train_metrics
        final_test_metrics = tuned_test_metrics
        selected_model = "tuned"
    else:
        final_estimator = baseline_estimator
        final_train_metrics = baseline_train_metrics
        final_test_metrics = baseline_test_metrics
        selected_model = "baseline"

    # Nihai model için pred / plot
    y_pred_train_final = final_estimator.predict(X_train)
    y_pred_test_final = final_estimator.predict(X_test)

    print(f"\n=== Final seçilen model: {selected_model} ===")
    print("Final train metrics:", final_train_metrics)
    print("Final test  metrics:", final_test_metrics)

    # -------------------------------------------------------------------------
    # 6) Model + metadata kaydet
    # -------------------------------------------------------------------------
    dump(final_estimator, artifacts_dir / "gbr_model.joblib")

    # Parametre özetleri
    inner_final = get_inner_estimator(final_estimator)
    final_params = inner_final.get_params()

    meta = {
        "model": "GradientBoostingRegressor",
        "used_log_target": bool(used_log),
        "selected_model": selected_model,
        "train_metrics": final_train_metrics,
        "test_metrics": final_test_metrics,
        "final_params": final_params,
        "baseline": {
            "params": baseline_params,
            "train_metrics": baseline_train_metrics,
            "test_metrics": baseline_test_metrics,
        },
        "shapes": {
            "X_train": list(X_train.shape),
            "X_test": list(X_test.shape),
            "y_train": list(y_train.shape),
            "y_test": list(y_test.shape),
        },
    }

    if tuned_estimator is not None:
        tuned_inner = get_inner_estimator(tuned_estimator)
        tuned_params = tuned_inner.get_params()
        meta["tuned"] = {
            "search_best_params": best_params,
            "inner_estimator_params": tuned_params,
            "cv_info": cv_info,
            "train_metrics": tuned_train_metrics,
            "test_metrics": tuned_test_metrics,
        }

    save_json(meta, artifacts_dir / "metrics_and_params.json")

    # -------------------------------------------------------------------------
    # 7) Nihai model için grafikler (test seti)
    # -------------------------------------------------------------------------
    plot_predictions(
        y_test,
        y_pred_test_final,
        artifacts_dir / "pred_vs_actual_test.png",
        title="Test: Predicted vs Actual (final GBR)",
    )
    plot_residuals(
        y_test,
        y_pred_test_final,
        artifacts_dir / "residuals_test",
        title_prefix="Test residuals (final GBR)",
    )

    # -------------------------------------------------------------------------
    # 8) Feature importance (nihai model üzerinden)
    # -------------------------------------------------------------------------
    inner = get_inner_estimator(final_estimator)
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

    print(f"\nSaved artifacts to: {artifacts_dir.resolve()}")


if __name__ == "__main__":
    main()
