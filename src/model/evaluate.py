"""
Model evaluation: SHAP feature importance and diagnostic plots.

Run:
    uv run python -m src.model.evaluate

Reads:
    models/xgb_v1.json
    models/feature_cols.json
    models/cv_results.parquet
    data/features/train_dataset.parquet
    data/features/test_dataset.parquet

Outputs (reports/figures/):
    shap_summary.png
    shap_beeswarm.png
    residual_plot.png
    actual_vs_predicted.png
    cv_fold_rmse.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MODELS_DIR  = Path(__file__).parents[2] / "models"
FEATURES_DIR = Path(__file__).parents[2] / "data" / "features"
FIGURES_DIR = Path(__file__).parents[2] / "reports" / "figures"

TARGET = "log_inflation_adjusted_fee"


def _prep_X(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = df[feature_cols].copy()
    X["pct_team_minutes"] = X["pct_team_minutes"].fillna(0)
    X["pct_team_goals"]   = X["pct_team_goals"].fillna(0)
    return X


def plot_cv_fold_rmse(cv_df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(cv_df["fold"], cv_df["rmse"], color="steelblue", edgecolor="white")
    ax.set_ylabel("RMSE (log-inflation space)")
    ax.set_title("Walk-forward CV — RMSE by fold")
    ax.set_xticklabels(cv_df["fold"], rotation=15, ha="right")
    mean_rmse = cv_df["rmse"].mean()
    ax.axhline(mean_rmse, color="red", linestyle="--", label=f"Mean {mean_rmse:.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "cv_fold_rmse.png", dpi=150)
    plt.close(fig)


def plot_actual_vs_predicted(y_true: pd.Series, y_pred: np.ndarray,
                             label: str, out_dir: Path) -> None:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, s=20, color="steelblue")
    lo = min(y_true.min(), y_pred.min()) - 0.2
    hi = max(y_true.max(), y_pred.max()) + 0.2
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax.set_xlabel("Actual log-inflation-adjusted fee")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{label}\nRMSE={rmse:.3f}  R²={r2:.3f}")
    fig.tight_layout()
    fig.savefig(out_dir / "actual_vs_predicted.png", dpi=150)
    plt.close(fig)


def plot_residuals(y_true: pd.Series, y_pred: np.ndarray, out_dir: Path) -> None:
    residuals = y_true.values - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y_pred, residuals, alpha=0.5, s=20, color="steelblue")
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residual (actual − predicted)")
    axes[0].set_title("Residuals vs Predicted")

    axes[1].hist(residuals, bins=30, color="steelblue", edgecolor="white")
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Residual")
    axes[1].set_title("Residual distribution")

    fig.tight_layout()
    fig.savefig(out_dir / "residual_plot.png", dpi=150)
    plt.close(fig)


def plot_shap(model: xgb.XGBRegressor, X: pd.DataFrame, out_dir: Path) -> None:
    # XGBoost 3.x native SHAP (bypasses shap.TreeExplainer version incompatibility)
    booster = model.get_booster()
    dmat = xgb.DMatrix(X, feature_names=list(X.columns))
    contribs = booster.predict(dmat, pred_contribs=True)  # shape (n, n_features+1)
    shap_matrix = contribs[:, :-1]
    base_value = float(contribs[0, -1])
    shap_values = shap.Explanation(
        values=shap_matrix,
        base_values=np.full(len(X), base_value),
        data=X.values,
        feature_names=list(X.columns),
    )

    # bar summary
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.plots.bar(shap_values, max_display=15, show=False, ax=ax)
    ax.set_title("SHAP feature importance (mean |SHAP|)")
    fig.tight_layout()
    fig.savefig(out_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # beeswarm
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title("SHAP beeswarm — feature impact on prediction")
    plt.tight_layout()
    fig.savefig(out_dir / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def evaluate(
    models_dir: Path = MODELS_DIR,
    features_dir: Path = FEATURES_DIR,
    figures_dir: Path = FIGURES_DIR,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    model = xgb.XGBRegressor()
    model.load_model(str(models_dir / "xgb_v1.json"))
    feature_cols = json.loads((models_dir / "feature_cols.json").read_text())
    cv_df = pd.read_parquet(models_dir / "cv_results.parquet")
    df_train = pd.read_parquet(features_dir / "train_dataset.parquet")
    df_test  = pd.read_parquet(features_dir / "test_dataset.parquet")

    X_train = _prep_X(df_train, feature_cols)
    X_test  = _prep_X(df_test,  feature_cols)
    y_train = df_train[TARGET]
    y_test  = df_test[TARGET]

    train_preds = model.predict(X_train)
    test_preds  = model.predict(X_test)

    print("Train metrics:")
    print(f"  RMSE={np.sqrt(mean_squared_error(y_train, train_preds)):.4f}"
          f"  MAE={mean_absolute_error(y_train, train_preds):.4f}"
          f"  R²={r2_score(y_train, train_preds):.4f}")

    print("2024 Hold-out metrics:")
    print(f"  RMSE={np.sqrt(mean_squared_error(y_test, test_preds)):.4f}"
          f"  MAE={mean_absolute_error(y_test, test_preds):.4f}"
          f"  R²={r2_score(y_test, test_preds):.4f}")

    # CV fold plot
    plot_cv_fold_rmse(cv_df, figures_dir)
    print(f"Saved cv_fold_rmse.png")

    # Actual vs predicted (test set)
    plot_actual_vs_predicted(y_test, test_preds, "2024 Hold-out", figures_dir)
    print(f"Saved actual_vs_predicted.png")

    # Residuals (test set)
    plot_residuals(y_test, test_preds, figures_dir)
    print(f"Saved residual_plot.png")

    # SHAP on train set (enough data, model trained on it)
    print("Computing SHAP values (train set)...")
    plot_shap(model, X_train, figures_dir)
    print(f"Saved shap_summary.png + shap_beeswarm.png")

    print(f"\nAll figures saved to {figures_dir}")


if __name__ == "__main__":
    evaluate()
