"""
XGBoost model with walk-forward CV and Optuna hyperparameter tuning.

Walk-forward folds (expanding window):
    Fold 1: train 2017-2019, validate 2020
    Fold 2: train 2017-2020, validate 2021
    Fold 3: train 2017-2021, validate 2022
    Fold 4: train 2017-2022, validate 2023
Hold-out test: 2024 (never seen during Optuna)

Run:
    uv run python -m src.model.train

Outputs:
    models/xgb_v1.json
    models/feature_cols.json
    models/cv_results.parquet
    data/features/train_dataset.parquet
    data/features/test_dataset.parquet
"""

import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES_DIR = Path(__file__).parents[2] / "data" / "features"
MODELS_DIR   = Path(__file__).parents[2] / "models"
DATASET_PATH = FEATURES_DIR / "transfer_dataset.parquet"

TARGET = "log_inflation_adjusted_fee"
HOLDOUT_YEAR = 2024
N_OPTUNA_TRIALS = 100

FEATURE_COLS = [
    # performance
    "goals_per90",
    "assists_per90",
    "goal_contributions_per90",
    "yellows_per90",
    "appearances",
    # player
    "age",
    "age_sq",
    "height_in_cm",
    "international_caps",
    "is_english",
    # position dummies (GK = reference)
    "pos_DEF",
    "pos_MID",
    "pos_FWD",
    # team context
    "team_goals_scored",
    "pct_team_minutes",
    "pct_team_goals",
    "club_transfer_spending_prior",
    # league/market
    "league_tier",
    "from_league_spending",
    "season_int",
    # contract (NaN until contract_scraper has been run; XGBoost handles natively)
    "contract_months_remaining",
    "has_contract_data",
]

# Walk-forward folds: (train_seasons_max, val_season)
_CV_FOLDS = [
    (2019, 2020),
    (2020, 2021),
    (2021, 2022),
    (2022, 2023),
]


def _prep(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURE_COLS].copy()
    # fill pct nulls (clubs with no prior-season data) with 0
    X["pct_team_minutes"] = X["pct_team_minutes"].fillna(0)
    X["pct_team_goals"]   = X["pct_team_goals"].fillna(0)
    y = df[TARGET]
    return X, y


def _cv_rmse(params: dict, df_train: pd.DataFrame) -> float:
    """Mean RMSE across walk-forward folds."""
    fold_rmses = []
    for train_max, val_year in _CV_FOLDS:
        fold_train = df_train[df_train["season_int"] <= train_max]
        fold_val   = df_train[df_train["season_int"] == val_year]
        if len(fold_train) == 0 or len(fold_val) == 0:
            continue
        X_tr, y_tr = _prep(fold_train)
        X_va, y_va = _prep(fold_val)
        model = xgb.XGBRegressor(
            **params,
            random_state=42,
            verbosity=0,
            eval_metric="rmse",
        )
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        preds = model.predict(X_va)
        fold_rmses.append(np.sqrt(mean_squared_error(y_va, preds)))
    return float(np.mean(fold_rmses))


def tune(df_train: pd.DataFrame, n_trials: int = N_OPTUNA_TRIALS) -> dict:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 50, 500),
            "max_depth":        trial.suggest_int("max_depth", 2, 6),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        }
        return _cv_rmse(params, df_train)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest CV RMSE: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study.best_params


def train(
    dataset_path: Path = DATASET_PATH,
    models_dir: Path = MODELS_DIR,
    features_dir: Path = FEATURES_DIR,
    n_trials: int = N_OPTUNA_TRIALS,
) -> tuple[xgb.XGBRegressor, pd.DataFrame]:
    df = pd.read_parquet(dataset_path)
    df = df.dropna(subset=[TARGET])

    df_train = df[df["season_int"] < HOLDOUT_YEAR].copy()
    df_test  = df[df["season_int"] == HOLDOUT_YEAR].copy()

    print(f"Train rows: {len(df_train)}  |  Test rows (2024 hold-out): {len(df_test)}")

    # --- walk-forward CV results (baseline, fold-by-fold) ---
    print("\nRunning walk-forward CV per fold (default params for baseline)...")
    baseline_params = {
        "n_estimators": 100, "max_depth": 3, "learning_rate": 0.1,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.1, "reg_lambda": 1.0, "min_child_weight": 5,
    }
    cv_records = []
    for train_max, val_year in _CV_FOLDS:
        fold_tr = df_train[df_train["season_int"] <= train_max]
        fold_va = df_train[df_train["season_int"] == val_year]
        X_tr, y_tr = _prep(fold_tr)
        X_va, y_va = _prep(fold_va)
        m = xgb.XGBRegressor(**baseline_params, random_state=42, verbosity=0)
        m.fit(X_tr, y_tr)
        p = m.predict(X_va)
        cv_records.append({
            "fold": f"train<={train_max}/val={val_year}",
            "n_train": len(fold_tr),
            "n_val": len(fold_va),
            "rmse": np.sqrt(mean_squared_error(y_va, p)),
            "mae":  mean_absolute_error(y_va, p),
            "r2":   r2_score(y_va, p),
        })
    cv_df = pd.DataFrame(cv_records)
    print("\nCV fold results (baseline params):")
    print(cv_df.to_string(index=False))

    # --- Optuna tuning ---
    print(f"\nRunning Optuna ({n_trials} trials)...")
    best_params = tune(df_train, n_trials=n_trials)

    # --- retrain on full 2017-2023 with best params ---
    print("\nRetraining on full 2017-2023...")
    X_full, y_full = _prep(df_train)
    final_model = xgb.XGBRegressor(
        **best_params,
        random_state=42,
        verbosity=0,
        eval_metric="rmse",
    )
    final_model.fit(X_full, y_full)

    # --- hold-out evaluation ---
    X_test, y_test = _prep(df_test)
    test_preds = final_model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    test_mae  = mean_absolute_error(y_test, test_preds)
    test_r2   = r2_score(y_test, test_preds)
    print(f"\n2024 Hold-out:  RMSE={test_rmse:.4f}  MAE={test_mae:.4f}  R²={test_r2:.4f}")

    # --- save ---
    models_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    final_model.save_model(str(models_dir / "xgb_v1.json"))
    (models_dir / "feature_cols.json").write_text(json.dumps(FEATURE_COLS))
    cv_df.to_parquet(models_dir / "cv_results.parquet", index=False)
    df_train.to_parquet(features_dir / "train_dataset.parquet", index=False)
    df_test.to_parquet(features_dir / "test_dataset.parquet", index=False)

    print(f"\nSaved model -> {models_dir / 'xgb_v1.json'}")
    return final_model, cv_df


if __name__ == "__main__":
    train()
