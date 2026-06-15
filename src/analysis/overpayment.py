"""
Over/underpayment analysis.

Task A — Transfer residuals: score every real transfer against the model.
Task B — Transductive fair value: apply model to all player-seasons to estimate
          what each player would fetch if sold today, benchmarked against TM.

Run:
    uv run python -m src.analysis.overpayment

Reads:
    models/xgb_v1.json
    models/feature_cols.json
    data/features/transfer_dataset.parquet
    data/features/feature_matrix.parquet
    data/transfermarkt.duckdb  (for inflation index)

Outputs:
    data/processed/transfer_residuals.parquet
    data/processed/player_season_fair_values.parquet
    reports/figures/overpayment_top20.png
    reports/figures/overpayment_by_league.png
    reports/figures/overpayment_by_position.png
    reports/figures/tm_vs_model.png
"""

import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from src.features.engineering import POSITION_MAP
from src.features.inflation import DB_PATH, build_fee_inflation_index
from src.model.train import FEATURE_COLS

MODELS_DIR   = Path(__file__).parents[2] / "models"
FEATURES_DIR = Path(__file__).parents[2] / "data" / "features"
PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"
FIGURES_DIR  = Path(__file__).parents[2] / "reports" / "figures"

TARGET = "log_inflation_adjusted_fee"


def _prep_X(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = df[feature_cols].copy()
    X["pct_team_minutes"] = X["pct_team_minutes"].fillna(0)
    X["pct_team_goals"]   = X["pct_team_goals"].fillna(0)
    return X


# ---------------------------------------------------------------------------
# Task A — transfer residuals
# ---------------------------------------------------------------------------

def compute_transfer_residuals(
    model: xgb.XGBRegressor,
    transfer_df: pd.DataFrame,
    feature_cols: list[str],
    inflation_index: dict[int, float],
) -> pd.DataFrame:
    df = transfer_df.copy()
    X = _prep_X(df, feature_cols)
    df["predicted"] = model.predict(X)
    df["residual"]  = df[TARGET] - df["predicted"]
    df["overpayment_factor"] = np.exp(df["residual"])   # >1 = overpaid, <1 = bargain

    # back to EUR
    df["inflation_median_fee"] = df["season_int"].map(inflation_index)
    df["predicted_fee_eur"] = (
        np.exp(df["predicted"]) * df["inflation_median_fee"]
    )
    return df


# ---------------------------------------------------------------------------
# Task B — transductive fair value on all player-seasons
# ---------------------------------------------------------------------------

def _build_transductive_df(
    model: xgb.XGBRegressor,
    feature_matrix: pd.DataFrame,
    feature_cols: list[str],
    inflation_index: dict[int, float],
) -> pd.DataFrame:
    """
    For each player-season S, predict what they'd fetch in window S+1.
    Only rows where all model features are non-null are scored.
    """
    df = feature_matrix.copy()

    # league_tier and position dummies may not be in feature_matrix — add if missing
    from src.features.transfer_features import LEAGUE_TIER
    if "league_tier" not in df.columns:
        df["league_tier"] = df["competition_id"].map(LEAGUE_TIER)

    for col in ["pos_DEF", "pos_MID", "pos_FWD"]:
        if col not in df.columns:
            grp = col.replace("pos_", "")
            df[col] = (df.get("position_group", pd.Series(dtype=str)) == grp).astype(int)

    if "position_group" not in df.columns:
        df["position_group"] = (
            df["position"].map(POSITION_MAP)
            .fillna(df.get("sub_position", pd.Series(dtype=str)).map(POSITION_MAP))
            .fillna("UNK")
        )

    # rename columns to match training feature names
    rename = {
        "club_transfer_spending": "club_transfer_spending_prior",
        "league_transfer_spending": "from_league_spending",
    }
    for old, new in rename.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    # season_int = season (games season label is the start year of the window)
    if "season_int" not in df.columns:
        df["season_int"] = df["season"]

    # keep only rows with all required features
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns for scoring: {missing_cols}")

    X = _prep_X(df, feature_cols)
    row_mask = X.notna().all(axis=1)
    df_scored = df[row_mask].copy()
    X_scored = X[row_mask]

    df_scored["predicted_log_inflation"] = model.predict(X_scored)

    # fair value in EUR: predicted log-inflation → multiply by that season's median fee
    df_scored["inflation_median_fee"] = df_scored["season_int"].map(
        lambda s: inflation_index.get(s, np.nan)
    )
    df_scored["predicted_fee_eur"] = (
        np.exp(df_scored["predicted_log_inflation"]) * df_scored["inflation_median_fee"]
    )

    # TM premium: TM value / model fair value
    mv = pd.to_numeric(df_scored["market_value_in_eur"], errors="coerce")
    df_scored["tm_premium"] = mv / df_scored["predicted_fee_eur"].replace(0, np.nan)

    return df_scored




# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_overpayment_top_n(
    residuals_df: pd.DataFrame,
    figures_dir: Path,
    n: int = 20,
) -> None:
    df = residuals_df.copy()
    df["label"] = df["player_name"] + " (" + df["season_int"].astype(str) + ")"

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # top-N overpaid
    over = df.nlargest(n, "residual")
    axes[0].barh(over["label"], over["overpayment_factor"], color="crimson", edgecolor="white")
    axes[0].set_xlabel("Overpayment factor (actual / model fair value)")
    axes[0].set_title(f"Top {n} most overpaid transfers")
    axes[0].invert_yaxis()
    axes[0].axvline(1, color="black", linestyle="--", lw=0.8)

    # top-N bargains
    under = df.nsmallest(n, "residual")
    axes[1].barh(under["label"], under["overpayment_factor"], color="steelblue", edgecolor="white")
    axes[1].set_xlabel("Overpayment factor (actual / model fair value)")
    axes[1].set_title(f"Top {n} biggest bargains")
    axes[1].invert_yaxis()
    axes[1].axvline(1, color="black", linestyle="--", lw=0.8)

    fig.suptitle("Transfer over/underpayment analysis")
    fig.tight_layout()
    fig.savefig(figures_dir / "overpayment_top20.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_residuals_by_league(residuals_df: pd.DataFrame, figures_dir: Path) -> None:
    league_labels = {"GB1": "PL", "ES1": "La Liga", "IT1": "Serie A", "L1": "Bundesliga", "FR1": "Ligue 1"}
    df = residuals_df.copy()
    df["league"] = df["from_competition_id"].map(league_labels)

    league_order = df.groupby("league")["residual"].median().sort_values().index.tolist()
    data = [df.loc[df["league"] == lg, "residual"].values for lg in league_order]

    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(data, tick_labels=league_order, patch_artist=True, notch=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
    ax.axhline(0, color="red", linestyle="--", lw=0.8)
    ax.set_ylabel("Residual (log-inflation space)")
    ax.set_title("Transfer residuals by selling league")
    fig.tight_layout()
    fig.savefig(figures_dir / "overpayment_by_league.png", dpi=150)
    plt.close(fig)


def plot_residuals_by_position(residuals_df: pd.DataFrame, figures_dir: Path) -> None:
    df = residuals_df.copy()
    pos_order = df.groupby("position_group")["residual"].median().sort_values().index.tolist()
    data = [df.loc[df["position_group"] == p, "residual"].values for p in pos_order]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, tick_labels=pos_order, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
    ax.axhline(0, color="red", linestyle="--", lw=0.8)
    ax.set_ylabel("Residual (log-inflation space)")
    ax.set_title("Transfer residuals by position group")
    fig.tight_layout()
    fig.savefig(figures_dir / "overpayment_by_position.png", dpi=150)
    plt.close(fig)


def plot_tm_vs_model(fair_values_df: pd.DataFrame, figures_dir: Path) -> None:
    df = fair_values_df.dropna(subset=["tm_premium", "market_value_in_eur"]).copy()
    mv = pd.to_numeric(df["market_value_in_eur"], errors="coerce")
    pred = df["predicted_fee_eur"]
    mask = (mv > 0) & (pred > 0)
    df = df[mask]
    mv, pred = mv[mask], pred[mask]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        np.log(pred), np.log(mv),
        alpha=0.15, s=8, color="steelblue",
    )
    lo = min(np.log(pred).min(), np.log(mv).min()) - 0.5
    hi = max(np.log(pred).max(), np.log(mv).max()) + 0.5
    ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="TM = Model")
    ax.set_xlabel("log(Model fair value EUR)")
    ax.set_ylabel("log(TM market value EUR)")
    ax.set_title("TM valuation vs Model fair value (all player-seasons)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "tm_vs_model.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_analysis(
    models_dir: Path = MODELS_DIR,
    features_dir: Path = FEATURES_DIR,
    processed_dir: Path = PROCESSED_DIR,
    figures_dir: Path = FIGURES_DIR,
    db_path: Path = DB_PATH,
) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    model = xgb.XGBRegressor()
    model.load_model(str(models_dir / "xgb_v1.json"))
    feature_cols = json.loads((models_dir / "feature_cols.json").read_text())

    con = duckdb.connect(str(db_path), read_only=True)
    inflation_index = build_fee_inflation_index(con)
    con.close()

    # --- Task A: transfer residuals ---
    transfer_df = pd.read_parquet(features_dir / "transfer_dataset.parquet")
    residuals_df = compute_transfer_residuals(model, transfer_df, feature_cols, inflation_index)
    residuals_df.to_parquet(processed_dir / "transfer_residuals.parquet", index=False)
    print(f"Transfer residuals saved ({len(residuals_df)} rows)")

    print("\nTop 10 overpaid transfers:")
    top10_over = residuals_df.nlargest(10, "residual")[
        ["player_name", "season_int", "from_competition_id", "position_group",
         "transfer_fee", "predicted_fee_eur", "overpayment_factor"]
    ]
    top10_over["transfer_fee_M"] = top10_over["transfer_fee"] / 1e6
    top10_over["predicted_M"]    = top10_over["predicted_fee_eur"] / 1e6
    print(top10_over[["player_name", "season_int", "from_competition_id",
                       "transfer_fee_M", "predicted_M", "overpayment_factor"]]
          .round(2).to_string(index=False))

    print("\nTop 10 bargain transfers:")
    top10_under = residuals_df.nsmallest(10, "residual")[
        ["player_name", "season_int", "from_competition_id", "position_group",
         "transfer_fee", "predicted_fee_eur", "overpayment_factor"]
    ]
    top10_under["transfer_fee_M"] = top10_under["transfer_fee"] / 1e6
    top10_under["predicted_M"]    = top10_under["predicted_fee_eur"] / 1e6
    print(top10_under[["player_name", "season_int", "from_competition_id",
                        "transfer_fee_M", "predicted_M", "overpayment_factor"]]
          .round(2).to_string(index=False))

    plot_overpayment_top_n(residuals_df, figures_dir)
    plot_residuals_by_league(residuals_df, figures_dir)
    plot_residuals_by_position(residuals_df, figures_dir)
    print("\nSaved overpayment_top20.png, overpayment_by_league.png, overpayment_by_position.png")

    # --- Task B: transductive fair value ---
    feature_matrix = pd.read_parquet(features_dir / "feature_matrix.parquet")
    fair_values_df = _build_transductive_df(model, feature_matrix, feature_cols, inflation_index)
    fair_values_df.to_parquet(processed_dir / "player_season_fair_values.parquet", index=False)
    print(f"\nFair values saved ({len(fair_values_df)} rows)")

    print("\nMost TM-overvalued players (TM >> model):")
    top_over_tm = fair_values_df.dropna(subset=["tm_premium"]).nlargest(8, "tm_premium")[
        ["player_name", "season_int", "competition_id", "position_group",
         "market_value_in_eur", "predicted_fee_eur", "tm_premium"]
    ]
    print(top_over_tm.round(2).to_string(index=False))

    plot_tm_vs_model(fair_values_df, figures_dir)
    print("\nSaved tm_vs_model.png")


if __name__ == "__main__":
    run_analysis()
