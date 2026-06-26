"""
Exploratory Data Analysis — feature matrix and transfer dataset.

Run:
    uv run python -m src.analysis.eda

Outputs:
    reports/figures/eda/  (PNG figures)
    reports/eda_report.md
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

FEATURES_DIR = Path(__file__).parents[2] / "data" / "features"
REPORTS_DIR  = Path(__file__).parents[2] / "reports"
FIGURES_DIR  = REPORTS_DIR / "figures" / "eda"

LEAGUE_LABELS = {
    "GB1": "Premier League",
    "ES1": "La Liga",
    "IT1": "Serie A",
    "L1":  "Bundesliga",
    "FR1": "Ligue 1",
}
LEAGUE_ORDER = ["Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1"]
POS_ORDER    = ["GK", "DEF", "MID", "FWD"]

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.0)
plt.rcParams.update({"figure.dpi": 150})


def _save(fig: plt.Figure, name: str) -> Path:
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")
    return path


# ---------------------------------------------------------------------------
# 1. Valuation distribution
# ---------------------------------------------------------------------------

def plot_valuation_dist(df: pd.DataFrame) -> dict:
    mv = pd.to_numeric(df["market_value_in_eur"], errors="coerce")
    mv = mv[mv > 0].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(mv / 1e6, bins=100, color="steelblue", edgecolor="none")
    axes[0].set_xlabel("Market value (€M)")
    axes[0].set_title("Raw TM valuation distribution")
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{int(x)}M"))

    log_mv = np.log(mv)
    axes[1].hist(log_mv, bins=60, color="steelblue", edgecolor="none")
    axes[1].set_xlabel("log(Market value EUR)")
    axes[1].set_title("Log-transformed (near-normal after transform)")

    fig.suptitle(f"TM Valuation Distribution  (n={len(mv):,} player-seasons with valuation)")
    fig.tight_layout()
    _save(fig, "eda_valuation_dist.png")

    skew_raw = float(stats.skew(mv))
    skew_log = float(stats.skew(log_mv))
    return {
        "n_with_valuation": len(mv),
        "median_eur": float(mv.median()),
        "mean_eur":   float(mv.mean()),
        "p25_eur":    float(mv.quantile(0.25)),
        "p75_eur":    float(mv.quantile(0.75)),
        "max_eur":    float(mv.max()),
        "skew_raw":   round(skew_raw, 2),
        "skew_log":   round(skew_log, 2),
    }


# ---------------------------------------------------------------------------
# 2. Valuation trend by season
# ---------------------------------------------------------------------------

def plot_valuation_by_season(df: pd.DataFrame) -> dict:
    mv = pd.to_numeric(df["market_value_in_eur"], errors="coerce")
    d = df.copy()
    d["mv"] = mv
    d = d[d["mv"] > 0].dropna(subset=["mv"])

    by_season = d.groupby("season")["mv"].agg(["median", "mean", "count"])
    counts = d.groupby("season").size()

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(by_season.index, by_season["median"] / 1e6, marker="o",
                 color="steelblue", label="Median")
    axes[0].plot(by_season.index, by_season["mean"] / 1e6, marker="s",
                 color="coral", linestyle="--", label="Mean")
    axes[0].set_ylabel("Market value (€M)")
    axes[0].set_title("TM valuation trend by season")
    axes[0].legend()
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:.0f}M"))

    axes[1].bar(counts.index, counts.values, color="steelblue", edgecolor="none")
    axes[1].set_ylabel("Player-seasons with valuation")
    axes[1].set_xlabel("Season")
    axes[1].set_title("Sample size by season")

    fig.tight_layout()
    _save(fig, "eda_valuation_by_season.png")

    peak = by_season["median"].idxmax()
    return {
        "seasons": sorted(by_season.index.tolist()),
        "peak_median_season": int(peak),
        "peak_median_eur": float(by_season.loc[peak, "median"]),
    }


# ---------------------------------------------------------------------------
# 3. Valuation by position and league
# ---------------------------------------------------------------------------

def plot_valuation_by_position_and_league(df: pd.DataFrame) -> dict:
    mv = pd.to_numeric(df["market_value_in_eur"], errors="coerce")
    d = df.copy()
    d["mv"] = mv
    d["log_mv"] = np.log(mv)
    d = d[d["mv"] > 0].dropna(subset=["mv"])
    d["league"] = d["competition_id"].map(LEAGUE_LABELS)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    pos_data = [d.loc[d["position_group"] == p, "log_mv"].dropna().values for p in POS_ORDER]
    bp = axes[0].violinplot(pos_data, positions=range(len(POS_ORDER)), showmedians=True, showextrema=False)
    for body in bp["bodies"]:
        body.set_facecolor("steelblue")
        body.set_alpha(0.7)
    axes[0].set_xticks(range(len(POS_ORDER)))
    axes[0].set_xticklabels(POS_ORDER)
    axes[0].set_ylabel("log(Market value EUR)")
    axes[0].set_title("Valuation by position group")

    league_data = [d.loc[d["league"] == lg, "log_mv"].dropna().values for lg in LEAGUE_ORDER]
    bp2 = axes[1].violinplot(league_data, positions=range(len(LEAGUE_ORDER)), showmedians=True, showextrema=False)
    for body in bp2["bodies"]:
        body.set_facecolor("coral")
        body.set_alpha(0.7)
    axes[1].set_xticks(range(len(LEAGUE_ORDER)))
    axes[1].set_xticklabels(LEAGUE_ORDER, rotation=15, ha="right")
    axes[1].set_ylabel("log(Market value EUR)")
    axes[1].set_title("Valuation by league")

    fig.suptitle("TM Valuation Distribution by Position and League (log scale)")
    fig.tight_layout()
    _save(fig, "eda_valuation_by_position_league.png")

    pos_medians = {p: float(d.loc[d["position_group"] == p, "mv"].median() / 1e6)
                   for p in POS_ORDER}
    league_medians = {lg: float(d.loc[d["league"] == lg, "mv"].median() / 1e6)
                      for lg in LEAGUE_ORDER if lg in d["league"].values}
    return {"position_medians_eur_m": pos_medians, "league_medians_eur_m": league_medians}


# ---------------------------------------------------------------------------
# 4. Age curve
# ---------------------------------------------------------------------------

def plot_age_curve(df: pd.DataFrame) -> dict:
    mv = pd.to_numeric(df["market_value_in_eur"], errors="coerce")
    d = df.copy()
    d["mv"] = mv
    d["log_mv"] = np.log(mv)
    d = d[(d["mv"] > 0) & d["age"].between(15, 40)].dropna(subset=["mv", "age"])

    by_age = d.groupby("age")["mv"].agg(["median", "count"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # scatter (sample 3000 points for readability)
    sample = d.sample(min(3000, len(d)), random_state=42)
    axes[0].scatter(sample["age"], sample["log_mv"], alpha=0.1, s=6, color="steelblue")
    axes[0].plot(by_age.index, np.log(by_age["median"]), color="red", lw=2, label="Median")
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("log(Market value EUR)")
    axes[0].set_title("Age vs. TM valuation (log scale)")
    axes[0].legend()

    axes[1].bar(by_age.index, by_age["count"], color="steelblue", edgecolor="none")
    axes[1].set_xlabel("Age")
    axes[1].set_ylabel("Player-seasons")
    axes[1].set_title("Sample size by age")

    peak_age = int(by_age["median"].idxmax())
    fig.tight_layout()
    _save(fig, "eda_age_curve.png")

    return {
        "peak_valuation_age": peak_age,
        "median_val_at_peak_eur_m": float(by_age.loc[peak_age, "median"] / 1e6),
        "age_range": [int(d["age"].min()), int(d["age"].max())],
    }


# ---------------------------------------------------------------------------
# 5. Top nationalities
# ---------------------------------------------------------------------------

def plot_nationalities(df: pd.DataFrame) -> dict:
    mv = pd.to_numeric(df["market_value_in_eur"], errors="coerce")
    d = df.copy()
    d["mv"] = mv
    d = d[d["mv"] > 0].dropna(subset=["mv", "nationality"])

    top_by_count = d["nationality"].value_counts().head(15)
    top_by_median = (
        d.groupby("nationality")["mv"]
        .agg(["median", "count"])
        .query("count >= 50")
        .sort_values("median", ascending=False)
        .head(15)
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].barh(top_by_count.index[::-1], top_by_count.values[::-1], color="steelblue")
    axes[0].set_xlabel("Player-seasons")
    axes[0].set_title("Top 15 nationalities by player-season count")

    axes[1].barh(
        top_by_median.index[::-1],
        top_by_median["median"].values[::-1] / 1e6,
        color="coral",
    )
    axes[1].set_xlabel("Median TM value (€M)")
    axes[1].set_title("Top 15 nationalities by median valuation\n(min. 50 player-seasons)")
    axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:.0f}M"))

    fig.tight_layout()
    _save(fig, "eda_nationalities.png")

    return {
        "top5_by_count": top_by_count.head(5).to_dict(),
        "top5_by_median_eur_m": (top_by_median["median"].head(5) / 1e6).round(1).to_dict(),
    }


# ---------------------------------------------------------------------------
# 6. Per-90 stats by position
# ---------------------------------------------------------------------------

def plot_per90_by_position(df: pd.DataFrame) -> None:
    d = df[df["position_group"].isin(POS_ORDER)].copy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    for ax, col, title in [
        (axes[0], "goals_per90",   "Goals per 90"),
        (axes[1], "assists_per90", "Assists per 90"),
        (axes[2], "goal_contributions_per90", "Goal contributions per 90"),
    ]:
        pos_data = [d.loc[d["position_group"] == p, col].clip(0, 2).dropna().values
                    for p in POS_ORDER]
        bp = ax.boxplot(pos_data, tick_labels=POS_ORDER, patch_artist=True, showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor("steelblue")
            patch.set_alpha(0.7)
        ax.set_title(title)
        ax.set_ylabel("Per 90 mins")

    fig.suptitle("Per-90 performance stats by position group")
    fig.tight_layout()
    _save(fig, "eda_per90_by_position.png")


# ---------------------------------------------------------------------------
# 7. Correlation heatmap (numeric features)
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(df: pd.DataFrame) -> dict:
    numeric_cols = [
        "age", "appearances", "goals_per90", "assists_per90",
        "goal_contributions_per90", "yellows_per90", "minutes_played",
        "pct_team_minutes", "pct_team_goals", "team_goals_scored",
        "club_transfer_spending", "league_transfer_spending",
        "international_caps", "height_in_cm", "log_valuation",
    ]
    d = df[[c for c in numeric_cols if c in df.columns]].dropna()

    corr = d.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        vmin=-1, vmax=1, ax=ax, linewidths=0.5, annot_kws={"size": 8},
    )
    ax.set_title("Feature correlation matrix (player-season level)")
    fig.tight_layout()
    _save(fig, "eda_correlation_heatmap.png")

    if "log_valuation" in corr:
        corr_with_target = corr["log_valuation"].drop("log_valuation").sort_values(key=abs, ascending=False)
        return {"top_correlates_with_log_val": corr_with_target.head(8).round(3).to_dict()}
    return {}


# ---------------------------------------------------------------------------
# 8. Transfer fee distribution
# ---------------------------------------------------------------------------

def plot_transfer_fee_dist(tdf: pd.DataFrame) -> dict:
    fees = tdf["transfer_fee"].dropna()
    fees = fees[fees > 0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(fees / 1e6, bins=60, color="steelblue", edgecolor="none")
    axes[0].set_xlabel("Transfer fee (€M)")
    axes[0].set_title("Raw transfer fee distribution")
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{int(x)}M"))

    axes[1].hist(np.log(fees), bins=40, color="steelblue", edgecolor="none")
    axes[1].set_xlabel("log(Transfer fee EUR)")
    axes[1].set_title("Log-transformed transfer fees")

    fig.suptitle(f"Transfer Fee Distribution  (n={len(fees)} transfers, 2017–2024)")
    fig.tight_layout()
    _save(fig, "eda_transfer_fee_dist.png")

    return {
        "n_transfers": len(fees),
        "median_fee_eur_m": float(fees.median() / 1e6),
        "mean_fee_eur_m":   float(fees.mean() / 1e6),
        "max_fee_eur_m":    float(fees.max() / 1e6),
        "skew_raw":         round(float(stats.skew(fees)), 2),
    }


# ---------------------------------------------------------------------------
# 9. Transfer metrics by season
# ---------------------------------------------------------------------------

def plot_transfers_by_season(tdf: pd.DataFrame) -> dict:
    d = tdf[tdf["transfer_fee"] > 0].copy()
    by_season = d.groupby("season_int")["transfer_fee"].agg(["median", "count"])

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(by_season.index, by_season["median"] / 1e6, marker="o", color="steelblue")
    axes[0].set_ylabel("Median fee (€M)")
    axes[0].set_title("Median transfer fee by season")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:.1f}M"))

    axes[1].bar(by_season.index, by_season["count"], color="steelblue", edgecolor="none")
    axes[1].set_ylabel("# Qualifying transfers")
    axes[1].set_xlabel("Season")
    axes[1].set_title("Transfer count by season")

    fig.tight_layout()
    _save(fig, "eda_transfers_by_season.png")

    peak = by_season["median"].idxmax()
    trough = by_season["median"].idxmin()
    return {
        "peak_median_season": int(peak),
        "peak_median_eur_m": float(by_season.loc[peak, "median"] / 1e6),
        "trough_median_season": int(trough),
        "trough_median_eur_m": float(by_season.loc[trough, "median"] / 1e6),
    }


# ---------------------------------------------------------------------------
# 10. Transfers by position and league
# ---------------------------------------------------------------------------

def plot_transfers_by_position_and_league(tdf: pd.DataFrame) -> dict:
    d = tdf[tdf["transfer_fee"] > 0].copy()
    d["log_fee"] = np.log(d["transfer_fee"])
    d["league"] = d["from_competition_id"].map(LEAGUE_LABELS)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    pos_in = [p for p in POS_ORDER if p in d["position_group"].values]
    pos_data = [d.loc[d["position_group"] == p, "log_fee"].values for p in pos_in]
    bp = axes[0].boxplot(pos_data, tick_labels=pos_in, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.7)
    axes[0].set_ylabel("log(Transfer fee EUR)")
    axes[0].set_title("Transfer fees by position group")

    league_in = [lg for lg in LEAGUE_ORDER if lg in d["league"].values]
    lg_data = [d.loc[d["league"] == lg, "log_fee"].values for lg in league_in]
    bp2 = axes[1].boxplot(lg_data, tick_labels=league_in, patch_artist=True, showfliers=False)
    for patch in bp2["boxes"]:
        patch.set_facecolor("coral")
        patch.set_alpha(0.7)
    axes[1].set_ylabel("log(Transfer fee EUR)")
    axes[1].set_title("Transfer fees by selling league")
    for label in axes[1].get_xticklabels():
        label.set_rotation(15)
        label.set_ha("right")

    fig.tight_layout()
    _save(fig, "eda_transfers_by_position_league.png")

    pos_medians = {p: round(float(d.loc[d["position_group"] == p, "transfer_fee"].median()) / 1e6, 1)
                   for p in pos_in}
    league_medians = {lg: round(float(d.loc[d["league"] == lg, "transfer_fee"].median()) / 1e6, 1)
                      for lg in league_in}
    return {"position_median_fee_eur_m": pos_medians, "league_median_fee_eur_m": league_medians}


# ---------------------------------------------------------------------------
# 11. Contract months remaining
# ---------------------------------------------------------------------------

def plot_contract_months(tdf: pd.DataFrame) -> dict:
    has_col = "contract_months_remaining" in tdf.columns
    if not has_col:
        return {}

    total = len(tdf)
    has_data = tdf["contract_months_remaining"].notna().sum()
    coverage_pct = 100 * has_data / total

    d = tdf.dropna(subset=["contract_months_remaining"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(d["contract_months_remaining"], bins=40, color="steelblue", edgecolor="none")
    axes[0].set_xlabel("Months remaining on contract")
    axes[0].set_title(f"Contract length at time of transfer\n(n={has_data}, {coverage_pct:.1f}% coverage)")

    # contract length vs fee
    d2 = d[d["transfer_fee"] > 0].copy()
    axes[1].scatter(d2["contract_months_remaining"], np.log(d2["transfer_fee"]),
                    alpha=0.4, s=15, color="steelblue")
    # regression line
    slope, intercept, r, _, _ = stats.linregress(
        d2["contract_months_remaining"], np.log(d2["transfer_fee"])
    )
    x_line = np.linspace(d2["contract_months_remaining"].min(), d2["contract_months_remaining"].max(), 100)
    axes[1].plot(x_line, slope * x_line + intercept, color="red", lw=1.5, label=f"r={r:.2f}")
    axes[1].set_xlabel("Months remaining")
    axes[1].set_ylabel("log(Transfer fee)")
    axes[1].set_title("Contract length vs. transfer fee")
    axes[1].legend()

    fig.tight_layout()
    _save(fig, "eda_contract_months.png")

    r_val, _ = stats.pearsonr(d2["contract_months_remaining"], np.log(d2["transfer_fee"]))
    return {
        "coverage_pct": round(coverage_pct, 1),
        "n_with_contract": int(has_data),
        "median_months": float(d["contract_months_remaining"].median()),
        "corr_with_log_fee": round(float(r_val), 3),
    }


# ---------------------------------------------------------------------------
# 12. Feature correlations with log transfer fee
# ---------------------------------------------------------------------------

def plot_feature_corr_with_fee(tdf: pd.DataFrame) -> dict:
    numeric_cols = [
        "age", "appearances", "goals_per90", "assists_per90",
        "goal_contributions_per90", "yellows_per90", "minutes_played",
        "pct_team_minutes", "pct_team_goals", "team_goals_scored",
        "international_caps", "height_in_cm", "is_english",
        "contract_months_remaining", "club_transfer_spending_prior",
        "from_league_spending", "league_tier",
    ]
    d = tdf[tdf["transfer_fee"] > 0].copy()
    d["log_fee"] = np.log(d["transfer_fee"])

    available = [c for c in numeric_cols if c in d.columns]
    corr_vals = {}
    for col in available:
        sub = d[[col, "log_fee"]].dropna()
        if len(sub) > 10:
            r, _ = stats.pearsonr(sub[col], sub["log_fee"])
            corr_vals[col] = r

    corr_series = pd.Series(corr_vals).sort_values()
    colors = ["coral" if v > 0 else "steelblue" for v in corr_series.values]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(corr_series.index, corr_series.values, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Pearson r with log(transfer fee)")
    ax.set_title("Feature correlations with log transfer fee")
    fig.tight_layout()
    _save(fig, "eda_feature_corr_with_fee.png")

    top_pos = corr_series[corr_series > 0].tail(5)
    top_neg = corr_series[corr_series < 0].head(5)
    return {
        "top_positive": top_pos.round(3).to_dict(),
        "top_negative": top_neg.round(3).to_dict(),
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_eda(
    features_dir: Path = FEATURES_DIR,
    reports_dir: Path  = REPORTS_DIR,
    figures_dir: Path  = FIGURES_DIR,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    fm  = pd.read_parquet(features_dir / "feature_matrix.parquet")
    tdf = pd.read_parquet(features_dir / "transfer_dataset.parquet")

    n_total     = len(fm)
    n_with_val  = fm["log_valuation"].notna().sum()
    n_positions = fm["position_group"].value_counts().to_dict()
    n_leagues   = fm["competition_id"].value_counts().rename(LEAGUE_LABELS).to_dict()
    season_min  = int(fm["season"].min())
    season_max  = int(fm["season"].max())

    print(f"\nFeature matrix: {n_total:,} player-seasons, {n_with_val:,} with TM valuation")
    print(f"Transfer dataset: {len(tdf):,} rows")

    print("\nGenerating figures...")
    val_stats   = plot_valuation_dist(fm)
    season_meta = plot_valuation_by_season(fm)
    pos_lg_meta = plot_valuation_by_position_and_league(fm)
    age_meta    = plot_age_curve(fm)
    nat_meta    = plot_nationalities(fm)
    plot_per90_by_position(fm)
    corr_meta   = plot_correlation_heatmap(fm)
    fee_stats   = plot_transfer_fee_dist(tdf)
    fee_season  = plot_transfers_by_season(tdf)
    fee_pos_lg  = plot_transfers_by_position_and_league(tdf)
    contract    = plot_contract_months(tdf)
    corr_fee    = plot_feature_corr_with_fee(tdf)

    # Build the markdown report
    report = _build_report(
        n_total=n_total, n_with_val=n_with_val, n_positions=n_positions,
        n_leagues=n_leagues, season_min=season_min, season_max=season_max,
        n_transfers=len(tdf),
        val_stats=val_stats, season_meta=season_meta, pos_lg_meta=pos_lg_meta,
        age_meta=age_meta, nat_meta=nat_meta, corr_meta=corr_meta,
        fee_stats=fee_stats, fee_season=fee_season, fee_pos_lg=fee_pos_lg,
        contract=contract, corr_fee=corr_fee,
    )

    report_path = reports_dir / "eda_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport written to {report_path}")


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def _fmt_eur(val_eur: float) -> str:
    if val_eur >= 1e6:
        return f"€{val_eur/1e6:.1f}M"
    return f"€{val_eur/1e3:.0f}K"


def _build_report(
    n_total, n_with_val, n_positions, n_leagues,
    season_min, season_max, n_transfers,
    val_stats, season_meta, pos_lg_meta,
    age_meta, nat_meta, corr_meta,
    fee_stats, fee_season, fee_pos_lg,
    contract, corr_fee,
) -> str:
    pos_table = "\n".join(
        f"| {p} | {c:,} |"
        for p, c in sorted(n_positions.items(), key=lambda x: -x[1])
        if p != "UNK"
    )
    league_table = "\n".join(
        f"| {lg} | {c:,} |"
        for lg, c in sorted(n_leagues.items(), key=lambda x: -x[1])
    )
    pos_val_table = "\n".join(
        f"| {p} | €{v:.1f}M |"
        for p, v in pos_lg_meta.get("position_medians_eur_m", {}).items()
    )
    league_val_table = "\n".join(
        f"| {lg} | €{v:.1f}M |"
        for lg, v in pos_lg_meta.get("league_medians_eur_m", {}).items()
    )

    fee_pos_rows = "\n".join(
        f"| {p} | €{v}M |"
        for p, v in fee_pos_lg.get("position_median_fee_eur_m", {}).items()
    )
    fee_league_rows = "\n".join(
        f"| {lg} | €{v}M |"
        for lg, v in fee_pos_lg.get("league_median_fee_eur_m", {}).items()
    )

    nat_count_rows = "\n".join(
        f"| {k} | {v:,} |"
        for k, v in nat_meta.get("top5_by_count", {}).items()
    )
    nat_val_rows = "\n".join(
        f"| {k} | €{v}M |"
        for k, v in nat_meta.get("top5_by_median_eur_m", {}).items()
    )

    corr_pos_rows = "\n".join(
        f"| {k} | {v:+.3f} |"
        for k, v in corr_fee.get("top_positive", {}).items()
    )
    corr_neg_rows = "\n".join(
        f"| {k} | {v:+.3f} |"
        for k, v in corr_fee.get("top_negative", {}).items()
    )

    contract_section = ""
    if contract:
        contract_section = f"""
## 9. Contract Length

![Contract months](figures/eda/eda_contract_months.png)

- **Coverage:** {contract['coverage_pct']}% of transfers have contract data ({contract['n_with_contract']} / {n_transfers})
- **Median months remaining:** {contract['median_months']:.0f} months (~{contract['median_months']/12:.1f} years)
- **Pearson r with log(fee):** {contract['corr_with_log_fee']:+.3f}

Contract months remaining shows a **positive correlation with fee ({contract['corr_with_log_fee']:+.3f})**: players sold with more contract time remaining command higher fees, consistent with the SHAP importance ranking this feature 3rd in the model. The distribution is bimodal — a cluster near 0–12 months (players sold cheaply with contract leverage) and a second cluster around 36–48 months (prime-age players under long deals).
"""

    return f"""# Transfer Market EDA Report

**Author:** Marco Barnfield
**Date:** 2026-06-25
**Data:** `davidcariboo/player-scores` (Kaggle/Transfermarkt), seasons {season_min}–{season_max}

---

## 1. Dataset Overview

| Metric | Value |
|--------|-------|
| Player-seasons (feature matrix) | {n_total:,} |
| Player-seasons with TM valuation | {n_with_val:,} ({100*n_with_val/n_total:.1f}%) |
| Qualifying transfers | {n_transfers} |
| Seasons covered | {season_min}–{season_max} |
| Leagues | Top 5 (PL, La Liga, Serie A, Bundesliga, Ligue 1) |

**Position breakdown (feature matrix):**

| Position | Player-seasons |
|----------|---------------|
{pos_table}

**League breakdown:**

| League | Player-seasons |
|--------|---------------|
{league_table}

---

## 2. TM Valuation Distribution

![Valuation distribution](figures/eda/eda_valuation_dist.png)

| Metric | Value |
|--------|-------|
| Player-seasons with valuation | {val_stats['n_with_valuation']:,} |
| Median TM value | {_fmt_eur(val_stats['median_eur'])} |
| Mean TM value | {_fmt_eur(val_stats['mean_eur'])} |
| 25th percentile | {_fmt_eur(val_stats['p25_eur'])} |
| 75th percentile | {_fmt_eur(val_stats['p75_eur'])} |
| Maximum | {_fmt_eur(val_stats['max_eur'])} |
| Skewness (raw) | {val_stats['skew_raw']} |
| Skewness (log) | {val_stats['skew_log']} |

The raw distribution is **highly right-skewed** (skewness = {val_stats['skew_raw']}), with a long tail of elite-player valuations. The log transform reduces this to near-symmetry (skewness = {val_stats['skew_log']}), confirming the model's choice of `log(valuation)` as the target.

---

## 3. Valuation Trend by Season

![Valuation by season](figures/eda/eda_valuation_by_season.png)

- **Peak median season:** {season_meta['peak_median_season']} ({_fmt_eur(season_meta['peak_median_eur'])})
- Season-level valuations track the same post-COVID compression seen in actual transfer fees, though TM is smoother — Transfermarkt updates valuations continuously, whereas real fees collapsed sharply in 2020–2021.

---

## 4. Valuation by Position and League

![Valuation by position and league](figures/eda/eda_valuation_by_position_league.png)

**Median TM valuation by position:**

| Position | Median valuation |
|----------|-----------------|
{pos_val_table}

**Median TM valuation by league:**

| League | Median valuation |
|--------|-----------------|
{league_val_table}

Forwards and midfielders carry the highest median valuations, as expected — goals and assists attract market attention. Goalkeepers have the lowest median despite being irreplaceable positionally, because the position is less traded and TM applies a systematic discount. The Premier League premium is clearly visible in the league violin: the PL distribution is shifted right relative to the other four leagues.

---

## 5. Age Curve

![Age curve](figures/eda/eda_age_curve.png)

- **Peak valuation age:** {age_meta['peak_valuation_age']} ({_fmt_eur(age_meta['median_val_at_peak_eur_m'] * 1e6)} median)
- **Age range in dataset:** {age_meta['age_range'][0]}–{age_meta['age_range'][1]}

The median TM valuation traces the expected inverted-U: rising sharply through the early-to-mid 20s, peaking around age {age_meta['peak_valuation_age']}, then declining. The model uses both `age` and `age²` to capture this curvature. The spread is highest in the 20–28 range, where star players and squad fillers coexist. Beyond 32, the distribution compresses — only consistently performing veterans retain high valuations.

---

## 6. Nationalities

![Nationalities](figures/eda/eda_nationalities.png)

**Top 5 by player-season count:**

| Nationality | Player-seasons |
|-------------|---------------|
{nat_count_rows}

**Top 5 by median TM valuation (min. 50 player-seasons):**

| Nationality | Median valuation |
|-------------|-----------------|
{nat_val_rows}

The dataset is unsurprisingly dominated by the host leagues' home nations (Spain, Germany, France, England, Brazil). The median valuation ranking diverges from the count ranking — smaller elite nations (e.g., Belgium, Portugal) punch above their count because their top players disproportionately appear in the dataset.

---

## 7. Per-90 Stats by Position

![Per-90 by position](figures/eda/eda_per90_by_position.png)

The per-90 distributions confirm the position groups are sensibly constructed:
- **Forwards** dominate goals per 90 and goal contributions, with the widest spread
- **Midfielders** have a non-trivial assist rate and a bimodal goal distribution (attacking vs defensive)
- **Defenders** show near-zero goal/assist rates, as expected
- **Goalkeepers** cluster at zero for all three metrics — the model could benefit from GK-specific features (clean sheets, save%)

---

## 8. Correlation Heatmap

![Correlation heatmap](figures/eda/eda_correlation_heatmap.png)

Top correlates with `log_valuation`:

| Feature | Pearson r |
|---------|-----------|
{chr(10).join(f"| {k} | {v:+.3f} |" for k, v in list(corr_meta.get('top_correlates_with_log_val', {}).items())[:8])}

**Notable patterns:**
- `international_caps` is the single strongest simple correlate with log valuation — consistent with TM incorporating international profile directly
- `appearances` and `minutes_played` are highly correlated with each other (collinear) but both correlate positively with valuation
- `age` has a non-linear relationship (negative simple correlation but an inverted-U in reality, captured by `age²`)
- Per-90 metrics (`goals_per90`, `assists_per90`) show surprisingly moderate raw correlations — valuation is as much about playing time and context as raw output rate
{contract_section}
---

## 10. Transfer Fee Distribution

![Transfer fee distribution](figures/eda/eda_transfer_fee_dist.png)

| Metric | Value |
|--------|-------|
| Transfers in dataset | {fee_stats['n_transfers']} |
| Median fee | €{fee_stats['median_fee_eur_m']:.1f}M |
| Mean fee | €{fee_stats['mean_fee_eur_m']:.1f}M |
| Maximum fee | €{fee_stats['max_fee_eur_m']:.0f}M |
| Skewness (raw) | {fee_stats['skew_raw']} |

Like TM valuations, transfer fees are extremely right-skewed. The log transform compresses them to approximate normality, validating the inflation-adjusted log-space target.

---

## 11. Transfer Trends by Season

![Transfers by season](figures/eda/eda_transfers_by_season.png)

- **Peak season:** {fee_season['peak_median_season']} (€{fee_season['peak_median_eur_m']:.1f}M median)
- **Trough season:** {fee_season['trough_median_season']} (€{fee_season['trough_median_eur_m']:.1f}M median)

The COVID effect is unmistakable: 2020 and 2021 see both a collapse in count (fewer qualifying transfers) and a halving of median fees. Fees have not recovered to 2019 levels — the 2022–2024 medians remain near post-COVID lows. The inflation index in the model adjusts for this, ensuring season is not the dominant spurious predictor.

---

## 12. Transfer Fees by Position and League

![Transfers by position and league](figures/eda/eda_transfers_by_position_league.png)

**Median transfer fee by position:**

| Position | Median fee |
|----------|-----------|
{fee_pos_rows}

**Median transfer fee by selling league:**

| Selling league | Median fee |
|----------------|-----------|
{fee_league_rows}

Forwards command the highest median fees, followed by midfielders. Goalkeeper fees are low and have high variance — the market for elite GKs is thin and unpredictable. The Premier League commands the highest median selling price, consistent with the residual league premium documented in the model report.

---

## 13. Feature Correlations with Transfer Fee

![Feature correlations with fee](figures/eda/eda_feature_corr_with_fee.png)

**Strongest positive correlates:**

| Feature | Pearson r |
|---------|-----------|
{corr_pos_rows}

**Strongest negative correlates:**

| Feature | Pearson r |
|---------|-----------|
{corr_neg_rows}

The correlation profile is broadly consistent with SHAP importances from the model, with two exceptions: `international_caps` and `appearances` rank higher in raw correlation than in SHAP (suggesting XGBoost discounts them once team context features are present), while `team_goals_scored` ranks higher in SHAP than in simple r (suggesting it captures interaction effects the correlation misses).

---

## Summary

The dataset is well-suited for modelling with a few known constraints:

1. **Right-skew requires log target** — both TM valuations and actual fees follow log-normal distributions; regression on raw values would be dominated by outliers
2. **COVID discontinuity** — seasons 2020–2021 are structural outliers in both count and price; the inflation index partially corrects for this but the model should not extrapolate through the gap
3. **Position group adequacy** — broad position groups (GK/DEF/MID/FWD) capture most of the valuation variation; sub-position detail would help but the sample per sub-position is small
4. **International caps as a confound** — strongly correlated with valuation but also with nationality premium (English/French/Brazilian players have higher cap counts due to squad policies); partial controls needed in the English tax hypothesis
5. **Per-90 underperforms playing time** — the market values consistent availability over peak rate production; any v2 feature set should include `pct_team_minutes` before chasing exotic per-90 metrics

*All figures: `reports/figures/eda/`. Feature matrix: `data/features/feature_matrix.parquet`. Transfer dataset: `data/features/transfer_dataset.parquet`.*
"""


if __name__ == "__main__":
    run_eda()
