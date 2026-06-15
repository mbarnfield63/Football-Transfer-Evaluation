# Transfer Valuation Model — Implementation Plan

> Written: 2026-06-14. Reference this file at session start to reconstruct implementation context.

## Context

The previous iteration predicted TM's published valuations — subjective market opinions, not ground truth. This plan rebuilds the model to predict **actual transfer fees** using only objective on-pitch performance and structural market features. TM valuations are deliberately excluded as a training signal; they are held out as the comparison baseline in the analysis.

**Two deliverables:**
1. A generalizable XGBoost model predicting an inflation-adjusted "fair fee" for any player given their stats and context
2. A transductive historical analysis: apply the model to all player-seasons to rank the biggest over/underpayments in transfer history (2017–2024)

Final output: `reports/valuation_report.md`

---

## Data Reality

- **Training rows**: ~444 transfers where the selling club is top-5 and the player has top-5 prior-season appearances data
- **Training set**: 2017–2023 (~390 rows); **Hold-out test**: 2024 (~54 rows); **Excluded**: 2025 (future projections in dataset)
- **Join strategy**: `player_id + prior_season`, no club constraint — gives 65% coverage (vs 49% with club constraint due to January movers)
- **January transfer handling**: Players with two clubs in a season → primary club = most minutes (QUALIFY + ROW_NUMBER in DuckDB)
- **Out of scope**: `other_to_top5` transfers — only 14/142 have prior top-5 stats, too sparse for v1

---

## Target Variable

**Primary**: `log_inflation_adjusted_fee = log(transfer_fee / seasonal_median_fee)`

Interpretation: log of how many multiples of the market median this transfer cost. 0 = exactly median; 2.3 = ~10× median.

**Why inflation-adjust**: Post-COVID structural compression dropped median fee from €4.6M (2019 peak) to €2.0M (2022). Without adjustment, `season` becomes a dominant spurious predictor.

Also store `log_transfer_fee = log(transfer_fee)` as an alternative.

Filter: `transfer_fee > 0`, seller is top-5, `season_int` 2017–2024.

---

## Feature Set (~20 features)

| Group | Features | Notes |
|---|---|---|
| Performance | `goals_per90`, `assists_per90`, `goal_contributions_per90`, `yellows_per90`, `appearances` | Prior season |
| Player | `age`, `age_sq`, `height_in_cm`, `international_caps`, `is_english` | Static + computed |
| Position | `pos_DEF`, `pos_MID`, `pos_FWD` (dummies, GK=reference) | |
| Team context | `team_goals_scored`, `pct_team_minutes`, `pct_team_goals`, `club_transfer_spending_prior` | Prior season, selling club |
| League/Market | `league_tier` (ordinal 1–5), `from_league_spending`, `season_int` | Selling league |

**Excluded**: `tm_value_at_transfer` (leakage), `to_league` (buyer's choice), raw `minutes_played` (correlated with appearances + pct_team_minutes).

`league_tier` is a **fixed constant dict** — not data-driven, to prevent leakage:
```python
LEAGUE_TIER = {"GB1": 1, "ES1": 2, "IT1": 3, "L1": 4, "FR1": 5}
```

---

## Build Order

Each module depends on the previous — implement in this exact order:

```
1. src/features/inflation.py          [MODIFY]  add fee inflation index functions
2. src/features/engineering.py        [MODIFY]  add team context features to parquet
3. src/features/transfer_features.py  [NEW]     assemble transfer-level training dataset
4. src/model/__init__.py              [NEW]     empty
   src/model/train.py                 [NEW]     XGBoost + walk-forward CV + Optuna
   src/model/evaluate.py              [NEW]     RMSE, R², SHAP, residual plots
5. src/analysis/__init__.py           [NEW]     empty
   src/analysis/overpayment.py        [NEW]     residuals + transductive fair value
6. reports/valuation_report.md        [NEW]     final deliverable (after model runs)
```

New directories to create: `src/model/`, `src/analysis/`, `models/`, `data/processed/`

---

## Module Specs

### 1. `src/features/inflation.py` — extend, do not remove existing functions

Add two new public functions at the bottom:

```python
def build_fee_inflation_index(con: duckdb.DuckDBPyConnection) -> dict[int, float]:
    """
    Median top-5 transfer fee per season.
    Scope: all transfers where at least one club is in a top-5 league.
    Returns {season_int: median_fee}.
    """

def inflation_adjust_fee(fee: float, season_int: int, index: dict[int, float]) -> float:
    """Return fee / index[season_int]."""
```

SQL pattern for `build_fee_inflation_index`:
```sql
SELECT
    (2000 + CAST(SPLIT_PART(t.transfer_season, '/', 1) AS INTEGER)) AS season_int,
    MEDIAN(t.transfer_fee) AS median_fee
FROM transfers t
JOIN clubs fc ON fc.club_id = t.from_club_id
JOIN clubs tc ON tc.club_id = t.to_club_id
WHERE t.transfer_fee > 0
  AND (fc.domestic_competition_id IN ('GB1','L1','ES1','IT1','FR1')
       OR tc.domestic_competition_id IN ('GB1','L1','ES1','IT1','FR1'))
GROUP BY season_int
ORDER BY season_int
```

### 2. `src/features/engineering.py` — add team context CTEs

Add to `_STATS_SQL` (new CTEs before the final SELECT, extend the final SELECT):

**`team_goals` CTE** — club-season goals from `games` table (home + away via FULL OUTER JOIN):
```sql
team_goals AS (
    WITH home_g AS (
        SELECT home_club_id AS club_id, competition_id, season,
               SUM(home_club_goals) AS goals
        FROM games
        WHERE competition_id IN ('GB1','L1','ES1','IT1','FR1')
          AND season BETWEEN 2016 AND 2025
        GROUP BY home_club_id, competition_id, season
    ),
    away_g AS (
        SELECT away_club_id AS club_id, competition_id, season,
               SUM(away_club_goals) AS goals
        FROM games
        WHERE competition_id IN ('GB1','L1','ES1','IT1','FR1')
          AND season BETWEEN 2016 AND 2025
        GROUP BY away_club_id, competition_id, season
    )
    SELECT
        COALESCE(h.club_id, a.club_id)              AS club_id,
        COALESCE(h.competition_id, a.competition_id) AS competition_id,
        COALESCE(h.season, a.season)                AS season,
        COALESCE(h.goals,0) + COALESCE(a.goals,0)  AS team_goals_scored
    FROM home_g h
    FULL OUTER JOIN away_g a
        ON h.club_id = a.club_id
        AND h.competition_id = a.competition_id
        AND h.season = a.season
)
```

**`team_totals` CTE** — club-season totals from `appearances` for player-share %:
```sql
team_totals AS (
    SELECT
        a.player_club_id  AS club_id,
        g.competition_id,
        g.season,
        SUM(a.minutes_played) AS team_total_minutes,
        SUM(a.goals)          AS team_total_goals,
        SUM(a.assists)        AS team_total_assists
    FROM appearances a
    JOIN games g ON g.game_id = a.game_id
    WHERE g.competition_id IN ('GB1','L1','ES1','IT1','FR1')
      AND g.season BETWEEN 2016 AND 2025
    GROUP BY a.player_club_id, g.competition_id, g.season
)
```

**`primary_club` CTE** — resolves January movers to single club via QUALIFY:
```sql
primary_club AS (
    SELECT a.player_id, g.competition_id, g.season, a.player_club_id AS primary_club_id
    FROM appearances a
    JOIN games g ON g.game_id = a.game_id
    WHERE g.competition_id IN ('GB1','L1','ES1','IT1','FR1')
      AND g.season BETWEEN 2016 AND 2025
    GROUP BY a.player_id, g.competition_id, g.season, a.player_club_id
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY a.player_id, g.competition_id, g.season
        ORDER BY SUM(a.minutes_played) DESC
    ) = 1
)
```

**`club_spending` CTE**:
```sql
club_spending AS (
    SELECT
        to_club_id AS club_id,
        (2000 + CAST(SPLIT_PART(transfer_season,'/',1) AS INTEGER)) AS season,
        SUM(transfer_fee) AS club_transfer_spending
    FROM transfers
    WHERE transfer_fee > 0
    GROUP BY to_club_id, season
)
```

**`league_spending` CTE**:
```sql
league_spending AS (
    SELECT
        c.domestic_competition_id AS competition_id,
        (2000 + CAST(SPLIT_PART(t.transfer_season,'/',1) AS INTEGER)) AS season,
        SUM(t.transfer_fee) AS league_transfer_spending
    FROM transfers t
    JOIN clubs c ON c.club_id = t.to_club_id
    WHERE c.domestic_competition_id IN ('GB1','L1','ES1','IT1','FR1')
      AND t.transfer_fee > 0
    GROUP BY c.domestic_competition_id, season
)
```

**New columns in final SELECT** (join `primary_club`, `team_goals`, `team_totals`, `club_spending`, `league_spending`):
```sql
pc.primary_club_id,
COALESCE(tg.team_goals_scored, 0)        AS team_goals_scored,
COALESCE(tt.team_total_minutes, 0)       AS team_total_minutes,
COALESCE(tt.team_total_goals, 0)         AS team_total_goals,
s.minutes_played * 100.0 / NULLIF(tt.team_total_minutes, 0) AS pct_team_minutes,
s.goals * 100.0 / NULLIF(tt.team_total_goals, 0)            AS pct_team_goals,
COALESCE(cs.club_transfer_spending, 0)   AS club_transfer_spending,
COALESCE(ls.league_transfer_spending, 0) AS league_transfer_spending
```

Join conditions in the FROM/JOIN block:
```sql
LEFT JOIN primary_club pc ON pc.player_id = s.player_id
    AND pc.competition_id = s.competition_id AND pc.season = s.season
LEFT JOIN team_goals tg ON tg.club_id = pc.primary_club_id
    AND tg.competition_id = s.competition_id AND tg.season = s.season
LEFT JOIN team_totals tt ON tt.club_id = pc.primary_club_id
    AND tt.competition_id = s.competition_id AND tt.season = s.season
LEFT JOIN club_spending cs ON cs.club_id = pc.primary_club_id AND cs.season = s.season
LEFT JOIN league_spending ls ON ls.competition_id = s.competition_id AND ls.season = s.season
```

All existing columns remain — changes are additive only.

### 3. `src/features/transfer_features.py` — new module

**Season alignment (critical)**:
```
Transfer in season_int = S  →  prior stats from season S-1
                            →  team quality from selling club in season S-1
                            →  league spending from season S (current window)
                            →  inflation index from season S
```

Top-level SQL CTE chain:
1. `top5_transfers` — qualifying transfers (fee > 0, from_club in top-5, season_int 2017–2024)
2. `player_attrs` — join players table for static attributes
3. `prior_primary_club` — QUALIFY pattern to get primary club in season S-1
4. `prior_stats` — SUM goals/assists/minutes/cards in season S-1 across top-5 competitions
5. `team_goals` / `team_totals` — from selling club in season S-1
6. `club_spending` — selling club spending in season S-1
7. `league_spending` — from_league spending in season S
8. `inflation_index` — median fee in season S

Final WHERE: `WHERE ps.player_id IS NOT NULL` (require prior-season top-5 data).

**Python post-processing** (`_add_python_features(df)`):
- Per-90 ratios (guard `minutes_played == 0`)
- `age_sq`, `position_group` (import `POSITION_MAP` from `engineering.py`)
- `is_english = (nationality == "England").astype(int)`
- `log_transfer_fee = np.log(transfer_fee)`
- `log_inflation_adjusted_fee = np.log(transfer_fee / inflation_median_fee)` ← **primary target**
- `league_tier` from `LEAGUE_TIER` constant dict
- Position dummies (`pos_DEF`, `pos_MID`, `pos_FWD`)

Output: `data/features/transfer_dataset.parquet`

### 4. `src/model/train.py` — new module

**Walk-forward CV (expanding window)**:

| Fold | Train | Validate | ~Train rows |
|---|---|---|---|
| 1 | 2017–2019 | 2020 | ~160 |
| 2 | 2017–2020 | 2021 | ~215 |
| 3 | 2017–2021 | 2022 | ~280 |
| 4 | 2017–2022 | 2023 | ~340 |

Hold-out test: **2024 only** — never seen during Optuna tuning.

**Optuna** (100 trials, objective = mean CV RMSE in log-inflation space):
```python
params = {
    "n_estimators":     trial.suggest_int("n_estimators", 50, 500),
    "max_depth":        trial.suggest_int("max_depth", 2, 6),  # conservative for ~400 rows
    "learning_rate":    trial.suggest_float("lr", 0.01, 0.3, log=True),
    "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
    "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
    "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
}
```

After Optuna: retrain on full 2017–2023, evaluate once on 2024.

**Saves**:
- `models/xgb_v1.json` — serialized model
- `models/feature_cols.json` — explicit feature column list (contract between train and inference)
- `data/features/train_dataset.parquet`
- `data/features/test_dataset.parquet`
- `models/cv_results.parquet`

### 5. `src/model/evaluate.py` — new module

Metrics: RMSE, MAE, R² in log-inflation space. Convert predictions back to EUR.

Figures → `reports/figures/`:
- `shap_summary.png`, `shap_beeswarm.png`
- `residual_plot.png`, `actual_vs_predicted.png`
- `cv_fold_rmse.png`

### 6. `src/analysis/overpayment.py` — new module

**Task A — Transfer residuals** (all real transfers):
```python
residual = log_inflation_adjusted_fee - predicted
overpayment_factor = np.exp(residual)   # >1 = overpaid, <1 = bargain
predicted_fee_eur = np.exp(predicted) * inflation_median_fee
```
Output: `data/processed/transfer_residuals.parquet`
Figures: top-20 overpaid, top-20 bargains, by league, by position

**Task B — Transductive fair value** (all player-seasons):
- For player in season S: predict fee in window S+1 using season S stats
- Compare to TM: `tm_premium = market_value_in_eur / predicted_fee_eur`
- Output: `data/processed/player_season_fair_values.parquet`
- Figures: TM-vs-model scatter, overvalued/undervalued rankings

Public functions:
- `compute_transfer_residuals(model, transfer_df, feature_cols, inflation_index)`
- `apply_to_all_player_seasons(model, feature_matrix, feature_cols, inflation_index)`
- `plot_overpayment_top_n(residuals_df, n=20)`
- `plot_residuals_by_league(df)`, `plot_residuals_by_position(df)`
- `plot_tm_vs_model(fair_values_df)`

---

## Inflation Story (key finding, lead the report with this)

| Season | Median fee | Index (2017=100) |
|---|---|---|
| 2017 | €3.9M | 100 |
| 2019 | €4.6M | ~118 (peak) |
| 2020 | €4.0M | ~103 (COVID start) |
| 2021 | €2.0M | ~51 (structural compression) |
| 2022–2024 | €2.0–2.4M | ~51–62 (compressed) |

A €10M deal in 2022 is economically equivalent to a ~€23M deal in 2019.

---

## Report Structure (`reports/valuation_report.md`)

1. Methodology Overview
2. Data Pipeline + Scope
3. Transfer Fee Inflation Index *(inflation_index_plot.png)*
4. Feature Engineering — table with expected SHAP directions
5. Model Architecture and Training — CV fold table
6. Hold-Out Performance (2024): RMSE, MAE, R² *(actual_vs_predicted.png)*
7. SHAP Feature Importance *(shap_summary.png, shap_beeswarm.png)*
8. Historical Transfer Residuals — top-10 overpaid, top-10 bargains *(overpayment_top20.png)*
9. Breakdown by Position *(overpayment_by_position.png)*
10. Breakdown by League *(overpayment_by_league.png)*
11. TM Valuation vs Model Fair Value — all player-seasons *(tm_vs_model.png)*
12. Limitations and v2 Roadmap (contract length, advanced stats via FBRef)

---

## Verification Checklist

1. `transfer_dataset.parquet` shape ~(400–500, ~25 cols); `log_inflation_adjusted_fee` range roughly -2 to +5; no 2024 stats appearing as prior-season data for 2024 transfers
2. Inflation index plot: peak 2019, COVID drop 2020–2021, compressed recovery — matches real market history
3. Correlation audit before training: positives expected = `international_caps`, `team_goals_scored`, `goals_per90`; negatives = `age`, `yellows_per90`
4. Baseline OLS before XGBoost: expect R² ~0.3–0.4; XGBoost should beat this
5. CV fold table: flag if val_rmse > 2× others for any single fold
6. SHAP directions: `age` negative, `goals_per90` positive (FWDs/MIDs), `team_goals_scored` positive, `is_english` positive, `pct_team_minutes` positive
7. Residual plot: roughly centered at zero, no funnel pattern
8. Named-entity face-validity: top-5 most overpaid should include recognisable expensive mistakes

---

## Toolchain Commands

```sh
uv run python -m src.features.engineering          # rebuild feature matrix with team context
uv run python -m src.features.transfer_features    # build transfer-level training dataset
uv run python -m src.model.train                   # Optuna + walk-forward CV + final model
uv run python -m src.model.evaluate                # SHAP + residual figures
uv run python -m src.analysis.overpayment          # over/underpayment analysis + report figures
```

---

## v1 Omissions (document in report, address in v2)

- **Contract length** — not in dataset; use age as rough proxy in v1
- **Advanced stats** (xG, xA, progressive carries) — soccerdata/FBRef available but requires re-implementing player ID linker; omitted from v1
- **other_to_top5 transfers** — only 14 rows with prior top-5 stats; too sparse for v1
