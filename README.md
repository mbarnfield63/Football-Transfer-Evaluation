# Football Transfer Valuation Engine

A data science project estimating what professional footballers *should* be worth based on on-pitch performance, age trajectory, and market context — then using that model to test two specific hypotheses about transfer market inefficiencies.

**Data:** Transfermarkt valuations + FBRef per-season stats across Europe's top 5 leagues, 2017–2025  
**Model target:** `log(market_value_€)` — coefficients read as percentage changes in valuation  
**Output:** Reports, SHAP plots, and hypothesis test figures in `reports/`

---

## Research Questions

**1. Is there an "English tax"?**  
Do English players command a systematic premium in Transfermarkt valuations, controlling for age, position, and on-pitch performance? Estimated via OLS with a nationality indicator.

**2. Does World Cup participation boost valuations?**  
Does playing significant minutes at a World Cup (≥ 300 min) cause a measurable uplift in market valuation in the following transfer window? Estimated via difference-in-differences comparing treated vs. untreated players before and after the 2022 WC.

---

## Methods

### Data Pipeline
- **Transfermarkt** — per-season valuation snapshots and actual transfer fees, via `soccerdata`
- **FBRef** — per-season player stats (goals, xG, progressive carries, etc. per 90), via `soccerdata`
- **Player linking** — fuzzy name matching with `rapidfuzz` to join both sources on a stable player ID

### Feature Engineering
- Lag features (1-season) to capture trajectory
- Per-90 ratios normalised for playing time
- Position group encoding (GK / DEF / MID / FWD)
- Relative value score: each valuation divided by the median for its position group × season, removing market inflation

### Modelling
- **OLS baseline** — interpretable coefficients, directly used for the English tax estimate
- **XGBoost regressor** — tuned with Optuna, SHAP values for feature attribution
- **Time-aware cross-validation** — train on seasons ≤ Y, validate on Y+1; no random splits

### Hypothesis Testing
- **English tax** — OLS with `is_english` indicator and full performance controls; output is the log-point premium with confidence intervals
- **World Cup effect** — DiD regression: `log_valuation ~ post + treated + post×treated + controls`; the interaction coefficient is the causal estimate

---

## Repository Structure

```
├── src/
│   ├── scraping/
│   │   ├── tm_scraper.py        # Transfermarkt valuations & fees
│   │   ├── fbref_scraper.py     # FBRef per-season player stats
│   │   └── player_linker.py     # Fuzzy cross-source player ID map
│   ├── features/
│   │   ├── engineering.py       # Lag features, per-90s, position encoding
│   │   └── inflation.py         # Relative value score
│   └── hypotheses/
│       ├── english_tax.py       # OLS nationality premium
│       └── wc_effect.py         # Difference-in-differences WC boost
├── data/
│   ├── raw/                     # Scraped parquet files (gitignored)
│   ├── processed/               # Cleaned & player-linked data (gitignored)
│   └── features/                # Final feature matrix (gitignored)
├── reports/
│   └── figures/                 # SHAP plots, residual plots, DiD figures
├── requirements.txt
└── CLAUDE.md
```

---

## Setup

Requires Python 3.10+ and [`uv`](https://github.com/astral-sh/uv).

```sh
uv venv
uv pip install -r requirements.txt
```

---

## Running the Pipeline

```sh
# 1. Scrape data (test on one league/season before full batch)
uv run python -m src.scraping.tm_scraper
uv run python -m src.scraping.fbref_scraper

# 2. Build the player ID map (run once, cached to data/processed/)
uv run python -m src.scraping.player_linker

# 3. Feature engineering
uv run python -m src.features.engineering
uv run python -m src.features.inflation

# 4. Hypothesis testing
uv run python -m src.hypotheses.english_tax
uv run python -m src.hypotheses.wc_effect
```

---

## Results

*To be populated once the full data pipeline is complete.*

**English tax estimate:** TBD  
**World Cup valuation boost:** TBD  

Key figures will be saved to `reports/figures/`:
- SHAP summary plot (feature importance)
- Residual plot by nationality (English tax visual)
- DiD parallel trends and post-treatment effect plot

---

## Data Sources

| Source | Access | Notes |
|--------|--------|-------|
| [Transfermarkt](https://www.transfermarkt.com) | `soccerdata` TM reader | Valuations, fees, nationality, position |
| [FBRef](https://fbref.com) | `soccerdata` FBref reader | Per-season stats, top 5 leagues |

Leagues: Premier League, Bundesliga, La Liga, Serie A, Ligue 1 — seasons 2017–2025.

Raw data is not committed to the repo (scraped on demand, gitignored).
