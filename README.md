# NBA Game Predictor

A machine learning system that predicts NBA game outcomes and compares predictions against live market odds from Polymarket and Kalshi.

## Quick Start

```bash
source .venv/bin/activate
pip install nba_api flask scikit-learn joblib requests websockets aiohttp python-dateutil xgboost

# Train the model (fetches ~10 seasons of data, takes several minutes)
python train_model.py

# Run the web app
python app.py
# → http://127.0.0.1:5001

# Optional: stream live market odds alongside predictions
python unify_live_feed.py
```

---

## Model Decisions

This section documents every significant modeling choice and why it was made. It is kept up to date as the project evolves.

---

### Algorithm: Gradient Boosting (XGBoost / sklearn GBM)

**Why not Logistic Regression (v1)?**
Logistic Regression is a linear model — it can only learn additive combinations of features. NBA outcomes depend on non-linear interactions: a well-rested team's advantage is bigger when playing a team on a back-to-back; a high-Elo team's home court edge amplifies against weaker opponents. Gradient Boosting learns these interaction effects natively through its tree structure.

**Why Gradient Boosting over Random Forest or a neural net?**
- Gradient Boosting typically outperforms Random Forest on tabular data with ~50 features because it trains sequentially, correcting prior errors.
- A neural net would require far more data and hyperparameter tuning to beat a well-configured GBM on a dataset of ~10k games.
- GBM provides feature importances, making the model auditable and debuggable.

**Hyperparameters chosen:**
- `n_estimators=500, learning_rate=0.05` — slow learning rate with more trees reduces overfitting vs. fewer trees at a higher rate
- `max_depth=4` — shallow trees encourage learning feature interactions without memorizing noise
- `subsample=0.8, colsample_bytree=0.8` — row and column subsampling further reduce overfitting (same principle as Random Forest's bagging)

---

### Feature Set

#### Elo Ratings (FiveThirtyEight Methodology)

Elo is a running strength-of-schedule-adjusted team rating updated after every game. It is the single most predictive feature in the model.

**Implementation details (matching FiveThirtyEight):**
- **K=20** — update factor; basketball has lower randomness than baseball/hockey so K is higher than those sports
- **Home court +100 Elo points** — equivalent to ~3.5 points on the spread for an average team; baked into the expected-outcome calculation
- **MOV multiplier: `(|MOV| + 3)^0.8 / (7.5 + 0.006 * |Elo_diff|)`** — diminishing returns for blowouts; prevents a team from inflating Elo by running up scores against weak opponents
- **Season regression: 75% prior Elo + 25% toward mean (1505)** — accounts for off-season roster turnover; without this, last season's champion stays overrated all the following season

Why Elo beats raw winning percentage: it implicitly encodes strength of schedule. A team that went 15-5 beating playoff teams is correctly rated higher than a team that went 15-5 beating lottery teams.

---

#### Four Factors (Dean Oliver Framework)

Dean Oliver's 2004 research identified four factors that collectively explain ~97% of the variance in team winning percentage. These are computed from basic box score stats (no extra API calls needed).

| Factor | Formula | Weight |
|---|---|---|
| Effective FG% (eFG%) | (FGM + 0.5 × FG3M) / FGA | 40% |
| Turnover Rate (TOV%) | TOV / (FGA + 0.44×FTA + TOV) | 25% |
| Offensive Rebound % (OREB%) | OREB / (OREB + Opp DREB) | 20% |
| Free Throw Rate (FTR) | FTA / FGA | 15% |

All four factors are computed for both the team **and their opponents**, giving eight Four Factor features per team. The opponent versions measure defensive quality (e.g., forcing turnovers, preventing offensive rebounds).

**Why include all eight instead of just the four offensive ones?** A team can win by being elite offensively (high eFG%) or elite defensively (forcing low opponent eFG%). The model needs both signals. Oliver's original weights apply to the offensive factors; the opponent factors carry the same relative weights but on the defensive side.

---

#### Net Rating (Pace-Normalized Efficiency)

Net Rating = Offensive Rating − Defensive Rating, where:
- Offensive Rating = Points scored per 100 possessions
- Defensive Rating = Points allowed per 100 possessions
- Possessions estimated via Oliver formula: `FGA − OREB + TOV + 0.44×FTA`

**Why pace-normalize?** Raw point differential conflates team quality with game pace. A team that wins 115-105 in a fast-paced 100-possession game is less dominant than a team that wins 105-95 in a slow 88-possession game — both have +10 margin but the latter achieved it in fewer opportunities. Net Rating removes this distortion.

Net Rating rolling L10 replaces raw Point Differential as the primary efficiency signal. Raw PPG and PAPG L10 are still included as secondary features.

---

#### Schedule and Fatigue Features

**Back-to-back flag (B2B):** 1 if the team played yesterday, else 0. Research consistently shows road teams on back-to-backs lose ~4-5% more than baseline. Vegas prices this explicitly at 1.5-3 points on the spread. The model learns this effect from the data rather than hard-coding it.

**Rest days (DAYS_REST):** Days since last game, clipped at 14. Captures diminishing returns on additional rest (a team with 5 days rest is not meaningfully fresher than one with 3).

**Games in last 7 days (GAMES_L7):** Counts prior games in the rolling 7-day window. This captures multi-game fatigue: a team playing their 4th game in 6 days is more fatigued than their simple last-rest-day number suggests.

**Dynamic B2B for upcoming games:** When predicting upcoming games, B2B and DAYS_REST are computed from each team's `LAST_GAME_DATE` stored in `team_features.pkl`. This means the app automatically reflects real schedule fatigue for games shown in the upcoming-games dashboard.

---

#### Home / Away Performance Splits

**Why separate home and road rolling windows?** Some teams are dramatically different at home vs. on the road — Denver's altitude advantage, Boston's crowd effects, Utah's historical home dominance. A team's overall rolling win rate blends these together. By maintaining separate L15 rolling windows for home games and road games, the model can distinguish a team that is 10-5 overall but 8-2 at home from one that is 10-5 but 5-5 at home.

The model uses:
- **Home team's home-game win rate and net rating (L15 home games)**
- **Away team's road-game win rate and net rating (L15 road games)**

These are the meaningful combinations: we want to know how the home team performs *at home* and how the away team performs *on the road*.

---

#### Win/Loss Streak

A team on a 7-game winning streak is in different form than their rolling averages suggest — the recent games carry more signal about momentum and chemistry. The streak feature encodes: positive integers = win streak length, negative integers = loss streak length. Shift(1) applied so the model sees the streak *entering* the game, not including it.

---

#### Shooting Profile: TS%, 3PAR, AST/TOV

- **True Shooting % (TS%):** `PTS / (2 × (FGA + 0.44×FTA))` — a more accurate shooting efficiency metric than FG% because it weighs 3-pointers and free throws correctly.
- **3-Point Attempt Rate (3PAR):** `FG3A / FGA` — captures playing style. High-3PAR teams have higher variance; style mismatches (e.g., a 3PAR-heavy team vs. elite 3-point defense) can be predictive.
- **AST/TOV Ratio:** Ball-movement quality and decision-making under pressure.

---

### Data

**Seasons:** 2015-16 through 2024-25 (10 seasons). Added 2023-24 and 2024-25 vs. v1's 8 seasons.

**Source:** `nba_api` LeagueGameFinder endpoint — no additional API keys or scrapers needed. All advanced stats (Four Factors, Net Rating, etc.) are computed from the basic box score columns returned by this endpoint.

**Why 2015 and not earlier?** The 3-point era accelerated meaningfully around 2015; earlier seasons have different pace and shot-selection distributions that could introduce noise. 10 seasons provides ~12,000 training games.

---

### Evaluation: Walk-Forward Cross-Validation

**Why not a simple 80/20 split?** A single time-based split gives one accuracy estimate, which can be lucky or unlucky depending on which season falls in the test set. Walk-forward CV tests on four consecutive held-out seasons:

| Fold | Train | Test |
|---|---|---|
| 1 | 2015–2020 | 2020-21 |
| 2 | 2015–2021 | 2021-22 |
| 3 | 2015–2022 | 2022-23 |
| 4 | 2015–2023 | 2023-24 |

This mirrors actual deployment: the model always predicts games using only past data. Mean accuracy across folds is the headline metric.

**Benchmarks:**
| Model | Accuracy |
|---|---|
| Naive (always pick home team) | ~59% |
| v1: Logistic Regression, 12 features | ~60.2% |
| v2 target (Four Factors + Elo + GBM) | ~66–68% |
| FiveThirtyEight Elo + RAPTOR | ~70–72% |
| Vegas closing lines (ceiling) | ~73–75% |

---

### Data Leakage Fix

**v1 bug:** `team_features.pkl` stored full-season averages per team. These averages included games played *after* the game being predicted, meaning the model had access to future information during training.

**v2 fix:** `team_features.pkl` stores the most recent rolling-window snapshot from each team's last game in the training data. Specifically:
- Overall features (Net Rating L10, eFG% L10, etc.) come from the team's last game row
- Home split features come from the team's last *home* game row
- Road split features come from the team's last *road* game row

The rolling transforms themselves use `shift(1)` throughout, ensuring each game's features are computed from prior games only.

---

### Live Market Integration

When `unify_live_feed.py` is running, the upcoming-games dashboard shows:
- **Model probability** from the ML model
- **Market probability** from Polymarket and/or Kalshi live orderbooks
- **Edge** = Model probability − Market probability

Positive edge means the model thinks a team is undervalued relative to the market — a potential betting signal. The market is the ceiling (~73–75% implied accuracy from closing lines), so large persistent edges often indicate model error rather than true mispricing. Use cautiously.

---

### Future Improvements

- [ ] Star player availability / injury flags (ESPN injury API or Rotowire scrape) — moves Vegas lines 4-8 points, highest-value signal not yet in model
- [ ] Opponent-adjusted net rating (SRS-style) — controls for schedule strength beyond Elo
- [ ] Player lineup quality score (sum of rotation players' rolling +/-)
- [ ] Travel distance proxy (city coordinate lookup) — cross-timezone travel impacts performance
- [ ] ELO + model ensemble weighting (FiveThirtyEight found 35% Elo + 65% player ratings optimal)
- [ ] Historical edge backtesting against closing lines

---

## File Structure

```
nba-game-predicter/
├── train_model.py              # Model training (run this first)
├── app.py                      # Flask web app (port 5001)
├── discover_polymarket_nba.py  # Find active NBA markets on Polymarket
├── discover_kalshi_nba.py      # Find active NBA markets on Kalshi
├── stream_polymarket.py        # Polymarket WebSocket price streamer
├── stream_kalshi.py            # Kalshi WebSocket price streamer
├── unify_live_feed.py          # Runs both streamers, writes JSONL
├── live_feed_reader.py         # JSONL reader for Flask
├── nba_predictor_model.pkl     # Trained model (generated by train_model.py)
├── team_features.pkl           # Per-team rolling features (generated)
├── feature_cols.pkl            # Feature column names (generated)
├── teams.pkl                   # NBA team list (generated)
└── templates/
    ├── index.html              # Manual prediction form
    ├── result.html             # Prediction result page
    └── upcoming.html           # Upcoming games + market odds dashboard
```
