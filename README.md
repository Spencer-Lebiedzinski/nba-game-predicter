# NBA Game Predictor

A machine learning system that predicts NBA game outcomes and compares predictions against live market odds from Polymarket and Kalshi.

The repository now ships **two** prediction backends side-by-side:

| Version | Backend | Algorithm | Status |
|---|---|---|---|
| v2 | `gbm` *(default)* | XGBoost / sklearn GradientBoosting on tabular rolling features | Production |
| v3 | `transformer` | Two-tower shared-weight Transformer over each team's recent game sequence (PyTorch) | Opt-in |

The Transformer is selected with the `MODEL_BACKEND=transformer` env var. See **v3: Transformer** below for the architecture, multi-task loss, and walk-forward results.

## Quick Start

```bash
source .venv/bin/activate
pip install nba_api flask scikit-learn joblib requests websockets aiohttp python-dateutil xgboost
pip install torch --index-url https://download.pytorch.org/whl/cpu   # only required for the Transformer backend

# Train the v2 GBM (fetches ~10 seasons via nba_api, takes several minutes)
cd models && python train_model.py

# Train the v3 Transformer (uses the cached games table after the first run)
cd models && python train_transformer.py

# Run the web app — defaults to the GBM
python app.py
# → http://127.0.0.1:5001

# Or switch to the Transformer backend (PowerShell):
$env:MODEL_BACKEND = "transformer"; python app.py
# bash/zsh:
MODEL_BACKEND=transformer python app.py

# Optional: stream live market odds alongside predictions
python unify_live_feed.py
```

---

## Model Decisions

This section documents every significant modeling choice and why it was made. It is kept up to date as the project evolves.

The repo has gone through three versions:

* **v1** — Logistic Regression on 12 hand-picked features. ~60.2% accuracy.
* **v2** — Gradient Boosting on ~50 derived features (Four Factors, Net Rating, Elo, schedule fatigue, home/road splits). The bulk of this README documents v2.
* **v3** — Two-tower Transformer encoder over each team's recent game sequence, with a multi-task win + score head. Lives in `models/transformer_*.py` and is selected via the `MODEL_BACKEND=transformer` env var. Documented in [**v3: Transformer**](#v3-transformer-two-tower-shared-weight-sequence-model) at the bottom of this section.

The v2 GBM remains the **default** backend so the app never breaks if the Transformer artifacts are absent; the Transformer is opt-in and explicitly logged at startup.

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

**Benchmarks (walk-forward CV unless noted):**

| Model | Accuracy | Notes |
|---|---|---|
| Naive — always pick home | ~59% | Home win rate baseline |
| v1 — Logistic Regression, 12 features | ~60.2% | Linear · pre-rebuild |
| v2 — GBM (Four Factors + Elo + GBM) | **~66–68%** | Tabular · ~50 features |
| **v3 — Transformer (two-tower, multi-task)** | **64.5% mean (66.2% best fold)** | Sequence model · 619,779 params · AUC 0.69, Brier 0.22 |
| FiveThirtyEight (Elo + RAPTOR) | ~70–72% | Industry reference |
| Vegas closing lines (ceiling) | ~73–75% | Practical ceiling |

**v3 Transformer walk-forward detail** *(from `transformer_run_summary.json`)*:

| Fold | Train | Test | Accuracy | LogLoss | AUC | Brier |
|---|---|---|---|---|---|---|
| 1 | 2015-2020 | 2021-22 | 66.23% | 0.6335 | 0.687 | 0.221 |
| 2 | 2015-2021 | 2022-23 | 62.31% | 0.6470 | 0.653 | 0.228 |
| 3 | 2015-2022 | 2023-24 | 64.46% | 0.6190 | 0.709 | 0.215 |
| 4 | 2015-2023 | 2024-25 | 64.95% | 0.6213 | 0.706 | 0.216 |
| **Mean** | | | **64.49%** | **0.6302** | — | — |

The Transformer is roughly at parity with the GBM on this dataset — which is the expected result for ~13k training games. Transformers shine when (a) data is much larger or (b) per-token features are richer than what the GBM already gets to use. The win here isn't a raw accuracy bump, it's that the same architecture **outputs a calibrated joint distribution over scores** (the auxiliary regression heads), which the GBM cannot — and this distribution is the conditioning input for the v4 diffusion model.

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

### v3: Transformer (Two-Tower, Shared-Weight Sequence Model)

The v2 GBM treats each game as a flat row of ~50 summary statistics. That works, but it discards the *order and texture* of how a team has been playing — a team that started 8-2 then went 2-8 has the same L10 win rate as one that did the reverse, but they are very different teams headed into game 21. v3 fixes that by treating each team's recent N=20 games as a **sequence** and learning the representation with a small Transformer encoder. This is also the natural backbone for the upcoming diffusion-based outcome model.

#### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  home last-20 games ──► TeamEncoder ──► [CLS] vec  (d=128)      │
│  away last-20 games ──► TeamEncoder ──► [CLS] vec  (d=128)      │   shared weights
│  game context (Elo, rest, B2B) ──► MLP ──► ctx vec (d=128)      │
│                                                                  │
│  concat (3*d) ──► shared head MLP ──► { win logit,              │
│                                          home score,             │
│                                          away score }            │
└─────────────────────────────────────────────────────────────────┘
```

* **`TeamEncoder`** is a pre-norm Transformer encoder (4 layers, d_model=128, 4 heads, FFN=256, dropout=0.1) applied **with shared weights** to both home and away sequences. Sharing is the right inductive bias — the meaning of "Boston's last 20 games" doesn't change based on whether Boston happens to be the home team in the *next* game (the venue of each historical game is already encoded inside the token as `IS_HOME`).
* **Learned `[CLS]` token at position 0**, with a learned positional embedding. Sequence length is fixed at 20 so sinusoidal encoding isn't worth the extra complexity.
* **Attention padding mask** (`src_key_padding_mask`) blocks the encoder from attending to left-padded positions for teams with <20 prior games (early-season examples).

#### Per-token feature schema (each token = one historical game)

22 features per game-token, mixing binary and continuous:

| Group | Features |
|---|---|
| Binary (not normalized) | `IS_HOME`, `WIN`, `B2B` |
| Box score outcome | `MARGIN`, `PTS`, `OPP_PTS` |
| Dean Oliver Four Factors (own + opp) | `EFG_PCT`, `OPP_EFG_PCT`, `TOV_PCT`, `OPP_TOV_PCT`, `OREB_PCT`, `OPP_OREB_PCT`, `FTR`, `OPP_FTR` |
| Efficiency / style | `TS_PCT`, `THREE_PAR`, `AST_TOV`, `NET_RATING`, `PACE` |
| Schedule | `DAYS_REST` |
| Opponent strength | `TEAM_ELO_PRE`, `OPP_ELO_PRE` |

`OPP_ELO_PRE` lets the encoder weight each historical game by strength of opposition — a 10-point win over a 1700-Elo team is very different from one over a 1400-Elo team, and the Transformer learns that gradient instead of us hand-coding strength-of-schedule.

Continuous features are z-scored using **training-set-only** statistics, saved in `transformer_norm_stats.pkl` so inference applies the same normalization. Binary features pass through with mean=0, std=1.

#### Game-context vector (8 features)

`HOME_ELO`, `AWAY_ELO`, `ELO_DIFF`, `HOME_DAYS_REST`, `AWAY_DAYS_REST`, `REST_DIFF`, `HOME_B2B`, `AWAY_B2B`. These are about the *upcoming* matchup; the per-token features are about history. Keeping them separate prevents the model from confusing "this game" with "a past game."

#### Multi-task loss

```
L = BCE(home_win) + 0.01 * MSE(home_score) + 0.01 * MSE(away_score)
```

* **Why predict scores at all?** Three reasons. (1) Score regression is a stronger learning signal than binary win/loss — it teaches the model the *magnitude* of expected dominance, not just the sign. (2) It regularizes the win head by forcing shared representations to support both tasks. (3) Predicted (home_score, away_score) gives the spread and total *for free*, which is exactly what the Polymarket/Kalshi UI needs.
* **Why weight scores at 0.01?** Score variance is ~150 pts² per side vs. BCE which is O(1). Without down-weighting, the MSE term dominates and the win head underfits. 0.01 keeps win classification as the primary objective while still extracting the auxiliary signal.

#### Training setup

* PyTorch, CPU-friendly. ~620k parameters — small enough to train without a GPU in tens of minutes per fold.
* AdamW, lr=3e-4, weight_decay=1e-2, batch_size=256, 30 epochs with patience=5 early stopping on held-out log-loss.
* Cosine LR schedule with 10% linear warmup.
* Gradient clipping at norm 1.0.
* Walk-forward CV folds match the v2 GBM exactly: hold out 2020-21, 2021-22, 2022-23, 2023-24 in turn. Normalization stats are fitted **only** on each fold's training seasons — fitting on the test season would leak global statistics into the model.

#### Inference path

At prediction time, `team_sequences.pkl` provides each team's most recent 20 game-tokens (raw, un-normalized). `app.py` builds the inference input by:
1. Loading the home and away sequences and their padding masks
2. Applying the saved z-score stats
3. Computing the upcoming-game context (rest days / B2B from each team's `LAST_GAME_DATE`, Elo from `team_features.pkl`)
4. Running a single forward pass under `torch.no_grad()`

If `team_sequences.pkl` is missing for a team, the app silently falls back to the GBM prediction so the UI never breaks.

#### Why this is a better foundation for v4 (diffusion / flow matching)

A score-regression head outputs a single best guess — but the actual distribution of (home_score, away_score) is bimodal-ish (blowouts vs. close games) and asymmetric in tails. A diffusion (or flow-matching) model conditioned on the same fused `[home_vec, away_vec, ctx_vec]` representation can learn the *full joint distribution* of outcomes, from which win probability, spread, and total fall out as marginal queries. v3 builds the conditioning embedding; v4 will replace the deterministic score heads with a generative one.

---

---

## Web Application

The Flask app is no longer "a form and a results page" — it's a full product surface for the ML engine. Everything lives behind a unified design system in `static/css/app.css` (no build step, no framework — vanilla CSS + a small JS module).

### Pages

| Route | What it does |
|---|---|
| `/` (dashboard) | Hero with active-model card, 4-up KPI strip (games today / edge bets / biggest edge / teams tracked), featured-matchup card showing the largest edge, today's slate grid, and a "Under the hood" capabilities row. |
| `/upcoming` | Filterable + sortable game list. Chips for **All / Best bets / High confidence / Toss-ups / Has market**. Sort by tip-off, edge, or confidence. Each card shows model + market split bars, edge meter with magnitude track, predicted score (Transformer backend), team form dots, B2B flags, streak pills, rest days. Auto-refresh every 30s. |
| `/predict` | Manual matchup builder. Two team selects with live logo previews, a side panel that explains what the active backend will output (different copy for GBM vs Transformer), and a single primary CTA. |
| `/predict` (POST → result) | Big matchup hero, semicircular win-probability gauge (animated SVG), confidence band, side-by-side stat comparison table with better/worse coloring, recent form rows. When the Transformer is active: also shows predicted final score, spread, and total. |
| `/team/<ABBR>` | Per-team page. Logo + name hero with stat pills, KPI strip (eFG%, TOV%, OREB%, Pace), three sparklines (points scored, margin, net rating) drawn client-side from the Transformer's `team_sequences.pkl`, home-vs-road split cards, and a sortable last-20-games table. |
| `/models` | Side-by-side model cards (GBM vs Transformer) with parameters, architecture summary, and an **Active / Standby** indicator that reflects the current backend. Walk-forward CV table with all folds + mean. Full benchmarks table. |

### Design system

- **Typography:** Inter (300-900, variable) for UI, JetBrains Mono for metrics. Tabular numerals enabled globally so number columns stay aligned.
- **Palette:** layered deep-slate backgrounds, electric-blue + violet brand colors, lime / amber / red for status. Subtle radial gradients in the hero and behind the brand mark. All colors live in CSS variables in `app.css`.
- **Components in the library:** `.card` (+ variants `card-glow`, `card-violet`), `.kpi`, `.pill` (+ accent / violet / lime / amber / red), `.prob-bar` with animated fill, `.score-pred` chips, `.gauge` (SVG semicircle), `.chip-group` filter, `.form-dot` win/loss markers, sparkline renderer, `.badge-confidence`, toast system, and a fully-keyboard-navigable command palette.
- **Motion:** entry animations are staggered via the `.stagger > *:nth-child(...)` selectors; probability bars animate their fill on load via a small JS bootstrapper; sparklines are drawn from `data-sparkline="..."` attributes.

### Command palette (Cmd/Ctrl + K)

A Linear-style overlay that searches across:
- **Pages** — jump to any route
- **Actions** — switch backend, with a "Switch to Transformer" / "Switch to GBM" entry depending on current state
- **Teams** — type any team name or abbreviation to jump to its detail page

Keyboard navigation: `↑`/`↓` to focus, `Enter` to run, `Esc` to close. Implementation in `static/js/app.js`.

### Backend switching

The top-right nav pill shows the active backend and which `[indicator]` color is pulsing. Clicking it calls `POST /api/backend` which writes a `backend` cookie (30-day TTL) and reloads. The app **eagerly loads both** GBM and Transformer artifacts at startup if available, so switching is instantaneous — no model load on toggle. If the Transformer artifacts are missing the switch is disabled with a tooltip explaining how to build them.

### JSON API

| Endpoint | Returns |
|---|---|
| `GET /api/upcoming` | List of today's games with model probabilities |
| `GET /api/predict?home=BOS&away=LAL` | Full prediction dict (incl. score predictions on Transformer backend) |
| `GET /api/backend` | Current backend + transformer-available flag |
| `POST /api/backend {"backend": "transformer"}` | Switch backend; sets cookie |

The `/api/predict` endpoint is what the command palette will use for inline previews in a future iteration.

### Screenshots

Reference screenshots of every page live in [`docs/screenshots/`](docs/screenshots/):
- `ui-dashboard.png` — homepage / dashboard
- `ui-upcoming.png` — game list (empty state shown)
- `ui-predict.png` — manual matchup builder
- `ui-result.png` / `ui-result-transformer.png` — GBM and Transformer prediction results, side-by-side
- `ui-team-2.png` — Boston Celtics team detail page with sparklines + last-20 table
- `ui-models.png` — models & metrics page
- `ui-cmdk.png` — command palette open

---

### Future Improvements

- [x] **v3: sequence model (Transformer) over recent game tokens** — shipped
- [x] **Full UI revamp** — design system, dashboard, command palette, models page, team detail pages — shipped
- [ ] **v4: diffusion / flow-matching over (home_score, away_score) joint distribution** — conditioned on the v3 Transformer's fused embedding; gives calibrated quantiles, not just point estimates
- [ ] **LLM agent layer** — tool-using agent that pulls injury reports + news (ESPN / Rotowire) and adjusts the base model's prediction with visible reasoning; RAG over recent commentary
- [ ] **Reinforcement learning bet sizer** — given model probability + live market odds, learn a Kelly-aware sizing policy with PPO
- [ ] **Uncertainty quantification on the GBM** — conformal prediction wrappers so v2 also outputs intervals, not just point probs
- [ ] **MLOps stack** — MLflow / Weights & Biases experiment tracking, GitHub Actions for nightly retrain + drift check, calibration-plot dashboard
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
├── app.py                            # Flask web app — dispatches to GBM or Transformer backend per cookie
├── live_feed_reader.py               # JSONL reader for Polymarket/Kalshi live feed
│
├── models/
│   ├── train_model.py                # v2 GBM training (XGBoost / sklearn GBM)
│   ├── train_transformer.py          # v3 Transformer training driver (walk-forward CV + final model)
│   ├── transformer_model.py          # PyTorch TeamEncoder + NBATransformer module definitions
│   ├── transformer_data.py           # Sequence dataset builder + normalization stats
│   ├── data_loader.py                # Cached nba_api fetcher + team-log builder (shared)
│   ├── cache/                        # Parquet caches of raw games + team_log (auto-generated)
│   ├── nba_predictor_model.pkl       # v2 GBM artifact
│   ├── feature_cols.pkl              # v2 feature column names
│   ├── team_features.pkl             # v2 per-team rolling-feature snapshot
│   ├── teams.pkl                     # NBA team abbreviation list
│   ├── transformer_model.pt          # v3 PyTorch weights
│   ├── transformer_config.pkl        # v3 architecture config
│   ├── transformer_norm_stats.pkl    # v3 z-score parameters
│   ├── team_sequences.pkl            # v3 last-20-game token sequence per team
│   └── transformer_run_summary.json  # v3 CV metrics + final-model metrics
│
├── templates/
│   ├── base.html                     # Layout: nav, command palette modal, toast wrap, footer
│   ├── dashboard.html                # `/` — hero, KPI strip, featured matchup, today's slate, capabilities
│   ├── upcoming.html                 # `/upcoming` — filterable + sortable game cards
│   ├── predict.html                  # `/predict` (GET) — manual matchup builder
│   ├── result.html                   # `/predict` (POST) — gauge + stat compare + score prediction
│   ├── team_detail.html              # `/team/<ABBR>` — team page with sparklines + last 20 games
│   ├── models.html                   # `/models` — GBM vs Transformer side-by-side + walk-forward CV
│   └── index.html                    # (legacy — no longer routed; kept for reference)
│
├── static/
│   ├── css/app.css                   # Design system (variables, components, animations)
│   ├── js/app.js                     # Command palette, sparkline renderer, toast, backend toggle
│   └── img/logo.png                  # Favicon / brand mark
│
├── market_agents/
│   ├── discover_polymarket_nba.py    # Find active NBA markets on Polymarket
│   ├── discover_kalshi_nba.py        # Find active NBA markets on Kalshi
│   ├── stream_polymarket.py          # Polymarket WebSocket price streamer
│   ├── stream_kalshi.py              # Kalshi WebSocket price streamer
│   └── unify_live_feed.py            # Runs both streamers, writes JSONL
│
└── docs/
    └── screenshots/                  # Reference screenshots of every UI page
```

**Backend selection:** `app.py` reads `MODEL_BACKEND` from the environment. Default is `gbm`. Set to `transformer` to use the v3 model. The GBM artifacts are always loaded as a fallback so prediction never fails when the Transformer is mid-retraining or its artifacts are missing.
