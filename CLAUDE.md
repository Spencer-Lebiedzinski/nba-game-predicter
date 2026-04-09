# Claude Instructions for NBA Game Predictor

## README Maintenance Rule

**Every time a model change is made, update the `README.md` Model Decisions section.**

This includes:
- Adding, removing, or renaming features — document what was added and **why**
- Changing the model algorithm or hyperparameters — document the reasoning
- Adding new data sources or seasons — note what was added
- Fixing bugs (especially data leakage) — describe the bug and the fix
- Accuracy changes — update the benchmarks table

The goal is that any person — including the project owner returning months later — can read the README and understand exactly what the model does and why each decision was made. Never leave the README behind when changing `train_model.py`.

## Project Context

- **What it is:** NBA game outcome predictor + live market odds comparison (Polymarket/Kalshi)
- **ML model:** `train_model.py` → trains gradient boosting on 10 seasons of NBA data → saves `.pkl` artifacts
- **Web app:** `app.py` → Flask on port 5001, loads `.pkl` artifacts, serves predictions
- **Market data:** `unify_live_feed.py` → optional streaming from Polymarket + Kalshi WebSockets
- **Current accuracy target:** ~66-68% (up from v1's 60.2%)

## Key Files

| File | Purpose |
|---|---|
| `train_model.py` | Full model training pipeline — edit this when changing features or algorithm |
| `app.py` | Flask app — `predict_game()` must stay compatible with `feature_cols.pkl` structure |
| `team_features.pkl` | Per-team rolling feature snapshots — lookup key is team abbreviation (e.g. "BOS") |
| `feature_cols.pkl` | Ordered list of feature column names the model expects |
| `README.md` | **Keep this updated** — model decisions and reasoning |

## pkl Artifact Interface

`predict_game(home_team, away_team)` in `app.py` assembles the feature vector by:
1. For each feature in `feature_cols`, strip the `HOME_` or `AWAY_` prefix
2. Look up the remaining key in `team_features[team_abbrev]`
3. Special case: `HOME_HOME_WIN_RATE_L15` → looks up `"HOME_WIN_RATE_L15"` in home team's dict

When adding new features to `train_model.py`, ensure the same key (without `HOME_`/`AWAY_` prefix) is stored in `team_features_out[abbrev]` in the save step.

## Data Leakage Rule

All rolling features in `train_model.py` **must use `shift(1)` before the rolling window**. This prevents a game's features from including data from that same game. Example:

```python
# CORRECT
tg["NET_RATING_L10"] = tg.groupby("TEAM_ID")["NET_RATING"].transform(
    lambda s: s.shift(1).rolling(10, min_periods=3).mean()
)

# WRONG — leaks current game into its own features
tg["NET_RATING_L10"] = tg.groupby("TEAM_ID")["NET_RATING"].transform(
    lambda s: s.rolling(10, min_periods=3).mean()
)
```

`team_features.pkl` must store last-N-game rolling values, not full-season averages.
