"""
NBA Game Predictor - Enhanced Model Training v2
===============================================
Improvements over v1:
  - Gradient Boosting (XGBoost or sklearn GBM) replaces Logistic Regression
  - Four Factors (Dean Oliver): eFG%, TOV%, OREB%, FTR + opponent versions
  - Elo ratings: FiveThirtyEight-style with MOV multiplier, home advantage,
    and season regression toward mean
  - Net Rating (pace-normalized) replaces raw point differential
  - True Shooting %, 3-Point Attempt Rate, AST/TOV ratio
  - Back-to-back flags (single rest day = B2B)
  - Home/away performance splits (separate rolling windows per venue type)
  - Win/loss streak going into each game
  - Games played in last 7 days (cumulative fatigue proxy)
  - Expanded data: 2015-2025 (added 2023-24 and 2024-25 seasons)
  - Walk-forward cross-validation across 4 held-out seasons
  - Fixed data leakage: team_features.pkl now stores last-N-game rolling
    values per team (not full-season averages which include future games)
"""

import time
import numpy as np
import pandas as pd
import joblib
from nba_api.stats.endpoints import leaguegamefinder
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

# ── MODEL SELECTION ──────────────────────────────────────────────────────────
# XGBoost preferred; falls back to sklearn GradientBoostingClassifier if not
# installed. Both outperform Logistic Regression on non-linear NBA patterns.
try:
    from xgboost import XGBClassifier
    def make_model():
        return XGBClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric='logloss', random_state=42, n_jobs=-1,
            verbosity=0
        )
    MODEL_NAME = "XGBoost"
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    def make_model():
        return GradientBoostingClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42
        )
    MODEL_NAME = "GradientBoosting (sklearn)"

print(f"Model: {MODEL_NAME}")

# ── 1. FETCH GAME DATA ───────────────────────────────────────────────────────
print("\nFetching game data (2015-2025)...")
all_dfs = []
for season in range(2015, 2025):
    season_str = f'{season}-{str(season+1)[-2:]}'
    print(f"  {season_str}...", end=" ", flush=True)
    gf = leaguegamefinder.LeagueGameFinder(season_nullable=season_str)
    s = gf.get_data_frames()[0]
    s["SEASON"] = season
    all_dfs.append(s)
    print("ok")
    time.sleep(0.6)  # respect nba_api rate limits

raw = pd.concat(all_dfs, ignore_index=True)
raw["GAME_DATE"] = pd.to_datetime(raw["GAME_DATE"])

# ── 2. BUILD HOME/AWAY GAME TABLE ────────────────────────────────────────────
# One row per game with HOME_ and AWAY_ prefixed columns.
KEEP = ["GAME_ID", "GAME_DATE", "SEASON", "TEAM_ID", "TEAM_ABBREVIATION",
        "PTS", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
        "OREB", "DREB", "AST", "TOV"]

ID_COLS = ["GAME_ID", "GAME_DATE", "SEASON"]
home_raw = raw[raw["MATCHUP"].str.contains("vs\\.", na=False)][KEEP].copy()
away_raw = raw[raw["MATCHUP"].str.contains("@", na=False)][KEEP].copy()
home_raw = home_raw.rename(columns={c: f"HOME_{c}" for c in KEEP if c not in ID_COLS})
away_raw = away_raw.rename(columns={c: f"AWAY_{c}" for c in KEEP if c not in ID_COLS})

games = (home_raw.merge(away_raw, on=ID_COLS, how="inner")
         .sort_values("GAME_DATE").reset_index(drop=True))
games["HOME_WIN"] = (games["HOME_PTS"] > games["AWAY_PTS"]).astype(int)
games["MOV"] = games["HOME_PTS"] - games["AWAY_PTS"]

print(f"\nGames loaded: {len(games)}  |  Home win rate: {games['HOME_WIN'].mean():.3f}")

# ── 3. ELO RATINGS ───────────────────────────────────────────────────────────
# FiveThirtyEight methodology:
#   K=20       — standard update factor for basketball
#   +100 home  — home court worth ~3.5 pts on spread
#   MOV mult   — diminishing returns for blowouts (prevents runaway Elo)
#   75/25 regression at season start — accounts for roster turnover
#
# Pre-game Elo is stored per game so training sees only prior information.
print("Computing Elo ratings...")

K         = 20
HOME_ADV  = 100
ELO0      = 1500
MEAN_ELO  = 1505
REGRESS   = 0.25

elo_ratings = {}
home_elos, away_elos = [], []
last_season = None

for _, row in games.iterrows():
    # Season boundary: regress all ratings toward mean
    if row["SEASON"] != last_season:
        last_season = row["SEASON"]
        if elo_ratings:
            elo_ratings = {
                t: (1 - REGRESS) * e + REGRESS * MEAN_ELO
                for t, e in elo_ratings.items()
            }

    hi = row["HOME_TEAM_ID"]
    ai = row["AWAY_TEAM_ID"]
    he = elo_ratings.get(hi, ELO0)
    ae = elo_ratings.get(ai, ELO0)

    home_elos.append(he)
    away_elos.append(ae)

    # Expected home win probability
    diff = (he + HOME_ADV) - ae
    exp_home = 1 / (1 + 10 ** (-diff / 400))

    # Margin of victory multiplier (FiveThirtyEight formula)
    mov = abs(row["MOV"])
    mov_mult = (mov + 3) ** 0.8 / (7.5 + 0.006 * abs(diff))

    # Update
    delta = K * mov_mult * (row["HOME_WIN"] - exp_home)
    elo_ratings[hi] = he + delta
    elo_ratings[ai] = ae - delta

games["HOME_ELO"] = home_elos
games["AWAY_ELO"] = away_elos
print(f"  Final Elo range: {min(elo_ratings.values()):.0f} – {max(elo_ratings.values()):.0f}")

# ── 4. PER-TEAM GAME LOG WITH ADVANCED STATS ─────────────────────────────────
def build_team_log(games):
    """
    Convert game table into a per-team log (two rows per game).
    Computes Dean Oliver's Four Factors, pace-normalized efficiency ratings,
    shooting profile stats, and schedule/fatigue metrics from basic box scores.
    No additional API calls needed — everything derived from leaguegamefinder.
    """
    sides = []
    for side, opp, is_home in [("HOME", "AWAY", 1), ("AWAY", "HOME", 0)]:
        src = games[[
            "GAME_ID", "GAME_DATE", "SEASON",
            f"{side}_TEAM_ID", f"{side}_TEAM_ABBREVIATION",
            f"{side}_PTS", f"{opp}_PTS",
            f"{side}_FGM", f"{side}_FGA", f"{side}_FG3M", f"{side}_FG3A",
            f"{side}_FTM", f"{side}_FTA", f"{side}_TOV", f"{side}_AST",
            f"{side}_OREB", f"{side}_DREB",
            f"{opp}_FGM", f"{opp}_FGA", f"{opp}_FG3M",
            f"{opp}_FTA", f"{opp}_TOV",
            f"{opp}_OREB", f"{opp}_DREB",
        ]].copy()
        src.columns = [
            "GAME_ID", "GAME_DATE", "SEASON",
            "TEAM_ID", "TEAM_ABBREV",
            "PTS", "OPP_PTS",
            "FGM", "FGA", "FG3M", "FG3A",
            "FTM", "FTA", "TOV", "AST",
            "OREB", "DREB",
            "OPP_FGM", "OPP_FGA", "OPP_FG3M",
            "OPP_FTA", "OPP_TOV",
            "OPP_OREB", "OPP_DREB",
        ]
        src["IS_HOME"] = is_home
        sides.append(src)

    tg = (pd.concat(sides, ignore_index=True)
          .sort_values(["TEAM_ID", "GAME_DATE"])
          .reset_index(drop=True))

    tg["PD"]  = tg["PTS"] - tg["OPP_PTS"]
    tg["WIN"] = (tg["PD"] > 0).astype(int)

    # Possession estimate (Oliver formula: FGA - OREB + TOV + 0.44*FTA)
    tg["POSS"]     = tg["FGA"]     - tg["OREB"]     + tg["TOV"]     + 0.44 * tg["FTA"]
    tg["OPP_POSS"] = tg["OPP_FGA"] - tg["OPP_OREB"] + tg["OPP_TOV"] + 0.44 * tg["OPP_FTA"]

    # ── Four Factors (Dean Oliver) ─────────────────────────────────────────
    # Weights: eFG% 40%, TOV% 25%, OREB% 20%, FTR 15%
    # These explain ~97% of winning percentage variance per Oliver's research.
    tg["EFG_PCT"]      = (tg["FGM"] + 0.5 * tg["FG3M"]) / tg["FGA"].clip(lower=1)
    tg["TOV_PCT"]      = tg["TOV"] / (tg["FGA"] + 0.44 * tg["FTA"] + tg["TOV"]).clip(lower=1)
    tg["OREB_PCT"]     = tg["OREB"] / (tg["OREB"] + tg["OPP_DREB"]).clip(lower=1)
    tg["FTR"]          = tg["FTA"] / tg["FGA"].clip(lower=1)

    # Opponent Four Factors (defensive quality)
    tg["OPP_EFG_PCT"]  = (tg["OPP_FGM"] + 0.5 * tg["OPP_FG3M"]) / tg["OPP_FGA"].clip(lower=1)
    tg["OPP_TOV_PCT"]  = tg["OPP_TOV"] / (tg["OPP_FGA"] + 0.44 * tg["OPP_FTA"] + tg["OPP_TOV"]).clip(lower=1)
    tg["OPP_OREB_PCT"] = tg["OPP_OREB"] / (tg["OPP_OREB"] + tg["DREB"]).clip(lower=1)
    tg["OPP_FTR"]      = tg["OPP_FTA"] / tg["OPP_FGA"].clip(lower=1)

    # ── Pace-normalized efficiency ─────────────────────────────────────────
    # Net Rating outperforms raw point differential because it controls for
    # game pace — a 108-100 game in 90 possessions means something different
    # than 108-100 in 100 possessions.
    tg["ORTG"]       = tg["PTS"]     / tg["POSS"].clip(lower=1) * 100
    tg["DRTG"]       = tg["OPP_PTS"] / tg["OPP_POSS"].clip(lower=1) * 100
    tg["NET_RATING"] = tg["ORTG"] - tg["DRTG"]

    # ── Shooting profile ───────────────────────────────────────────────────
    tg["TS_PCT"]    = tg["PTS"] / (2 * (tg["FGA"] + 0.44 * tg["FTA"])).clip(lower=1)
    tg["THREE_PAR"] = tg["FG3A"] / tg["FGA"].clip(lower=1)
    tg["AST_TOV"]   = tg["AST"] / tg["TOV"].clip(lower=1)
    tg["PACE"]      = (tg["POSS"] + tg["OPP_POSS"]) / 2

    # ── Schedule / fatigue ─────────────────────────────────────────────────
    tg["DAYS_REST"] = (tg.groupby("TEAM_ID")["GAME_DATE"]
                       .diff().dt.days.fillna(2).clip(upper=14))
    tg["B2B"] = (tg["DAYS_REST"] <= 1).astype(int)

    # Games in last 7 days: counts prior games within rolling 7-day window.
    # This captures multi-game fatigue better than a single rest-day number.
    tg = tg.sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)
    gl7 = {}
    for _, grp in tg.groupby("TEAM_ID", sort=False):
        dates = grp["GAME_DATE"].values
        for i, idx in enumerate(grp.index):
            diffs = (dates[i] - dates[:i]) / np.timedelta64(1, "D")
            gl7[idx] = int(np.sum((diffs > 0) & (diffs <= 7)))
    tg["GAMES_L7"] = pd.Series(gl7)

    # ── Win/loss streak (going INTO each game) ─────────────────────────────
    # Positive = win streak length, negative = loss streak length.
    def streak_series(wins):
        out, cur = [], 0
        for w in wins:
            cur = max(1, cur + 1) if w else min(-1, cur - 1)
            out.append(cur)
        return out

    tg["_streak_now"] = tg.groupby("TEAM_ID")["WIN"].transform(
        lambda s: pd.Series(streak_series(s.tolist()), index=s.index)
    )
    # shift(1) so we use the streak ENTERING the game, not after it
    tg["STREAK"] = tg.groupby("TEAM_ID")["_streak_now"].shift(1).fillna(0)
    tg.drop(columns=["_streak_now"], inplace=True)

    return tg


tg = build_team_log(games)
print(f"Per-team log: {len(tg)} rows")

# ── 5. ROLLING WINDOW FEATURES (L10, shift(1) prevents leakage) ──────────────
print("Computing rolling features (L10)...")
W, MINP = 10, 3

ROLL_COLS = [
    "PD", "WIN", "NET_RATING",
    "EFG_PCT", "OPP_EFG_PCT",
    "TOV_PCT", "OPP_TOV_PCT",
    "OREB_PCT", "OPP_OREB_PCT",
    "FTR", "OPP_FTR",
    "TS_PCT", "THREE_PAR", "PACE",
    "PTS", "OPP_PTS", "AST_TOV",
]

for col in ROLL_COLS:
    tg[f"{col}_L10"] = tg.groupby("TEAM_ID")[col].transform(
        lambda s: s.shift(1).rolling(W, min_periods=MINP).mean()
    )

# Standard deviation of point differential — measures consistency
tg["PD_STD_L10"] = tg.groupby("TEAM_ID")["PD"].transform(
    lambda s: s.shift(1).rolling(W, min_periods=MINP).std()
)

# ── 6. HOME / AWAY PERFORMANCE SPLITS (L15 window) ───────────────────────────
# Some teams have dramatically different home vs road performance. We compute
# separate rolling windows for home-only and road-only games to capture this.
print("Computing home/away split features (L15)...")
W2, MINP2 = 15, 5

home_only = tg[tg["IS_HOME"] == 1].copy()
home_only["HOME_WIN_RATE_L15"]    = home_only.groupby("TEAM_ID")["WIN"].transform(
    lambda s: s.shift(1).rolling(W2, min_periods=MINP2).mean())
home_only["HOME_NET_RATING_L15"]  = home_only.groupby("TEAM_ID")["NET_RATING"].transform(
    lambda s: s.shift(1).rolling(W2, min_periods=MINP2).mean())

road_only = tg[tg["IS_HOME"] == 0].copy()
road_only["ROAD_WIN_RATE_L15"]    = road_only.groupby("TEAM_ID")["WIN"].transform(
    lambda s: s.shift(1).rolling(W2, min_periods=MINP2).mean())
road_only["ROAD_NET_RATING_L15"]  = road_only.groupby("TEAM_ID")["NET_RATING"].transform(
    lambda s: s.shift(1).rolling(W2, min_periods=MINP2).mean())

tg = tg.merge(
    home_only[["GAME_ID", "TEAM_ID", "HOME_WIN_RATE_L15", "HOME_NET_RATING_L15"]],
    on=["GAME_ID", "TEAM_ID"], how="left"
)
tg = tg.merge(
    road_only[["GAME_ID", "TEAM_ID", "ROAD_WIN_RATE_L15", "ROAD_NET_RATING_L15"]],
    on=["GAME_ID", "TEAM_ID"], how="left"
)

# ── 7. MERGE FEATURES INTO GAME TABLE ────────────────────────────────────────
print("Merging features into game table...")

# All per-team features to join (without HOME_/AWAY_ prefix yet)
TEAM_FEAT_COLS = (
    [f"{c}_L10" for c in ROLL_COLS] +
    ["PD_STD_L10", "STREAK", "DAYS_REST", "B2B", "GAMES_L7",
     "HOME_WIN_RATE_L15", "HOME_NET_RATING_L15",
     "ROAD_WIN_RATE_L15", "ROAD_NET_RATING_L15"]
)

for side, id_col in [("HOME", "HOME_TEAM_ID"), ("AWAY", "AWAY_TEAM_ID")]:
    games = (games
             .merge(tg[["GAME_ID", "TEAM_ID"] + TEAM_FEAT_COLS],
                    left_on=["GAME_ID", id_col],
                    right_on=["GAME_ID", "TEAM_ID"],
                    how="left")
             .rename(columns={f: f"{side}_{f}" for f in TEAM_FEAT_COLS})
             .drop(columns=["TEAM_ID"]))

# ── 8. FEATURE COLUMNS ───────────────────────────────────────────────────────
# Base features applied to both teams (HOME_ and AWAY_ prefixed).
# Split features are included selectively: home team's home splits and
# away team's road splits — the meaningful combinations.
BASE_FEATS = (
    [f"{c}_L10" for c in ROLL_COLS] +
    ["PD_STD_L10", "STREAK", "DAYS_REST", "B2B", "GAMES_L7"]
)

feature_cols = (
    ["HOME_ELO", "AWAY_ELO"] +
    [f"HOME_{f}" for f in BASE_FEATS] +
    [f"AWAY_{f}" for f in BASE_FEATS] +
    # Home team's home-game performance + Away team's road-game performance
    ["HOME_HOME_WIN_RATE_L15", "HOME_HOME_NET_RATING_L15",
     "AWAY_ROAD_WIN_RATE_L15", "AWAY_ROAD_NET_RATING_L15"]
)

# Drop early-season games that lack enough rolling history
required = ["HOME_NET_RATING_L10", "AWAY_NET_RATING_L10",
            "HOME_EFG_PCT_L10",    "AWAY_EFG_PCT_L10"]
model_df = games.dropna(subset=required).copy()
X = model_df[feature_cols].fillna(0)
y = model_df["HOME_WIN"]

print(f"Training samples: {len(model_df)} / {len(games)}")
print(f"Features: {len(feature_cols)}")

# ── 9. WALK-FORWARD CROSS-VALIDATION ─────────────────────────────────────────
# Train on all prior seasons, test on the next season. This mirrors how the
# model is actually deployed — never seeing future data during training.
print("\nWalk-forward cross-validation:")
cv_accs = []
for test_yr in range(2021, 2025):
    tr = model_df["SEASON"] < test_yr
    te = model_df["SEASON"] == test_yr
    if te.sum() < 200:
        continue
    clf = make_model()
    clf.fit(X[tr], y[tr])
    probs = clf.predict_proba(X[te])[:, 1]
    preds = (probs >= 0.5).astype(int)
    acc   = accuracy_score(y[te], preds)
    auc   = roc_auc_score(y[te], probs)
    base  = y[te].mean()
    print(f"  {test_yr}-{str(test_yr+1)[-2:]}: Acc={acc:.4f}  AUC={auc:.4f}  Baseline={base:.4f}")
    cv_accs.append(acc)

print(f"  Mean CV Accuracy: {np.mean(cv_accs):.4f}  (v1 baseline: ~0.6024)")

# ── 10. FINAL MODEL (trained on all available data) ───────────────────────────
print("\nTraining final model on all data...")
final_model = make_model()
final_model.fit(X, y)

if hasattr(final_model, "feature_importances_"):
    imp = pd.Series(final_model.feature_importances_, index=feature_cols)
    print("\nTop 15 features by importance:")
    for feat, val in imp.nlargest(15).items():
        print(f"  {feat:<48} {val:.4f}")

# ── 11. SAVE ARTIFACTS ───────────────────────────────────────────────────────
print("\nSaving artifacts...")
joblib.dump(final_model, "nba_predictor_model.pkl")
joblib.dump(feature_cols, "feature_cols.pkl")

teams_list = sorted(raw["TEAM_ABBREVIATION"].unique())
joblib.dump(teams_list, "teams.pkl")

# team_features.pkl: store the most recent rolling-window values per team.
#
# LEAKAGE FIX: v1 saved full-season averages which included games played
# after the date being predicted. We now store the last computed rolling
# snapshot from each team's final game in the training data.
team_id_to_abbrev = raw.set_index("TEAM_ID")["TEAM_ABBREVIATION"].to_dict()

latest_overall   = tg.sort_values("GAME_DATE").groupby("TEAM_ID").last()
latest_home_split = (tg[tg["IS_HOME"] == 1].sort_values("GAME_DATE")
                     .groupby("TEAM_ID")[["HOME_WIN_RATE_L15", "HOME_NET_RATING_L15"]].last())
latest_road_split = (tg[tg["IS_HOME"] == 0].sort_values("GAME_DATE")
                     .groupby("TEAM_ID")[["ROAD_WIN_RATE_L15", "ROAD_NET_RATING_L15"]].last())

STORE_COLS = BASE_FEATS  # same keys predict_game() will look up (without HOME_/AWAY_ prefix)

team_features_out = {}
for tid, abbrev in team_id_to_abbrev.items():
    if tid not in latest_overall.index:
        continue
    row = latest_overall.loc[tid]
    d = {col: float(row.get(col, 0) or 0) for col in STORE_COLS}
    d["ELO"] = float(elo_ratings.get(tid, ELO0))
    d["LAST_GAME_DATE"] = str(row["GAME_DATE"].date()) if pd.notna(row.get("GAME_DATE")) else ""

    # Home/road splits from their respective last home/road game
    if tid in latest_home_split.index:
        hr = latest_home_split.loc[tid]
        d["HOME_WIN_RATE_L15"]   = float(hr.get("HOME_WIN_RATE_L15",  0) or 0)
        d["HOME_NET_RATING_L15"] = float(hr.get("HOME_NET_RATING_L15", 0) or 0)
    else:
        d["HOME_WIN_RATE_L15"]   = 0.0
        d["HOME_NET_RATING_L15"] = 0.0

    if tid in latest_road_split.index:
        rr = latest_road_split.loc[tid]
        d["ROAD_WIN_RATE_L15"]   = float(rr.get("ROAD_WIN_RATE_L15",  0) or 0)
        d["ROAD_NET_RATING_L15"] = float(rr.get("ROAD_NET_RATING_L15", 0) or 0)
    else:
        d["ROAD_WIN_RATE_L15"]   = 0.0
        d["ROAD_NET_RATING_L15"] = 0.0

    team_features_out[abbrev] = d

joblib.dump(team_features_out, "team_features.pkl")

# team_recent.pkl: last 5 game results per team for form display in the UI
team_recent_out = {}
for tid, grp in tg.sort_values("GAME_DATE").groupby("TEAM_ID"):
    abbrev = team_id_to_abbrev.get(tid)
    if abbrev:
        team_recent_out[abbrev] = [
            {"result": "W" if row["WIN"] == 1 else "L",
             "pts": int(row["PTS"]),
             "opp_pts": int(row["OPP_PTS"])}
            for _, row in grp.tail(5).iterrows()
        ]
joblib.dump(team_recent_out, "team_recent.pkl")

print("Saved: nba_predictor_model.pkl, feature_cols.pkl, teams.pkl, team_features.pkl, team_recent.pkl")
print("\nDone!")
