"""
NBA Data Loader (cached)
========================
Shared data prep for the Transformer training pipeline. Pulls 10 seasons of
games from nba_api once, derives the same advanced stats used by the v2 GBM
(Four Factors, Net Rating, pace, Elo, schedule fatigue), and caches the
results to parquet so subsequent runs skip the slow network fetch.

This module intentionally duplicates a small amount of logic from
`train_model.py`'s `build_team_log` rather than importing it. Reasons:
  1. `train_model.py` runs the whole pipeline at import time (it's a script),
     so importing from it would trigger an unwanted fetch.
  2. Decoupling lets the Transformer pipeline evolve (e.g. different cache
     schemas) without risk of breaking the GBM trainer.

Public API:
    load_games_and_team_log(seasons=range(2015, 2025), cache_dir="cache",
                            force_refresh=False) -> (games_df, team_log_df)

Cached files:
    cache/raw_games.parquet     — one row per game, HOME_/AWAY_ prefixed
    cache/team_log.parquet      — two rows per game (one per team) with
                                  per-game advanced stats and pre-game Elo
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ── Elo constants (match train_model.py / FiveThirtyEight) ──────────────────
K_FACTOR    = 20
HOME_ADV    = 100
ELO_INIT    = 1500
MEAN_ELO    = 1505
SEASON_REG  = 0.25  # 25% regression toward mean between seasons


# ── Fetch raw games via nba_api ─────────────────────────────────────────────
KEEP_COLS = [
    "GAME_ID", "GAME_DATE", "SEASON", "TEAM_ID", "TEAM_ABBREVIATION",
    "PTS", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
    "OREB", "DREB", "AST", "TOV",
]
ID_COLS = ["GAME_ID", "GAME_DATE", "SEASON"]


def _fetch_raw(seasons) -> pd.DataFrame:
    from nba_api.stats.endpoints import leaguegamefinder

    frames = []
    for season in seasons:
        season_str = f"{season}-{str(season + 1)[-2:]}"
        print(f"  fetching {season_str}...", end=" ", flush=True)
        gf = leaguegamefinder.LeagueGameFinder(season_nullable=season_str)
        s = gf.get_data_frames()[0]
        s["SEASON"] = season
        frames.append(s)
        print("ok")
        time.sleep(0.6)  # respect nba_api rate limits
    raw = pd.concat(frames, ignore_index=True)
    raw["GAME_DATE"] = pd.to_datetime(raw["GAME_DATE"])
    return raw


def _build_games(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per game with HOME_/AWAY_ prefixed stat columns + Elo."""
    home_raw = raw[raw["MATCHUP"].str.contains("vs\\.", na=False)][KEEP_COLS].copy()
    away_raw = raw[raw["MATCHUP"].str.contains("@",     na=False)][KEEP_COLS].copy()
    home_raw = home_raw.rename(columns={c: f"HOME_{c}" for c in KEEP_COLS if c not in ID_COLS})
    away_raw = away_raw.rename(columns={c: f"AWAY_{c}" for c in KEEP_COLS if c not in ID_COLS})

    games = (home_raw.merge(away_raw, on=ID_COLS, how="inner")
                     .sort_values("GAME_DATE")
                     .reset_index(drop=True))
    games["HOME_WIN"] = (games["HOME_PTS"] > games["AWAY_PTS"]).astype(int)
    games["MOV"]      = games["HOME_PTS"] - games["AWAY_PTS"]

    # Walking Elo ratings. Pre-game values are recorded so training never sees
    # the rating updated with the current game's result.
    elo = {}
    home_pre, away_pre = [], []
    last_season = None
    for _, row in games.iterrows():
        if row["SEASON"] != last_season:
            last_season = row["SEASON"]
            if elo:
                elo = {t: (1 - SEASON_REG) * v + SEASON_REG * MEAN_ELO for t, v in elo.items()}

        hi, ai = row["HOME_TEAM_ID"], row["AWAY_TEAM_ID"]
        he = elo.get(hi, ELO_INIT)
        ae = elo.get(ai, ELO_INIT)
        home_pre.append(he)
        away_pre.append(ae)

        diff = (he + HOME_ADV) - ae
        exp_home = 1 / (1 + 10 ** (-diff / 400))
        mov_mult = (abs(row["MOV"]) + 3) ** 0.8 / (7.5 + 0.006 * abs(diff))
        delta    = K_FACTOR * mov_mult * (row["HOME_WIN"] - exp_home)
        elo[hi] = he + delta
        elo[ai] = ae - delta

    games["HOME_ELO"] = home_pre
    games["AWAY_ELO"] = away_pre
    return games


def _build_team_log(games: pd.DataFrame) -> pd.DataFrame:
    """
    Two rows per game (home perspective, away perspective). Adds per-game
    Four Factors, Net Rating, pace, shooting profile, and schedule fatigue.
    The opponent's pre-game Elo is attached so the Transformer can weigh
    each historical game by strength of opposition.
    """
    sides = []
    for side, opp, is_home in [("HOME", "AWAY", 1), ("AWAY", "HOME", 0)]:
        cols = [
            "GAME_ID", "GAME_DATE", "SEASON",
            f"{side}_TEAM_ID", f"{side}_TEAM_ABBREVIATION",
            f"{side}_PTS", f"{opp}_PTS",
            f"{side}_FGM", f"{side}_FGA", f"{side}_FG3M", f"{side}_FG3A",
            f"{side}_FTM", f"{side}_FTA", f"{side}_TOV", f"{side}_AST",
            f"{side}_OREB", f"{side}_DREB",
            f"{opp}_FGM", f"{opp}_FGA", f"{opp}_FG3M",
            f"{opp}_FTA", f"{opp}_TOV",
            f"{opp}_OREB", f"{opp}_DREB",
            f"{side}_ELO", f"{opp}_ELO",
        ]
        src = games[cols].copy()
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
            "TEAM_ELO_PRE", "OPP_ELO_PRE",
        ]
        src["IS_HOME"] = is_home
        sides.append(src)

    tg = (pd.concat(sides, ignore_index=True)
            .sort_values(["TEAM_ID", "GAME_DATE"])
            .reset_index(drop=True))

    tg["MARGIN"] = tg["PTS"] - tg["OPP_PTS"]
    tg["WIN"]    = (tg["MARGIN"] > 0).astype(int)

    # Possession estimate (Oliver)
    tg["POSS"]     = tg["FGA"]     - tg["OREB"]     + tg["TOV"]     + 0.44 * tg["FTA"]
    tg["OPP_POSS"] = tg["OPP_FGA"] - tg["OPP_OREB"] + tg["OPP_TOV"] + 0.44 * tg["OPP_FTA"]

    # Four Factors
    tg["EFG_PCT"]      = (tg["FGM"]     + 0.5 * tg["FG3M"])     / tg["FGA"].clip(lower=1)
    tg["TOV_PCT"]      =  tg["TOV"]     / (tg["FGA"]     + 0.44 * tg["FTA"]     + tg["TOV"]).clip(lower=1)
    tg["OREB_PCT"]     =  tg["OREB"]    / (tg["OREB"]    + tg["OPP_DREB"]).clip(lower=1)
    tg["FTR"]          =  tg["FTA"]     /  tg["FGA"].clip(lower=1)
    tg["OPP_EFG_PCT"]  = (tg["OPP_FGM"] + 0.5 * tg["OPP_FG3M"]) / tg["OPP_FGA"].clip(lower=1)
    tg["OPP_TOV_PCT"]  =  tg["OPP_TOV"] / (tg["OPP_FGA"] + 0.44 * tg["OPP_FTA"] + tg["OPP_TOV"]).clip(lower=1)
    tg["OPP_OREB_PCT"] =  tg["OPP_OREB"]/ (tg["OPP_OREB"]+ tg["DREB"]).clip(lower=1)
    tg["OPP_FTR"]      =  tg["OPP_FTA"] /  tg["OPP_FGA"].clip(lower=1)

    # Pace-normalized efficiency
    tg["ORTG"]       = tg["PTS"]     / tg["POSS"].clip(lower=1)     * 100
    tg["DRTG"]       = tg["OPP_PTS"] / tg["OPP_POSS"].clip(lower=1) * 100
    tg["NET_RATING"] = tg["ORTG"] - tg["DRTG"]
    tg["PACE"]       = (tg["POSS"] + tg["OPP_POSS"]) / 2

    # Shooting profile
    tg["TS_PCT"]    = tg["PTS"] / (2 * (tg["FGA"] + 0.44 * tg["FTA"])).clip(lower=1)
    tg["THREE_PAR"] = tg["FG3A"] / tg["FGA"].clip(lower=1)
    tg["AST_TOV"]   = tg["AST"]  / tg["TOV"].clip(lower=1)

    # Schedule / fatigue
    tg["DAYS_REST"] = (tg.groupby("TEAM_ID")["GAME_DATE"]
                         .diff().dt.days.fillna(2).clip(upper=14))
    tg["B2B"]       = (tg["DAYS_REST"] <= 1).astype(int)

    return tg


# ── Public API ──────────────────────────────────────────────────────────────
def load_games_and_team_log(
    seasons=range(2015, 2025),
    cache_dir: str | Path = "cache",
    force_refresh: bool = False,
):
    """
    Returns (games_df, team_log_df). Caches both to parquet under `cache_dir`.

    Cache invalidation is purely time-based via `force_refresh=True`. The
    season range is encoded in the parquet filename so different ranges get
    their own caches automatically.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    season_tag = f"{min(seasons)}_{max(seasons)}"
    games_path = cache_dir / f"raw_games_{season_tag}.parquet"
    log_path   = cache_dir / f"team_log_{season_tag}.parquet"

    if not force_refresh and games_path.exists() and log_path.exists():
        print(f"Loading cached games + team log ({games_path.name})")
        games = pd.read_parquet(games_path)
        team_log = pd.read_parquet(log_path)
        games["GAME_DATE"]    = pd.to_datetime(games["GAME_DATE"])
        team_log["GAME_DATE"] = pd.to_datetime(team_log["GAME_DATE"])
        return games, team_log

    print(f"Fetching {len(list(seasons))} seasons from nba_api...")
    raw = _fetch_raw(seasons)
    print("Building game table with Elo...")
    games = _build_games(raw)
    print("Building per-team game log with advanced stats...")
    team_log = _build_team_log(games)

    games.to_parquet(games_path, index=False)
    team_log.to_parquet(log_path, index=False)
    print(f"Cached to {cache_dir}/")

    return games, team_log


if __name__ == "__main__":
    games, team_log = load_games_and_team_log()
    print(f"\nGames: {len(games):,}  |  Team-log rows: {len(team_log):,}")
    print(f"Date range: {games['GAME_DATE'].min().date()} -> {games['GAME_DATE'].max().date()}")
    print(f"Home win rate: {games['HOME_WIN'].mean():.3f}")
