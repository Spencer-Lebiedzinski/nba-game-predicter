"""
Sequence dataset for the NBA Transformer
========================================
Converts the per-team game log into the (home_seq, away_seq, context, labels)
tuples the Transformer trains on.

Each "example" is one upcoming game. For that game we look up:
  - The home team's most recent SEQ_LEN games BEFORE the prediction date
  - The away team's most recent SEQ_LEN games BEFORE the prediction date
  - Game-level context (Elo, rest-day diffs, B2B flags)

Per-token feature schema (one historical game = one token, F columns):
    binary:    IS_HOME, WIN, B2B
    continuous: MARGIN, PTS, OPP_PTS,
                EFG_PCT, OPP_EFG_PCT, TOV_PCT, OPP_TOV_PCT,
                OREB_PCT, OPP_OREB_PCT, FTR, OPP_FTR,
                TS_PCT, THREE_PAR, AST_TOV, NET_RATING, PACE,
                DAYS_REST, TEAM_ELO_PRE, OPP_ELO_PRE

The continuous features are z-scored using training-set statistics, which are
saved alongside the model so inference uses the same normalization.

Public API:
    TOKEN_FEATURES                  — ordered list of per-token feature names
    BINARY_TOKEN_FEATURES           — subset that should NOT be z-scored
    CONTEXT_FEATURES                — ordered list of game-context features
    SEQ_LEN                         — fixed sequence length (default 20)
    MIN_HISTORY                     — minimum prior games required per team
    build_training_arrays(games, team_log, train_seasons) -> dict
    build_team_recent_sequences(team_log, n=SEQ_LEN) -> dict[abbrev -> ndarray]
    apply_norm(arr, stats)          — z-score continuous features in place
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


SEQ_LEN     = 20    # last-N games per team in each sequence
MIN_HISTORY = 10    # skip examples where either team has fewer prior games

TOKEN_FEATURES = [
    # binary (do not normalize)
    "IS_HOME", "WIN", "B2B",
    # continuous
    "MARGIN", "PTS", "OPP_PTS",
    "EFG_PCT", "OPP_EFG_PCT", "TOV_PCT", "OPP_TOV_PCT",
    "OREB_PCT", "OPP_OREB_PCT", "FTR", "OPP_FTR",
    "TS_PCT", "THREE_PAR", "AST_TOV", "NET_RATING", "PACE",
    "DAYS_REST", "TEAM_ELO_PRE", "OPP_ELO_PRE",
]
BINARY_TOKEN_FEATURES = {"IS_HOME", "WIN", "B2B"}

CONTEXT_FEATURES = [
    "HOME_ELO", "AWAY_ELO", "ELO_DIFF",
    "HOME_DAYS_REST", "AWAY_DAYS_REST", "REST_DIFF",
    "HOME_B2B", "AWAY_B2B",
]


# ── Normalization helpers ────────────────────────────────────────────────────
@dataclass
class NormStats:
    """Z-score parameters for continuous features. Binary features pass through."""
    token_mean: np.ndarray  # shape (F,) — zero entries for binary features
    token_std:  np.ndarray  # shape (F,) — one  entries for binary features
    ctx_mean:   np.ndarray  # shape (C,)
    ctx_std:    np.ndarray  # shape (C,)

    def to_dict(self) -> dict:
        return {
            "token_mean": self.token_mean.tolist(),
            "token_std":  self.token_std.tolist(),
            "ctx_mean":   self.ctx_mean.tolist(),
            "ctx_std":    self.ctx_std.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NormStats":
        return cls(
            token_mean=np.array(d["token_mean"], dtype=np.float32),
            token_std =np.array(d["token_std"],  dtype=np.float32),
            ctx_mean  =np.array(d["ctx_mean"],   dtype=np.float32),
            ctx_std   =np.array(d["ctx_std"],    dtype=np.float32),
        )


def _fit_token_norm(token_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-feature mean/std on training tokens, leaving binaries as (0,1)."""
    mean = token_array.reshape(-1, token_array.shape[-1]).mean(axis=0)
    std  = token_array.reshape(-1, token_array.shape[-1]).std(axis=0)
    std  = np.where(std < 1e-6, 1.0, std)  # guard against zero-variance columns
    for i, name in enumerate(TOKEN_FEATURES):
        if name in BINARY_TOKEN_FEATURES:
            mean[i] = 0.0
            std[i]  = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def _fit_ctx_norm(ctx_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = ctx_array.mean(axis=0)
    std  = ctx_array.std(axis=0)
    std  = np.where(std < 1e-6, 1.0, std)
    # B2B columns are binary
    for i, name in enumerate(CONTEXT_FEATURES):
        if name.endswith("_B2B"):
            mean[i] = 0.0
            std[i]  = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_norm(tokens: np.ndarray, ctx: np.ndarray, stats: NormStats):
    """Z-score in place. Operates on stacked (..., F) and (..., C) arrays."""
    tokens[...] = (tokens - stats.token_mean) / stats.token_std
    ctx[...]    = (ctx    - stats.ctx_mean)   / stats.ctx_std


# ── History lookup ──────────────────────────────────────────────────────────
def _team_history_index(team_log: pd.DataFrame) -> dict:
    """
    Pre-build per-team arrays sorted by date so we can slice last-N quickly
    instead of filtering the dataframe inside the per-game loop.

    Returns: {team_id: {"dates": ndarray, "tokens": ndarray (n_games, F)}}
    """
    out = {}
    grouped = team_log.sort_values(["TEAM_ID", "GAME_DATE"]).groupby("TEAM_ID", sort=False)
    for team_id, grp in grouped:
        tokens = grp[TOKEN_FEATURES].to_numpy(dtype=np.float32)
        dates  = grp["GAME_DATE"].to_numpy(dtype="datetime64[ns]")
        out[int(team_id)] = {"dates": dates, "tokens": tokens}
    return out


def _last_n_before(history: dict, before_date: np.datetime64, n: int = SEQ_LEN):
    """Return (tokens, n_valid) for the n most recent games strictly before `before_date`."""
    dates = history["dates"]
    end = int(np.searchsorted(dates, before_date, side="left"))
    start = max(0, end - n)
    chunk = history["tokens"][start:end]
    n_valid = chunk.shape[0]
    if n_valid < n:
        pad = np.zeros((n - n_valid, len(TOKEN_FEATURES)), dtype=np.float32)
        chunk = np.concatenate([pad, chunk], axis=0)  # left-pad so recent games are at the end
    return chunk, n_valid


# ── Training array construction ─────────────────────────────────────────────
def _context_row(row: pd.Series) -> np.ndarray:
    home_rest = float(row.get("HOME_DAYS_REST", 2) or 2)
    away_rest = float(row.get("AWAY_DAYS_REST", 2) or 2)
    home_b2b  = float(row.get("HOME_B2B", 0)      or 0)
    away_b2b  = float(row.get("AWAY_B2B", 0)      or 0)
    return np.array([
        float(row["HOME_ELO"]),
        float(row["AWAY_ELO"]),
        float(row["HOME_ELO"]) - float(row["AWAY_ELO"]),
        home_rest,
        away_rest,
        home_rest - away_rest,
        home_b2b,
        away_b2b,
    ], dtype=np.float32)


def _attach_rest_b2b_to_games(games: pd.DataFrame, team_log: pd.DataFrame) -> pd.DataFrame:
    """
    The game table doesn't carry rest/B2B columns directly. Pull them from the
    team_log (where each team-game already has DAYS_REST and B2B precomputed).
    """
    home_side = team_log[team_log["IS_HOME"] == 1][["GAME_ID", "DAYS_REST", "B2B"]].rename(
        columns={"DAYS_REST": "HOME_DAYS_REST", "B2B": "HOME_B2B"}
    )
    away_side = team_log[team_log["IS_HOME"] == 0][["GAME_ID", "DAYS_REST", "B2B"]].rename(
        columns={"DAYS_REST": "AWAY_DAYS_REST", "B2B": "AWAY_B2B"}
    )
    return games.merge(home_side, on="GAME_ID", how="left").merge(away_side, on="GAME_ID", how="left")


def build_training_arrays(
    games: pd.DataFrame,
    team_log: pd.DataFrame,
    fit_seasons: Iterable[int],
    use_seasons: Iterable[int] | None = None,
) -> dict:
    """
    Build train-ready numpy arrays.

    Args:
        games:        game-level table from data_loader
        team_log:     per-team-game log from data_loader
        fit_seasons:  seasons used to fit normalization stats (typically the
                      training seasons; never the test season — fitting on test
                      data would leak global statistics)
        use_seasons:  seasons to include in the returned arrays. Defaults to all
                      seasons present in `games`.

    Returns a dict containing:
        x_home, x_away   — (N, SEQ_LEN, F) float32
        mask_home, mask_away — (N, SEQ_LEN) bool (True at padded positions)
        ctx              — (N, C) float32
        y_win            — (N,) int64
        y_home_score     — (N,) float32
        y_away_score     — (N,) float32
        season           — (N,) int64
        norm_stats       — NormStats fitted on fit_seasons
    """
    fit_seasons = set(int(s) for s in fit_seasons)
    if use_seasons is None:
        use_seasons = sorted(games["SEASON"].unique())
    use_seasons = set(int(s) for s in use_seasons)

    games = _attach_rest_b2b_to_games(games, team_log)
    history = _team_history_index(team_log)

    n_features = len(TOKEN_FEATURES)
    n_ctx      = len(CONTEXT_FEATURES)

    x_home_list, x_away_list = [], []
    mask_home_list, mask_away_list = [], []
    ctx_list = []
    y_win_list, y_hs_list, y_as_list = [], [], []
    season_list = []
    fit_token_pool: list[np.ndarray] = []
    fit_ctx_pool:   list[np.ndarray] = []

    skipped_no_history = 0
    skipped_missing_team = 0

    for _, row in games.iterrows():
        season = int(row["SEASON"])
        if season not in use_seasons:
            continue

        h_id = int(row["HOME_TEAM_ID"])
        a_id = int(row["AWAY_TEAM_ID"])
        if h_id not in history or a_id not in history:
            skipped_missing_team += 1
            continue

        game_date = np.datetime64(row["GAME_DATE"], "ns")
        h_seq, h_valid = _last_n_before(history[h_id], game_date)
        a_seq, a_valid = _last_n_before(history[a_id], game_date)

        if h_valid < MIN_HISTORY or a_valid < MIN_HISTORY:
            skipped_no_history += 1
            continue

        ctx = _context_row(row)

        x_home_list.append(h_seq)
        x_away_list.append(a_seq)
        # mask: True at positions that are padding (i.e. left-padded zeros)
        h_mask = np.zeros(SEQ_LEN, dtype=bool)
        a_mask = np.zeros(SEQ_LEN, dtype=bool)
        h_mask[:SEQ_LEN - h_valid] = True
        a_mask[:SEQ_LEN - a_valid] = True
        mask_home_list.append(h_mask)
        mask_away_list.append(a_mask)
        ctx_list.append(ctx)
        y_win_list.append(int(row["HOME_WIN"]))
        y_hs_list.append(float(row["HOME_PTS"]))
        y_as_list.append(float(row["AWAY_PTS"]))
        season_list.append(season)

        # Collect tokens from fit-seasons (only the non-padded positions) for
        # normalization fitting.
        if season in fit_seasons:
            if h_valid > 0:
                fit_token_pool.append(h_seq[SEQ_LEN - h_valid:])
            if a_valid > 0:
                fit_token_pool.append(a_seq[SEQ_LEN - a_valid:])
            fit_ctx_pool.append(ctx)

    if not x_home_list:
        raise RuntimeError("No training examples produced — check season filters and history threshold.")

    x_home   = np.stack(x_home_list)
    x_away   = np.stack(x_away_list)
    mask_home = np.stack(mask_home_list)
    mask_away = np.stack(mask_away_list)
    ctx      = np.stack(ctx_list)
    y_win    = np.array(y_win_list, dtype=np.int64)
    y_hs     = np.array(y_hs_list,  dtype=np.float32)
    y_as     = np.array(y_as_list,  dtype=np.float32)
    seasons  = np.array(season_list, dtype=np.int64)

    fit_tokens = np.concatenate(fit_token_pool, axis=0)
    fit_ctx    = np.stack(fit_ctx_pool)
    t_mean, t_std = _fit_token_norm(fit_tokens)
    c_mean, c_std = _fit_ctx_norm(fit_ctx)
    stats = NormStats(token_mean=t_mean, token_std=t_std,
                      ctx_mean=c_mean,   ctx_std=c_std)

    # Apply normalization to all examples. Padded zeros become slightly negative
    # after subtraction, but the attention mask will block those positions so
    # the model never sees them — no need to special-case.
    x_home_norm = (x_home - stats.token_mean) / stats.token_std
    x_away_norm = (x_away - stats.token_mean) / stats.token_std
    ctx_norm    = (ctx    - stats.ctx_mean)   / stats.ctx_std

    print(f"  built {len(y_win):,} examples  |  "
          f"skipped: {skipped_no_history} (history), {skipped_missing_team} (team-id)")

    return {
        "x_home":    x_home_norm.astype(np.float32),
        "x_away":    x_away_norm.astype(np.float32),
        "mask_home": mask_home,
        "mask_away": mask_away,
        "ctx":       ctx_norm.astype(np.float32),
        "y_win":     y_win,
        "y_home_score": y_hs,
        "y_away_score": y_as,
        "season":    seasons,
        "norm_stats": stats,
    }


# ── Inference-time sequence builder ─────────────────────────────────────────
def build_team_recent_sequences(team_log: pd.DataFrame, n: int = SEQ_LEN) -> dict:
    """
    For each team, capture its most recent n games (raw, un-normalized) so the
    Flask app can build inference inputs without re-querying the full team_log
    dataframe.

    Returns a dict mapping team_abbrev (e.g. "BOS") → dict with keys:
        tokens:        ndarray (n, F) of raw token features (zeros pad the front)
        n_valid:       int — number of real games (<= n)
        last_game_date: str ISO date of the most recent game
    """
    out = {}
    team_log = team_log.sort_values(["TEAM_ID", "GAME_DATE"])
    for team_id, grp in team_log.groupby("TEAM_ID", sort=False):
        recent = grp.tail(n)
        tokens = recent[TOKEN_FEATURES].to_numpy(dtype=np.float32)
        n_valid = tokens.shape[0]
        if n_valid < n:
            pad = np.zeros((n - n_valid, len(TOKEN_FEATURES)), dtype=np.float32)
            tokens = np.concatenate([pad, tokens], axis=0)
        abbrev = recent["TEAM_ABBREV"].iloc[-1]
        out[str(abbrev)] = {
            "tokens": tokens,
            "n_valid": int(n_valid),
            "last_game_date": str(recent["GAME_DATE"].iloc[-1].date()),
        }
    return out
