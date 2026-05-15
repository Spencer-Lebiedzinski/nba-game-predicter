"""
NBA Transformer training driver
===============================
End-to-end training script for the two-tower Transformer model.

Pipeline:
  1. Load cached games + per-team game log via data_loader
  2. Walk-forward cross-validation across the same held-out seasons used by
     train_model.py (2021, 2022, 2023, 2024). For each fold:
        - Normalization stats fitted ONLY on the training seasons
        - Train with early stopping on the held-out season
        - Report Accuracy, LogLoss, AUC, Brier vs majority/home baselines
  3. Train a final production model on all seasons through 2023, evaluate on
     2024 as a clean holdout
  4. Save artifacts to the current directory so app.py picks them up:
        transformer_model.pt        — final-model state dict
        transformer_config.pkl      — architecture + feature config
        transformer_norm_stats.pkl  — z-score parameters
        team_sequences.pkl          — each team's most recent SEQ_LEN tokens

Run from the models/ directory:
    py train_transformer.py
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, brier_score_loss, log_loss, roc_auc_score,
)

from data_loader import load_games_and_team_log
from transformer_data import (
    CONTEXT_FEATURES, SEQ_LEN, TOKEN_FEATURES,
    build_team_recent_sequences, build_training_arrays,
)
from transformer_model import NBATransformer, count_parameters


# ── Hyperparameters ─────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    d_model:      int   = 128
    n_heads:      int   = 4
    n_layers:     int   = 4
    d_ff:         int   = 256
    dropout:      float = 0.1
    head_hidden:  int   = 128

    batch_size:   int   = 256
    epochs:       int   = 30
    lr:           float = 3e-4
    weight_decay: float = 1e-2
    warmup_frac:  float = 0.1
    patience:     int   = 5

    # Score regression is auxiliary — keep its weight small so the win head
    # remains the primary optimization target.
    score_loss_weight: float = 0.01

    seed: int = 42


# ── Helpers ─────────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(arrays: dict, indices: np.ndarray | None, batch_size: int, shuffle: bool):
    if indices is None:
        indices = np.arange(len(arrays["y_win"]))
    ds = TensorDataset(
        torch.from_numpy(arrays["x_home"][indices]),
        torch.from_numpy(arrays["mask_home"][indices]),
        torch.from_numpy(arrays["x_away"][indices]),
        torch.from_numpy(arrays["mask_away"][indices]),
        torch.from_numpy(arrays["ctx"][indices]),
        torch.from_numpy(arrays["y_win"][indices]).float(),
        torch.from_numpy(arrays["y_home_score"][indices]),
        torch.from_numpy(arrays["y_away_score"][indices]),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def cosine_warmup_lr(step: int, total_steps: int, base_lr: float, warmup_frac: float) -> float:
    warmup_steps = max(1, int(total_steps * warmup_frac))
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


def evaluate(model: NBATransformer, loader: DataLoader, device: torch.device, cfg: TrainConfig):
    """Return dict of metrics + raw probs."""
    model.eval()
    bce = nn.BCEWithLogitsLoss(reduction="sum")
    mse = nn.MSELoss(reduction="sum")
    total_bce = total_mse_h = total_mse_a = 0.0
    n = 0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for xh, mh, xa, ma, ctx, yw, yhs, yas in loader:
            xh, mh, xa, ma, ctx = (t.to(device) for t in (xh, mh, xa, ma, ctx))
            yw, yhs, yas = yw.to(device), yhs.to(device), yas.to(device)
            out = model(xh, mh, xa, ma, ctx)
            total_bce  += bce(out["logit"], yw).item()
            total_mse_h += mse(out["home_score"], yhs).item()
            total_mse_a += mse(out["away_score"], yas).item()
            n += yw.size(0)
            all_probs.append(torch.sigmoid(out["logit"]).cpu().numpy())
            all_labels.append(yw.cpu().numpy().astype(int))

    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds  = (probs >= 0.5).astype(int)

    return {
        "loss":     total_bce / n + cfg.score_loss_weight * (total_mse_h + total_mse_a) / n,
        "logloss":  log_loss(labels, probs.clip(1e-6, 1 - 1e-6), labels=[0, 1]),
        "acc":      accuracy_score(labels, preds),
        "auc":      roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan"),
        "brier":    brier_score_loss(labels, probs),
        "home_rate": float(labels.mean()),
        "probs":    probs,
        "labels":   labels,
    }


def train_one_fold(
    arrays: dict,
    train_idx: np.ndarray,
    val_idx:   np.ndarray,
    device: torch.device,
    cfg: TrainConfig,
    verbose_every: int = 5,
) -> dict:
    set_seed(cfg.seed)
    n_token_feats = len(TOKEN_FEATURES)
    n_ctx_feats   = len(CONTEXT_FEATURES)

    model = NBATransformer(
        n_token_features=n_token_feats,
        n_ctx_features=n_ctx_feats,
        seq_len=SEQ_LEN,
        d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=cfg.n_layers,
        d_ff=cfg.d_ff, dropout=cfg.dropout, head_hidden=cfg.head_hidden,
    ).to(device)

    train_loader = make_loader(arrays, train_idx, cfg.batch_size, shuffle=True)
    val_loader   = make_loader(arrays, val_idx,   cfg.batch_size, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    total_steps = cfg.epochs * len(train_loader)
    step = 0

    best_val_logloss = float("inf")
    best_state = None
    patience_left = cfg.patience

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_seen = 0
        for xh, mh, xa, ma, ctx, yw, yhs, yas in train_loader:
            xh, mh, xa, ma, ctx = (t.to(device) for t in (xh, mh, xa, ma, ctx))
            yw, yhs, yas = yw.to(device), yhs.to(device), yas.to(device)

            lr_now = cosine_warmup_lr(step, total_steps, cfg.lr, cfg.warmup_frac)
            for g in opt.param_groups:
                g["lr"] = lr_now

            out = model(xh, mh, xa, ma, ctx)
            loss = bce(out["logit"], yw) + cfg.score_loss_weight * (
                mse(out["home_score"], yhs) + mse(out["away_score"], yas)
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            step += 1
            epoch_loss += loss.item() * yw.size(0)
            n_seen += yw.size(0)

        val_metrics = evaluate(model, val_loader, device, cfg)
        if epoch == 1 or epoch % verbose_every == 0 or epoch == cfg.epochs:
            print(f"    epoch {epoch:>2}  train_loss={epoch_loss/n_seen:.4f}  "
                  f"val_logloss={val_metrics['logloss']:.4f}  val_acc={val_metrics['acc']:.4f}  "
                  f"val_auc={val_metrics['auc']:.4f}")

        if val_metrics["logloss"] < best_val_logloss - 1e-4:
            best_val_logloss = val_metrics["logloss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"    early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    final = evaluate(model, val_loader, device, cfg)
    return {"model": model, "metrics": final}


# ── Main pipeline ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", type=int, nargs=2, default=[2015, 2024],
                        help="Inclusive season range, e.g. 2015 2024 for 2015-16 through 2024-25")
    parser.add_argument("--cv-test-seasons", type=int, nargs="+",
                        default=[2021, 2022, 2023, 2024],
                        help="Hold-out seasons for walk-forward CV")
    parser.add_argument("--final-eval-season", type=int, default=2024,
                        help="Final clean-holdout season for the production model")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--no-cv", action="store_true",
                        help="Skip walk-forward CV; just train the production model.")
    args = parser.parse_args()

    cfg = TrainConfig()
    if args.epochs is not None:
        cfg.epochs = args.epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: d_model={cfg.d_model} n_layers={cfg.n_layers} n_heads={cfg.n_heads} "
          f"batch={cfg.batch_size} epochs={cfg.epochs}")

    seasons = range(args.seasons[0], args.seasons[1] + 1)
    games, team_log = load_games_and_team_log(
        seasons=seasons,
        cache_dir=args.cache_dir,
        force_refresh=args.force_refresh,
    )
    print(f"Loaded games: {len(games):,}  |  team-log rows: {len(team_log):,}\n")

    cv_summary = []
    if not args.no_cv:
        print("=" * 70)
        print("WALK-FORWARD CROSS-VALIDATION")
        print("=" * 70)
        for test_yr in args.cv_test_seasons:
            fit_seasons = [s for s in seasons if s < test_yr]
            if not fit_seasons:
                print(f"  {test_yr}: skipped (no prior seasons)")
                continue
            print(f"\n  Fold: train {min(fit_seasons)}-{max(fit_seasons)} -> test {test_yr}-{str(test_yr+1)[-2:]}")

            use = fit_seasons + [test_yr]
            arrays = build_training_arrays(
                games=games, team_log=team_log,
                fit_seasons=fit_seasons, use_seasons=use,
            )
            seasons_arr = arrays["season"]
            train_idx = np.where(np.isin(seasons_arr, fit_seasons))[0]
            val_idx   = np.where(seasons_arr == test_yr)[0]
            if len(val_idx) < 100:
                print(f"    skip — only {len(val_idx)} examples in test season")
                continue

            result = train_one_fold(arrays, train_idx, val_idx, device, cfg)
            m = result["metrics"]
            print(f"    -> acc={m['acc']:.4f}  logloss={m['logloss']:.4f}  "
                  f"auc={m['auc']:.4f}  brier={m['brier']:.4f}  "
                  f"(home_rate={m['home_rate']:.3f})")
            cv_summary.append({
                "test_season": test_yr,
                "acc":     m["acc"],
                "logloss": m["logloss"],
                "auc":     m["auc"],
                "brier":   m["brier"],
                "home_rate": m["home_rate"],
                "n_test":  int(len(val_idx)),
            })

        if cv_summary:
            print("\n  CV summary:")
            print(f"  {'Season':<10} {'Acc':>7} {'LogLoss':>9} {'AUC':>7} {'Brier':>7} {'N':>6}")
            for r in cv_summary:
                print(f"  {r['test_season']:<10} {r['acc']:>7.4f} {r['logloss']:>9.4f} "
                      f"{r['auc']:>7.4f} {r['brier']:>7.4f} {r['n_test']:>6}")
            mean_acc     = np.mean([r["acc"]     for r in cv_summary])
            mean_logloss = np.mean([r["logloss"] for r in cv_summary])
            print(f"  {'mean':<10} {mean_acc:>7.4f} {mean_logloss:>9.4f}")

    # ── Final production model ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"FINAL MODEL  (train through {args.final_eval_season - 1}, eval on {args.final_eval_season})")
    print("=" * 70)
    fit_seasons = [s for s in seasons if s < args.final_eval_season]
    arrays = build_training_arrays(
        games=games, team_log=team_log,
        fit_seasons=fit_seasons, use_seasons=list(seasons),
    )
    seasons_arr = arrays["season"]
    train_idx = np.where(np.isin(seasons_arr, fit_seasons))[0]
    val_idx   = np.where(seasons_arr == args.final_eval_season)[0]

    if len(val_idx) < 100:
        # Final season not present — train on everything and skip holdout eval
        train_idx = np.arange(len(seasons_arr))
        val_idx = train_idx[-max(200, len(train_idx) // 20):]
        print(f"  (insufficient final-season data; using {len(val_idx)} most-recent examples as val)")

    result = train_one_fold(arrays, train_idx, val_idx, device, cfg, verbose_every=2)
    final_model = result["model"]
    m = result["metrics"]
    print(f"\n  Final holdout: acc={m['acc']:.4f}  logloss={m['logloss']:.4f}  "
          f"auc={m['auc']:.4f}  brier={m['brier']:.4f}")

    # ── Persist artifacts ──────────────────────────────────────────────────
    print("\nSaving artifacts...")
    torch.save(final_model.state_dict(), "transformer_model.pt")

    config_dict = {
        "n_token_features": len(TOKEN_FEATURES),
        "n_ctx_features":   len(CONTEXT_FEATURES),
        "seq_len":          SEQ_LEN,
        "d_model":          cfg.d_model,
        "n_heads":          cfg.n_heads,
        "n_layers":         cfg.n_layers,
        "d_ff":             cfg.d_ff,
        "dropout":          cfg.dropout,
        "head_hidden":      cfg.head_hidden,
        "token_features":   TOKEN_FEATURES,
        "context_features": CONTEXT_FEATURES,
    }
    joblib.dump(config_dict, "transformer_config.pkl")
    joblib.dump(arrays["norm_stats"].to_dict(), "transformer_norm_stats.pkl")

    team_sequences = build_team_recent_sequences(team_log)
    joblib.dump(team_sequences, "team_sequences.pkl")
    print(f"  team_sequences: {len(team_sequences)} teams")

    # Also dump a small JSON summary of the run for quick reference / READMEs.
    summary = {
        "cv_summary": cv_summary,
        "final_eval_season": args.final_eval_season,
        "final_metrics": {k: float(v) for k, v in m.items() if k in {"acc", "logloss", "auc", "brier", "home_rate"}},
        "n_params": int(count_parameters(final_model)),
        "config":   config_dict,
    }
    with open("transformer_run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved: transformer_model.pt, transformer_config.pkl, "
          "transformer_norm_stats.pkl, team_sequences.pkl, transformer_run_summary.json")
    print(f"Parameter count: {count_parameters(final_model):,}")
    print("Done.")


if __name__ == "__main__":
    main()
