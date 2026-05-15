"""
Two-tower NBA Transformer
=========================
Encodes each team's recent game history with a shared-weight Transformer
encoder, then fuses the two team embeddings with a learned game-context
vector and predicts:

    1. P(home wins)        — sigmoid head, BCE loss
    2. home final score    — regression head, MSE loss
    3. away final score    — regression head, MSE loss

Score regression is an auxiliary task. It regularizes the win head, provides
the spread and total for the betting UI, and gives downstream work (e.g. the
diffusion outcome model planned next) calibrated mean predictions to build on.

Design notes:

* Shared `TeamEncoder` for home and away — the meaning of a team's recent form
  is invariant to whether that team happens to be playing at home in the
  upcoming game (the IS_HOME flag inside each token already encodes the
  venue of each historical game).

* Learned [CLS] token + learned positional embedding. The sequence length is
  fixed and short (20), so sinusoidal positional encoding isn't necessary.

* `src_key_padding_mask` blocks attention to padded positions so early-season
  examples (teams with fewer than 20 prior games) don't poison the encoder.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class TeamEncoder(nn.Module):
    """Transformer encoder over a team's recent N-game token sequence."""

    def __init__(
        self,
        n_features: int,
        seq_len: int = 20,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.in_proj = nn.Linear(n_features, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_emb   = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_emb,   std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm trains more stably for small models
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens:   (B, S, F) float
            pad_mask: (B, S) bool — True at padded positions
        Returns:
            (B, d_model) — pooled team embedding from the [CLS] token
        """
        B = tokens.size(0)
        x = self.in_proj(tokens)                            # (B, S, d)
        cls = self.cls_token.expand(B, -1, -1)              # (B, 1, d)
        x = torch.cat([cls, x], dim=1)                      # (B, S+1, d)
        x = x + self.pos_emb[:, : x.size(1), :]

        # Prepend False for the CLS slot (CLS is always real, never padded).
        cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=pad_mask.device)
        full_mask = torch.cat([cls_pad, pad_mask], dim=1)   # (B, S+1)

        x = self.encoder(x, src_key_padding_mask=full_mask)
        return self.out_norm(x[:, 0])                       # CLS pooled


class NBATransformer(nn.Module):
    """Two-tower (shared) Transformer + context MLP → win / score heads."""

    def __init__(
        self,
        n_token_features: int,
        n_ctx_features: int,
        seq_len: int = 20,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        head_hidden: int = 128,
    ):
        super().__init__()
        self.team_encoder = TeamEncoder(
            n_features=n_token_features,
            seq_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.ctx_proj = nn.Sequential(
            nn.Linear(n_ctx_features, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        fused_dim = 3 * d_model
        self.shared_head = nn.Sequential(
            nn.Linear(fused_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Three output heads. The score heads predict raw points (typically 90-130).
        self.win_head        = nn.Linear(head_hidden, 1)
        self.home_score_head = nn.Linear(head_hidden, 1)
        self.away_score_head = nn.Linear(head_hidden, 1)

    def forward(
        self,
        x_home: torch.Tensor,
        mask_home: torch.Tensor,
        x_away: torch.Tensor,
        mask_away: torch.Tensor,
        ctx: torch.Tensor,
    ) -> dict:
        h_vec = self.team_encoder(x_home, mask_home)   # (B, d)
        a_vec = self.team_encoder(x_away, mask_away)   # (B, d)
        c_vec = self.ctx_proj(ctx)                     # (B, d)
        z = torch.cat([h_vec, a_vec, c_vec], dim=-1)
        z = self.shared_head(z)
        return {
            "logit":      self.win_head(z).squeeze(-1),
            "home_score": self.home_score_head(z).squeeze(-1),
            "away_score": self.away_score_head(z).squeeze(-1),
        }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
