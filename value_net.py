"""GEN-GBC value surrogate: GEN-CIM mean pooling + (256,128) ReLU MLP.

Public forward/predict return RAW ordered-pair internal-node GBC units.
The MLP learns standardized targets; float64 calibration buffers travel with
state_dict. Calibration is frozen after bootstrap, including online updates.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def compute_seed_embedding(h_v: Tensor, seed_set) -> Tensor:
    return compute_seed_embedding_batch(h_v, [seed_set])[0]


def compute_seed_embedding_batch(h_v: Tensor, seed_sets: Sequence) -> Tensor:
    if h_v.ndim != 2 or not h_v.is_floating_point():
        raise ValueError("h_v must be a floating [N,d] tensor")
    if not seed_sets:
        return h_v.new_empty((0, h_v.size(1)))
    rows = []
    for s in seed_sets:
        ids = sorted(s.nodes)
        if ids and (ids[0] < 0 or ids[-1] >= len(h_v)):
            raise ValueError("Seed IDs outside embedding matrix")
        rows.append(h_v[ids].mean(0) if ids else h_v.new_zeros(h_v.size(1)))
    return torch.stack(rows)


class ValueNetwork(nn.Module):
    def __init__(self, embed_dim: int = 128, hidden_dims=(256, 128),
                 use_context: bool = False, dropout: float = 0.0):
        super().__init__()
        if embed_dim < 1 or any(h < 1 for h in hidden_dims) or not 0 <= dropout < 1:
            raise ValueError("Invalid ValueNetwork dimensions/dropout")
        self.embed_dim, self.hidden_dims = embed_dim, tuple(hidden_dims)
        self.use_context, self.dropout = use_context, dropout
        layers, width = [], embed_dim * (2 if use_context else 1)
        for h in hidden_dims:
            layers.extend([nn.Linear(width, h), nn.ReLU()])
            if dropout:
                layers.append(nn.Dropout(dropout))
            width = h
        layers.append(nn.Linear(width, 1))
        self.mlp = nn.Sequential(*layers)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        self.register_buffer("target_mean", torch.tensor(0., dtype=torch.float64))
        self.register_buffer("target_scale", torch.tensor(1., dtype=torch.float64))
        self.register_buffer("calibrated", torch.tensor(False))

    @torch.no_grad()
    def calibrate(self, targets: Tensor, normalize: bool = True) -> None:
        """Set once. Changing statistics later would change EVERY raw prediction."""
        if bool(self.calibrated):
            return
        values = targets.detach().to(self.target_mean.device, torch.float64)
        if not values.numel() or not bool(torch.isfinite(values).all()):
            raise ValueError("Calibration requires finite nonempty targets")
        if normalize:
            self.target_mean.copy_(values.mean())
            std = values.std(unbiased=False)
            self.target_scale.copy_(std if std > 1e-8 else std.new_tensor(1.))
        self.calibrated.fill_(True)

    def forward_normalized(self, h_S: Tensor, h_G: Optional[Tensor] = None) -> Tensor:
        param = next(self.parameters())
        h_S = h_S.to(device=param.device, dtype=param.dtype)
        if h_S.shape[-1] != self.embed_dim:
            raise ValueError("Embedding dimension does not match ValueNetwork")
        if self.use_context:
            if h_G is None:
                raise ValueError("h_G is required when use_context=True")
            h_G = h_G.to(device=param.device, dtype=param.dtype)
            if h_G.shape != h_S.shape:
                h_G = h_G.expand_as(h_S)
            h_S = torch.cat([h_S, h_G], -1)
        return self.mlp(h_S).squeeze(-1)

    def forward(self, h_S: Tensor, h_G: Optional[Tensor] = None) -> Tensor:
        return (self.forward_normalized(h_S, h_G).to(torch.float64)
                * self.target_scale + self.target_mean)

    @torch.no_grad()
    def predict(self, h_v: Tensor, seed_set, h_G: Optional[Tensor] = None) -> float:
        return self.predict_batch(h_v, [seed_set], h_G)[0]

    @torch.no_grad()
    def predict_batch(self, h_v: Tensor, seed_sets: Sequence,
                      h_G: Optional[Tensor] = None) -> list[float]:
        if not seed_sets:
            return []
        was_training = self.training
        self.eval()
        try:
            scores = self(compute_seed_embedding_batch(h_v, seed_sets), h_G)
            if not bool(torch.isfinite(scores).all()):
                raise FloatingPointError("Nonfinite ValueNet prediction")
            return scores.cpu().tolist()
        finally:
            self.train(was_training)


@dataclass
class TrainConfig:
    lr: float = 5e-4
    weight_decay: float = 1e-5
    epochs: int = 200
    batch_size: int = 32
    patience: int = 30
    normalize_targets: bool = True
    log_every: int = 100
    lambda_rank: float = 0.5
    rank_pairs_per_batch: int = 32
    seed: int = 42


class ValueNetTrainer:
    def __init__(self, model: ValueNetwork, config: Optional[TrainConfig] = None,
                 device=None):
        self.config = config or TrainConfig()
        cfg = self.config
        if (cfg.lr <= 0 or cfg.weight_decay < 0 or cfg.epochs < 1 or
                cfg.batch_size < 1 or cfg.patience < 1 or cfg.lambda_rank < 0 or
                cfg.rank_pairs_per_batch < 0 or cfg.log_every < 0):
            raise ValueError("Invalid training configuration")
        self.device = torch.device(device or next(model.parameters()).device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                          weight_decay=cfg.weight_decay)
        self.train_losses = []
        self.fit_calls = 0

    def fit(self, h_S_batch: Tensor, targets: Tensor,
            sample_weights: Optional[Tensor] = None,
            h_G_batch: Optional[Tensor] = None) -> list[float]:
        cfg, n = self.config, len(h_S_batch)
        x = h_S_batch.detach().to(self.device, torch.float32)
        y = targets.detach().to(self.device, torch.float64)
        w = (torch.ones(n, device=self.device) if sample_weights is None
             else sample_weights.detach().to(self.device, torch.float32))
        if (not n or y.shape != (n,) or w.shape != (n,) or
                not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all()) or
                not bool(torch.isfinite(w).all()) or not bool((w > 0).all())):
            raise ValueError("Training requires finite nonempty aligned data/positive weights")
        context = h_G_batch.detach().to(self.device) if h_G_batch is not None else None
        self.model.calibrate(y, cfg.normalize_targets)
        # Subtract/scale in float64 BEFORE converting to model precision.
        yn = ((y - self.model.target_mean) / self.model.target_scale).float()
        gen = torch.Generator(device=self.device).manual_seed(cfg.seed + self.fit_calls)
        self.fit_calls += 1
        best, stale, snapshot = float("inf"), 0, None
        self.train_losses = []
        self.model.train()
        for epoch in range(1, cfg.epochs + 1):
            order = torch.randperm(n, generator=gen, device=self.device)
            total = 0.
            for start in range(0, n, cfg.batch_size):
                idx = order[start:start + cfg.batch_size]
                ctx = context[idx] if context is not None and context.ndim == 2 else context
                pred = self.model.forward_normalized(x[idx], ctx)
                # GEN-CIM uses mean(w * residual^2), preserving w=1 vs .3.
                mse = (w[idx] * (pred - yn[idx]).square()).mean()
                rank = pred.new_tensor(0.)
                b = len(idx)
                if cfg.lambda_rank and b > 1 and cfg.rank_pairs_per_batch:
                    count = min(cfg.rank_pairs_per_batch, b * (b - 1) // 2)
                    ri = torch.randint(b, (count,), generator=gen, device=self.device)
                    rj = torch.randint(b, (count,), generator=gen, device=self.device)
                    # Exact/raw float64 decides ties; no fake ranking on equal labels.
                    sign = (y[idx][ri] - y[idx][rj]).sign().float()
                    valid = (ri != rj) & (sign != 0)
                    if bool(valid.any()):
                        ri, rj, sign = ri[valid], rj[valid], sign[valid]
                        pair_w = (w[idx][ri] * w[idx][rj]).sqrt()
                        rank = (pair_w * F.softplus(-sign * (pred[ri] - pred[rj]))).mean()
                loss = mse + cfg.lambda_rank * rank
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite ValueNet loss")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                self.optimizer.step()
                total += loss.item() * b
            avg = total / n
            self.train_losses.append(avg)
            if avg < best:
                best, stale = avg, 0
                snapshot = (copy.deepcopy(self.model.state_dict()),
                            copy.deepcopy(self.optimizer.state_dict()))
            else:
                stale += 1
            if cfg.log_every and (epoch == 1 or epoch % cfg.log_every == 0):
                print(f"[ValueNet] epoch={epoch}/{cfg.epochs} loss={avg:.6f} best={best:.6f}", flush=True)
            if stale >= cfg.patience:
                print(f"[ValueNet] Early stop epoch={epoch}; best_loss={best:.6f}", flush=True)
                break
        if snapshot is not None:
            self.model.load_state_dict(snapshot[0])
            self.optimizer.load_state_dict(snapshot[1])
        self.model.eval()
        print(f"[ValueNet] Fit xong: n={n}, epochs={len(self.train_losses)}, "
              f"best_loss={best:.6f}, raw_mean={self.model.target_mean.item():.6f}, "
              f"raw_scale={self.model.target_scale.item():.6f}", flush=True)
        return self.train_losses

    def online_update(self, h_S: Tensor, target: float, h_G=None,
                      lr_multiplier: float = 0.1, n_steps: int = 5) -> float:
        """Optional Stage C correction; the orchestrator follows source (off).

        Never update calibration from one observation: this would rescale all
        predictions before the MLP could learn that change.
        """
        if not bool(self.model.calibrated):
            raise RuntimeError("Bootstrap before online_update")
        if n_steps < 1 or lr_multiplier <= 0:
            raise ValueError("Invalid online-update configuration")
        x = h_S.detach().to(self.device).unsqueeze(0)
        y = torch.tensor([target], dtype=torch.float64, device=self.device)
        if not bool(torch.isfinite(y).all()):
            raise ValueError("target must be finite")
        yn = ((y - self.model.target_mean) / self.model.target_scale).float()
        rates = [p['lr'] for p in self.optimizer.param_groups]
        self.model.train()
        try:
            for p, lr in zip(self.optimizer.param_groups, rates):
                p['lr'] = lr * lr_multiplier
            for _ in range(n_steps):
                loss = (self.model.forward_normalized(x, h_G) - yn).square().mean()
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                self.optimizer.step()
        finally:
            for p, lr in zip(self.optimizer.param_groups, rates):
                p['lr'] = lr
            self.model.eval()
        return loss.item()

    @torch.no_grad()
    def evaluate(self, h_S_batch: Tensor, targets: Tensor, h_G_batch=None) -> dict:
        self.model.eval()
        y = targets.to(self.device, torch.float64)
        pred = self.model(h_S_batch, h_G_batch)
        res = (pred - y).square()
        ss = (y - y.mean()).square().sum()
        return dict(mse=res.mean().item(), mae=(pred-y).abs().mean().item(),
                    r2=1-res.sum().item()/ss.item() if ss > 0 else 0.)

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(model_state=self.model.state_dict(),
                        optimizer_state=self.optimizer.state_dict(),
                        train_config=asdict(self.config), fit_calls=self.fit_calls,
                        model_config=dict(embed_dim=self.model.embed_dim,
                                          hidden_dims=self.model.hidden_dims,
                                          use_context=self.model.use_context,
                                          dropout=self.model.dropout)), path)

    def load(self, path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt['model_state'])
        self.config = TrainConfig(**ckpt['train_config'])
        self.optimizer.load_state_dict(ckpt['optimizer_state'])
        self.fit_calls = ckpt.get('fit_calls', 0)
        self.model.eval()
