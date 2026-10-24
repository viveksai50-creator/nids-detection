"""Variational autoencoder anomaly detector. Score = reconstruction MSE + beta * KL (using mu, no sampling)."""
from __future__ import annotations

import math
import time
import torch
from torch import nn

from src import config as C
from src.models.base import BaseDetector
from src.models.autoencoder import AutoencoderDetector
from src.models.torch_utils import (EarlyStopper, batched_apply, device,
                                    iterate_minibatches, mlp, set_seed,
                                    to_tensor)


class VAE(nn.Module):
    def __init__(self, d: int, hidden: int, latent: int):
        super().__init__()
        self.enc = mlp([d, hidden, hidden])
        self.mu = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)
        self.dec = mlp([latent, hidden, d])

    def encode(self, x):
        h = torch.relu(self.enc(x))
        return self.mu(h), self.logvar(h)

    def forward(self, x, sample: bool = True):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if sample else mu
        return self.dec(z), mu, logvar

    def row_loss(self, x, beta: float, sample: bool = True):
        recon, mu, logvar = self.forward(x, sample=sample)
        mse = ((recon - x) ** 2).mean(dim=1)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1) / x.shape[1]
        return mse + beta * kl


class VAEDetector(AutoencoderDetector):
    name = "vae"
    benign_only = True

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        set_seed(self.seed)
        self.dev = device()
        hp = self.hp
        self.beta = float(hp.get("beta", 1.0))
        Xb = to_tensor(self.benign(X_train, y_train), self.dev)
        Xv = to_tensor(self.benign(X_val, y_val), self.dev) if X_val is not None else None
        self.net = VAE(Xb.shape[1], int(hp.get("hidden", 128)), int(hp.get("latent", 16))).to(self.dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=float(hp.get("lr", 1e-3)), weight_decay=1e-5)
        gen = torch.Generator().manual_seed(self.seed)
        self.shuffle_generator = gen
        stopper = EarlyStopper()
        self.history = []
        start_epoch = self._load_checkpoint(opt, stopper)
        if stopper.bad >= stopper.patience:
            start_epoch = int(hp.get("epochs", C.MAX_EPOCHS))
        for epoch in range(start_epoch, int(hp.get("epochs", C.MAX_EPOCHS))):
            epoch_start = time.perf_counter()
            self.net.train()
            total, count = 0.0, 0
            for xb in iterate_minibatches(Xb, C.TORCH_BATCH, shuffle=True, gen=gen):
                opt.zero_grad()
                loss = self.net.row_loss(xb, self.beta).mean()
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(xb)
                count += len(xb)
            train_loss = total / max(1, count)
            val_loss = float(self._val_loss(Xv)) if Xv is not None else train_loss
            if not math.isfinite(train_loss) or not math.isfinite(val_loss):
                raise FloatingPointError("non-finite VAE training or benign-validation loss")
            self.history.append({"epoch": epoch, "train": train_loss, "val": val_loss,
                                 "seconds": time.perf_counter() - epoch_start})
            should_stop = stopper.step(val_loss, self.net)
            if hasattr(self, "checkpoint_path"):
                self._save_checkpoint(epoch, opt, stopper)
            print(f"  VAE epoch {epoch + 1}: train={train_loss:.6g} val={val_loss:.6g}", flush=True)
            if should_stop:
                break
        stopper.restore(self.net)
        self.net.eval()

    @torch.no_grad()
    def _val_loss(self, X):
        self.net.eval()
        tot, n = 0.0, 0
        for i in range(0, X.shape[0], 4096):
            xb = X[i : i + 4096]
            tot += float(self.net.row_loss(xb, self.beta, sample=False).sum())
            n += len(xb)
        return tot / max(1, n)

    def score(self, X):
        return batched_apply(lambda xb: self.net.row_loss(xb, self.beta, sample=False), X, self.dev)
