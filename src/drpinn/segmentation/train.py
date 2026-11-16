"""Training loop with checkpoint/resume (survives Colab disconnects)."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .. import __version__
from .data import PatchSampler, SegSample
from .unet import UNet, UNetConfig


@dataclass
class TrainConfig:
    iterations: int = 4000
    batch: int = 16
    patch: int = 96
    lr: float = 1e-3
    weight_decay: float = 1e-4
    vessel_fraction: float = 0.5
    av_loss_weight: float = 1.0
    save_every: int = 250
    seed: int = 0
    unet: UNetConfig = field(default_factory=UNetConfig)
    av_balance: bool = False  # weight artery and vein pixels equally (arteries are thinner, so fewer pixels)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["unet"] = self.unet.to_dict()
        if not self.av_balance:  # keep the settings of older checkpoints identical, so they still resume
            d.pop("av_balance")
        return d


def loss_fn(logits: torch.Tensor, y: torch.Tensor, w: torch.Tensor, av_weight: float = 1.0,
            av_balance: bool = False) -> torch.Tensor:
    """Vessel: weighted BCE + soft Dice inside the FOV. Artery/vein: BCE on labelled vessel pixels.

    With `av_balance`, artery-only and vein-only pixels each carry half of the
    A/V loss, so the larger vein area does not pull every doubtful pixel to vein.
    """
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    v_w = w[:, 0]
    vessel_bce = (bce[:, 0] * v_w).sum() / v_w.sum().clamp_min(1.0)
    p = torch.sigmoid(logits[:, 0]) * v_w
    t = y[:, 0] * v_w
    dice = 1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1)
    loss = vessel_bce + dice
    av_w = w[:, 1:]
    if av_w.sum() > 0:
        if av_balance:
            art = (y[:, 1] > 0.5) & (y[:, 2] < 0.5)
            vein = (y[:, 2] > 0.5) & (y[:, 1] < 0.5)
            n_a, n_v = (art.float() * w[:, 1]).sum(), (vein.float() * w[:, 1]).sum()
            scale = torch.ones_like(w[:, 1])
            if n_a > 0 and n_v > 0:
                scale = torch.where(art, 0.5 * (n_a + n_v) / n_a, scale)
                scale = torch.where(vein, 0.5 * (n_a + n_v) / n_v, scale)
            av_w = av_w * scale.unsqueeze(1)
        loss = loss + av_weight * (bce[:, 1:] * av_w).sum() / av_w.sum()
    return loss


def _atomic_save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[UNet, dict]:
    """Model + metadata. Raises FileNotFoundError with a clear message when missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"model weights not found: {path} — train them first (notebook step 12)")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = UNet(UNetConfig(**ckpt["config"]["unet"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def train(samples: list[SegSample], config: TrainConfig, ckpt_path: str | Path, device: str = "cpu",
          meta: dict | None = None, log_every: int = 100, use_av: bool = True,
          stop_after: int | None = None) -> tuple[UNet, dict]:
    """Train (or resume) a model; the checkpoint at `ckpt_path` is updated as training goes.

    `stop_after` ends this call after that many steps (checkpoint saved, not
    complete); calling again resumes. A completed checkpoint is reused as is.
    """
    ckpt_path = Path(ckpt_path)
    torch.manual_seed(config.seed)
    model = UNet(config.unet).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(s, config.iterations) / config.iterations)))
    use_amp = str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    step, history = 0, []
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        if ck.get("config") != config.to_dict():
            raise ValueError(f"{ckpt_path} was trained with different settings; delete it or change --ckpt-dir")
        model.load_state_dict(ck["model"])
        if ck.get("complete"):
            model.eval()
            return model, ck
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        step, history = ck["step"], ck.get("history", [])
        print(f"  resuming {ckpt_path.name} from step {step}", flush=True)
    sampler = PatchSampler(samples, config.patch, config.vessel_fraction, seed=config.seed * 1000 + step, use_av=use_av)
    meta = {**(meta or {}), "train_ids": [s.image_id for s in samples], "drpinn_version": __version__,
            "torch_version": torch.__version__, "device": str(device)}

    def checkpoint(complete: bool):
        state = {"model": model.state_dict(), "step": step, "history": history, "config": config.to_dict(),
                 "meta": meta, "complete": complete}
        if not complete:  # optimizer state is only needed to resume; finished models stay small
            state.update(optimizer=opt.state_dict(), scheduler=sched.state_dict())
        _atomic_save(state, ckpt_path)

    model.train()
    t0, running = time.perf_counter(), []
    start = step
    while step < config.iterations:
        if stop_after is not None and step - start >= stop_after:
            checkpoint(False)
            model.eval()
            return model, torch.load(ckpt_path, map_location=device, weights_only=False)
        x, y, w = (torch.from_numpy(a).to(device) for a in sampler.batch(config.batch))
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        loss = loss_fn(logits.float(), y, w, config.av_loss_weight, config.av_balance)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1
        running.append(float(loss.detach()))
        if step % log_every == 0 or step == config.iterations:
            history.append({"step": step, "loss": float(np.mean(running))})
            rate = len(running) / max(time.perf_counter() - t0, 1e-9)
            print(f"  step {step}/{config.iterations}  loss {history[-1]['loss']:.4f}  ({rate:.1f} it/s)", flush=True)
            running, t0 = [], time.perf_counter()
        if step % config.save_every == 0 and step < config.iterations:
            checkpoint(False)
    checkpoint(True)
    model.eval()
    return model, torch.load(ckpt_path, map_location=device, weights_only=False)


def write_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=float))
