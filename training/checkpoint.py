"""Everything needed to carry on as if the run had never stopped.

A checkpoint that only holds the weights is enough to keep the network but
not enough to keep the run: the optimizer's momentum, the learning rate
schedule, the place in the data and the random state all decide what the next
step does. Losing them means the resumed run is a different run that happens
to start from the same weights, which makes any measurement across a stop
meaningless.

Writes go to a temporary file and are then renamed over the real one, because
a stop can arrive during a save and a half-written checkpoint is worse than
no checkpoint at all.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

LATEST = "checkpoint.pt"


def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save(run_dir: str | Path, payload: dict, keep_every: int | None = None) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / LATEST
    temporary = run_dir / (LATEST + ".tmp")

    torch.save(payload, temporary)
    os.replace(temporary, target)

    # A dated copy every so often, so that a run which went wrong somewhere
    # can be restarted from before it did rather than from nothing.
    step = payload.get("step", 0)
    if keep_every and step and step % keep_every == 0:
        torch.save(payload, run_dir / f"checkpoint-{step:09d}.pt")
    return target


def load(path: str | Path) -> dict:
    # weights_only=False: a checkpoint holds numpy and Python RNG state as
    # well as tensors, and it is a file this project wrote itself.
    return torch.load(path, map_location="cpu", weights_only=False)


def build_model(state: dict):
    """The model the checkpoint was trained with, shaped from its own args.

    The width is a run-time choice on this side and a compile-time one in the
    engine, so anything that loads a checkpoint has to take the shape from the
    checkpoint rather than from model.FT_DIM. Older checkpoints have no
    ft_dim in their args and were all trained at the default.
    """
    from . import model as model_module

    args = state.get("args") or {}
    net = model_module.HalfKP(ft_dim=args.get("ft_dim", model_module.FT_DIM))
    net.load_state_dict(state["model"])
    return net


def latest(run_dir: str | Path) -> Path | None:
    path = Path(run_dir) / LATEST
    return path if path.exists() else None
