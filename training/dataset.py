"""Streaming, resumable reader for the .bin files luna-datagen writes.

RAM is 32GB and a serious data set is larger than that, so nothing is ever
loaded whole. Records are fixed at 84 bytes, which is what makes both halves
of that possible: seeking to sample N is a multiplication, so the loader can
be told to carry on from exactly where a stopped run left off.

Shuffling still has to happen, because consecutive samples come from the same
game and a batch of one game's positions is a batch of almost the same
position. The compromise is the usual one: read a large contiguous chunk,
shuffle inside it, and shuffle the order the chunks are read in. Both
shufflings are derived from a seed and the epoch number, so they replay
identically on resume.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np

from . import features

SAMPLE_DTYPE = np.dtype(
    [
        ("bona", "<u2", (features.ACTIVE_FEATURES,)),
        ("king_black", "u1"),
        ("king_white", "u1"),
        ("side_to_move", "u1"),
        ("result", "i1"),
        ("score", "<i2"),
        ("ply", "<u2"),
    ]
)

SAMPLE_BYTES = SAMPLE_DTYPE.itemsize
assert SAMPLE_BYTES == 84, f"record layout drifted from the engine's: {SAMPLE_BYTES}"

# `result` is +1, 0 or -1 for a game black won, drew or lost. This is a fourth
# value meaning "there was no game": positions lifted out of a search tree by
# `luna-datagen --label` have none, and the engine writes 0 for them, which the
# loss would otherwise read as a draw. src/datagen/datagen.cpp says so in as
# many words -- "fine to measure against and wrong to train on".
#
# Nothing in the engine writes this. It is stamped on by whatever builds a
# mixed file (see docs/nnue-plan.md step 2), and training/model.py drops the
# result term for the samples carrying it.
RESULT_UNKNOWN = 2


@dataclass
class Batch:
    """One batch, still as numpy. train.py moves it to the GPU."""

    black: np.ndarray  # (B, 38) int32 feature indices
    white: np.ndarray  # (B, 38) int32
    side_to_move: np.ndarray  # (B,) int8, 0 black
    score: np.ndarray  # (B,) int16, black's point of view
    result: np.ndarray  # (B,) int8, +1 black won

    def __len__(self) -> int:
        return len(self.side_to_move)


class SampleFiles:
    """Several .bin files addressed as one array of samples."""

    def __init__(self, paths: Sequence[str]):
        if not paths:
            raise ValueError("no data files given")
        self.paths = [str(p) for p in paths]
        self.counts = []
        for path in self.paths:
            size = os.path.getsize(path)
            if size % SAMPLE_BYTES != 0:
                raise ValueError(
                    f"{path} is {size} bytes, not a whole number of {SAMPLE_BYTES}-byte samples"
                )
            self.counts.append(size // SAMPLE_BYTES)
        self.offsets = np.cumsum([0] + self.counts)
        self.total = int(self.offsets[-1])
        if self.total == 0:
            raise ValueError("the data files are empty")

    def __len__(self) -> int:
        return self.total

    def read(self, start: int, count: int) -> np.ndarray:
        """`count` samples from global index `start`, crossing files as needed."""
        out = np.empty(count, dtype=SAMPLE_DTYPE)
        written = 0
        while written < count:
            index = int(np.searchsorted(self.offsets, start + written, side="right") - 1)
            local = start + written - int(self.offsets[index])
            take = min(count - written, self.counts[index] - local)
            with open(self.paths[index], "rb") as handle:
                handle.seek(local * SAMPLE_BYTES)
                raw = handle.read(take * SAMPLE_BYTES)
            if len(raw) != take * SAMPLE_BYTES:
                raise IOError(f"{self.paths[index]} ended early")
            out[written : written + take] = np.frombuffer(raw, dtype=SAMPLE_DTYPE)
            written += take
        return out


def to_batch(samples: np.ndarray) -> Batch:
    black, white = features.halfkp_indices(
        samples["bona"], samples["king_black"], samples["king_white"]
    )
    return Batch(
        black=black,
        white=white,
        side_to_move=samples["side_to_move"].astype(np.int8),
        score=samples["score"].astype(np.int32),
        result=samples["result"].astype(np.int32),
    )


def sample_evenly(paths: Sequence[str], count: int) -> Batch:
    """`count` samples spread over the files, as one batch.

    For quantization's bias correction, which needs a few thousand positions
    that look like the ones the network will meet and needs them cheaply.

    Evenly spread rather than the first N, because the front of a datagen file
    is the opening moves of the first games and nothing else. Nothing random is
    involved: the same files and the same count give the same batch every time,
    so a checkpoint exported twice is the same network twice.
    """
    files = SampleFiles(paths)
    index = np.linspace(0, len(files) - 1, min(count, len(files))).astype(np.int64)
    return to_batch(np.concatenate([files.read(int(i), 1) for i in index]))


class StreamingLoader:
    """Batches, in an order that a stopped run can pick up again exactly.

    The position is (epoch, chunk, batch), and everything else about the order
    follows from the seed and the epoch, so those three numbers are the whole
    of the loader's state.
    """

    def __init__(
        self,
        paths: Sequence[str],
        batch_size: int,
        chunk_samples: int = 1 << 20,
        seed: int = 0,
        epochs: int | None = None,
    ):
        self.files = SampleFiles(paths)
        self.batch_size = batch_size
        self.chunk_samples = chunk_samples
        self.seed = seed
        self.epochs = epochs

        self.n_chunks = max(1, (len(self.files) + chunk_samples - 1) // chunk_samples)
        self.epoch = 0
        self.chunk = 0
        self.batch = 0

    # --- resume ---------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "chunk": self.chunk,
            "batch": self.batch,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "chunk_samples": self.chunk_samples,
            "total": len(self.files),
        }

    def load_state_dict(self, state: dict) -> None:
        # The order depends on all of these, so a run resumed with different
        # ones is not the run that was stopped. Say so rather than quietly
        # replaying data or skipping it.
        for key in ("seed", "batch_size", "chunk_samples", "total"):
            mine = getattr(self, key, None) if key != "total" else len(self.files)
            if key in state and state[key] != mine:
                raise ValueError(
                    f"cannot resume: {key} was {state[key]} and is now {mine}. "
                    "Start a new run, or put the old value back."
                )
        self.epoch = state["epoch"]
        self.chunk = state["chunk"]
        self.batch = state["batch"]

    # --- iteration ------------------------------------------------------

    def _chunk_order(self, epoch: int) -> np.ndarray:
        return np.random.default_rng([self.seed, epoch, 1]).permutation(self.n_chunks)

    def _within_chunk(self, epoch: int, chunk: int, count: int) -> np.ndarray:
        return np.random.default_rng([self.seed, epoch, 2, chunk]).permutation(count)

    def batches_per_epoch(self) -> int:
        """Approximate: the last chunk is short, and its tail batch is dropped."""
        full = len(self.files) // self.chunk_samples
        rest = len(self.files) % self.chunk_samples
        return full * (self.chunk_samples // self.batch_size) + rest // self.batch_size

    def __iter__(self) -> Iterator[Batch]:
        while self.epochs is None or self.epoch < self.epochs:
            order = self._chunk_order(self.epoch)
            while self.chunk < self.n_chunks:
                start = int(order[self.chunk]) * self.chunk_samples
                count = min(self.chunk_samples, len(self.files) - start)
                if count < self.batch_size:
                    self.chunk += 1
                    self.batch = 0
                    continue
                samples = self.files.read(start, count)
                shuffled = samples[self._within_chunk(self.epoch, self.chunk, count)]

                n_batches = count // self.batch_size
                while self.batch < n_batches:
                    at = self.batch * self.batch_size
                    # Advance before yielding: a checkpoint taken by the
                    # consumer has to record the batch after this one, or a
                    # resumed run trains on it twice.
                    self.batch += 1
                    yield to_batch(shuffled[at : at + self.batch_size])
                self.chunk += 1
                self.batch = 0
            self.epoch += 1
            self.chunk = 0
            self.batch = 0
