"""Float model -> quantized .nnue file, and the integer forward pass again.

The second half of this file is a deliberate duplicate of what
src/nnue/network.cpp does. It exists so training/verify.py can compare the two
number for number: if the trainer and the engine disagree about what a network
computes, the strength measured in training is not the strength that plays,
and there is no way to find that out from either side alone.

The scheme is documented in src/nnue/network.hpp; the constants below are the
same ones.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from . import features

FILE_MAGIC = b"LUNANNUE"
FILE_VERSION = 1
HEADER_BYTES = 32

ACTIVATION_SCALE = 127
WEIGHT_SCALE_BITS = 6
WEIGHT_SCALE = 1 << WEIGHT_SCALE_BITS
HIDDEN_BIAS_SCALE = ACTIVATION_SCALE * WEIGHT_SCALE  # 8128
PONANZA_CONSTANT = 600
FV_SCALE = 16
OUTPUT_BIAS_SCALE = PONANZA_CONSTANT * FV_SCALE  # 9600
OUTPUT_WEIGHT_SCALE = OUTPUT_BIAS_SCALE / ACTIVATION_SCALE
MAX_EVAL_SCORE = 16000


@dataclass
class QuantizedNet:
    ft_weight: np.ndarray  # (FEATURE_DIMS, ft_dim) int16
    ft_bias: np.ndarray  # (ft_dim,) int16
    l1_weight: np.ndarray  # (l1_out, 2*ft_dim) int8
    l1_bias: np.ndarray  # (l1_out,) int32
    l2_weight: np.ndarray  # (l2_out, l1_out) int8
    l2_bias: np.ndarray  # (l2_out,) int32
    l3_weight: np.ndarray  # (l2_out,) int8
    l3_bias: np.ndarray  # (1,) int32

    @property
    def ft_dim(self) -> int:
        return self.ft_bias.shape[0]


def _round_clip(values, scale, dtype):
    """Scale, round to nearest, and report anything that had to be clipped."""
    info = np.iinfo(dtype)
    scaled = np.rint(np.asarray(values, dtype=np.float64) * scale)
    clipped = int(np.count_nonzero((scaled < info.min) | (scaled > info.max)))
    return np.clip(scaled, info.min, info.max).astype(dtype), clipped


def quantize(model) -> tuple[QuantizedNet, dict]:
    """The model as integers, plus a count of weights that had to be clipped.

    A non-zero count is not fatal but is worth knowing about: it means the
    float network was using a weight the engine cannot represent, and the two
    have started to disagree. model.clip_weights during training is what keeps
    it at zero.
    """
    import torch

    with torch.no_grad():
        state = {name: value.detach().cpu().numpy() for name, value in model.state_dict().items()}

    report = {}
    ft_weight, report["ft_weight"] = _round_clip(state["ft.weight"], ACTIVATION_SCALE, np.int16)
    ft_bias, report["ft_bias"] = _round_clip(state["ft_bias"], ACTIVATION_SCALE, np.int16)
    l1_weight, report["l1_weight"] = _round_clip(state["l1.weight"], WEIGHT_SCALE, np.int8)
    l1_bias, report["l1_bias"] = _round_clip(state["l1.bias"], HIDDEN_BIAS_SCALE, np.int32)
    l2_weight, report["l2_weight"] = _round_clip(state["l2.weight"], WEIGHT_SCALE, np.int8)
    l2_bias, report["l2_bias"] = _round_clip(state["l2.bias"], HIDDEN_BIAS_SCALE, np.int32)
    l3_weight, report["l3_weight"] = _round_clip(
        state["out.weight"][0], OUTPUT_WEIGHT_SCALE, np.int8
    )
    l3_bias, report["l3_bias"] = _round_clip(state["out.bias"], OUTPUT_BIAS_SCALE, np.int32)

    return (
        QuantizedNet(
            ft_weight=ft_weight,
            ft_bias=ft_bias,
            l1_weight=l1_weight,
            l1_bias=l1_bias,
            l2_weight=l2_weight,
            l2_bias=l2_bias,
            l3_weight=l3_weight,
            l3_bias=l3_bias,
        ),
        report,
    )


def write(net: QuantizedNet, path: str) -> None:
    header = struct.pack(
        "<8sIIIIII",
        FILE_MAGIC,
        FILE_VERSION,
        features.FEATURE_DIMS,
        net.ft_dim,
        net.l1_weight.shape[0],
        net.l2_weight.shape[0],
        0,
    )
    assert len(header) == HEADER_BYTES
    with open(path, "wb") as handle:
        handle.write(header)
        # The order the engine reads them in: bias then weights, layer by
        # layer, every matrix row-major by output.
        handle.write(net.ft_bias.astype("<i2").tobytes())
        handle.write(net.ft_weight.astype("<i2").tobytes())
        handle.write(net.l1_bias.astype("<i4").tobytes())
        handle.write(net.l1_weight.astype("<i1").tobytes())
        handle.write(net.l2_bias.astype("<i4").tobytes())
        handle.write(net.l2_weight.astype("<i1").tobytes())
        handle.write(net.l3_bias.astype("<i4").tobytes())
        handle.write(net.l3_weight.astype("<i1").tobytes())


def read(path: str) -> QuantizedNet:
    with open(path, "rb") as handle:
        header = handle.read(HEADER_BYTES)
        if len(header) != HEADER_BYTES:
            raise ValueError(f"{path}: too short to hold a header")
        magic, version, feature_dims, ft_dim, l1_out, l2_out, _ = struct.unpack("<8sIIIIII", header)
        if magic != FILE_MAGIC:
            raise ValueError(f"{path}: not a luna nnue file")
        if version != FILE_VERSION:
            raise ValueError(f"{path}: version {version}, expected {FILE_VERSION}")
        if feature_dims != features.FEATURE_DIMS:
            raise ValueError(f"{path}: {feature_dims} features, expected {features.FEATURE_DIMS}")

        def take(count, dtype):
            raw = handle.read(count * np.dtype(dtype).itemsize)
            if len(raw) != count * np.dtype(dtype).itemsize:
                raise ValueError(f"{path}: ended in the middle of the weights")
            return np.frombuffer(raw, dtype=dtype).copy()

        ft_bias = take(ft_dim, "<i2")
        ft_weight = take(feature_dims * ft_dim, "<i2").reshape(feature_dims, ft_dim)
        l1_bias = take(l1_out, "<i4")
        l1_weight = take(l1_out * 2 * ft_dim, "<i1").reshape(l1_out, 2 * ft_dim)
        l2_bias = take(l2_out, "<i4")
        l2_weight = take(l2_out * l1_out, "<i1").reshape(l2_out, l1_out)
        l3_bias = take(1, "<i4")
        l3_weight = take(l2_out, "<i1")

    return QuantizedNet(
        ft_weight=ft_weight,
        ft_bias=ft_bias,
        l1_weight=l1_weight,
        l1_bias=l1_bias,
        l2_weight=l2_weight,
        l2_bias=l2_bias,
        l3_weight=l3_weight,
        l3_bias=l3_bias,
    )


def random_net(seed: int = 0, ft_dim: int = 256, l1_out: int = 32, l2_out: int = 32):
    """An untrained network, for checking that the two sides agree.

    Nothing here has to have learned anything: the point of the cross-check is
    the arithmetic, and random weights exercise it as well as trained ones and
    are a great deal quicker to come by. The feature transformer stays small
    so that 38 of its rows plus a bias cannot overflow the engine's int16
    accumulator, which is a property of a trained network too.
    """
    rng = np.random.default_rng(seed)
    return QuantizedNet(
        ft_weight=rng.integers(-40, 41, size=(features.FEATURE_DIMS, ft_dim), dtype=np.int16),
        ft_bias=rng.integers(-127, 128, size=ft_dim, dtype=np.int16),
        l1_weight=rng.integers(-127, 128, size=(l1_out, 2 * ft_dim)).astype(np.int8),
        l1_bias=rng.integers(-4000, 4001, size=l1_out, dtype=np.int32),
        l2_weight=rng.integers(-127, 128, size=(l2_out, l1_out)).astype(np.int8),
        l2_bias=rng.integers(-4000, 4001, size=l2_out, dtype=np.int32),
        l3_weight=rng.integers(-127, 128, size=l2_out).astype(np.int8),
        l3_bias=rng.integers(-9600, 9601, size=1, dtype=np.int32),
    )


# --- the integer forward pass ---------------------------------------------


def _clipped_relu_16(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0, ACTIVATION_SCALE).astype(np.uint8)


def _clipped_relu_32(values: np.ndarray) -> np.ndarray:
    # An arithmetic right shift, which is what C++ does to a negative int32
    # since C++20 and what numpy does to a signed integer array.
    return np.clip(values >> WEIGHT_SCALE_BITS, 0, ACTIVATION_SCALE).astype(np.uint8)


def _trunc_div(value: np.ndarray, divisor: int) -> np.ndarray:
    """C++ integer division, which truncates towards zero rather than down."""
    return np.trunc(value.astype(np.float64) / divisor).astype(np.int64)


def accumulate(net: QuantizedNet, indices: np.ndarray) -> np.ndarray:
    """One half of the accumulator for a batch: (B, 38) indices -> (B, ft_dim).

    int32 here rather than int16, and clipped afterwards, so that an overflow
    in the engine's int16 accumulator shows up in verify.py as a mismatch
    instead of being reproduced.
    """
    return net.ft_weight[indices].sum(axis=1, dtype=np.int32) + net.ft_bias.astype(np.int32)


def evaluate(net: QuantizedNet, black: np.ndarray, white: np.ndarray, stm: np.ndarray):
    """Scores in engine units, exactly as src/nnue/network.cpp computes them."""
    black = np.atleast_2d(np.asarray(black, dtype=np.int64))
    white = np.atleast_2d(np.asarray(white, dtype=np.int64))
    stm = np.atleast_1d(np.asarray(stm, dtype=np.int64))

    black_half = accumulate(net, black)
    white_half = accumulate(net, white)
    pick = stm[:, None] == 1
    own = np.where(pick, white_half, black_half)
    opponent = np.where(pick, black_half, white_half)

    x = np.concatenate([_clipped_relu_16(own), _clipped_relu_16(opponent)], axis=1)
    x = _clipped_relu_32(
        x.astype(np.int32) @ net.l1_weight.astype(np.int32).T + net.l1_bias.astype(np.int32)
    )
    x = _clipped_relu_32(
        x.astype(np.int32) @ net.l2_weight.astype(np.int32).T + net.l2_bias.astype(np.int32)
    )
    raw = x.astype(np.int32) @ net.l3_weight.astype(np.int32) + net.l3_bias.astype(np.int32)
    return np.clip(_trunc_div(raw, FV_SCALE), -MAX_EVAL_SCORE, MAX_EVAL_SCORE)
