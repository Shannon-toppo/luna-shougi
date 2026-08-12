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
# Still 1. PONANZA_CONSTANT changing is not a format change: it is folded into
# the output layer's weights and biases here, and the engine divides by
# FV_SCALE either way. The version is for the layout.
FILE_VERSION = 1
HEADER_BYTES = 32

ACTIVATION_SCALE = 127
WEIGHT_SCALE_BITS = 6
WEIGHT_SCALE = 1 << WEIGHT_SCALE_BITS
HIDDEN_BIAS_SCALE = ACTIVATION_SCALE * WEIGHT_SCALE  # 8128
# Engine points per unit of the network's output. This is a trainer-side
# choice and the engine never sees it: it is folded into the output layer
# below, so a network built at any value scores correctly through the same
# code. Keep it in step with training/model.py, which is where the loss uses
# it and where the reasoning for the value is written down.
PONANZA_CONSTANT = 1800
FV_SCALE = 16
OUTPUT_BIAS_SCALE = PONANZA_CONSTANT * FV_SCALE  # 28800
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


def forward(net: QuantizedNet, black: np.ndarray, white: np.ndarray, stm: np.ndarray):
    """The integer pipeline stage by stage: (L1 sum, L2 sum, output sum).

    The two hidden ones are the int32 sums before the shift and the clamp, at
    scale 8128; the last is before the division by kFvScale, at scale 9600.

    Split out of evaluate() rather than written twice because correct_biases
    needs to see the numbers on the way through, and a second copy of this
    arithmetic would be a second thing for the engine to disagree with. This
    one is what verify.py holds src/nnue/network.cpp to.
    """
    black = np.atleast_2d(np.asarray(black, dtype=np.int64))
    white = np.atleast_2d(np.asarray(white, dtype=np.int64))
    stm = np.atleast_1d(np.asarray(stm, dtype=np.int64))

    black_half = accumulate(net, black)
    white_half = accumulate(net, white)
    pick = stm[:, None] == 1
    own = np.where(pick, white_half, black_half)
    opponent = np.where(pick, black_half, white_half)

    x = np.concatenate([_clipped_relu_16(own), _clipped_relu_16(opponent)], axis=1)
    l1 = x.astype(np.int32) @ net.l1_weight.astype(np.int32).T + net.l1_bias.astype(np.int32)
    x = _clipped_relu_32(l1)
    l2 = x.astype(np.int32) @ net.l2_weight.astype(np.int32).T + net.l2_bias.astype(np.int32)
    x = _clipped_relu_32(l2)
    out = x.astype(np.int32) @ net.l3_weight.astype(np.int32) + net.l3_bias.astype(np.int32)
    return l1, l2, out


def evaluate(net: QuantizedNet, black: np.ndarray, white: np.ndarray, stm: np.ndarray):
    """Scores in engine units, exactly as src/nnue/network.cpp computes them."""
    raw = forward(net, black, white, stm)[2]
    return np.clip(_trunc_div(raw, FV_SCALE), -MAX_EVAL_SCORE, MAX_EVAL_SCORE)


# --- bias correction -------------------------------------------------------


def _float_stages(model, black: np.ndarray, white: np.ndarray, stm: np.ndarray):
    """The float model's three pre-activations on the same positions.

    Taken off the model with forward hooks rather than by writing the forward
    pass out again here. The correction is measured against what training
    actually optimized, and a third copy of the network -- after model.py and
    src/nnue/network.cpp -- would be a third thing to keep in step.
    """
    import torch

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    captured: dict = {}
    handles = [
        module.register_forward_hook(
            lambda _module, _inputs, output, key=key: captured.__setitem__(
                key, output.detach().to("cpu", torch.float64).numpy()
            )
        )
        for key, module in (("l1", model.l1), ("l2", model.l2), ("out", model.out))
    ]
    try:
        with torch.no_grad():
            model(
                torch.as_tensor(np.asarray(black), dtype=torch.long, device=device),
                torch.as_tensor(np.asarray(white), dtype=torch.long, device=device),
                torch.as_tensor(np.asarray(stm), dtype=torch.float32, device=device),
            )
    finally:
        for handle in handles:
            handle.remove()
        if was_training:
            model.train()
    return captured["l1"], captured["l2"], captured["out"][:, 0]


def correct_biases(net: QuantizedNet, model, black, white, stm) -> dict:
    """Move each layer's integer bias so the layer's mean output is right again.

    Rounding a weight to int8 is unbiased across weights, but a layer is not a
    weight. L1 and L2 are dense: every one of their inputs is used by every
    position, so their rounding error is one fixed linear functional of an
    activation vector that barely moves from position to position. It is
    therefore a near-constant, not noise -- a different constant for every
    network, of either sign, and around thirty engine points wide. The feature
    transformer behaves properly by comparison, because each position sums a
    different 38 of its rows and the error really does average out.

    A constant is the one shape of error this evaluation cannot afford. It is
    from the side to move, so negamax flips its sign every ply and it lands as
    a tempo bonus at every node.

    So: run `black`, `white`, `stm` through both the float model and the
    quantized one, take the mean error at each layer, and subtract it from that
    layer's integer bias. The biases are int32 with room to spare, which is
    what makes this free. Layer by layer and in order, because correcting L1
    changes what L2 sees; by the last layer the mean output error on these
    positions is nulled outright, and the shift and the truncation get absorbed
    along with everything else.

    The engine is untouched: this changes the numbers in the file, not the
    arithmetic that reads them, so training/verify.py still matches exactly.

    Calibrate on training positions and measure on held-out ones -- a mean
    fitted on the holdout and then reported there says nothing.
    """
    info = np.iinfo(np.int32)
    float_stages = _float_stages(model, black, white, stm)
    scales = (HIDDEN_BIAS_SCALE, HIDDEN_BIAS_SCALE, OUTPUT_BIAS_SCALE)
    names = ("l1_bias", "l2_bias", "l3_bias")

    def mean_output_error() -> float:
        """Where the network's score sits against the model's, in engine points."""
        raw = forward(net, black, white, stm)[2]
        return float(np.mean(raw - OUTPUT_BIAS_SCALE * float_stages[2]) / FV_SCALE)

    report = {"before": mean_output_error()}
    for index, (name, scale) in enumerate(zip(names, scales)):
        error = forward(net, black, white, stm)[index] - scale * float_stages[index]
        shift = np.rint(np.mean(error, axis=0))
        corrected = np.clip(getattr(net, name) - shift, info.min, info.max)
        setattr(net, name, corrected.astype(np.int32))
        report[name] = float(np.max(np.abs(shift)))
    report["after"] = mean_output_error()
    return report
