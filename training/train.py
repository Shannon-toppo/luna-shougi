"""Train a HalfKP network, stoppably.

    python -m training.train --data data/gen1.bin --run runs/net1 --steps 400000

Stopping and starting again:

    python -m training.control runs/net1 stop     # or just Ctrl+C
    python -m training.train --run runs/net1 --resume

--resume takes every argument from the checkpoint, so the second command needs
nothing but the run directory. Anything given on the command line alongside
--resume overrides what was saved, which is how a learning rate or a step
budget gets changed halfway through a run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import checkpoint, control, dataset, model as model_module, quantize


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="training.train", description=__doc__)
    parser.add_argument("--run", required=True, help="run directory: checkpoints, logs, control")
    parser.add_argument("--data", nargs="+", help="one or more .bin files from luna-datagen")
    parser.add_argument("--resume", action="store_true", help="carry on from the checkpoint")

    parser.add_argument("--steps", type=int, default=400_000, help="total optimizer steps")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-gamma", type=float, default=0.99,
                        help="learning rate decay applied every --lr-every steps")
    parser.add_argument("--lr-every", type=int, default=2000)
    parser.add_argument("--lambda-score", type=float, default=0.7,
                        help="weight of the search score against the game result")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk-samples", type=int, default=1 << 20,
                        help="samples read at a time and shuffled together")

    parser.add_argument("--save-every", type=int, default=2000, help="steps between checkpoints")
    parser.add_argument("--keep-every", type=int, default=50_000,
                        help="steps between checkpoints that are kept rather than replaced")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--export", help="also write a .nnue file every --save-every steps")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


# Arguments that decide the shape of the run rather than how it is driven.
# Changing one of these on a resume changes what is being trained, so they are
# taken from the checkpoint unless they were given again explicitly.
_STRUCTURAL = ("data", "batch_size", "seed", "chunk_samples")


def merge_with_checkpoint(args, saved: dict, given: set[str]) -> argparse.Namespace:
    for key, value in saved.items():
        if key in given:
            continue
        if hasattr(args, key):
            setattr(args, key, value)
    for key in _STRUCTURAL:
        if key in given and key in saved and getattr(args, key) != saved[key]:
            print(
                f"warning: --{key.replace('_', '-')} differs from the checkpoint "
                f"({saved[key]!r} -> {getattr(args, key)!r}); the data order will not "
                "match the stopped run",
                file=sys.stderr,
            )
    return args


def explicitly_given(argv: list[str]) -> set[str]:
    return {
        token.lstrip("-").split("=")[0].replace("-", "_")
        for token in argv
        if token.startswith("--")
    }


def to_device(batch: dataset.Batch, device: torch.device):
    return (
        torch.from_numpy(batch.black.astype(np.int64)).to(device, non_blocking=True),
        torch.from_numpy(batch.white.astype(np.int64)).to(device, non_blocking=True),
        torch.from_numpy(batch.side_to_move.astype(np.float32)).to(device, non_blocking=True),
        torch.from_numpy(batch.score.astype(np.float32)).to(device, non_blocking=True),
        torch.from_numpy(batch.result.astype(np.float32)).to(device, non_blocking=True),
    )


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    given = explicitly_given(argv)

    run_dir = Path(args.run)
    run_dir.mkdir(parents=True, exist_ok=True)

    state = None
    if args.resume:
        path = checkpoint.latest(run_dir)
        if path is None:
            print(f"{run_dir}: nothing to resume from", file=sys.stderr)
            return 1
        state = checkpoint.load(path)
        args = merge_with_checkpoint(args, state["args"], given)
        print(f"resuming {path} at step {state['step']}")

    if not args.data:
        print("--data is required for a new run", file=sys.stderr)
        return 1

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed & 0xFFFFFFFF)

    net = model_module.HalfKP().to(device)
    # Two optimizers because the feature transformer's gradient is sparse and
    # everything else's is not. SparseAdam only touches the rows a batch
    # actually used; plain Adam would rewrite all 32 million every step.
    dense = [value for name, value in net.named_parameters() if name != "ft.weight"]
    optimizer = torch.optim.Adam(dense, lr=args.lr)
    sparse_optimizer = torch.optim.SparseAdam([net.ft.weight], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_every,
                                                gamma=args.lr_gamma)
    sparse_scheduler = torch.optim.lr_scheduler.StepLR(sparse_optimizer,
                                                       step_size=args.lr_every,
                                                       gamma=args.lr_gamma)
    loader = dataset.StreamingLoader(
        args.data,
        batch_size=args.batch_size,
        chunk_samples=args.chunk_samples,
        seed=args.seed,
    )

    step = 0
    if state is not None:
        net.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        sparse_optimizer.load_state_dict(state["sparse_optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        sparse_scheduler.load_state_dict(state["sparse_scheduler"])
        loader.load_state_dict(state["loader"])
        checkpoint.restore_rng(state["rng"])
        step = state["step"]

    driver = control.Control(run_dir)
    # A leftover stop file from the run that just ended would stop this one on
    # its first step. Asking for a stop again is one command; noticing why a
    # fresh run exited instantly is ten minutes.
    driver.clear()
    driver.install_signal_handlers()

    log_path = run_dir / "log.jsonl"
    print(
        f"{len(loader.files):,} samples, {loader.batches_per_epoch():,} batches per epoch, "
        f"device {device}"
    )

    def snapshot() -> dict:
        return {
            "step": step,
            "model": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "sparse_optimizer": sparse_optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "sparse_scheduler": sparse_scheduler.state_dict(),
            "loader": loader.state_dict(),
            "rng": checkpoint.rng_state(),
            "args": vars(args),
        }

    def export_network() -> None:
        if not args.export:
            return
        quantized, clipped = quantize.quantize(net)
        quantize.write(quantized, args.export)
        total = sum(clipped.values())
        if total:
            print(f"  {total} weights had to be clipped to fit the quantization: {clipped}")

    running_loss = 0.0
    running_count = 0
    started = time.time()
    stopped_early = False

    # An explicit iterator, and every reason to stop checked before the next
    # batch is pulled. A `for batch in loader` loop that breaks afterwards has
    # already taken a batch out of the stream, and the checkpoint would then
    # say that batch was trained on when it was not: a resumed run would skip
    # it, and would no longer be the run that was stopped.
    batches = iter(loader)
    while step < args.steps:
        action = driver.poll()
        while action == control.PAUSE:
            # Checkpoint once on the way into the pause, then idle. The GPU is
            # free while this loop spins; the process is only holding its
            # place in the data.
            checkpoint.save(run_dir, snapshot(), keep_every=args.keep_every)
            print(f"paused at step {step}; remove {driver.dir / control.PAUSE} to carry on")
            while driver.poll() == control.PAUSE:
                time.sleep(1.0)
            action = driver.poll()
            if action == control.RUN:
                print("resumed")
        if action == control.STOP:
            stopped_early = True
            break

        try:
            batch = next(batches)
        except StopIteration:
            print(f"ran out of data at step {step}")
            break

        black, white, stm, score, result = to_device(batch, device)
        prediction = net(black, white, stm)
        loss = model_module.loss(prediction, score, result, stm, args.lambda_score)

        optimizer.zero_grad(set_to_none=True)
        sparse_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        sparse_optimizer.step()
        scheduler.step()
        sparse_scheduler.step()
        net.clip_weights()

        step += 1
        running_loss += float(loss.detach())
        running_count += 1

        if step % args.log_every == 0:
            average = running_loss / max(1, running_count)
            rate = step / max(1e-9, time.time() - started)
            record = {
                "step": step,
                "epoch": loader.epoch,
                "loss": average,
                "lr": scheduler.get_last_lr()[0],
                "steps_per_second": rate,
            }
            print(
                f"step {step:>8}  epoch {loader.epoch}  loss {average:.6f}  "
                f"lr {record['lr']:.2e}  {rate:.1f} steps/s"
            )
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            running_loss = 0.0
            running_count = 0

        if step % args.save_every == 0:
            checkpoint.save(run_dir, snapshot(), keep_every=args.keep_every)
            export_network()

    checkpoint.save(run_dir, snapshot(), keep_every=args.keep_every)
    export_network()

    if stopped_early:
        print(f"stopped at step {step}. Carry on with:")
        print(f"  python -m training.train --run {run_dir} --resume")
    else:
        print(f"finished at step {step}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
