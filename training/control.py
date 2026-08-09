"""Stopping and pausing a training run from outside it.

Training a net runs for hours, on the same machine that has to be used for
other things. Waiting for an epoch boundary to get the GPU back, or killing
the process and losing the last hour, are both bad answers, so a run can be
paused and stopped at any point and picked up again exactly where it was.

Three ways in, all landing in the same place:

  Ctrl+C            asks for a stop; the current step finishes, a checkpoint
                    is written, the process exits 0. A second Ctrl+C gives up
                    on the checkpoint and exits immediately.
  a `pause` file    the loop stops stepping and idles until the file is
                    removed. The GPU is left alone; the process keeps its
                    place in the data.
  a `stop` file     the same as Ctrl+C, but from another shell or a script.

Files rather than signals because the machine that does the training runs
Windows, which has no SIGUSR1 and no way to send one process a custom signal
from another. A file works the same on both platforms and can be created by
anything, including a text editor.

    python -m training.control runs/net1 pause
    python -m training.control runs/net1 resume
    python -m training.control runs/net1 stop
    python -m training.control runs/net1 status
"""

from __future__ import annotations

import signal
import sys
from pathlib import Path

RUN = "run"
PAUSE = "pause"
STOP = "stop"


class Control:
    def __init__(self, run_dir: str | Path):
        self.dir = Path(run_dir) / "control"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.interrupted = False
        self._previous = {}

    # --- the training side ----------------------------------------------

    def install_signal_handlers(self) -> None:
        def handler(signum, frame):  # noqa: ARG001
            if self.interrupted:
                # The first Ctrl+C is a request; the second is an order. A
                # checkpoint write that is itself stuck should not make the
                # process unkillable.
                print("\ninterrupted twice, exiting without a checkpoint", file=sys.stderr)
                sys.exit(130)
            self.interrupted = True
            print("\nstopping after this step; press Ctrl+C again to give up on the checkpoint",
                  file=sys.stderr)

        for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            sig = getattr(signal, name, None)
            if sig is None:
                continue
            try:
                self._previous[sig] = signal.signal(sig, handler)
            except (OSError, ValueError):
                # Not every signal can be caught on every platform, and a
                # missing one only costs that one way of asking to stop.
                pass

    def poll(self) -> str:
        if self.interrupted or (self.dir / STOP).exists():
            return STOP
        if (self.dir / PAUSE).exists():
            return PAUSE
        return RUN

    def clear(self) -> None:
        """Remove the sentinels, so the next run does not stop immediately."""
        for name in (PAUSE, STOP):
            (self.dir / name).unlink(missing_ok=True)

    # --- the other shell ------------------------------------------------

    def request(self, what: str) -> None:
        (self.dir / what).write_text("", encoding="utf-8")

    def withdraw(self, what: str) -> None:
        (self.dir / what).unlink(missing_ok=True)

    def status(self) -> str:
        return self.poll()


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[2] not in ("pause", "resume", "stop", "status"):
        print(__doc__.strip().splitlines()[-5:][0], file=sys.stderr)
        print("usage: python -m training.control RUN_DIR pause|resume|stop|status",
              file=sys.stderr)
        return 1

    control = Control(argv[1])
    command = argv[2]
    if command == "pause":
        control.request(PAUSE)
        print(f"{argv[1]}: paused (resume to carry on)")
    elif command == "resume":
        control.withdraw(PAUSE)
        print(f"{argv[1]}: resumed")
    elif command == "stop":
        control.request(STOP)
        print(f"{argv[1]}: stopping after the current step")
    else:
        print(f"{argv[1]}: {control.status()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
