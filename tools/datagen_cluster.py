"""Spread luna-datagen over several machines through one shared directory.

  datagen_cluster.py init DIR --mode selfplay --chunks 1050 --games-per-chunk 125
                     --seed-base 100000 -- --depth 6 --nodes 2000000
  datagen_cluster.py init DIR --mode label --lines-per-chunk 10000 -- --depth 6
  datagen_cluster.py work DIR --name i3-8100T --concurrency 4
  datagen_cluster.py status DIR
  datagen_cluster.py pause DIR NAME [--now]      NAME may be "all"
  datagen_cluster.py resume DIR NAME
  datagen_cluster.py reap DIR
  datagen_cluster.py merge DIR --out data/gen11

DIR is one generation's directory on the share. Before init it holds eval.nnue,
bin/linux/luna-datagen and/or bin/windows/luna-datagen.exe, and for --mode
label also source.sfen. Everything after "--" is passed to luna-datagen as is;
the worker adds --eval-file, --out, --sfen-out, --stop-file, --concurrency and
the chunk's own range itself.

A chunk is taken with mkdir, which the file server does atomically over NFS
and SMB alike, and only a chunk that ran to the end is published. Whatever is
stopped half way is thrown away and goes back to the pool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from array import array
from pathlib import Path

RECORD_BYTES = 84

MANIFEST = "manifest.json"
EVAL = "eval.nnue"
SOURCE = "source.sfen"

# The worker sets these itself. A manifest carrying one would either be
# overridden without a word or fight the worker over the chunk's range.
RESERVED = {
    "--out", "--append", "--sfen-out", "--stop-file", "--eval-file",
    "--concurrency", "--seed", "--games", "--label",
}

# What two workers must agree on. "chunks" is left out so that a run can be
# extended by editing the manifest while workers are up.
FIXED_KEYS = (
    "mode", "args", "eval_sha256", "datagen_sha256", "games_per_chunk",
    "seed_base", "lines", "lines_per_chunk", "source_sha256",
)


# ---------------------------------------------------------------- small helpers


def platform_key() -> str:
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def binary_path(root: Path, key: str) -> Path:
    name = "luna-datagen.exe" if key == "windows" else "luna-datagen"
    return root / "bin" / key / name


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def chunk_name(number: int) -> str:
    return f"{number:04d}"


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_json_atomic(path: Path, data) -> None:
    # Written beside the target and renamed over it, so that a reader on
    # another machine never sees half a file.
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def count_lines(path: Path) -> int:
    count = 0
    last = b"\n"
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            count += block.count(b"\n")
            last = block[-1:]
    return count + (0 if last == b"\n" else 1)


def default_local(name: str) -> Path:
    return Path(tempfile.gettempdir()) / f"luna-worker-{name}"


def log(message: str) -> None:
    print(time.strftime("%H:%M:%S ") + message, flush=True)


# ------------------------------------------------------------------- claims


def claims_dir(root: Path) -> Path:
    return root / "claims"


def done_dir(root: Path) -> Path:
    return root / "done"


def write_heartbeat(claim: Path, token: str, name: str, progress: str) -> None:
    write_json_atomic(
        claim / "heartbeat",
        {"time": time.time(), "worker": name, "token": token, "progress": progress},
    )


def try_claim(root: Path, number: int, name: str):
    """Take a chunk. Returns the claim's token, or None if someone has it."""
    claim = claims_dir(root) / chunk_name(number)
    try:
        claim.mkdir()
    except FileExistsError:
        return None
    token = secrets.token_hex(8)
    write_json_atomic(claim / "owner", {"name": name, "token": token})
    write_heartbeat(claim, token, name, "claimed")
    return token


def owns(root: Path, number: int, token: str) -> bool:
    owner = read_json(claims_dir(root) / chunk_name(number) / "owner")
    return bool(owner) and owner.get("token") == token


def release(root: Path, number: int, token: str) -> bool:
    """Give a chunk back to the pool, but only if the claim is still ours.

    A worker that was declared dead and came back would otherwise delete the
    claim of whoever took the chunk over, and a third worker would take it.
    """
    if not owns(root, number, token):
        return False
    shutil.rmtree(claims_dir(root) / chunk_name(number), ignore_errors=True)
    return True


def publish(root: Path, number: int, token: str, files: dict) -> bool:
    """Move a finished chunk into done/. False if it is not ours to publish.

    The files go into one directory that is renamed into place, so they appear
    together. Renaming onto a directory that is already there fails, so of two
    copies of one chunk only the first is kept.
    """
    if not owns(root, number, token):
        return False
    name = chunk_name(number)
    staging = done_dir(root) / f".tmp-{name}-{token}"
    staging.mkdir(parents=True, exist_ok=True)
    try:
        for target, source in files.items():
            shutil.copyfile(source, staging / target)
            if (staging / target).stat().st_size != Path(source).stat().st_size:
                raise OSError(f"copy of {target} came out the wrong size")
        if not owns(root, number, token):
            return False
        try:
            os.rename(staging, done_dir(root) / name)
        except OSError:
            if (done_dir(root) / name).exists():
                return False
            raise
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    # The claim stays, marked done. Deleting it would let a worker whose
    # listing of done/ is a minute old take the chunk again.
    (claims_dir(root) / name / "done").write_text(token + "\n", encoding="utf-8")
    return True


def is_finished(root: Path, number: int) -> bool:
    name = chunk_name(number)
    return (claims_dir(root) / name / "done").exists() or (done_dir(root) / name).exists()


def claim_age(claim: Path, now: float) -> float:
    beat = read_json(claim / "heartbeat")
    if beat and isinstance(beat.get("time"), (int, float)):
        return now - beat["time"]
    try:
        return now - claim.stat().st_mtime
    except OSError:
        return 0.0


def reap(root: Path, stale_after: float, now: float | None = None, only_name: str | None = None):
    """Free the chunks whose worker stopped writing heartbeats.

    The claim is renamed aside rather than deleted, because a rename either
    happens or does not: of two workers reaping at once, one wins. With
    only_name, the claims of that worker are freed whatever their age, which
    is what a worker does at start-up with the ones its previous run left.
    """
    now = time.time() if now is None else now
    freed = []
    try:
        entries = list(os.scandir(claims_dir(root)))
    except FileNotFoundError:
        return freed
    for entry in entries:
        if not entry.name.isdigit():
            continue
        number = int(entry.name)
        claim = Path(entry.path)
        if is_finished(root, number):
            continue
        if only_name is not None:
            owner = read_json(claim / "owner")
            if not owner or owner.get("name") != only_name:
                continue
        elif claim_age(claim, now) < stale_after:
            continue
        aside = claims_dir(root) / f"{entry.name}.stale.{int(now)}.{secrets.token_hex(3)}"
        try:
            os.rename(claim, aside)
        except OSError:
            continue
        freed.append(number)
    return sorted(freed)


# ----------------------------------------------------------------- manifest


def load_manifest(root: Path) -> dict:
    manifest = read_json(root / MANIFEST)
    if not isinstance(manifest, dict):
        raise SystemExit(f"{root / MANIFEST}: missing or not JSON")
    return manifest


def chunk_range(manifest: dict, number: int):
    if manifest["mode"] == "selfplay":
        size = manifest["games_per_chunk"]
        return manifest["seed_base"] + number * size, size
    size = manifest["lines_per_chunk"]
    start = number * size
    return start, min(manifest["lines"], start + size)


class LineIndex:
    """Byte offsets of every line in a file, so a slice is one seek away."""

    def __init__(self, path: Path):
        self.path = path
        self.offsets = array("Q", [0])
        with open(path, "rb") as handle:
            for line in handle:
                self.offsets.append(self.offsets[-1] + len(line))
        if len(self.offsets) > 1 and self.offsets[-1] == self.offsets[-2]:
            self.offsets.pop()

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def write_slice(self, start: int, end: int, target: Path) -> None:
        with open(self.path, "rb") as source, open(target, "wb") as out:
            source.seek(self.offsets[start])
            out.write(source.read(self.offsets[end] - self.offsets[start]))


def chunk_command(manifest, number, binary, eval_path, work: Path, concurrency, index=None):
    command = [
        str(binary),
        "--out", str(work / "chunk.bin"),
        "--sfen-out", str(work / "chunk.sfen"),
        "--eval-file", str(eval_path),
        "--concurrency", str(concurrency),
        "--stop-file", str(work.parent / "stop"),
    ]
    first, second = chunk_range(manifest, number)
    if manifest["mode"] == "selfplay":
        command += ["--seed", str(first), "--games", str(second)]
    else:
        index.write_slice(first, second, work / "input.sfen")
        command += ["--label", str(work / "input.sfen")]
    return command + list(manifest["args"])


def parse_report(text: str) -> dict:
    report = {"stopped": "stopped early" in text}
    for line in text.splitlines():
        words = line.split()
        if line.endswith("positions written") and words and words[0].isdigit():
            if len(words) >= 5 and words[1] == "games,":
                report["games"] = int(words[0])
                report["positions"] = int(words[2])
            else:
                report["positions"] = int(words[0])
        elif line.endswith("produced nothing (unparsable, or not a full board)"):
            report["skipped"] = int(words[0])
    return report


# ------------------------------------------------------------------- worker


class Interrupted(Exception):
    pass


class Worker:
    def __init__(self, root: Path, name: str, concurrency: int, local: Path,
                 heartbeat: float, stale_after: float, idle_wait: float,
                 pause_while: list):
        self.root = root
        self.name = name
        self.concurrency = concurrency
        self.local = local
        self.heartbeat = heartbeat
        self.stale_after = stale_after
        self.idle_wait = idle_wait
        self.check_every = min(10.0, heartbeat)
        self.pause_while = pause_while
        self.signalled = False
        self.failures = 0
        self.manifest = None
        self.binary = None
        self.eval_path = None
        self.index = None

    # Set-up: refuse to run with anything that differs from the manifest.

    def prepare(self) -> int:
        manifest = load_manifest(self.root)
        key = platform_key()
        expected = manifest.get("datagen_sha256", {}).get(key)
        if not expected:
            log(f"the manifest has no luna-datagen for {key}; taking nothing")
            return 2
        self.local.mkdir(parents=True, exist_ok=True)
        try:
            self.binary = self.copy_checked(binary_path(self.root, key), expected, "luna-datagen")
            self.eval_path = self.copy_checked(self.root / EVAL, manifest["eval_sha256"], EVAL)
            if manifest["mode"] == "label":
                source = self.copy_checked(self.root / SOURCE, manifest["source_sha256"], SOURCE)
                self.index = LineIndex(source)
                if len(self.index) != manifest["lines"]:
                    log(f"{SOURCE} has {len(self.index)} lines, the manifest says {manifest['lines']}")
                    return 2
        except ValueError as error:
            log(f"{error}; taking nothing")
            return 2
        if key != "windows":
            self.binary.chmod(0o755)
        self.manifest = manifest
        return 0

    def copy_checked(self, source: Path, expected: str, label: str) -> Path:
        target = self.local / source.name
        if target.exists() and sha256_of(target) == expected:
            return target
        if not source.exists():
            raise ValueError(f"{source} is missing")
        if sha256_of(source) != expected:
            raise ValueError(f"{label} on the share does not match the manifest's sha256")
        shutil.copyfile(source, target)
        if sha256_of(target) != expected:
            raise ValueError(f"local copy of {label} does not match the manifest's sha256")
        return target

    def manifest_still_matches(self) -> bool:
        current = read_json(self.root / MANIFEST)
        if not isinstance(current, dict):
            # Most likely the share is away. Wait for it like any other error.
            raise OSError(f"cannot read {self.root / MANIFEST}")
        if any(current.get(key) != self.manifest.get(key) for key in FIXED_KEYS):
            return False
        self.manifest["chunks"] = current["chunks"]
        return True

    # Markers that stop the worker.

    def marker(self, kind: str) -> bool:
        return any((self.root / kind / who).exists() for who in (self.name, "all"))

    def busy_process(self):
        for process in self.pause_while:
            if process_running(process):
                return process
        return None

    def paused(self) -> bool:
        return self.marker("pause") or self.marker("abort") or self.busy_process() is not None

    def sleep(self, seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            if self.signalled:
                raise Interrupted
            time.sleep(min(0.5, max(0.0, end - time.time())))

    # The loop.

    def run(self) -> int:
        status = self.prepare()
        if status:
            return status
        mine = reap(self.root, self.stale_after, only_name=self.name)
        if mine:
            log(f"freed {len(mine)} chunk(s) an earlier run of {self.name} left: {mine}")
        was_paused = False
        try:
            while True:
                if self.signalled:
                    raise Interrupted
                try:
                    if not self.manifest_still_matches():
                        log("the manifest changed under this worker; stopping")
                        return 2
                    if self.paused():
                        if not was_paused:
                            log("paused")
                        was_paused = True
                        self.sleep(self.idle_wait)
                        continue
                    if was_paused:
                        log("resumed")
                    was_paused = False
                    taken = self.claim_next()
                    if taken is None:
                        if all(is_finished(self.root, n) for n in range(self.manifest["chunks"])):
                            log("every chunk is done")
                            return 0
                        self.sleep(self.idle_wait)
                        continue
                except OSError as error:
                    log(f"shared directory: {error}; waiting")
                    self.sleep(self.idle_wait)
                    continue
                self.run_chunk(*taken)
                if self.failures >= 3:
                    log("three chunks in a row failed; stopping")
                    return 1
        except Interrupted:
            log("interrupted")
            return 130

    def claim_next(self):
        try:
            taken = {entry.name for entry in os.scandir(claims_dir(self.root))}
        except FileNotFoundError:
            claims_dir(self.root).mkdir(parents=True, exist_ok=True)
            taken = set()
        for number in range(self.manifest["chunks"]):
            if chunk_name(number) in taken:
                continue
            token = try_claim(self.root, number, self.name)
            if token:
                return number, token
        for number in reap(self.root, self.stale_after):
            log(f"chunk {chunk_name(number)} had no heartbeat for {self.stale_after:.0f}s; taking it back")
            token = try_claim(self.root, number, self.name)
            if token:
                return number, token
        return None

    def run_chunk(self, number: int, token: str) -> None:
        name = chunk_name(number)
        work = self.local / "chunk"
        stop = self.local / "stop"
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        if stop.exists():
            stop.unlink()

        command = chunk_command(self.manifest, number, self.binary, self.eval_path, work,
                                self.concurrency, self.index)
        log(f"chunk {name}: started")
        started = time.time()
        output = open(work / "log.txt", "wb")
        flags = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0) if platform_key() == "windows" else 0
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, creationflags=flags)
        stopping = False
        last_beat = last_check = time.time()
        claim = claims_dir(self.root) / name

        def halt():
            # Killed rather than asked: luna-datagen looks at its stop file
            # only between games, so a clean stop waits for the longest game
            # in flight, a minute or more at depth 6. The output is thrown
            # away either way.
            process.kill()
            process.wait()

        try:
            while process.poll() is None:
                time.sleep(0.5)
                now = time.time()
                if self.signalled or stop.exists():
                    stopping = True
                    halt()
                    break
                if now - last_check >= self.check_every:
                    last_check = now
                    try:
                        busy = self.busy_process()
                        if self.marker("abort") or busy:
                            log(f"chunk {name}: stopping now ({busy or 'abort'})")
                            stopping = True
                            halt()
                            break
                    except OSError:
                        pass
                if now - last_beat >= self.heartbeat:
                    last_beat = now
                    try:
                        write_heartbeat(claim, token, self.name, tail(work / "log.txt"))
                    except OSError:
                        pass  # the share is down; the chunk carries on locally
        except KeyboardInterrupt:
            self.signalled = True
            stopping = True
            halt()
        finally:
            output.close()

        seconds = time.time() - started
        problem = None if not stopping else "stopped"
        if problem is None and process.returncode != 0:
            problem = f"luna-datagen exited with {process.returncode}"
        info = None
        if problem is None:
            problem, info = self.check_output(number, work, seconds)

        if problem is None:
            files = {"chunk.bin": work / "chunk.bin", "chunk.sfen": work / "chunk.sfen",
                     "chunk.json": work / "chunk.json"}
            write_json_atomic(work / "chunk.json", info)
            try:
                if publish(self.root, number, token, files):
                    log(f"chunk {name}: published, {info['positions']} positions in {seconds:.0f}s")
                else:
                    log(f"chunk {name}: taken over by another worker; output thrown away")
                self.failures = 0
            except OSError as error:
                log(f"chunk {name}: could not publish ({error}); output thrown away")
                self.give_back(number, token)
        else:
            log(f"chunk {name}: {problem}; output thrown away")
            if problem != "stopped":
                self.failures += 1
            self.give_back(number, token)
        shutil.rmtree(work, ignore_errors=True)
        if self.signalled:
            raise Interrupted

    def give_back(self, number: int, token: str) -> None:
        try:
            release(self.root, number, token)
        except OSError:
            pass  # left to go stale and be reaped

    def check_output(self, number: int, work: Path, seconds: float):
        report = parse_report((work / "log.txt").read_text(encoding="utf-8", errors="replace"))
        if report["stopped"]:
            return "stopped early", None
        size = (work / "chunk.bin").stat().st_size
        if size % RECORD_BYTES:
            return f"chunk.bin is {size} bytes, not a whole number of records", None
        positions = size // RECORD_BYTES
        if report.get("positions") != positions:
            return f"luna-datagen reported {report.get('positions')} positions, the file holds {positions}", None
        lines = count_lines(work / "chunk.sfen")
        if lines != positions:
            return f"chunk.sfen has {lines} lines for {positions} positions", None
        info = {"chunk": number, "worker": self.name, "platform": platform_key(),
                "positions": positions, "seconds": round(seconds, 1)}
        if self.manifest["mode"] == "selfplay":
            if report.get("games") != self.manifest["games_per_chunk"]:
                return f"{report.get('games')} games played, not {self.manifest['games_per_chunk']}", None
            info["games"] = report["games"]
        else:
            first, end = chunk_range(self.manifest, number)
            info["lines"] = end - first
            info["skipped"] = report.get("skipped", 0)
        return None, info


def tail(path: Path) -> str:
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - 400))
            lines = [line for line in handle.read().decode("utf-8", "replace").splitlines() if line.strip()]
        return lines[-1].strip() if lines else ""
    except OSError:
        return ""


def process_running(name: str) -> bool:
    if platform_key() == "windows":
        result = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {name}", "/NH"],
                                capture_output=True, text=True)
        return name.lower() in result.stdout.lower()
    return subprocess.run(["pgrep", "-x", name], capture_output=True).returncode == 0


# ----------------------------------------------------------------- commands


def command_init(options, extra) -> int:
    root = Path(options.dir)
    if (root / MANIFEST).exists() and not options.calibrate:
        print(f"{root / MANIFEST} already exists. Edit \"chunks\" in it to extend a run.")
        return 1
    bad = sorted(set(extra) & RESERVED)
    if bad:
        print(f"the worker sets {', '.join(bad)} itself; leave it out of the arguments")
        return 1
    if not (root / EVAL).exists():
        print(f"{root / EVAL} is missing")
        return 1
    binaries = {key: sha256_of(binary_path(root, key))
                for key in ("linux", "windows", "macos") if binary_path(root, key).exists()}
    if not binaries:
        print(f"no luna-datagen under {root / 'bin'}")
        return 1

    manifest = {"mode": options.mode, "commit": options.commit or git_commit()}
    if options.mode == "selfplay":
        if not (options.chunks and options.games_per_chunk and options.seed_base is not None):
            print("--mode selfplay needs --chunks, --games-per-chunk and --seed-base")
            return 1
        manifest.update(chunks=options.chunks, games_per_chunk=options.games_per_chunk,
                        seed_base=options.seed_base)
    else:
        if not options.lines_per_chunk or not (root / SOURCE).exists():
            print(f"--mode label needs --lines-per-chunk and {root / SOURCE}")
            return 1
        lines = len(LineIndex(root / SOURCE))
        manifest.update(chunks=math.ceil(lines / options.lines_per_chunk), lines=lines,
                        lines_per_chunk=options.lines_per_chunk,
                        source_sha256=sha256_of(root / SOURCE))
    manifest.update(args=extra, eval_sha256=sha256_of(root / EVAL), datagen_sha256=binaries)

    if options.calibrate:
        return calibrate(root, manifest, options.concurrency)

    claims_dir(root).mkdir(exist_ok=True)
    done_dir(root).mkdir(exist_ok=True)
    (root / "pause").mkdir(exist_ok=True)
    (root / "abort").mkdir(exist_ok=True)
    write_json_atomic(root / MANIFEST, manifest)
    print(f"wrote {root / MANIFEST}: {manifest['chunks']} chunks, luna-datagen for {', '.join(binaries)}")
    if options.mode == "selfplay":
        last = manifest["seed_base"] + manifest["chunks"] * manifest["games_per_chunk"] - 1
        print(f"games use seeds {manifest['seed_base']} to {last}. Check that no earlier")
        print("generation and no holdout used any of them.")
    return 0


def calibrate(root: Path, manifest: dict, concurrency: int) -> int:
    key = platform_key()
    if key not in manifest["datagen_sha256"]:
        print(f"no luna-datagen for {key} under {root / 'bin'}")
        return 1
    with tempfile.TemporaryDirectory() as scratch:
        work = Path(scratch) / "chunk"
        work.mkdir()
        index = LineIndex(root / SOURCE) if manifest["mode"] == "label" else None
        command = chunk_command(manifest, 0, binary_path(root, key), root / EVAL, work,
                                concurrency, index)
        print("running chunk 0 here, nothing is written to the share")
        started = time.time()
        status = subprocess.run(command).returncode
        seconds = time.time() - started
    if status:
        return status
    unit, size = (("games", manifest["games_per_chunk"]) if manifest["mode"] == "selfplay"
                  else ("lines", manifest["lines_per_chunk"]))
    print(f"\n{size} {unit} took {seconds:.0f}s at concurrency {concurrency}")
    print(f"for 10 minutes a chunk on this machine: about {round(size * 600 / seconds)} {unit}")
    return 0


def git_commit() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True)
        return result.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def command_work(options) -> int:
    worker = Worker(Path(options.dir), options.name, options.concurrency,
                    Path(options.local) if options.local else default_local(options.name),
                    options.heartbeat, options.stale_after, options.idle_wait,
                    options.pause_while or [])

    def on_term(signum, frame):
        worker.signalled = True

    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, on_term)
    try:
        return worker.run()
    except KeyboardInterrupt:
        return 130


def command_status(options) -> int:
    root = Path(options.dir)
    manifest = load_manifest(root)
    now = time.time()
    chunks = manifest["chunks"]
    done = {int(entry.name) for entry in os.scandir(done_dir(root))
            if entry.name.isdigit() and int(entry.name) < chunks}
    running, stale = [], []
    for entry in os.scandir(claims_dir(root)):
        if not entry.name.isdigit() or int(entry.name) in done:
            continue
        claim = Path(entry.path)
        if (claim / "done").exists():
            continue
        beat = read_json(claim / "heartbeat") or {}
        age = claim_age(claim, now)
        row = (entry.name, beat.get("worker", "?"), age, beat.get("progress", ""))
        (stale if age >= options.stale_after else running).append(row)

    print(f"{root.name}: {manifest['mode']}, {chunks} chunks")
    print(f"done {len(done)}, running {len(running)}, stale {len(stale)}, "
          f"free {chunks - len(done) - len(running) - len(stale)}")
    for kind in ("pause", "abort"):
        markers = sorted(path.name for path in (root / kind).glob("*")) if (root / kind).exists() else []
        if markers:
            print(f"{kind}: {', '.join(markers)}")
    for title, rows in (("running", running), ("stale", stale)):
        if rows:
            print(f"\n{title}:")
            for chunk, who, age, progress in sorted(rows):
                print(f"  {chunk}  {who:<12} {age:6.0f}s  {progress}")
    per_worker = {}
    for number in done:
        info = read_json(done_dir(root) / chunk_name(number) / "chunk.json") or {}
        per_worker[info.get("worker", "?")] = per_worker.get(info.get("worker", "?"), 0) + 1
    if per_worker:
        print("\ndone by:")
        for who, count in sorted(per_worker.items(), key=lambda item: -item[1]):
            print(f"  {who:<12} {count}")
    return 0


def command_pause(options) -> int:
    root = Path(options.dir)
    (root / "pause").mkdir(exist_ok=True)
    (root / "pause" / options.name).touch()
    if options.now:
        (root / "abort").mkdir(exist_ok=True)
        (root / "abort" / options.name).touch()
        # On the worker's own machine the stop file is right here, and going
        # through it does not wait for the share's caches.
        local = Path(options.local) if options.local else default_local(options.name)
        if options.name != "all" and local.is_dir():
            (local / "stop").touch()
    print(f"{options.name}: {'stopping now' if options.now else 'stopping after the current chunk'}")
    return 0


def command_resume(options) -> int:
    root = Path(options.dir)
    for kind in ("pause", "abort"):
        marker = root / kind / options.name
        if marker.exists():
            marker.unlink()
    print(f"{options.name}: resumed")
    return 0


def command_reap(options) -> int:
    freed = reap(Path(options.dir), options.stale_after)
    print(f"freed {len(freed)} chunk(s){': ' + ' '.join(map(chunk_name, freed)) if freed else ''}")
    return 0


def command_merge(options) -> int:
    root = Path(options.dir)
    manifest = load_manifest(root)
    chunks = manifest["chunks"]
    present = [n for n in range(chunks) if (done_dir(root) / chunk_name(n)).is_dir()]
    missing = sorted(set(range(chunks)) - set(present))
    if missing and not options.allow_gaps:
        print(f"{len(missing)} of {chunks} chunks are not done: {describe(missing)}")
        print("nothing written. --allow-gaps merges what there is.")
        return 1

    infos = []
    for number in present:
        folder = done_dir(root) / chunk_name(number)
        info = read_json(folder / "chunk.json") or {}
        size = (folder / "chunk.bin").stat().st_size
        if info.get("positions") is None or size != info["positions"] * RECORD_BYTES:
            print(f"chunk {chunk_name(number)}: chunk.bin does not match chunk.json; nothing written")
            return 1
        infos.append((number, info))

    out = Path(options.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    targets = {suffix: out.with_name(out.name + suffix) for suffix in (".bin", ".sfen")}
    staging = {suffix: path.with_name(path.name + ".partial") for suffix, path in targets.items()}
    lines = 0
    with open(staging[".bin"], "wb") as bin_out, open(staging[".sfen"], "wb") as sfen_out:
        for number, info in infos:
            folder = done_dir(root) / chunk_name(number)
            with open(folder / "chunk.bin", "rb") as source:
                shutil.copyfileobj(source, bin_out, 1 << 20)
            with open(folder / "chunk.sfen", "rb") as source:
                for block in iter(lambda: source.read(1 << 20), b""):
                    lines += block.count(b"\n")
                    sfen_out.write(block)
    positions = sum(info["positions"] for _, info in infos)
    if lines != positions:
        print(f"the .sfen files hold {lines} lines for {positions} positions; nothing written")
        for path in staging.values():
            path.unlink()
        return 1
    for suffix in staging:
        os.replace(staging[suffix], targets[suffix])

    print(f"{targets['.bin']}: {positions} positions from {len(infos)} chunks")
    if missing:
        print(f"left out {len(missing)} chunks: {describe(missing)}")
    if manifest["mode"] == "label":
        skipped = [(number, info.get("skipped", 0)) for number, info in infos if info.get("skipped")]
        total_lines = sum(info.get("lines", 0) for _, info in infos)
        total_skipped = sum(count for _, count in skipped)
        print(f"{total_lines} input lines: {positions} labelled, {total_skipped} unparsable, "
              f"{total_lines - positions - total_skipped} blank or comments")
        for number, count in skipped:
            print(f"  chunk {chunk_name(number)}: {count} unparsable")
    return 0


def describe(numbers) -> str:
    shown = " ".join(chunk_name(n) for n in numbers[:20])
    return shown + (f" ... ({len(numbers)} in all)" if len(numbers) > 20 else "")


# --------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="datagen_cluster.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    def with_dir(name, help_text):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("dir", help="the generation's directory on the share")
        return sub

    init = with_dir("init", "write manifest.json")
    init.add_argument("--mode", choices=("selfplay", "label"), required=True)
    init.add_argument("--chunks", type=int, help="selfplay: number of chunks")
    init.add_argument("--games-per-chunk", type=int)
    init.add_argument("--seed-base", type=int, help="seed of the first game of chunk 0")
    init.add_argument("--lines-per-chunk", type=int, help="label: lines of source.sfen per chunk")
    init.add_argument("--commit", help="commit the binaries were built from (default: git HEAD)")
    init.add_argument("--calibrate", action="store_true",
                      help="run chunk 0 here and time it instead of writing the manifest")
    init.add_argument("--concurrency", type=int, default=os.cpu_count() or 1,
                      help="for --calibrate (default: every CPU)")

    work = with_dir("work", "take chunks until every one is done")
    work.add_argument("--name", required=True, help="this worker's name, used by pause and resume")
    work.add_argument("--concurrency", type=int, required=True)
    work.add_argument("--local", help="local working directory (default: under the temp directory)")
    work.add_argument("--pause-while", action="append", metavar="PROCESS",
                      help="stop at once while this process runs, carry on after; repeatable")
    work.add_argument("--heartbeat", type=float, default=60.0, help="seconds between heartbeats")
    work.add_argument("--stale-after", type=float, default=900.0,
                      help="seconds without a heartbeat before a claim is taken back")
    work.add_argument("--idle-wait", type=float, default=30.0,
                      help="seconds between looks while paused or waiting")

    status = with_dir("status", "show progress")
    status.add_argument("--stale-after", type=float, default=900.0)

    pause = with_dir("pause", "stop a worker after its chunk, or at once with --now")
    pause.add_argument("name", help="worker name, or all")
    pause.add_argument("--now", action="store_true", help="throw the running chunk away and stop")
    pause.add_argument("--local", help="the worker's --local, if it was given one")

    resume = with_dir("resume", "let a paused worker carry on")
    resume.add_argument("name", help="worker name, or all")

    reap_parser = with_dir("reap", "take back chunks whose worker stopped beating")
    reap_parser.add_argument("--stale-after", type=float, default=900.0)

    merge = with_dir("merge", "join the finished chunks in order")
    merge.add_argument("--out", required=True, help="writes OUT.bin and OUT.sfen")
    merge.add_argument("--allow-gaps", action="store_true", help="merge even with chunks missing")
    return parser


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if "--" in argv:
        split = argv.index("--")
        argv, extra = argv[:split], argv[split + 1:]
    options = build_parser().parse_args(argv)
    if extra and options.command != "init":
        print("arguments after -- are only for init")
        return 1
    handlers = {
        "init": lambda: command_init(options, extra),
        "work": lambda: command_work(options),
        "status": lambda: command_status(options),
        "pause": lambda: command_pause(options),
        "resume": lambda: command_resume(options),
        "reap": lambda: command_reap(options),
        "merge": lambda: command_merge(options),
    }
    return handlers[options.command]()


if __name__ == "__main__":
    raise SystemExit(main())
