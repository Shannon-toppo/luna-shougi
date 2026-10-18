"""Tests for datagen_cluster.py, against a stand-in for luna-datagen.

  python -m unittest tools/test_datagen_cluster.py

The stand-in is a Python script with the same command line and the same
closing lines on stdout. Every game or line it handles is appended to a log,
so a test can tell whether any chunk ran twice.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datagen_cluster as cluster  # noqa: E402

SCRIPT = Path(__file__).resolve().parent / "datagen_cluster.py"

FAKE = textwrap.dedent('''\
    #!{python}
    import os, sys, time
    args = sys.argv[1:]
    opts = dict(zip(args[0::2], args[1::2]))
    delay = float(os.environ.get("FAKE_DELAY", "0"))
    stop = opts["--stop-file"]
    log = open(os.environ["FAKE_LOG"], "a")
    out = open(opts["--out"], "wb")
    sfen = open(opts["--sfen-out"], "w")
    def stopped():
        return os.path.exists(stop)
    if "--label" in opts:
        written = skipped = 0
        for line in open(opts["--label"]):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if stopped():
                print("stopped early; the positions that were reached are all in the output")
                sys.exit(0)
            time.sleep(delay)
            if line.startswith("bad"):
                skipped += 1
                continue
            log.write("line " + line + "\\n"); log.flush()
            out.write(b"L" * 84)
            sfen.write(line + "\\n")
            written += 1
        print(f"{{written}} positions written")
        if skipped:
            print(f"{{skipped}} lines produced nothing (unparsable, or not a full board)")
        sys.exit(0)
    seed, games = int(opts["--seed"]), int(opts["--games"])
    for game in range(games):
        if stopped():
            print(f"{{game}} games, {{game * 2}} positions written")
            print("stopped early; rerun with --append to carry on")
            sys.exit(0)
        time.sleep(delay)
        log.write(f"seed {{seed + game}}\\n"); log.flush()
        for ply in range(2):
            out.write(b"G" * 84)
            sfen.write(f"seed{{seed + game}} ply{{ply}}\\n")
    print(f"{{games}} games, {{games * 2}} positions written")
    print("black 1 - white 1 - draw 0")
''')


class ClusterTest(unittest.TestCase):
    def setUp(self):
        if cluster.platform_key() == "windows":
            self.skipTest("the stand-in for luna-datagen is a shebang script")
        self.tmp = Path(tempfile.mkdtemp(prefix="luna-cluster-"))
        self.root = self.tmp / "share" / "gen"
        binary = cluster.binary_path(self.root, cluster.platform_key())
        binary.parent.mkdir(parents=True)
        binary.write_text(FAKE.format(python=sys.executable))
        binary.chmod(0o755)
        (self.root / cluster.EVAL).write_bytes(b"net")
        self.games_log = self.tmp / "games.log"
        self.games_log.touch()
        os.environ["FAKE_LOG"] = str(self.games_log)
        os.environ["FAKE_DELAY"] = "0"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # helpers

    def init_selfplay(self, chunks=6, games=3):
        status = cluster.main(["init", str(self.root), "--mode", "selfplay", "--chunks", str(chunks),
                               "--games-per-chunk", str(games), "--seed-base", "100",
                               "--", "--depth", "1"])
        self.assertEqual(status, 0)

    def worker(self, name="a", **overrides):
        settings = dict(heartbeat=0.2, stale_after=900.0, idle_wait=0.1, pause_while=[])
        settings.update(overrides)
        return cluster.Worker(self.root, name, 1, self.tmp / f"local-{name}", **settings)

    def spawn(self, name):
        return subprocess.Popen(
            [sys.executable, str(SCRIPT), "work", str(self.root), "--name", name,
             "--concurrency", "1", "--local", str(self.tmp / f"local-{name}"),
             "--heartbeat", "0.2", "--idle-wait", "0.1"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def logged(self):
        return self.games_log.read_text().split("\n")[:-1]

    def wait_for(self, condition, seconds=20.0):
        end = time.time() + seconds
        while time.time() < end:
            if condition():
                return
            time.sleep(0.05)
        self.fail("timed out")

    # tests

    def test_two_workers_never_run_the_same_chunk(self):
        os.environ["FAKE_DELAY"] = "0.02"
        self.init_selfplay(chunks=12, games=3)
        workers = [self.spawn("a"), self.spawn("b")]
        for process in workers:
            output, _ = process.communicate(timeout=60)
            self.assertEqual(process.returncode, 0, output)
        seeds = self.logged()
        self.assertEqual(len(seeds), 36)
        self.assertEqual(sorted(seeds), sorted(f"seed {100 + n}" for n in range(36)))
        self.assertEqual(cluster.main(["merge", str(self.root), "--out", str(self.tmp / "all")]), 0)
        sfens = (self.tmp / "all.sfen").read_text().split("\n")[:-1]
        self.assertEqual(sfens[:2], ["seed100 ply0", "seed100 ply1"])
        self.assertEqual(sfens[-1], "seed135 ply1")
        self.assertEqual((self.tmp / "all.bin").stat().st_size, 72 * 84)

    def test_a_finished_chunk_is_not_taken_again(self):
        self.init_selfplay(chunks=2)
        self.assertEqual(self.worker().run(), 0)
        before = self.logged()
        # A listing of done/ a minute old looks like this. The claim, kept
        # and marked done, is what keeps the chunk from running again.
        shutil.rmtree(self.root / "done" / "0001")
        self.assertEqual(self.worker("b").run(), 0)
        self.assertEqual(self.logged(), before)

    def test_a_worker_that_was_taken_over_neither_publishes_nor_releases(self):
        self.init_selfplay()
        old = cluster.try_claim(self.root, 0, "old")
        claim = self.root / "claims" / "0000"
        os.utime(claim / "heartbeat")
        self.assertEqual(cluster.reap(self.root, 60, now=time.time() + 3600), [0])
        new = cluster.try_claim(self.root, 0, "new")
        self.assertIsNotNone(new)

        payload = self.tmp / "payload"
        payload.write_bytes(b"x" * 84)
        self.assertFalse(cluster.publish(self.root, 0, old, {"chunk.bin": payload}))
        self.assertFalse((self.root / "done" / "0000").exists())
        self.assertFalse(cluster.release(self.root, 0, old))
        self.assertTrue(claim.exists())
        self.assertTrue(cluster.owns(self.root, 0, new))

    def test_only_the_first_publish_of_a_chunk_is_kept(self):
        self.init_selfplay()
        first = self.tmp / "first"
        second = self.tmp / "second"
        first.write_bytes(b"1" * 84)
        second.write_bytes(b"2" * 84)
        token = cluster.try_claim(self.root, 0, "a")
        self.assertTrue(cluster.publish(self.root, 0, token, {"chunk.bin": first}))
        # The same chunk run a second time, by a worker holding the claim.
        (self.root / "claims" / "0000" / "done").unlink()
        cluster.write_json_atomic(self.root / "claims" / "0000" / "owner", {"name": "b", "token": "t2"})
        self.assertFalse(cluster.publish(self.root, 0, "t2", {"chunk.bin": second}))
        self.assertEqual((self.root / "done" / "0000" / "chunk.bin").read_bytes(), b"1" * 84)
        self.assertEqual([p.name for p in (self.root / "done").iterdir()], ["0000"])

    def test_merge_stops_on_a_missing_chunk(self):
        self.init_selfplay(chunks=3)
        self.assertEqual(self.worker().run(), 0)
        shutil.rmtree(self.root / "done" / "0001")
        out = self.tmp / "merged"
        self.assertEqual(cluster.main(["merge", str(self.root), "--out", str(out)]), 1)
        self.assertFalse(Path(str(out) + ".bin").exists())
        self.assertEqual(cluster.main(["merge", str(self.root), "--out", str(out), "--allow-gaps"]), 0)
        self.assertEqual(Path(str(out) + ".bin").stat().st_size, 2 * 6 * 84)

    def test_a_worker_with_a_different_network_takes_nothing(self):
        self.init_selfplay()
        (self.root / cluster.EVAL).write_bytes(b"another net")
        self.assertEqual(self.worker().run(), 2)
        self.assertEqual(list((self.root / "claims").iterdir()), [])
        self.assertEqual(self.logged(), [])

    def test_a_worker_with_a_different_binary_takes_nothing(self):
        self.init_selfplay()
        binary = cluster.binary_path(self.root, cluster.platform_key())
        binary.write_text(binary.read_text() + "# rebuilt\n")
        self.assertEqual(self.worker().run(), 2)
        self.assertEqual(list((self.root / "claims").iterdir()), [])

    def test_reap_frees_only_silent_unfinished_claims(self):
        self.init_selfplay()
        for number in range(3):
            cluster.try_claim(self.root, number, "a")
        (self.root / "claims" / "0002" / "done").write_text("x\n")
        cluster.write_heartbeat(self.root / "claims" / "0001", "t", "a", "")
        old = time.time() - 3600
        beat = self.root / "claims" / "0000" / "heartbeat"
        cluster.write_json_atomic(beat, {"time": old, "worker": "a", "token": "t", "progress": ""})
        cluster.write_json_atomic(self.root / "claims" / "0002" / "heartbeat",
                                  {"time": old, "worker": "a", "token": "t", "progress": ""})
        self.assertEqual(cluster.reap(self.root, 900), [0])
        self.assertTrue((self.root / "claims" / "0001").exists())
        self.assertTrue((self.root / "claims" / "0002").exists())

    def test_label_chunks_join_back_into_source_order(self):
        lines = [f"pos{n}" for n in range(23)]
        lines[5] = "bad line"
        lines[9] = ""
        (self.root / cluster.SOURCE).write_text("\n".join(lines) + "\n")
        status = cluster.main(["init", str(self.root), "--mode", "label", "--lines-per-chunk", "5",
                               "--", "--depth", "1"])
        self.assertEqual(status, 0)
        self.assertEqual(cluster.load_manifest(self.root)["chunks"], 5)
        self.assertEqual(self.worker().run(), 0)
        out = self.tmp / "labelled"
        self.assertEqual(cluster.main(["merge", str(self.root), "--out", str(out)]), 0)
        expected = [line for line in lines if line and not line.startswith("bad")]
        self.assertEqual(Path(str(out) + ".sfen").read_text().split("\n")[:-1], expected)
        info = cluster.read_json(self.root / "done" / "0001" / "chunk.json")
        self.assertEqual(info["skipped"], 1)

    def test_pause_now_gives_the_chunk_back_and_resume_finishes(self):
        os.environ["FAKE_DELAY"] = "0.1"
        self.init_selfplay(chunks=2, games=30)
        process = self.spawn("a")
        self.wait_for(lambda: (self.root / "claims" / "0000").exists())
        self.assertEqual(cluster.main(["pause", str(self.root), "a", "--now",
                                       "--local", str(self.tmp / "local-a")]), 0)
        self.wait_for(lambda: not (self.root / "claims" / "0000").exists())
        self.assertFalse((self.root / "done" / "0000").exists())
        self.assertEqual(cluster.main(["resume", str(self.root), "a"]), 0)
        os.environ["FAKE_DELAY"] = "0"
        output, _ = process.communicate(timeout=60)
        self.assertEqual(process.returncode, 0, output)
        self.assertIn("paused", output)
        self.assertTrue((self.root / "done" / "0001").is_dir())

    def test_a_restarted_worker_frees_its_own_claims_at_once(self):
        self.init_selfplay(chunks=2)
        cluster.try_claim(self.root, 1, "a")
        self.assertEqual(self.worker("a").run(), 0)
        self.assertTrue((self.root / "done" / "0001").is_dir())

    def test_init_refuses_options_the_worker_sets(self):
        status = cluster.main(["init", str(self.root), "--mode", "selfplay", "--chunks", "1",
                               "--games-per-chunk", "1", "--seed-base", "0",
                               "--", "--eval-file", "x.nnue"])
        self.assertEqual(status, 1)
        self.assertFalse((self.root / cluster.MANIFEST).exists())


if __name__ == "__main__":
    unittest.main()
