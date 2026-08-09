"""Fail on raw tabs and carriage returns inside tracked source files.

This exists because of a bug that got all the way to CI: a patch wrote a real
tab and a real carriage return into a string literal instead of the two-
character escapes `\\t` and `\\r`. MSVC accepted it — a tab is legal inside a
string literal, and it quietly dropped the stray carriage return, so the
Windows build compiled and passed its tests while computing something slightly
different from what the source appeared to say. clang on macOS treated the
carriage return as a line terminator and failed with an unterminated string.

That is the worst shape a bug can have: green on the machine the work is done
on, red somewhere else, and the difference invisible in a diff. Two lines of
checking is a cheap way not to meet it twice.

Line endings are not the target here. Git normalizes those on the way in, so a
carriage return that survives into a committed file is inside the content.
"""

from __future__ import annotations

import subprocess
import sys

SUFFIXES = (".cpp", ".hpp", ".h", ".py", ".md", ".txt", ".yml", ".yaml", ".cmake")
NAMES = ("CMakeLists.txt",)

TAB = chr(9)
CARRIAGE_RETURN = chr(13)


def tracked_files() -> list[str]:
    listing = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=True
    ).stdout
    return [
        path
        for path in listing.splitlines()
        if path.endswith(SUFFIXES) or path.rsplit("/", 1)[-1] in NAMES
    ]


def main() -> int:
    problems = []
    for path in tracked_files():
        with open(path, "rb") as handle:
            text = handle.read().decode("utf-8", errors="replace")
        # Split on newlines first, so that only characters *within* a line are
        # judged; a file's line endings are git's business, not this script's.
        for number, line in enumerate(text.split("\n"), start=1):
            line = line.rstrip(CARRIAGE_RETURN)
            found = [
                name
                for name, char in (("tab", TAB), ("carriage return", CARRIAGE_RETURN))
                if char in line
            ]
            if found:
                problems.append(f"{path}:{number}: raw {' and '.join(found)}")

    if problems:
        print("Raw control characters in source. Did an escape lose a backslash?")
        for problem in problems:
            print(f"  {problem}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
