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

The second check has the same shape and the same reason. `--help` writes to
sys.stdout, and on a Japanese Windows console that is CP932. An em dash is not
a character CP932 has, so an em dash anywhere argparse prints turns `--help`,
of all commands, into a traceback. It works everywhere the author looked,
because a UTF-8 terminal encodes it without complaint. So the strings argparse
prints are held to ASCII. Only those: a comment is never printed, and the
Japanese in training/csa_to_data.py encodes into CP932 perfectly well anyway.
"""

from __future__ import annotations

import ast
import subprocess
import sys

SUFFIXES = (".cpp", ".hpp", ".h", ".py", ".md", ".txt", ".yml", ".yaml", ".cmake")
NAMES = ("CMakeLists.txt",)

TAB = chr(9)
CARRIAGE_RETURN = chr(13)

# Calls whose text reaches the terminal by way of print_help(), and the
# keywords of theirs that hold that text.
ARGPARSE_CALLS = (
    "ArgumentParser",
    "add_argument",
    "add_argument_group",
    "add_mutually_exclusive_group",
    "add_parser",
    "add_subparsers",
)
HELP_KEYWORDS = ("help", "description", "epilog", "usage", "metavar", "prog", "title")


def tracked_files() -> list[str]:
    listing = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=True
    ).stdout
    return [
        path
        for path in listing.splitlines()
        if path.endswith(SUFFIXES) or path.rsplit("/", 1)[-1] in NAMES
    ]


def control_characters(path: str, text: str) -> list[str]:
    problems = []
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
    return problems


def called_name(call: ast.Call) -> str:
    """The bare name of what is being called, however it was reached."""
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return getattr(call.func, "id", "")


def outside_ascii(text: str) -> list[str]:
    return sorted({f"U+{ord(char):04X}" for char in set(text) if ord(char) > 127})


def help_text(path: str, text: str) -> list[str]:
    try:
        tree = ast.parse(text, filename=path)
    except SyntaxError as error:
        return [f"{path}:{error.lineno}: will not parse: {error.msg}"]

    problems = []

    def check(line: int, what: str, string: str) -> None:
        found = outside_ascii(string)
        if found:
            problems.append(f"{path}:{line}: {what} holds {' '.join(found)}")

    # Every command line entry point here passes its module docstring in as the
    # parser description, or prints a slice of it as a usage message, so the
    # docstring of a module that names __doc__ at all is printed text.
    docstring = ast.get_docstring(tree, clean=False)
    prints_docstring = any(
        isinstance(node, ast.Name) and node.id == "__doc__" for node in ast.walk(tree)
    )
    if docstring and prints_docstring:
        check(tree.body[0].lineno, "the module docstring, which this module prints", docstring)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or called_name(node) not in ARGPARSE_CALLS:
            continue
        for keyword in node.keywords:
            value = keyword.value
            if keyword.arg not in HELP_KEYWORDS or not isinstance(value, ast.Constant):
                continue
            if isinstance(value.value, str):
                check(value.lineno, f"{keyword.arg}=", value.value)

    return problems


def main() -> int:
    control_problems = []
    help_problems = []
    for path in tracked_files():
        with open(path, "rb") as handle:
            text = handle.read().decode("utf-8", errors="replace")
        control_problems += control_characters(path, text)
        if path.endswith(".py"):
            help_problems += help_text(path, text)

    if control_problems:
        print("Raw control characters in source. Did an escape lose a backslash?")
        for problem in control_problems:
            print(f"  {problem}")
    if help_problems:
        print("Non-ASCII in text that --help prints. A Japanese Windows console is")
        print("CP932, which has no em dash and no curly quotes, so --help raises there.")
        for problem in help_problems:
            print(f"  {problem}")

    return 1 if control_problems or help_problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
