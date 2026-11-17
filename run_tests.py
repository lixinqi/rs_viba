#!/usr/bin/env python3
"""Run the viba test suites against the Rust implementation.

Every tests/test_*.py is a standalone script: it prints one summary line and
exits non-zero when it has a failure.  This runner keeps that contract and
prints a table at the end.

    python3 run_tests.py                 # every suite
    python3 run_tests.py pattern values  # suites whose name contains a word
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TESTS = ROOT / "tests"


def suites(selected):
    names = sorted(path.stem for path in TESTS.glob("test_*.py"))
    if not selected:
        return names
    return [name for name in names if any(word in name for word in selected)]


def main(argv):
    selected = argv[1:]
    names = suites(selected)
    if not names:
        print("no suites matched", file=sys.stderr)
        return 2
    failed = []
    total = 0
    for name in names:
        started = time.time()
        process = subprocess.run(
            [sys.executable, str(TESTS / f"{name}.py")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        elapsed = time.time() - started
        lines = [line for line in process.stdout.splitlines() if line.strip()]
        summary = lines[-1] if lines else "(no output)"
        if process.returncode != 0 and process.stderr.strip():
            lines = [line for line in (process.stdout + "\n" + process.stderr).splitlines() if line.strip()]
            summary = lines[-1] if lines else summary
        status = "ok" if process.returncode == 0 else "FAIL"
        print(f"{status:4} {name:42} {elapsed:6.1f}s  {summary}")
        if process.returncode != 0:
            failed.append(name)
            tail = lines[-12:]
            if tail:
                print("     " + "\n     ".join(tail))
        total += 1
    print()
    print(f"{total - len(failed)}/{total} suites passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
