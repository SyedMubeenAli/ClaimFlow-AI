"""
Runs every test without needing pytest:   python -m tests.runner
(pytest also works if you have it:        python -m pytest tests)
"""

from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

from tests.helpers import Skip


def main() -> int:
    passed, failures, skipped = 0, [], []
    for path in sorted(Path(__file__).parent.glob("test_*.py")):
        module = importlib.import_module(f"tests.{path.stem}")
        for name in sorted(n for n in dir(module) if n.startswith("test_")):
            function = getattr(module, name)
            if not callable(function):
                continue
            try:
                function()
                passed += 1
            except Skip as why:
                skipped.append(f"{path.stem}.{name}: {why}")
            except Exception:
                failures.append(f"{path.stem}.{name}")
                print(f"FAIL {path.stem}.{name}")
                traceback.print_exc()
    total = passed + len(failures)
    print(f"\n{passed} of {total} tests passed" + (f", {len(skipped)} skipped" if skipped else ""))
    for line in skipped:
        print("  skipped", line)
    if failures:
        print("Failed:", ", ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
