#!/usr/bin/env python3
"""
Legacy wrapper - delegates to ``src.cli.validate.main``.

Kept for backward compatibility with scripts that call this file directly.
Prefer using the ``sam3-validate`` console_script entry point instead.

Author: Ahmed Hany ElBamby
Date: 06-02-2026
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cli.validate import main  # noqa: E402 - path manipulation must precede


if __name__ == "__main__":
    sys.exit(main())
