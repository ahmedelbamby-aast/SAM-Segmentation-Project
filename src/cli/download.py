"""
CLI entry point: sam3-download — Download SAM 3 model weights.

Downloads the SAM 3 model from Hugging Face Hub and saves it to
the models/ directory (or a custom path).

Author: Ahmed Hany ElBamby
Date: 23-02-2026
"""
import argparse
import os
import sys
from pathlib import Path
from typing import List

from ..logging_system import LoggingSystem, trace
from ..model_downloader import HFModelDownloader, download_sam3_model

_logger = LoggingSystem.get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for sam3-download."""
    parser = argparse.ArgumentParser(
        prog="sam3-download",
        description="Download SAM 3 model weights from Hugging Face Hub.",
    )
    parser.add_argument(
        "--token",
        metavar="HF_TOKEN",
        help="Hugging Face access token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        metavar="DIR",
        help="Directory to save model files (default: models/)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show download status only; do not download",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser


@trace
def main(argv: List[str] = None) -> int:
    """
    Entry point for sam3-download command.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 = success, 1 = error).
    """
    # 1. Parse args
    parser = build_parser()
    args = parser.parse_args(argv)

    # 2. Init LoggingSystem FIRST
    LoggingSystem.initialize(level=args.log_level)
    _logger.info("sam3-download starting")

    output_dir = Path(args.output_dir)

    # 3. Status check mode
    if args.status:
        model_path = output_dir / "sam3.pt"
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"✓ Model found: {model_path} ({size_mb:.1f} MB)")
        else:
            print(f"✗ Model not found at: {model_path}")
        return 0

    # 4. Resolve token
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print(
            "✗ No Hugging Face token provided.\n"
            "  Set HF_TOKEN env var or pass --token <TOKEN>",
            file=sys.stderr,
        )
        return 1

    # 5. Download
    print(f"📤 Downloading SAM3 model to: {output_dir.absolute()}")
    try:
        model_path = download_sam3_model(token=token, output_dir=output_dir)
        print(f"✓ Model saved to: {model_path}")
        _logger.info("Model downloaded successfully: %s", model_path)
        return 0
    except Exception as exc:
        _logger.error("Download failed: %s", exc)
        print(f"✗ Download failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
