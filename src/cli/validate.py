"""
CLI entry point: sam3-validate — Compare input/output datasets.

Runs Validator to identify images present in input but missing from
the annotation output, and optionally caches them for reprocessing.

Author: Ahmed Hany ElBamby
Date: 23-02-2026
"""
import argparse
import sys
from pathlib import Path
from typing import List

from ..logging_system import LoggingSystem, trace
from ..config_manager import load_config
from ..validator import Validator

_logger = LoggingSystem.get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for sam3-validate."""
    parser = argparse.ArgumentParser(
        prog="sam3-validate",
        description="Compare input/output datasets and report missing images.",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        metavar="PATH",
        help="Path to config.yaml (default: config/config.yaml)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation comparison",
    )
    parser.add_argument(
        "--cache-missing",
        action="store_true",
        help="Cache missing images in SQLite for later batch reprocessing",
    )
    parser.add_argument(
        "--input-dir",
        metavar="DIR",
        help="Override config input_dir",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        help="Override config output_dir",
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
    Entry point for sam3-validate command.

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
    _logger.info("sam3-validate starting", extra={"config": args.config})

    # 3. Load config
    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"✗ Config error: {exc}", file=sys.stderr)
        return 1

    # 4. Apply CLI overrides
    if args.input_dir:
        config.pipeline.input_dir = Path(args.input_dir)
    if args.output_dir:
        config.pipeline.output_dir = Path(args.output_dir)

    if not args.validate and not args.cache_missing:
        print("No action specified. Use --validate or --cache-missing.")
        parser.print_help()
        return 0

    # 5. Wire concrete class (ISP: validator needs pipeline + progress config slices)
    validator = Validator(config)

    # 6. Run validation
    if args.validate:
        print(f"🔍 Comparing:\n  Input:  {config.pipeline.input_dir}\n  Output: {config.pipeline.output_dir}")
        try:
            # compare_datasets() returns ValidationResult with input/output/missing counts
            result = validator.compare_datasets()
            print("\n" + result.summary())
            if result.is_complete:
                print("✓ Dataset is complete — no missing images.")
            else:
                print(f"⚠️  {result.missing_count} image(s) missing from output.")

            if args.cache_missing and result.missing_count > 0:
                # cache_missing_images(result, job_name) requires a job name for the DB
                job_name = getattr(args, 'job_name', None) or 'validate'
                cached = validator.cache_missing_images(result, job_name)
                print(f"✓ Cached {cached} missing images for reprocessing.")

            _logger.info("sam3-validate finished", extra={
                "input_count": result.input_count,
                "output_count": result.output_count,
                "missing_count": result.missing_count,
            })
            return 0 if result.is_complete else 1
        except Exception as exc:
            _logger.error("Validation failed: %s", exc)
            print(f"✗ Validation error: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
