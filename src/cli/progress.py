"""
CLI entry point: sam3-progress — Inspect job progress.

Displays progress statistics for a running or completed pipeline job
stored in the SQLite progress database.

Author: Ahmed Hany ElBamby
Date: 23-02-2026
"""
import argparse
import sys
from pathlib import Path
from typing import List

from ..logging_system import LoggingSystem, trace
from ..config_manager import load_config
from ..progress_tracker import ProgressTracker

_logger = LoggingSystem.get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for sam3-progress."""
    parser = argparse.ArgumentParser(
        prog="sam3-progress",
        description="Show progress for a SAM3 pipeline job.",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        metavar="PATH",
        help="Path to config.yaml (default: config/config.yaml)",
    )
    parser.add_argument(
        "--job-name",
        required=True,
        metavar="NAME",
        help="Job name to inspect",
    )
    parser.add_argument(
        "--reset-stuck",
        action="store_true",
        help="Reset images stuck in 'processing' state back to 'pending'",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser


def _print_progress(job_name: str, tracker: ProgressTracker) -> None:
    """Print formatted progress report for a job.

    Args:
        job_name: Name of the job to report.
        tracker: Initialised ProgressTracker instance.
    """
    job_id = tracker.get_job_id(job_name)
    if job_id is None:
        print(f"✗ Job '{job_name}' not found in database.")
        return

    progress = tracker.get_progress(job_id)
    split_progress = tracker.get_progress_by_split(job_id)

    total = progress.get("total_images", 0)
    processed = progress.get("processed_count", 0)
    errors = progress.get("error_count", 0)
    pending = progress.get("pending_count", 0)
    pct = (processed / total * 100) if total > 0 else 0.0

    sep = "=" * 52
    print(f"\n{sep}")
    print(f"  JOB: {job_name}")
    print(sep)
    print(f"  Total images : {total}")
    print(f"  Processed    : {processed}  ({pct:.1f}%)")
    print(f"  Pending      : {pending}")
    print(f"  Errors       : {errors}")

    if split_progress:
        print()
        print("  By Split:")
        for split, data in split_progress.items():
            done = data.get("completed", 0)
            pend = data.get("pending", 0)
            proc = data.get("processing", 0)
            err = data.get("error", 0)
            stuck_str = f"  ⚠ {proc} stuck" if proc > 0 else ""
            print(f"    {split:<8}: {done} done, {pend} pending, {err} errors{stuck_str}")
    print(sep + "\n")


@trace
def main(argv: List[str] = None) -> int:
    """
    Entry point for sam3-progress command.

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
    _logger.info("sam3-progress starting", extra={"job_name": args.job_name})

    # 3. Load config (only progress slice is consumed)
    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"✗ Config error: {exc}", file=sys.stderr)
        return 1

    # 4. Wire concrete class (ISP: only config.progress slice is needed)
    db_path = Path(config.progress.db_path)
    tracker = ProgressTracker(db_path)

    # 5. Optional: reset stuck images
    if args.reset_stuck:
        job_id = tracker.get_job_id(args.job_name)
        if job_id is not None:
            # reset_processing_images() moves 'processing' → 'pending'
            tracker.reset_processing_images(job_id)
            print("✓ Stuck images reset to 'pending'.")
        else:
            print(f"✗ Job '{args.job_name}' not found.")
            return 1

    # 6. Display progress
    _print_progress(args.job_name, tracker)
    return 0


if __name__ == "__main__":
    sys.exit(main())
