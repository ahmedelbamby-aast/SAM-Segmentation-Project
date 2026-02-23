"""
CLI entry point: sam3-upload — Upload annotation batches to Roboflow.

Runs DistributedUploader against the annotation output directory.
Consumes only the roboflow + progress config slices (ISP).

Author: Ahmed Hany ElBamby
Date: 23-02-2026
"""
import argparse
import sys
from pathlib import Path
from typing import List

from ..logging_system import LoggingSystem, trace
from ..config_manager import load_config
from ..roboflow_uploader import DistributedUploader
from ..progress_tracker import ProgressTracker
from ..progress_display import ModuleProgressManager

_logger = LoggingSystem.get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for sam3-upload."""
    parser = argparse.ArgumentParser(
        prog="sam3-upload",
        description="Upload annotation batches to Roboflow.",
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
        help="Job name to upload batches for",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        help="Override output_dir",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List pending batches without uploading",
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
    Entry point for sam3-upload command.

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
    _logger.info("sam3-upload starting", extra={"job_name": args.job_name})

    # 3. Load config
    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"✗ Config error: {exc}", file=sys.stderr)
        return 1

    # 4. Apply CLI overrides
    if args.output_dir:
        config.pipeline.output_dir = Path(args.output_dir)

    if not config.roboflow.enabled:
        print("⚠️  Roboflow upload is disabled in config (roboflow.enabled: false).")
        return 0

    # 5. Wire concrete classes (ISP: uploader consumes config.roboflow + config.progress)
    db_path = Path(config.progress.db_path)
    tracker = ProgressTracker(db_path)

    # 6. Dry-run: list pending batches
    if args.dry_run:
        job_id = tracker.get_job_id(args.job_name)
        if job_id is None:
            print(f"✗ Job '{args.job_name}' not found.")
            return 1
        pending = tracker.get_pending_batches(job_id)
        print(f"📦 Pending batches for '{args.job_name}': {len(pending)}")
        for b in pending[:20]:
            print(f"  - {b}")
        if len(pending) > 20:
            print(f"  … and {len(pending) - 20} more")
        return 0

    # 7. Wire uploader (ISP: roboflow config slice + tracker)
    uploader = DistributedUploader(config.roboflow, tracker)

    job_id = tracker.get_job_id(args.job_name)
    if job_id is None:
        print(f"✗ Job '{args.job_name}' not found in database.")
        return 1

    pending_batches = tracker.get_pending_batches(job_id)
    total_batches = len(pending_batches)
    print(f"📤 Uploading {total_batches} batch(es) for job '{args.job_name}'")

    if total_batches == 0:
        print("✓ No pending batches to upload.")
        return 0

    # 8. Upload with progress display
    errors = 0
    with ModuleProgressManager() as mgr:
        mgr.start_stage("Upload", total=total_batches)
        for batch in pending_batches:
            batch_id = batch.get("id", batch.get("batch_id", 0))
            batch_dir = Path(batch.get("batch_dir", ""))
            split = batch.get("split", "train")
            label = f"batch-{batch_id}"
            mgr.on_item_start(label)
            try:
                # queue_batch() enqueues for async background upload
                uploader.queue_batch(batch_dir=batch_dir, batch_id=batch_id, split=split)
                mgr.on_item_complete(label)
            except Exception as exc:
                _logger.error("Queue error for batch %s: %s", batch_id, exc)
                mgr.on_item_error(label, exc)
                errors += 1
        mgr.finish_stage("Upload")

    uploader.shutdown(wait=True)

    # 9. Summary
    uploaded = total_batches - errors
    print(f"\n✓ Upload complete: {uploaded}/{total_batches} batches uploaded, {errors} errors")
    _logger.info("sam3-upload finished", extra={
        "total_batches": total_batches,
        "uploaded": uploaded,
        "errors": errors,
    })
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
