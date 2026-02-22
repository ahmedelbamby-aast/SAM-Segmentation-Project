"""
CLI entry point: sam3-filter — Categorise images by detection presence.

Moves images with no teacher/student detections to the 'neither' folder.
Consumes only the pipeline config slice (ISP).

Author: Ahmed Hany ElBamby
Date: 23-02-2026
"""
import argparse
import sys
from pathlib import Path
from typing import List

from ..logging_system import LoggingSystem, trace
from ..config_manager import load_config
from ..result_filter import ResultFilter
from ..progress_display import ModuleProgressManager

_logger = LoggingSystem.get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for sam3-filter."""
    parser = argparse.ArgumentParser(
        prog="sam3-filter",
        description="Categorise images with/without detections.",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        metavar="PATH",
        help="Path to config.yaml (default: config/config.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        help="Override output_dir",
    )
    parser.add_argument(
        "--neither-dir",
        metavar="DIR",
        help="Override neither_dir (default: output_dir/neither)",
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
    Entry point for sam3-filter command.

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
    _logger.info("sam3-filter starting", extra={"config": args.config})

    # 3. Load config
    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"✗ Config error: {exc}", file=sys.stderr)
        return 1

    # 4. Apply CLI overrides
    if args.output_dir:
        config.pipeline.output_dir = Path(args.output_dir)
    if args.neither_dir:
        config.pipeline.neither_dir = Path(args.neither_dir)

    # 5. Wire concrete class (ISP: filter only needs pipeline config slice)
    result_filter = ResultFilter(config.pipeline)
    print(f"📁 Neither folder: {result_filter.neither_dir}")

    # 6. Collect annotation TXT files to audit for empty annotations
    output_dir = config.pipeline.output_dir
    label_files: List[Path] = list(output_dir.rglob("labels/*.txt"))
    total = len(label_files)
    print(f"📂 Found {total} label files to audit")

    if total == 0:
        print("⚠️  No label files found — run sam3-segment + sam3-annotate first.")
        return 0

    # 7. Audit labels: empty file → no detections
    moved = 0
    errors = 0
    with ModuleProgressManager() as mgr:
        mgr.start_stage("Filter", total=total)
        for label_path in label_files:
            mgr.on_item_start(str(label_path))
            try:
                # Infer corresponding image path
                img_rel = label_path.relative_to(output_dir)
                img_path = output_dir / str(img_rel).replace("labels", "images").replace(".txt", ".jpg")

                content = label_path.read_text(encoding="utf-8").strip()
                has_detections = bool(content)
                if not has_detections and img_path.exists():
                    result_filter.filter_result(img_path, result=None, copy_to_neither=True)
                    moved += 1
                mgr.on_item_complete(str(label_path))
            except Exception as exc:
                _logger.warning("Filter error for %s: %s", label_path, exc)
                mgr.on_item_error(str(label_path), exc)
                errors += 1
        mgr.finish_stage("Filter")

    # 8. Summary
    stats = result_filter.stats
    print(
        f"\n✓ Filter complete: {stats.total_processed} processed, "
        f"{stats.with_detections} with detections, "
        f"{stats.no_detections} without ({moved} moved to neither), "
        f"{errors} errors"
    )
    _logger.info("sam3-filter finished", extra={
        "total": total,
        "moved": moved,
        "errors": errors,
        "detection_rate": stats.detection_rate,
    })
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
