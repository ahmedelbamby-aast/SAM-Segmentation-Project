"""
CLI entry point: sam3-preprocess — Validate and resize images.

Scans the input directory, validates images, and produces a
pre-processed copy at the target resolution.

Author: Ahmed Hany ElBamby
Date: 23-02-2026
"""
import argparse
import sys
from pathlib import Path
from typing import List

from ..logging_system import LoggingSystem, trace
from ..config_manager import load_config
from ..preprocessor import ImagePreprocessor
from ..progress_display import ModuleProgressManager

_logger = LoggingSystem.get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for sam3-preprocess."""
    parser = argparse.ArgumentParser(
        prog="sam3-preprocess",
        description="Validate and resize images for the SAM3 pipeline.",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        metavar="PATH",
        help="Path to config.yaml (default: config/config.yaml)",
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
        "--fast",
        action="store_true",
        help="Skip cv2.imread validation (faster scan)",
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
    Entry point for sam3-preprocess command.

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
    _logger.info("sam3-preprocess starting", extra={"config": args.config})

    # 3. Load config (only pipeline slice is consumed by ImagePreprocessor)
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

    # 5. Wire concrete class (ISP: preprocessor only needs pipeline config slice)
    preprocessor = ImagePreprocessor(config)
    if args.fast:
        preprocessor.set_fast_scan(True)

    # 6. Discover images
    _logger.info("Scanning input directory: %s", config.pipeline.input_dir)
    image_paths = preprocessor.scan_directory(config.pipeline.input_dir)
    total = len(image_paths)
    print(f"📂 Found {total} images in {config.pipeline.input_dir}")

    if total == 0:
        print("⚠️  No images found — check input_dir in config.")
        return 0

    # 7. Run preprocessing with progress display
    errors = 0
    with ModuleProgressManager() as mgr:
        mgr.start_stage("Preprocess", total=total)
        for img_path in image_paths:
            mgr.on_item_start(str(img_path))
            try:
                preprocessor.validate_image(img_path)
                mgr.on_item_complete(str(img_path))
            except Exception as exc:
                _logger.warning("Preprocess error for %s: %s", img_path, exc)
                mgr.on_item_error(str(img_path), exc)
                errors += 1
        mgr.finish_stage("Preprocess")
        stats = mgr.get_stage_stats("Preprocess")

    # 8. Summary
    completed = stats.get("completed", 0) if stats else total - errors
    print(f"\n✓ Preprocessing complete: {completed}/{total} valid, {errors} errors")
    _logger.info(
        "sam3-preprocess finished",
        extra={"total": total, "completed": completed, "errors": errors},
    )
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
