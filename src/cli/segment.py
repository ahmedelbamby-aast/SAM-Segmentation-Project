"""
CLI entry point: sam3-segment — Run SAM 3 inference on images.

Wires GPUStrategy + ClassRegistry + ParallelProcessor (or SequentialProcessor)
and runs inference over the pre-processed image directory.

Author: Ahmed Hany ElBamby
Date: 23-02-2026
"""
import argparse
import sys
from pathlib import Path
from typing import List

from ..logging_system import LoggingSystem, trace
from ..config_manager import load_config
from ..class_registry import ClassRegistry
from ..parallel_processor import create_processor
from ..progress_display import ModuleProgressManager

_logger = LoggingSystem.get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for sam3-segment."""
    parser = argparse.ArgumentParser(
        prog="sam3-segment",
        description="Run SAM 3 segmentation inference on images.",
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
        "--device",
        metavar="DEVICE",
        help="Override model device (auto, cpu, cuda:0, …)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        metavar="N",
        help="Override number of parallel workers",
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
    Entry point for sam3-segment command.

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
    _logger.info("sam3-segment starting", extra={"config": args.config})

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
    if args.device:
        config.model.device = args.device
    if args.workers is not None:
        config.model.parallel_workers = args.workers

    # 5. Build ClassRegistry (single source of truth for class names/IDs)
    registry = ClassRegistry.from_config(config.model)
    _logger.info("ClassRegistry: %s", registry)

    # 6. Discover image paths from input directory
    input_dir = config.pipeline.input_dir
    supported = {f.lower() for f in config.pipeline.supported_formats}
    image_paths: List[Path] = [
        p for p in input_dir.rglob("*") if p.suffix.lower() in supported
    ]
    total = len(image_paths)
    print(f"📂 Found {total} images in {input_dir}")
    if total == 0:
        print("⚠️  No images found — check input_dir in config.")
        return 0

    # 7. Wire Processor via factory (ISP: processor gets config.model + config.gpu)
    processor = create_processor(config, registry)
    processor.start()

    # 8. Run inference with progress display
    processed = 0
    errors = 0
    with ModuleProgressManager() as mgr:
        mgr.start_stage("Segment", total=total)
        try:
            for result in processor.process_batch(image_paths, callback=mgr):
                processed += 1
        except Exception as exc:
            _logger.error("Segment batch error: %s", exc)
            errors += 1
        mgr.finish_stage("Segment")

    processor.shutdown(wait=True)

    # 9. Summary
    print(f"\n✓ Segmentation complete: {processed}/{total} processed, {errors} errors")
    _logger.info(
        "sam3-segment finished",
        extra={"total": total, "processed": processed, "errors": errors},
    )
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
