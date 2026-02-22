"""
CLI entry point: sam3-annotate — Write YOLOv11 annotation files.

Runs AnnotationWriter to produce per-image TXT polygon files and
data.yaml for dataset configuration.

Author: Ahmed Hany ElBamby
Date: 23-02-2026
"""
import argparse
import sys
from pathlib import Path
from typing import List

from ..logging_system import LoggingSystem, trace
from ..config_manager import load_config
from ..annotation_writer import AnnotationWriter
from ..class_registry import ClassRegistry
from ..progress_display import ModuleProgressManager

_logger = LoggingSystem.get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for sam3-annotate."""
    parser = argparse.ArgumentParser(
        prog="sam3-annotate",
        description="Write YOLOv11 annotation files from segmentation results.",
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
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser


@trace
def main(argv: List[str] = None) -> int:
    """
    Entry point for sam3-annotate command.

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
    _logger.info("sam3-annotate starting", extra={"config": args.config})

    # 3. Load config
    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"✗ Config error: {exc}", file=sys.stderr)
        return 1

    # 4. Apply CLI overrides
    if args.output_dir:
        config.pipeline.output_dir = Path(args.output_dir)

    # 5. Wire concrete class
    #    ISP note: AnnotationWriter consumes config.pipeline + config.model slices.
    #    ClassRegistry is the canonical source for class names (not hardcoded strings).
    registry = ClassRegistry.from_config(config.model)
    writer = AnnotationWriter(config)
    print(f"📝 Output directory: {config.pipeline.output_dir}")
    print(f"   Classes:         {registry.class_names}")

    # 6. Write data.yaml (dataset configuration)
    try:
        writer.write_data_yaml()
        print("✓ data.yaml written")
    except Exception as exc:
        _logger.error("Failed to write data.yaml: %s", exc)
        print(f"✗ Failed to write data.yaml: {exc}", file=sys.stderr)
        return 1

    _logger.info("sam3-annotate: data.yaml written", extra={
        "output_dir": str(config.pipeline.output_dir),
        "class_names": registry.class_names,
    })
    print(
        "\nℹ️  To write per-image annotation files run sam3-pipeline, "
        "which passes SegmentationResult objects directly to AnnotationWriter."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
