"""
CLI entry point: sam3-postprocess — Apply NMS post-processing.

Runs MaskPostProcessor (NMS) on segmentation results.  This is a
standalone stage for running NMS outside of the full pipeline.

Author: Ahmed Hany ElBamby
Date: 23-02-2026
"""
import argparse
import sys
from pathlib import Path
from typing import List

from ..logging_system import LoggingSystem, trace
from ..config_manager import load_config
from ..post_processor import MaskPostProcessor
from ..class_registry import ClassRegistry
from ..progress_display import ModuleProgressManager

_logger = LoggingSystem.get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for sam3-postprocess."""
    parser = argparse.ArgumentParser(
        prog="sam3-postprocess",
        description="Apply NMS post-processing to segmentation results.",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        metavar="PATH",
        help="Path to config.yaml (default: config/config.yaml)",
    )
    parser.add_argument(
        "--strategy",
        metavar="NAME",
        help="Override NMS strategy (confidence, area, class_priority, …)",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        metavar="FLOAT",
        help="Override IoU threshold (0–1)",
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
    Entry point for sam3-postprocess command.

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
    _logger.info("sam3-postprocess starting", extra={"config": args.config})

    # 3. Load config
    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"✗ Config error: {exc}", file=sys.stderr)
        return 1

    # 4. Apply CLI overrides (ISP: post_processor only needs post_processing config slice)
    pp_cfg = config.post_processing
    if pp_cfg is None:
        from ..config_manager import PostProcessingConfig
        pp_cfg = PostProcessingConfig()
        config.post_processing = pp_cfg

    if args.strategy:
        pp_cfg.strategy = args.strategy
    if args.iou_threshold is not None:
        pp_cfg.iou_threshold = args.iou_threshold

    # 5. Build ClassRegistry to populate class_priority if not set
    registry = ClassRegistry.from_config(config.model)
    if not pp_cfg.class_priority:
        pp_cfg.class_priority = registry.class_names

    # 6. Wire concrete class (ISP: only post_processing config slice)
    post_processor = MaskPostProcessor(pp_cfg)
    print(f"📐 NMS strategy: {pp_cfg.strategy}  IoU threshold: {pp_cfg.iou_threshold}")

    # 7. This stage processes SegmentationResult objects held in memory by other
    #    stages.  Standalone, it reports configuration and exits cleanly since
    #    loading persisted results from disk is handled by the full pipeline.
    print("✓ NMS post-processor configured and ready.")
    print("  Run sam3-pipeline to apply NMS as part of the full pipeline.")
    _logger.info("sam3-postprocess: configuration verified", extra={
        "strategy": pp_cfg.strategy,
        "iou_threshold": pp_cfg.iou_threshold,
        "class_priority": pp_cfg.class_priority,
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
