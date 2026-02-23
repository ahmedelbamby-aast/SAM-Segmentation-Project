"""
CLI entry point: sam3-pipeline — Run the full segmentation pipeline.

Thin orchestrator that wires all concrete classes and delegates to
SegmentationPipeline.run().  This is the primary entry point for
end-to-end batch processing.

Author: Ahmed Hany ElBamby
Date: 23-02-2026
"""
import argparse
import sys
from pathlib import Path
from typing import List

from ..logging_system import LoggingSystem, trace
from ..config_manager import load_config, validate_config
from ..pipeline import SegmentationPipeline
from ..class_registry import ClassRegistry
from ..post_processor import MaskPostProcessor
from ..progress_tracker import ProgressTracker
from ..roboflow_uploader import DistributedUploader
from ..preprocessor import ImagePreprocessor
from ..progress_display import ModuleProgressManager
from ..utils import format_duration

_logger = LoggingSystem.get_logger(__name__)


_BANNER = """\
╔═══════════════════════════════════════════════════════════════╗
║          SAM 3 Segmentation Pipeline                          ║
║          Teacher/Student Detection & Annotation               ║
╚═══════════════════════════════════════════════════════════════╝"""


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for sam3-pipeline."""
    parser = argparse.ArgumentParser(
        prog="sam3-pipeline",
        description="Run the full SAM3 segmentation pipeline end-to-end.",
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
        help="Unique name for this processing job (used for resume + progress tracking)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted job from its last checkpoint",
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
    Entry point for sam3-pipeline command.

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
    _logger.info(
        "sam3-pipeline starting",
        extra={"job_name": args.job_name, "resume": args.resume},
    )

    print(_BANNER)
    print(f"\nJob: {args.job_name}  |  Resume: {args.resume}\n")

    # 3. Load config (full config — pipeline needs all slices)
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

    # 5. Validate config; log warnings
    for warning in validate_config(config):
        _logger.warning("Config warning: %s", warning)
        print(f"⚠️  {warning}")

    # 6. Build ClassRegistry — single source of truth for class names/IDs
    registry = ClassRegistry.from_config(config.model)

    # 7. Build optional post_processor (ISP: only post_processing config slice)
    post_processor: MaskPostProcessor = None
    if config.post_processing and config.post_processing.enabled:
        if not config.post_processing.class_priority:
            config.post_processing.class_priority = registry.class_names
        post_processor = MaskPostProcessor(config.post_processing)

    # 8. Build ProgressTracker (ISP: only progress config slice)
    tracker = ProgressTracker(Path(config.progress.db_path))

    # 9. Build DistributedUploader if Roboflow is enabled
    uploader: DistributedUploader = None
    if config.roboflow.enabled:
        uploader = DistributedUploader(config.roboflow, tracker)

    # 10. Build ImagePreprocessor (ISP: only pipeline config slice)
    preprocessor = ImagePreprocessor(config.pipeline)

    # 11. Wire SegmentationPipeline with all injected dependencies
    pipeline = SegmentationPipeline(
        config,
        registry=registry,
        preprocessor=preprocessor,
        tracker=tracker,
        uploader=uploader,
        post_processor=post_processor,
    )

    # 12. Run pipeline
    try:
        stats = pipeline.run(args.job_name, resume=args.resume)

        # 13. Print summary
        sep = "=" * 60
        print(f"\n{sep}\nPROCESSING COMPLETE\n{sep}")
        print(f"  Job Name:     {stats.get('job_name', args.job_name)}")
        print(f"  Total Images: {stats.get('total_images', '?')}")
        print(f"  Processed:    {stats.get('processed', '?')}")
        print(f"  Errors:       {stats.get('errors', 0)}")
        print(f"  Duration:     {stats.get('duration', '?')}")
        print()
        print("Annotations by split:")
        for split, data in stats.get("annotations", {}).items():
            imgs = data.get("images", 0)
            anns = data.get("annotations", 0)
            print(f"  {split}: {imgs} images, {anns} annotations")
        print(f"\nOutput saved to: {config.pipeline.output_dir}")
        print(f"{sep}\n")

        _logger.info("sam3-pipeline finished", extra=stats)
        return 0

    except KeyboardInterrupt:
        _logger.info("Pipeline interrupted by user; progress saved.")
        print("\n⚠️  Interrupted. Progress saved. Use --resume to continue.")
        return 1
    except Exception as exc:
        _logger.exception("Pipeline failed: %s", exc)
        print(f"✗ Pipeline failed: {exc}", file=sys.stderr)
        return 1
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    sys.exit(main())
