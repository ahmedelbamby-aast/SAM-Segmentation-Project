#!/usr/bin/env python3
"""
CLI entry point for SAM 3 segmentation pipeline.

Author: Ahmed Hany ElBamby
Date: 06-02-2026
"""
import argparse
import logging
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_manager import load_config, validate_config
from src.pipeline import SegmentationPipeline
from src.progress_tracker import ProgressTracker
from src.utils import setup_logging


def print_banner():
    """Print application banner."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║          SAM 3 Segmentation Pipeline                          ║
║          Teacher/Student Detection & Annotation               ║
╚═══════════════════════════════════════════════════════════════╝
    """)


def cmd_run(args, config):
    """Run the segmentation pipeline."""
    logger = logging.getLogger(__name__)
    
    print_banner()
    
    # Validate config
    warnings = validate_config(config)
    for warning in warnings:
        logger.warning(f"Config warning: {warning}")
    
    # Initialize pipeline
    pipeline = SegmentationPipeline(config)
    
    try:
        # Run pipeline
        stats = pipeline.run(args.job_name, resume=args.resume)
        
        # Print summary
        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)
        print(f"  Job Name:     {stats['job_name']}")
        print(f"  Total Images: {stats['total_images']}")
        print(f"  Processed:    {stats['processed']}")
        print(f"  Errors:       {stats['errors']}")
        print(f"  Duration:     {stats['duration']}")
        print()
        print("Annotations by split:")
        for split, data in stats.get('annotations', {}).items():
            print(f"  {split}: {data.get('images', 0)} images, {data.get('annotations', 0)} annotations")
        print()
        print(f"Output saved to: {config.pipeline.output_dir}")
        print("=" * 60)
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Progress saved.")
        print("\nInterrupted. Progress has been saved. Use --resume to continue.")
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)
    finally:
        pipeline.cleanup()


def cmd_status(args, config):
    """Show job status."""
    tracker = ProgressTracker(Path(config.progress.db_path))
    
    job_id = tracker.get_job_id(args.job_name)
    
    if job_id is None:
        print(f"Job '{args.job_name}' not found.")
        return
    
    progress = tracker.get_progress(job_id)
    split_progress = tracker.get_progress_by_split(job_id)
    pending_batches = tracker.get_pending_batches(job_id)
    uploaded_batches = tracker.get_uploaded_batches(job_id)
    
    print("\n" + "=" * 50)
    print(f"JOB STATUS: {args.job_name}")
    print("=" * 50)
    print(f"  Total images:  {progress.get('total_images', 0)}")
    print(f"  Processed:     {progress.get('processed_count', 0)}")
    print(f"  Errors:        {progress.get('error_count', 0)}")
    print(f"  Pending:       {progress.get('pending_count', 0)}")
    
    # Progress percentage
    total = progress.get('total_images', 0)
    processed = progress.get('processed_count', 0)
    if total > 0:
        pct = (processed / total) * 100
        print(f"  Progress:      {pct:.1f}%")
    
    print()
    print("By Split:")
    stuck_count = 0
    for split, data in split_progress.items():
        completed = data.get('completed', 0)
        pending = data.get('pending', 0)
        processing = data.get('processing', 0)
        errors = data.get('error', 0)
        stuck_count += processing
        status_str = f"{completed} done, {pending} pending, {errors} errors"
        if processing > 0:
            status_str += f", {processing} STUCK"
        print(f"  {split}: {status_str}")
    
    if stuck_count > 0:
        print()
        print(f"  ⚠️  {stuck_count} images stuck in 'processing' state!")
        print(f"     Run: python scripts/run_pipeline.py --reset-stuck --job-name {args.job_name}")
    
    print()
    print("Uploads:")
    print(f"  Completed: {len(uploaded_batches)}")
    print(f"  Pending:   {len(pending_batches)}")
    print("=" * 50)
    
    tracker.close()


def cmd_retry_uploads(args, config):
    """Retry failed Roboflow uploads."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Retrying failed uploads for job: {args.job_name}")
    
    pipeline = SegmentationPipeline(config)
    
    try:
        job_id = pipeline.tracker.get_job_id(args.job_name)
        
        if job_id is None:
            print(f"Job '{args.job_name}' not found.")
            return
        
        pending = pipeline.tracker.get_pending_batches(job_id)
        
        if not pending:
            print("No pending batches to upload.")
            return
        
        print(f"Found {len(pending)} pending batches. Retrying...")
        
        for batch in pending:
            pipeline.uploader.queue_batch(
                config.pipeline.output_dir,
                batch['id']
            )
        
        pipeline.uploader.wait_for_uploads()
        
        stats = pipeline.uploader.get_stats()
        print(f"\nUpload complete. Succeeded: {stats['completed']}, Failed: {stats['failed']}")
        
    finally:
        pipeline.cleanup()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='SAM 3 Segmentation Pipeline - Process and annotate images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start new job
  python run_pipeline.py --job-name batch_001

  # Resume interrupted job
  python run_pipeline.py --resume --job-name batch_001

  # Check status
  python run_pipeline.py --status --job-name batch_001

  # Retry failed uploads
  python run_pipeline.py --retry-uploads --job-name batch_001
        """
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    parser.add_argument(
        '--job-name', 
        type=str, 
        required=True,
        help='Unique name for this processing job'
    )
    
    parser.add_argument(
        '--resume', 
        action='store_true',
        help='Resume an interrupted job'
    )
    
    parser.add_argument(
        '--status', 
        action='store_true',
        help='Show job status and exit'
    )
    
    parser.add_argument(
        '--retry-uploads', 
        action='store_true',
        help='Retry failed Roboflow uploads'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default=None,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Override log level from config'
    )
    
    parser.add_argument(
        '--force-scan',
        action='store_true',
        help='Force full dataset rescan (ignore cache)'
    )
    
    parser.add_argument(
        '--cache-info',
        action='store_true',
        help='Show dataset cache info and exit'
    )
    
    parser.add_argument(
        '--reset-stuck',
        action='store_true',
        help='Reset images stuck in processing state back to pending'
    )
    
    parser.add_argument(
        '--reset-errors',
        action='store_true',
        help='Reset images with errors back to pending for retry'
    )
    
    args = parser.parse_args()
    
    # Find config file
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    # Load config
    try:
        config = load_config(str(config_path))
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Setup logging
    log_level = args.log_level or config.progress.log_level
    setup_logging(
        log_file=Path(config.progress.log_file),
        level=log_level
    )
    
    # Route to appropriate command
    if args.cache_info:
        cmd_cache_info(config)
    elif args.reset_stuck:
        # Reset stuck images
        from src.progress_tracker import ProgressTracker
        tracker = ProgressTracker(Path(config.progress.db_path))
        job_id = tracker.get_job_id(args.job_name)
        if job_id is None:
            print(f"Job '{args.job_name}' not found.")
        else:
            count = tracker.reset_stuck_images(job_id)
            if count > 0:
                print(f"Reset {count} stuck images to 'pending' state.")
                print(f"Run the pipeline again to process them.")
            else:
                print("No stuck images found.")
        tracker.close()
    elif args.reset_errors:
        # Reset error images to pending for retry
        from src.progress_tracker import ProgressTracker
        tracker = ProgressTracker(Path(config.progress.db_path))
        job_id = tracker.get_job_id(args.job_name)
        if job_id is None:
            print(f"Job '{args.job_name}' not found.")
        else:
            count = tracker.reset_error_images(job_id)
            if count > 0:
                print(f"Reset {count} error images to 'pending' state.")
                print(f"Run the pipeline again to retry them.")
            else:
                print("No error images found.")
        tracker.close()
    elif args.status:
        cmd_status(args, config)
    elif args.retry_uploads:
        cmd_retry_uploads(args, config)
    else:
        # Handle force-scan by invalidating cache first
        if args.force_scan:
            from src.dataset_cache import DatasetCache
            cache = DatasetCache()
            if cache.invalidate_cache(config.pipeline.input_dir):
                print("Cache invalidated - will perform full rescan")
        cmd_run(args, config)


def cmd_cache_info(config):
    """Show dataset cache information."""
    from src.dataset_cache import DatasetCache
    
    cache = DatasetCache()
    info = cache.get_cache_info(config.pipeline.input_dir)
    
    print("\nDataset Cache Information")
    print("=" * 50)
    
    if info is None:
        print("No cache found for this dataset.")
        print(f"Dataset: {config.pipeline.input_dir}")
    else:
        print(f"Dataset:      {info['input_dir']}")
        print(f"Total files:  {info['total_files']}")
        print(f"Total size:   {info['total_size_mb']:.1f} MB")
        print(f"Cached at:    {info['cached_at']}")
        print(f"Cache file:   {info['cache_file']}")
        print("\nFiles by split:")
        for split, count in info.get('splits', {}).items():
            print(f"  {split}: {count}")
    
    print("=" * 50)


if __name__ == '__main__':
    main()
