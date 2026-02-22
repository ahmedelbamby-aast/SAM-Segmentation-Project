#!/usr/bin/env python3
"""
CLI entry point for dataset validation.

Author: Ahmed Hany ElBamby
Date: 06-02-2026
"""
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_manager import load_config
from src.validator import Validator
from src.utils import setup_logging


def print_banner():
    """Print application banner."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║          SAM 3 Dataset Validator                              ║
║          Input/Output Comparison & Missing Image Cache        ║
╚═══════════════════════════════════════════════════════════════╝
    """)


def cmd_validate(args, config):
    """Run validation and show report."""
    validator = Validator(config)
    
    try:
        result = validator.run(
            job_name=args.job_name,
            cache_results=args.cache
        )
        
        # Exit with code based on missing images
        if not args.cache and result.missing_count > 0:
            print(f"\n💡 Tip: Re-run with --cache to save missing images for later processing")
        
        return 0 if result.is_complete else 1
    finally:
        validator.close()


def cmd_show_cached(args, config):
    """Show cached missing images."""
    validator = Validator(config)
    
    try:
        if args.job_name:
            # Show specific job
            cached = validator.get_cached_missing_images(args.job_name, unprocessed_only=not args.all)
            
            if not cached:
                print(f"No cached images found for job '{args.job_name}'")
                return 0
            
            print(f"\nCached images for job '{args.job_name}':")
            print("=" * 60)
            
            # Group by split
            by_split = {}
            for path, split in cached:
                if split not in by_split:
                    by_split[split] = []
                by_split[split].append(path)
            
            for split, paths in sorted(by_split.items()):
                print(f"\n{split}: ({len(paths)} images)")
                if args.verbose:
                    for p in paths[:20]:  # Limit output
                        print(f"  - {p}")
                    if len(paths) > 20:
                        print(f"  ... and {len(paths) - 20} more")
            
            print(f"\nTotal: {len(cached)} cached images")
        else:
            # Show all jobs
            jobs = validator.get_validation_jobs()
            
            if not jobs:
                print("No validation jobs found in cache.")
                return 0
            
            print("\nValidation Cache Summary:")
            print("=" * 70)
            print(f"{'Job Name':<35} {'Total':<8} {'Pending':<8} {'Processed':<10}")
            print("-" * 70)
            
            for job in jobs:
                print(f"{job['job_name']:<35} {job['total']:<8} {job['pending']:<8} {job['processed']:<10}")
            
            print("=" * 70)
            print(f"\nUse --job-name <name> to see details for a specific job")
        
        return 0
    finally:
        validator.close()


def cmd_clear_cache(args, config):
    """Clear cached images."""
    validator = Validator(config)
    
    try:
        if not args.job_name:
            print("Error: --job-name is required for --clear-cache")
            return 1
        
        if not args.force:
            confirm = input(f"Clear all cached images for job '{args.job_name}'? [y/N]: ")
            if confirm.lower() != 'y':
                print("Cancelled.")
                return 0
        
        deleted = validator.clear_validation_cache(args.job_name)
        print(f"Cleared {deleted} cached entries for job '{args.job_name}'")
        return 0
    finally:
        validator.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='SAM 3 Dataset Validator - Compare input/output and cache missing images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate datasets and show report
  python run_validator.py --validate

  # Validate and cache missing images for later processing
  python run_validator.py --validate --cache --job-name missing_batch_1

  # Show cached missing images
  python run_validator.py --show-cached
  python run_validator.py --show-cached --job-name missing_batch_1 --verbose

  # Clear cache for a job
  python run_validator.py --clear-cache --job-name missing_batch_1

  # Then process cached missing images with the pipeline
  python run_pipeline.py --job-name missing_batch_1 --resume
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
        help='Name for the validation job'
    )
    
    # Commands
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run validation: compare input/output and show report'
    )
    
    parser.add_argument(
        '--cache',
        action='store_true',
        help='Cache missing images for later processing (use with --validate)'
    )
    
    parser.add_argument(
        '--show-cached',
        action='store_true',
        help='Show cached missing images'
    )
    
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear cached images for a job (requires --job-name)'
    )
    
    # Options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed output (e.g., list individual files)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Show all cached images (including processed ones)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Skip confirmation prompts'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='WARNING',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Log level (default: WARNING for cleaner output)'
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
    setup_logging(
        log_file=Path(config.progress.log_file),
        level=args.log_level
    )
    
    # Route to appropriate command
    if args.validate:
        print_banner()
        sys.exit(cmd_validate(args, config))
    elif args.show_cached:
        sys.exit(cmd_show_cached(args, config))
    elif args.clear_cache:
        sys.exit(cmd_clear_cache(args, config))
    else:
        # Default: show help
        parser.print_help()
        print("\n💡 Try: python run_validator.py --validate")


if __name__ == '__main__':
    main()
