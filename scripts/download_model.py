#!/usr/bin/env python3
"""
CLI script to download SAM 3 model from Hugging Face.

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

from src.model_downloader import HFModelDownloader, download_sam3_model


def setup_logging(level: str = "INFO"):
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def print_banner():
    """Print application banner."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║          SAM 3 Model Downloader                               ║
║          Download from Hugging Face                           ║
╚═══════════════════════════════════════════════════════════════╝
    """)


def cmd_download(args):
    """Download the model."""
    logger = logging.getLogger(__name__)
    
    print_banner()
    
    output_dir = Path(args.output_dir)
    print(f"Output directory: {output_dir.absolute()}")
    print()
    
    # Get token
    token = args.token or os.environ.get("HF_TOKEN")
    
    if not token:
        print("ERROR: No Hugging Face token provided.")
        print()
        print("Please provide a token using one of these methods:")
        print("  1. Set environment variable: export HF_TOKEN=your_token")
        print("  2. Pass via argument: --token your_token")
        print("  3. Login via CLI: huggingface-cli login")
        print()
        print("Get your token at: https://huggingface.co/settings/tokens")
        sys.exit(1)
    
    # Create downloader
    downloader = HFModelDownloader(
        output_dir=output_dir,
        token=token,
        repo_id=args.repo_id
    )
    
    # Check auth
    print("Checking authentication...")
    if not downloader.check_auth():
        print("ERROR: Authentication failed. Please check your token.")
        sys.exit(1)
    print()
    
    # Get model info
    if not args.skip_info:
        print("Fetching model information...")
        info = downloader.get_model_info()
        if info:
            print(f"  Repository: {info['id']}")
            print(f"  Last Modified: {info['last_modified']}")
            print(f"  Private: {info['private']}")
        print()
    
    # Download
    print("Starting download...")
    print("Note: Large files may take a while. Downloads are resumable.")
    print()
    
    results = downloader.download_model(
        include_optional=args.include_optional,
        force=args.force
    )
    
    # Report results
    print()
    print("=" * 50)
    print("DOWNLOAD RESULTS")
    print("=" * 50)
    
    success_count = 0
    for filename, path in results.items():
        if path:
            size = path.stat().st_size / 1e9
            print(f"  ✓ {filename} ({size:.2f} GB)")
            success_count += 1
        else:
            print(f"  ✗ {filename} (FAILED)")
    
    print()
    
    # Verify
    if downloader.verify_model():
        print("✓ Model verification passed!")
        print(f"  Model saved to: {output_dir.absolute()}")
    else:
        print("✗ Model verification failed!")
        sys.exit(1)
    
    print("=" * 50)


def cmd_status(args):
    """Show download status."""
    output_dir = Path(args.output_dir)
    
    if not output_dir.exists():
        print(f"Output directory does not exist: {output_dir}")
        return
    
    downloader = HFModelDownloader(output_dir=output_dir)
    status = downloader.get_download_status()
    
    print()
    print("MODEL FILE STATUS")
    print("=" * 50)
    
    for filename, info in status.items():
        required = "required" if info['required'] else "optional"
        if info['exists']:
            size_gb = info.get('size_gb', 0)
            print(f"  ✓ {filename} ({size_gb:.2f} GB) [{required}]")
        else:
            print(f"  ✗ {filename} (missing) [{required}]")
    
    print("=" * 50)


def cmd_list(args):
    """List files in repository."""
    token = args.token or os.environ.get("HF_TOKEN")
    
    if not token:
        print("ERROR: Token required to list repository files.")
        sys.exit(1)
    
    downloader = HFModelDownloader(
        output_dir=Path(args.output_dir),
        token=token,
        repo_id=args.repo_id
    )
    
    print()
    print(f"FILES IN {downloader.repo_id}")
    print("=" * 50)
    
    files = downloader.list_files()
    for f in files:
        print(f"  {f}")
    
    print("=" * 50)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Download SAM 3 model from Hugging Face',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Authentication:
  Provide your Hugging Face token using one of:
    1. --token argument
    2. HF_TOKEN environment variable
    3. huggingface-cli login

Get your token at: https://huggingface.co/settings/tokens

Examples:
  # Download with token argument
  python download_model.py --token hf_xxxxxxxxxxxx

  # Download using environment variable
  export HF_TOKEN=hf_xxxxxxxxxxxx
  python download_model.py

  # Check download status
  python download_model.py --status

  # List repository files
  python download_model.py --list --token hf_xxxxxxxxxxxx
        """
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./models',
        help='Directory to save model files (default: ./models)'
    )
    
    parser.add_argument(
        '--token',
        type=str,
        default=None,
        help='Hugging Face authentication token'
    )
    
    parser.add_argument(
        '--repo-id',
        type=str,
        default=None,
        help='Override default repository ID (default: facebook/sam3)'
    )
    
    parser.add_argument(
        '--include-optional',
        action='store_true',
        help='Also download optional files (config.json, tokenizer.json)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Re-download even if files exist'
    )
    
    parser.add_argument(
        '--skip-info',
        action='store_true',
        help='Skip displaying repository info'
    )
    
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show status of downloaded files and exit'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List files in the repository and exit'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Log level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    
    # Route to appropriate command
    if args.status:
        cmd_status(args)
    elif args.list:
        cmd_list(args)
    else:
        cmd_download(args)


if __name__ == '__main__':
    main()
