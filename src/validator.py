"""
Dataset validator - compares input/output datasets and caches missing images.

Author: Ahmed Hany ElBamby
Date: 06-02-2026
"""
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from .config_manager import Config

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Holds validation comparison results."""
    input_count: int
    output_count: int
    missing_images: List[Path]
    missing_by_split: Dict[str, List[Path]] = field(default_factory=dict)
    
    @property
    def missing_count(self) -> int:
        """Total number of missing images."""
        return len(self.missing_images)
    
    @property
    def is_complete(self) -> bool:
        """True if no missing images found."""
        return self.missing_count == 0
    
    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"Input images:  {self.input_count}",
            f"Output images: {self.output_count}",
            f"Missing:       {self.missing_count}",
        ]
        if self.missing_by_split:
            lines.append("\nMissing by split:")
            for split, images in self.missing_by_split.items():
                lines.append(f"  {split}: {len(images)}")
        return "\n".join(lines)


class Validator:
    """
    Compares input/output datasets and caches missing images for batch processing.
    
    Features:
    - Scans input directory respecting config settings
    - Scans output directory for generated annotations
    - Identifies missing images (in input but not in output)
    - Caches missing images in SQLite for later processing
    - Supports standalone CLI and pipeline integration
    """
    
    def __init__(self, config: Config, db_path: Path = None):
        """
        Initialize validator with configuration.
        
        Args:
            config: Configuration object with pipeline and model settings
            db_path: Optional custom database path (default: config.progress.db_path)
        """
        self.config = config
        self.input_dir = Path(config.pipeline.input_dir)
        self.output_dir = Path(config.pipeline.output_dir)
        self.supported_formats = set(config.pipeline.supported_formats)
        self.input_mode = config.pipeline.input_mode
        
        # Database for caching
        self.db_path = db_path or Path(config.progress.db_path)
        self._conn = None
        self._init_db()
        
        logger.info(f"Validator initialized - input: {self.input_dir}, output: {self.output_dir}")
    
    @property
    def conn(self):
        """Get database connection."""
        import sqlite3
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path), timeout=30.0)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn
    
    def _init_db(self):
        """Initialize validation cache schema."""
        schema = """
        CREATE TABLE IF NOT EXISTS validation_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_name TEXT NOT NULL,
            path TEXT NOT NULL,
            split TEXT,
            cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            processed INTEGER DEFAULT 0,
            UNIQUE(job_name, path)
        );
        CREATE INDEX IF NOT EXISTS idx_validation_job ON validation_cache(job_name);
        CREATE INDEX IF NOT EXISTS idx_validation_processed ON validation_cache(job_name, processed);
        """
        self.conn.executescript(schema)
        self.conn.commit()
    
    def scan_input_directory(self) -> Dict[str, List[Path]]:
        """
        Scan input directory for images, respecting config settings.
        
        Returns:
            Dictionary mapping split names to list of image paths
        """
        files_by_split: Dict[str, List[Path]] = {}
        
        # Check if input directory exists
        if not self.input_dir.exists():
            logger.warning(f"Input directory does not exist: {self.input_dir}")
            print(f"⚠️  Input directory not found: {self.input_dir}")
            return files_by_split
        
        print(f"📂 Scanning input directory: {self.input_dir}")
        print(f"   Input mode: {self.input_mode}")
        print(f"   Supported formats: {self.supported_formats}")
        
        if self.input_mode == "pre-split":
            # Expect train/valid/test subdirectories (and optionally neither)
            splits = ["train", "valid", "test", "neither"]
            for split in splits:
                split_dir = self.input_dir / split
                if split_dir.exists():
                    files_by_split[split] = self._scan_folder(split_dir)
                    print(f"   {split}/: {len(files_by_split[split])} images")
                else:
                    if split != "neither":  # neither is optional in input
                        print(f"   {split}/: directory not found")
        else:
            # Flat mode - all images in one directory
            files_by_split["all"] = self._scan_folder(self.input_dir)
            print(f"   all: {len(files_by_split['all'])} images")
        
        total = sum(len(v) for v in files_by_split.values())
        logger.info(f"Input scan complete: {total} images found")
        return files_by_split
    
    def _scan_folder(self, folder: Path) -> List[Path]:
        """Scan a folder recursively for supported image files."""
        images = []
        if not folder.exists():
            return images
        
        for fmt in self.supported_formats:
            # Use recursive glob to find images in subdirectories too
            # Non-recursive: folder.glob(f"*{fmt}")
            # Recursive: folder.rglob(f"*{fmt}") or folder.glob(f"**/*{fmt}")
            images.extend(folder.rglob(f"*{fmt}"))
            # Also check uppercase extensions
            images.extend(folder.rglob(f"*{fmt.upper()}"))
        
        # Filter to ensure we only have files (not directories)
        images = [p for p in images if p.is_file()]
        
        return sorted(set(images))
    
    def scan_output_directory(self) -> Dict[str, Set[str]]:
        """
        Scan output directory for generated annotation files.
        
        Returns:
            Dictionary mapping split names to set of processed image stems
        """
        processed_by_split: Dict[str, Set[str]] = {}
        
        # Output structure: output_dir/{split}/labels/*.txt (or images/ for neither)
        splits = ["train", "valid", "test"]
        for split in splits:
            labels_dir = self.output_dir / split / "labels"
            processed_stems = set()
            
            if labels_dir.exists():
                for txt_file in labels_dir.glob("*.txt"):
                    # Get the stem (filename without extension)
                    processed_stems.add(txt_file.stem)
            
            processed_by_split[split] = processed_stems
            if processed_stems:
                logger.debug(f"Found {len(processed_stems)} annotations in {split}/labels")
        
        # Also scan neither folder (images with no detections)
        # neither folder has images/ not labels/, so we scan image files directly
        neither_dir = self.output_dir / "neither" / "images"
        neither_stems = set()
        if neither_dir.exists():
            for fmt in self.supported_formats:
                for img_file in neither_dir.glob(f"*{fmt}"):
                    neither_stems.add(img_file.stem)
                for img_file in neither_dir.glob(f"*{fmt.upper()}"):
                    neither_stems.add(img_file.stem)
            processed_by_split["neither"] = neither_stems
            if neither_stems:
                logger.debug(f"Found {len(neither_stems)} images in neither/images")
        else:
            processed_by_split["neither"] = set()
        
        total = sum(len(v) for v in processed_by_split.values())
        logger.info(f"Output scan complete: {total} annotations/images found (including neither)")
        return processed_by_split
    
    def compare_datasets(self) -> ValidationResult:
        """
        Compare input vs output directories to identify missing images.
        
        Returns:
            ValidationResult with comparison details
        """
        input_files = self.scan_input_directory()
        output_stems = self.scan_output_directory()
        
        missing_images: List[Path] = []
        missing_by_split: Dict[str, List[Path]] = {}
        
        # Get stems from neither folder (images with no detections are NOT missing)
        neither_stems = output_stems.get("neither", set())
        
        # For pre-split mode, compare each split
        if self.input_mode == "pre-split":
            for split in ["train", "valid", "test", "neither"]:
                input_for_split = input_files.get(split, [])
                output_for_split = output_stems.get(split, set())
                
                split_missing = []
                for img_path in input_for_split:
                    # Image is not missing if it's in the output split OR in neither folder
                    if img_path.stem not in output_for_split and img_path.stem not in neither_stems:
                        split_missing.append(img_path)
                
                if split_missing:
                    missing_by_split[split] = split_missing
                    missing_images.extend(split_missing)
        else:
            # Flat mode - check against all output splits (including neither)
            all_output_stems = set()
            for stems in output_stems.values():
                all_output_stems.update(stems)
            
            for img_path in input_files.get("all", []):
                if img_path.stem not in all_output_stems:
                    missing_images.append(img_path)
            
            if missing_images:
                missing_by_split["all"] = missing_images
        
        input_count = sum(len(v) for v in input_files.values())
        output_count = sum(len(v) for v in output_stems.values())
        
        result = ValidationResult(
            input_count=input_count,
            output_count=output_count,
            missing_images=missing_images,
            missing_by_split=missing_by_split
        )
        
        logger.info(f"Comparison complete: {result.missing_count} missing images")
        return result
    
    def cache_missing_images(self, result: ValidationResult, job_name: str) -> int:
        """
        Store missing images in database for later processing.
        
        Args:
            result: ValidationResult with missing images
            job_name: Unique name for this validation batch
            
        Returns:
            Number of images cached (total in cache, not just newly added)
        """
        if not result.missing_images:
            logger.info("No missing images to cache")
            return 0
        
        # Prepare data for bulk insert
        data = []
        for split, images in result.missing_by_split.items():
            for img_path in images:
                data.append((job_name, str(img_path), split))
        
        # Bulk insert with IGNORE for duplicates
        cursor = self.conn.cursor()
        cursor.executemany(
            "INSERT OR IGNORE INTO validation_cache (job_name, path, split) VALUES (?, ?, ?)",
            data
        )
        self.conn.commit()
        
        # Get actual count in cache (not just newly inserted)
        cursor.execute(
            "SELECT COUNT(*) FROM validation_cache WHERE job_name = ? AND processed = 0",
            (job_name,)
        )
        cached_count = cursor.fetchone()[0]
        
        logger.info(f"Cached {cached_count} missing images for job '{job_name}'")
        return cached_count
        return cached_count
    
    def get_cached_missing_images(self, job_name: str, unprocessed_only: bool = True) -> List[Tuple[Path, str]]:
        """
        Retrieve cached missing images for a validation job.
        
        Args:
            job_name: Validation job name
            unprocessed_only: If True, only return images not yet processed
            
        Returns:
            List of (path, split) tuples
        """
        if unprocessed_only:
            cursor = self.conn.execute(
                "SELECT path, split FROM validation_cache WHERE job_name = ? AND processed = 0",
                (job_name,)
            )
        else:
            cursor = self.conn.execute(
                "SELECT path, split FROM validation_cache WHERE job_name = ?",
                (job_name,)
            )
        
        return [(Path(row['path']), row['split']) for row in cursor.fetchall()]
    
    def mark_cached_processed(self, job_name: str, paths: List[Path]) -> int:
        """
        Mark cached images as processed.
        
        Args:
            job_name: Validation job name
            paths: List of image paths to mark as processed
            
        Returns:
            Number of images marked
        """
        data = [(job_name, str(p)) for p in paths]
        cursor = self.conn.cursor()
        cursor.executemany(
            "UPDATE validation_cache SET processed = 1 WHERE job_name = ? AND path = ?",
            data
        )
        self.conn.commit()
        return cursor.rowcount
    
    def clear_validation_cache(self, job_name: str) -> int:
        """
        Clear all cached images for a validation job.
        
        Args:
            job_name: Validation job name
            
        Returns:
            Number of records deleted
        """
        cursor = self.conn.execute(
            "DELETE FROM validation_cache WHERE job_name = ?",
            (job_name,)
        )
        self.conn.commit()
        deleted = cursor.rowcount
        logger.info(f"Cleared {deleted} cached entries for job '{job_name}'")
        return deleted
    
    def get_validation_jobs(self) -> List[Dict[str, Any]]:
        """
        Get summary of all validation jobs in the cache.
        
        Returns:
            List of job summaries with counts
        """
        cursor = self.conn.execute("""
            SELECT 
                job_name,
                COUNT(*) as total,
                SUM(CASE WHEN processed = 0 THEN 1 ELSE 0 END) as pending,
                SUM(CASE WHEN processed = 1 THEN 1 ELSE 0 END) as processed,
                MIN(cached_at) as first_cached,
                MAX(cached_at) as last_cached
            FROM validation_cache
            GROUP BY job_name
            ORDER BY last_cached DESC
        """)
        
        return [dict(row) for row in cursor.fetchall()]
    
    def run(self, job_name: str = None, cache_results: bool = True) -> ValidationResult:
        """
        Run full validation workflow: scan, compare, and optionally cache.
        
        Args:
            job_name: Name for the validation job (default: auto-generated)
            cache_results: Whether to cache missing images in database
            
        Returns:
            ValidationResult with comparison details
        """
        if job_name is None:
            job_name = f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting validation job: {job_name}")
        
        # Compare datasets
        result = self.compare_datasets()
        
        # Print summary
        print("\n" + "=" * 50)
        print("VALIDATION RESULTS")
        print("=" * 50)
        print(result.summary())
        print("=" * 50 + "\n")
        
        # Cache if requested and there are missing images
        if cache_results and result.missing_count > 0:
            cached = self.cache_missing_images(result, job_name)
            print(f"✓ Cached {cached} missing images for job '{job_name}'")
            print(f"  Run pipeline with --job-name {job_name} to process them")
        elif result.is_complete:
            print("✓ All input images have been processed!")
        
        return result
    
    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
