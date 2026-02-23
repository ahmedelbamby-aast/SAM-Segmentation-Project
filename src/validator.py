"""
Dataset validator — compares input/output datasets and caches missing images.

SRP breakdown:
  - :class:`ValidationCache` — all SQLite persistence for validation results only
  - :class:`Validator`       — pure scan/compare logic; delegates cache I/O to
    :class:`ValidationCache`

ISP: ``Validator`` accepts only ``pipeline_config`` (not the full ``Config``).
The caller supplies ``db_path`` explicitly.

Author: Ahmed Hany ElBamby
Date: 06-02-2026
"""
import sqlite3
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from .logging_system import LoggingSystem

logger = LoggingSystem.get_logger(__name__)



# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

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
        """Generate a human-readable summary string."""
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


# ---------------------------------------------------------------------------
# ValidationCache — SRP: SQLite caching of missing images only
# ---------------------------------------------------------------------------

class ValidationCache:
    """Store and retrieve missing-image lists in SQLite.

    Single Responsibility: all SQLite operations for validation caching.
    ``Validator`` owns an instance and delegates persistence here.
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS validation_cache (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        job_name  TEXT NOT NULL,
        path      TEXT NOT NULL,
        split     TEXT,
        cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        processed INTEGER DEFAULT 0,
        UNIQUE(job_name, path)
    );
    CREATE INDEX IF NOT EXISTS idx_validation_job ON validation_cache(job_name);
    CREATE INDEX IF NOT EXISTS idx_validation_processed
        ON validation_cache(job_name, processed);
    """

    def __init__(self, db_path: Path) -> None:
        """Open (or create) the SQLite database.

        Args:
            db_path: Path to the SQLite file.  Parent directory is created
                automatically.
        """
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()
        logger.debug(f"ValidationCache opened at {db_path}")

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    @property
    def conn(self) -> sqlite3.Connection:
        """Lazy-open connection (WAL mode)."""
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path), timeout=30.0)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _init_db(self) -> None:
        """Create schema if it does not exist yet."""
        self.conn.executescript(self._SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Cache writes
    # ------------------------------------------------------------------

    def store(self, result: "ValidationResult", job_name: str) -> int:
        """Bulk-insert missing images; ignore duplicates.

        Args:
            result: :class:`ValidationResult` containing missing images.
            job_name: Unique name for this validation batch.

        Returns:
            Number of images currently pending (not yet processed) for
            *job_name*.
        """
        if not result.missing_images:
            logger.info("No missing images to cache")
            return 0

        data = [
            (job_name, str(img_path), split)
            for split, images in result.missing_by_split.items()
            for img_path in images
        ]

        cursor = self.conn.cursor()
        cursor.executemany(
            "INSERT OR IGNORE INTO validation_cache "
            "(job_name, path, split) VALUES (?, ?, ?)",
            data,
        )
        self.conn.commit()

        cursor.execute(
            "SELECT COUNT(*) FROM validation_cache "
            "WHERE job_name = ? AND processed = 0",
            (job_name,),
        )
        cached_count: int = cursor.fetchone()[0]
        logger.info(
            f"Cached {cached_count} missing images for job '{job_name}'"
        )
        return cached_count

    def mark_processed(self, job_name: str, paths: List[Path]) -> int:
        """Mark a list of images as processed.

        Args:
            job_name: Validation job name.
            paths: Image paths to mark.

        Returns:
            Number of rows updated.
        """
        data = [(job_name, str(p)) for p in paths]
        cursor = self.conn.cursor()
        cursor.executemany(
            "UPDATE validation_cache SET processed = 1 "
            "WHERE job_name = ? AND path = ?",
            data,
        )
        self.conn.commit()
        return cursor.rowcount

    def clear(self, job_name: str) -> int:
        """Delete all cached entries for a job.

        Args:
            job_name: Validation job name.

        Returns:
            Number of records deleted.
        """
        cursor = self.conn.execute(
            "DELETE FROM validation_cache WHERE job_name = ?",
            (job_name,),
        )
        self.conn.commit()
        deleted: int = cursor.rowcount
        logger.info(
            f"Cleared {deleted} cached entries for job '{job_name}'"
        )
        return deleted

    # ------------------------------------------------------------------
    # Cache reads
    # ------------------------------------------------------------------

    def retrieve(
        self,
        job_name: str,
        unprocessed_only: bool = True,
    ) -> List[Tuple[Path, str]]:
        """Retrieve cached missing images for a validation job.

        Args:
            job_name: Validation job name.
            unprocessed_only: If ``True``, only return unprocessed images.

        Returns:
            List of ``(path, split)`` tuples.
        """
        if unprocessed_only:
            cursor = self.conn.execute(
                "SELECT path, split FROM validation_cache "
                "WHERE job_name = ? AND processed = 0",
                (job_name,),
            )
        else:
            cursor = self.conn.execute(
                "SELECT path, split FROM validation_cache WHERE job_name = ?",
                (job_name,),
            )
        return [
            (Path(row["path"]), row["split"]) for row in cursor.fetchall()
        ]

    def list_jobs(self) -> List[Dict[str, Any]]:
        """Return summary statistics for all validation jobs.

        Returns:
            List of dicts with keys ``job_name``, ``total``, ``pending``,
            ``processed``, ``first_cached``, ``last_cached``.
        """
        cursor = self.conn.execute("""
            SELECT
                job_name,
                COUNT(*) AS total,
                SUM(CASE WHEN processed = 0 THEN 1 ELSE 0 END) AS pending,
                SUM(CASE WHEN processed = 1 THEN 1 ELSE 0 END) AS processed,
                MIN(cached_at)  AS first_cached,
                MAX(cached_at)  AS last_cached
            FROM validation_cache
            GROUP BY job_name
            ORDER BY last_cached DESC
        """)
        return [dict(row) for row in cursor.fetchall()]


# ---------------------------------------------------------------------------
# Validator — SRP: scan + compare only; delegates persistence to cache
# ---------------------------------------------------------------------------

class Validator:
    """Compare input/output datasets and cache missing images for reprocessing.

    ISP: accepts only a ``pipeline_config``
    (:class:`~src.config_manager.PipelineConfig` slice), not the full
    ``Config`` object.

    Responsibilities:
    - Scan input directory tree.
    - Scan output annotation directory tree.
    - Identify images present in input but absent from output.
    - Delegate caching/retrieval to an internal :class:`ValidationCache`.
    """

    def __init__(
        self,
        pipeline_config,
        db_path: Optional[Path] = None,
    ) -> None:
        """Initialise validator.

        Args:
            pipeline_config: :class:`~src.config_manager.PipelineConfig`
                slice (ISP — only the pipeline config, not the full Config
                object).
            db_path: Path to the SQLite database.  Defaults to
                ``<output_dir>/validation_cache.db`` when ``None``.
        """
        self.input_dir = Path(pipeline_config.input_dir)
        self.output_dir = Path(pipeline_config.output_dir)
        self.supported_formats: Set[str] = set(
            pipeline_config.supported_formats
        )
        self.input_mode: str = pipeline_config.input_mode

        if db_path is None:
            db_path = self.output_dir / "validation_cache.db"

        self.cache = ValidationCache(db_path)
        logger.info(
            f"Validator initialised — input: {self.input_dir}, "
            f"output: {self.output_dir}"
        )

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def scan_input_directory(self) -> Dict[str, List[Path]]:
        """Scan the input directory for images.

        Returns:
            Dict mapping split names to sorted lists of image paths.
        """
        files_by_split: Dict[str, List[Path]] = {}

        if not self.input_dir.exists():
            logger.warning(
                f"Input directory does not exist: {self.input_dir}"
            )
            return files_by_split

        logger.info(
            f"Scanning input: {self.input_dir} "
            f"(mode={self.input_mode}, formats={self.supported_formats})"
        )

        if self.input_mode == "pre-split":
            for split in ["train", "valid", "test", "neither"]:
                split_dir = self.input_dir / split
                if split_dir.exists():
                    files_by_split[split] = self._scan_folder(split_dir)
                    logger.debug(
                        f"  {split}/: {len(files_by_split[split])} images"
                    )
                elif split != "neither":
                    logger.debug(f"  {split}/: directory not found")
        else:
            files_by_split["all"] = self._scan_folder(self.input_dir)
            logger.debug(f"  all: {len(files_by_split['all'])} images")

        total = sum(len(v) for v in files_by_split.values())
        logger.info(f"Input scan complete: {total} images found")
        return files_by_split

    def _scan_folder(self, folder: Path) -> List[Path]:
        """Recursively scan *folder* for supported image files."""
        images: List[Path] = []
        if not folder.exists():
            return images
        for fmt in self.supported_formats:
            images.extend(folder.rglob(f"*{fmt}"))
            images.extend(folder.rglob(f"*{fmt.upper()}"))
        return sorted({p for p in images if p.is_file()})

    def scan_output_directory(self) -> Dict[str, Set[str]]:
        """Scan the output directory for generated annotation files.

        Returns:
            Dict mapping split names to sets of image stems that have been
            processed.
        """
        processed_by_split: Dict[str, Set[str]] = {}

        for split in ["train", "valid", "test"]:
            labels_dir = self.output_dir / split / "labels"
            stems: Set[str] = set()
            if labels_dir.exists():
                for txt_file in labels_dir.glob("*.txt"):
                    stems.add(txt_file.stem)
            processed_by_split[split] = stems
            if stems:
                logger.debug(
                    f"Found {len(stems)} annotations in {split}/labels"
                )

        neither_dir = self.output_dir / "neither" / "images"
        neither_stems: Set[str] = set()
        if neither_dir.exists():
            for fmt in self.supported_formats:
                for img_file in neither_dir.glob(f"*{fmt}"):
                    neither_stems.add(img_file.stem)
                for img_file in neither_dir.glob(f"*{fmt.upper()}"):
                    neither_stems.add(img_file.stem)
            if neither_stems:
                logger.debug(
                    f"Found {len(neither_stems)} images in neither/images"
                )
        processed_by_split["neither"] = neither_stems

        total = sum(len(v) for v in processed_by_split.values())
        logger.info(
            f"Output scan complete: {total} annotations/images "
            "(including neither)"
        )
        return processed_by_split

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare_datasets(self) -> ValidationResult:
        """Compare input vs output directories.

        Returns:
            :class:`ValidationResult` with missing-image details.
        """
        input_files = self.scan_input_directory()
        output_stems = self.scan_output_directory()

        missing_images: List[Path] = []
        missing_by_split: Dict[str, List[Path]] = {}
        neither_stems = output_stems.get("neither", set())

        if self.input_mode == "pre-split":
            for split in ["train", "valid", "test", "neither"]:
                split_missing: List[Path] = [
                    img
                    for img in input_files.get(split, [])
                    if img.stem not in output_stems.get(split, set())
                    and img.stem not in neither_stems
                ]
                if split_missing:
                    missing_by_split[split] = split_missing
                    missing_images.extend(split_missing)
        else:
            all_stems: Set[str] = set()
            for stems in output_stems.values():
                all_stems.update(stems)
            flat_missing = [
                img
                for img in input_files.get("all", [])
                if img.stem not in all_stems
            ]
            if flat_missing:
                missing_by_split["all"] = flat_missing
                missing_images.extend(flat_missing)

        result = ValidationResult(
            input_count=sum(len(v) for v in input_files.values()),
            output_count=sum(len(v) for v in output_stems.values()),
            missing_images=missing_images,
            missing_by_split=missing_by_split,
        )
        logger.info(
            f"Comparison complete: {result.missing_count} missing images"
        )
        return result

    # ------------------------------------------------------------------
    # Public delegation helpers (backward-compatible API)
    # ------------------------------------------------------------------

    def cache_missing_images(
        self, result: ValidationResult, job_name: str
    ) -> int:
        """Delegate to :meth:`ValidationCache.store`."""
        return self.cache.store(result, job_name)

    def get_cached_missing_images(
        self,
        job_name: str,
        unprocessed_only: bool = True,
    ) -> List[Tuple[Path, str]]:
        """Delegate to :meth:`ValidationCache.retrieve`."""
        return self.cache.retrieve(job_name, unprocessed_only)

    def mark_cached_processed(
        self, job_name: str, paths: List[Path]
    ) -> int:
        """Delegate to :meth:`ValidationCache.mark_processed`."""
        return self.cache.mark_processed(job_name, paths)

    def clear_validation_cache(self, job_name: str) -> int:
        """Delegate to :meth:`ValidationCache.clear`."""
        return self.cache.clear(job_name)

    def get_validation_jobs(self) -> List[Dict[str, Any]]:
        """Delegate to :meth:`ValidationCache.list_jobs`."""
        return self.cache.list_jobs()

    # ------------------------------------------------------------------
    # High-level run helper
    # ------------------------------------------------------------------

    def run(
        self,
        job_name: Optional[str] = None,
        cache_results: bool = True,
    ) -> ValidationResult:
        """Scan, compare, and optionally cache missing images.

        Args:
            job_name: Unique validation job name (auto-generated when
                ``None``).
            cache_results: Whether to persist missing images to the database.

        Returns:
            :class:`ValidationResult` with comparison details.
        """
        if job_name is None:
            job_name = (
                f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

        logger.info(f"Starting validation job: {job_name}")
        result = self.compare_datasets()
        logger.info(f"Validation summary:\n{result.summary()}")

        if cache_results and result.missing_count > 0:
            cached = self.cache_missing_images(result, job_name)
            logger.info(
                f"Cached {cached} missing images for job '{job_name}' — "
                f"run pipeline with --job-name {job_name} to process them"
            )
        elif result.is_complete:
            logger.info("All input images have been processed!")

        return result

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying :class:`ValidationCache` connection."""
        self.cache.close()
