"""
Dataset cache for avoiding rescans of unchanged datasets.

Author: Ahmed Hany ElBamby
Date: 06-02-2026
"""
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime

from .logging_system import LoggingSystem, trace

_logger = LoggingSystem.get_logger(__name__)


@dataclass
class DatasetFingerprint:
    """Fingerprint of a dataset for change detection."""
    input_dir: str
    total_files: int
    total_size_bytes: int
    dir_hash: str  # Hash of file paths + sizes + mtimes
    splits: Dict[str, int]  # Count per split
    created_at: str
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DatasetFingerprint':
        return cls(**data)


class DatasetCache:
    """
    Cache dataset scan results to avoid rescanning unchanged datasets.
    
    Features:
    - Stores file list and metadata
    - Detects changes via directory fingerprint
    - Supports different datasets by path
    """
    
    CACHE_VERSION = 1
    
    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        """
        Initialize dataset cache.
        
        Args:
            cache_dir: Directory to store cache files (default: ./db/cache/)
        """
        self.cache_dir = Path(cache_dir or "./db/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        _logger.debug("DatasetCache initialized at %s", self.cache_dir)
    
    def _get_cache_path(self, input_dir: Path) -> Path:
        """Get cache file path for a dataset directory."""
        # Create unique cache filename based on path
        dir_hash = hashlib.md5(str(input_dir.absolute()).encode()).hexdigest()[:12]
        return self.cache_dir / f"dataset_{dir_hash}.json"
    
    def _compute_fingerprint(
        self, 
        input_dir: Path, 
        file_info: Dict[str, List[Tuple[Path, int, float]]]
    ) -> DatasetFingerprint:
        """
        Compute fingerprint of current dataset state.
        
        Args:
            input_dir: Input directory path
            file_info: Dict of split -> list of (path, size, mtime)
            
        Returns:
            DatasetFingerprint object
        """
        # Collect all file info for hashing
        all_entries = []
        total_files = 0
        total_size = 0
        splits_count = {}
        
        for split, files in file_info.items():
            splits_count[split] = len(files)
            for path, size, mtime in files:
                # Truncate mtime to integer seconds to avoid floating point precision issues
                mtime_int = int(mtime)
                all_entries.append(f"{path}:{size}:{mtime_int}")
                total_files += 1
                total_size += size
        
        # Sort for consistent hash
        all_entries.sort()
        dir_hash = hashlib.sha256("\n".join(all_entries).encode()).hexdigest()[:32]
        
        return DatasetFingerprint(
            input_dir=str(input_dir.absolute()),
            total_files=total_files,
            total_size_bytes=total_size,
            dir_hash=dir_hash,
            splits=splits_count,
            created_at=datetime.now().isoformat()
        )
    
    def _quick_scan(self, input_dir: Path, splits: List[str]) -> Dict[str, List[Tuple[Path, int, float]]]:
        """
        Quick scan of directory to get file metadata without validation.
        
        Args:
            input_dir: Input directory
            splits: List of split names to scan
            
        Returns:
            Dict of split -> list of (path, size, mtime)
        """
        result = {}
        
        for split in splits:
            split_dir = input_dir / split
            if not split_dir.exists():
                result[split] = []
                continue
            
            files = []
            for f in split_dir.rglob("*"):
                if f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}:
                    try:
                        stat = f.stat()
                        files.append((f, stat.st_size, stat.st_mtime))
                    except OSError:
                        pass
            
            result[split] = files
        
        return result
    
    @trace
    def check_cache(self, input_dir: Path, splits: Optional[List[str]] = None) -> Tuple[bool, Optional[Dict], str]:
        """
        Check if cached scan results are still valid.
        
        Args:
            input_dir: Dataset input directory
            splits: Split folder names (default: ['train', 'valid', 'test'])
            
        Returns:
            Tuple of (is_valid, cached_data, reason)
            - is_valid: True if cache is valid and can be used
            - cached_data: The cached file list if valid, else None
            - reason: Human-readable explanation
        """
        splits = splits or ['train', 'valid', 'test']
        input_dir = Path(input_dir)
        cache_path = self._get_cache_path(input_dir)
        
        # Check if cache file exists
        if not cache_path.exists():
            return False, None, "No cache found - first run for this dataset"
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            _logger.warning("Cache file corrupted: %s", e)
            return False, None, "Cache file corrupted"
        
        # Check cache version
        if cache_data.get('version') != self.CACHE_VERSION:
            return False, None, "Cache version mismatch - will rescan"
        
        # Load stored fingerprint
        stored_fp = DatasetFingerprint.from_dict(cache_data['fingerprint'])
        
        # Check if same directory
        if stored_fp.input_dir != str(input_dir.absolute()):
            return False, None, "Different dataset directory"
        
        # Quick scan current state
        _logger.info("Checking dataset for changes...")
        current_info = self._quick_scan(input_dir, splits)
        current_fp = self._compute_fingerprint(input_dir, current_info)
        
        # Compare fingerprints
        if stored_fp.total_files != current_fp.total_files:
            diff = current_fp.total_files - stored_fp.total_files
            sign = "+" if diff > 0 else ""
            return False, None, f"File count changed ({sign}{diff} files)"
        
        if stored_fp.dir_hash != current_fp.dir_hash:
            return False, None, "Files modified since last scan"
        
        # Cache is valid!
        _logger.info("Cache valid - %d files, skipping full scan", stored_fp.total_files)
        return True, cache_data['files'], "Cache valid - no changes detected"
    
    @trace
    def save_cache(
        self, 
        input_dir: Path, 
        files_by_split: Dict[str, List[Path]],
        splits: Optional[List[str]] = None,
    ) -> Path:
        """
        Save scan results to cache.
        
        Args:
            input_dir: Dataset input directory
            files_by_split: Dict of split -> list of valid image paths
            splits: Split folder names
            
        Returns:
            Path to cache file
        """
        splits = splits or ['train', 'valid', 'test']
        input_dir = Path(input_dir)
        cache_path = self._get_cache_path(input_dir)
        
        # Collect file info for fingerprint
        file_info = {}
        for split in splits:
            files = files_by_split.get(split, [])
            info_list = []
            for f in files:
                try:
                    stat = f.stat()
                    info_list.append((f, stat.st_size, stat.st_mtime))
                except OSError:
                    pass
            file_info[split] = info_list
        
        # Compute fingerprint
        fingerprint = self._compute_fingerprint(input_dir, file_info)
        
        # Prepare cache data
        cache_data = {
            'version': self.CACHE_VERSION,
            'fingerprint': fingerprint.to_dict(),
            'files': {
                split: [str(p) for p in paths]
                for split, paths in files_by_split.items()
            }
        }
        
        # Save cache
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2)
        
        _logger.info("Saved dataset cache: %d files", fingerprint.total_files)
        return cache_path
    
    @trace
    def invalidate_cache(self, input_dir: Path) -> bool:
        """
        Invalidate cache for a dataset directory.
        
        Args:
            input_dir: Dataset directory
            
        Returns:
            True if cache was deleted
        """
        cache_path = self._get_cache_path(Path(input_dir))
        if cache_path.exists():
            cache_path.unlink()
            _logger.info("Cache invalidated for %s", input_dir)
            return True
        return False
    
    @trace
    def get_cache_info(self, input_dir: Path) -> Optional[Dict[str, Any]]:
        """
        Get information about cached dataset.
        
        Args:
            input_dir: Dataset directory
            
        Returns:
            Cache info or None if no cache
        """
        cache_path = self._get_cache_path(Path(input_dir))
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            fp = cache_data.get('fingerprint', {})
            return {
                'cache_file': str(cache_path),
                'input_dir': fp.get('input_dir'),
                'total_files': fp.get('total_files'),
                'total_size_mb': fp.get('total_size_bytes', 0) / 1e6,
                'splits': fp.get('splits'),
                'cached_at': fp.get('created_at')
            }
        except (json.JSONDecodeError, IOError):
            return None

    # ------------------------------------------------------------------
    # Stats pattern
    # ------------------------------------------------------------------

    @trace
    def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics.

        Returns:
            Dict with ``cache_dir`` and ``cached_datasets`` count.
        """
        cached = list(self.cache_dir.glob("*.json")) if self.cache_dir.exists() else []
        return {
            "cache_dir": str(self.cache_dir),
            "cached_datasets": len(cached),
        }

    @trace
    def reset_stats(self) -> None:
        """No-op — :class:`DatasetCache` has no mutable counters."""
