"""
Hugging Face model downloader with authentication support.

Author: Ahmed Hany ElBamby
Date: 06-02-2026
"""
import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelFile:
    """Represents a model file to download."""
    filename: str
    required: bool = True
    description: str = ""


class HFModelDownloader:
    """
    Download SAM 3 model from Hugging Face with authentication.
    
    Supports:
    - Token via environment variable (HF_TOKEN)
    - Token via argument
    - Resume interrupted downloads
    - Progress tracking
    """
    
    # SAM 3 model repository on Hugging Face
    REPO_ID = "facebook/sam3"
    
    # Files to download
    MODEL_FILES = [
        ModelFile("sam3.pt", required=True, description="Main SAM 3 model weights (3.45 GB)"),
        ModelFile("config.json", required=False, description="Model configuration"),
        ModelFile("tokenizer.json", required=False, description="Tokenizer for text prompts"),
    ]
    
    def __init__(
        self, 
        output_dir: Path,
        token: Optional[str] = None,
        repo_id: Optional[str] = None
    ):
        """
        Initialize downloader.
        
        Args:
            output_dir: Directory to save model files
            token: Hugging Face token (uses HF_TOKEN env var if not provided)
            repo_id: Override default repository ID
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.repo_id = repo_id or self.REPO_ID
        self.token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        
        self._hf_api = None
        self._ensure_hf_hub()
    
    def _ensure_hf_hub(self):
        """Ensure huggingface_hub is installed."""
        try:
            import huggingface_hub
            self._hf_api = huggingface_hub
        except ImportError:
            raise ImportError(
                "huggingface_hub is required for model downloading. "
                "Install it with: pip install huggingface_hub"
            )
    
    def check_auth(self) -> bool:
        """
        Check if authentication is valid.
        
        Returns:
            True if authenticated, False otherwise
        """
        if not self.token:
            logger.warning("No Hugging Face token provided")
            return False
        
        try:
            # Try to get user info to validate token
            api = self._hf_api.HfApi(token=self.token)
            user_info = api.whoami()
            logger.info(f"Authenticated as: {user_info.get('name', 'Unknown')}")
            return True
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return False
    
    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the model repository.
        
        Returns:
            Repository info or None if not accessible
        """
        try:
            api = self._hf_api.HfApi(token=self.token)
            info = api.repo_info(repo_id=self.repo_id, repo_type="model")
            return {
                'id': info.id,
                'sha': info.sha,
                'private': info.private,
                'downloads': getattr(info, 'downloads', 'N/A'),
                'last_modified': str(info.last_modified) if info.last_modified else 'Unknown'
            }
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return None
    
    def list_files(self) -> List[str]:
        """
        List all files in the model repository.
        
        Returns:
            List of filenames
        """
        try:
            api = self._hf_api.HfApi(token=self.token)
            files = api.list_repo_files(repo_id=self.repo_id, repo_type="model")
            return files
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            return []
    
    def download_file(
        self, 
        filename: str,
        force: bool = False,
        show_progress: bool = True
    ) -> Optional[Path]:
        """
        Download a single file from the repository.
        
        Args:
            filename: Name of file to download
            force: Re-download even if file exists
            show_progress: Show download progress bar
            
        Returns:
            Path to downloaded file or None if failed
        """
        dest_path = self.output_dir / filename
        
        # Check if already exists
        if dest_path.exists() and not force:
            logger.info(f"File already exists: {dest_path}")
            return dest_path
        
        try:
            logger.info(f"Downloading {filename} from {self.repo_id}...")
            
            # Use hf_hub_download for resumable downloads
            downloaded_path = self._hf_api.hf_hub_download(
                repo_id=self.repo_id,
                filename=filename,
                token=self.token,
                local_dir=str(self.output_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            
            logger.info(f"Downloaded: {downloaded_path}")
            return Path(downloaded_path)
            
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            return None
    
    def download_model(
        self, 
        include_optional: bool = False,
        force: bool = False
    ) -> Dict[str, Optional[Path]]:
        """
        Download all model files.
        
        Args:
            include_optional: Also download optional files
            force: Re-download existing files
            
        Returns:
            Dictionary mapping filename to download path (None if failed)
        """
        results = {}
        
        for model_file in self.MODEL_FILES:
            if not model_file.required and not include_optional:
                logger.debug(f"Skipping optional file: {model_file.filename}")
                continue
            
            logger.info(f"Processing: {model_file.filename} - {model_file.description}")
            path = self.download_file(model_file.filename, force=force)
            results[model_file.filename] = path
        
        return results
    
    def verify_model(self) -> bool:
        """
        Verify that required model files exist and are valid.
        
        Returns:
            True if all required files are present and valid
        """
        required_files = [f for f in self.MODEL_FILES if f.required]
        
        for model_file in required_files:
            path = self.output_dir / model_file.filename
            
            if not path.exists():
                logger.error(f"Required file missing: {model_file.filename}")
                return False
            
            # Check file is not empty
            if path.stat().st_size == 0:
                logger.error(f"File is empty: {model_file.filename}")
                return False
            
            logger.info(f"Verified: {model_file.filename} ({path.stat().st_size / 1e9:.2f} GB)")
        
        return True
    
    def get_download_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of model files.
        
        Returns:
            Dictionary with status for each file
        """
        status = {}
        
        for model_file in self.MODEL_FILES:
            path = self.output_dir / model_file.filename
            
            if path.exists():
                size = path.stat().st_size
                status[model_file.filename] = {
                    'exists': True,
                    'size_bytes': size,
                    'size_gb': size / 1e9,
                    'required': model_file.required
                }
            else:
                status[model_file.filename] = {
                    'exists': False,
                    'required': model_file.required
                }
        
        return status


def download_sam3_model(
    output_dir: str = "./models",
    token: Optional[str] = None,
    include_optional: bool = False,
    force: bool = False
) -> bool:
    """
    Convenience function to download SAM 3 model.
    
    Args:
        output_dir: Directory to save model
        token: Hugging Face token
        include_optional: Download optional files
        force: Re-download existing files
        
    Returns:
        True if download successful
    """
    downloader = HFModelDownloader(
        output_dir=Path(output_dir),
        token=token
    )
    
    # Check authentication
    if not downloader.check_auth():
        logger.error("Authentication required. Set HF_TOKEN environment variable or pass token.")
        return False
    
    # Download model
    results = downloader.download_model(
        include_optional=include_optional,
        force=force
    )
    
    # Check results
    failed = [f for f, p in results.items() if p is None]
    if failed:
        logger.error(f"Failed to download: {failed}")
        return False
    
    # Verify
    if not downloader.verify_model():
        logger.error("Model verification failed")
        return False
    
    logger.info("Model download complete!")
    return True
