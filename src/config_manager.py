"""
Configuration management with validation and environment variable support.

Author: Ahmed Hany ElBamby
Date: 06-02-2026
"""
import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Any, Dict


@dataclass
class PipelineConfig:
    """Pipeline configuration settings."""
    input_dir: Path
    output_dir: Path
    resolution: int
    supported_formats: List[str]
    num_workers: int
    input_mode: str = "pre-split"  # "pre-split" or "flat"
    train_percent: float = 100.0  # 0-100, percentage of train images (0=skip, 0.5=0.5%)
    valid_percent: float = 100.0  # 0-100, percentage of valid images (0=skip, 0.5=0.5%)
    test_percent: float = 100.0   # 0-100, percentage of test images (0=skip, 0.5=0.5%)
    neither_dir: Optional[Path] = None  # Custom path for 'neither' folder (default: output_dir/neither)

    def __post_init__(self):
        self.input_dir = Path(self.input_dir)
        self.output_dir = Path(self.output_dir)
        if self.input_mode not in ("pre-split", "flat"):
            raise ValueError(f"input_mode must be 'pre-split' or 'flat', got '{self.input_mode}'")
        # Clamp percentages to valid range 0-100
        self.train_percent = float(max(0, min(100, self.train_percent)))
        self.valid_percent = float(max(0, min(100, self.valid_percent)))
        self.test_percent = float(max(0, min(100, self.test_percent)))
        # Set neither_dir to output_dir/neither if not specified
        if self.neither_dir is not None:
            self.neither_dir = Path(self.neither_dir)
        else:
            self.neither_dir = self.output_dir / "neither"


@dataclass
class ModelConfig:
    """Model configuration settings."""
    path: Path
    confidence: float
    prompts: List[str]
    half_precision: bool
    device: str
    parallel_workers: int = 1  # Number of parallel inference processes

    def __post_init__(self):
        self.path = Path(self.path)
        if self.parallel_workers < 1:
            self.parallel_workers = 1


@dataclass
class SplitConfig:
    """Data split configuration."""
    train: float
    valid: float
    test: float
    seed: int

    def __post_init__(self):
        total = self.train + self.valid + self.test
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")


@dataclass
class ProgressConfig:
    """Progress tracking configuration."""
    db_path: Path
    checkpoint_interval: int
    log_file: Path
    log_level: str

    def __post_init__(self):
        self.db_path = Path(self.db_path)
        self.log_file = Path(self.log_file)


@dataclass
class RoboflowProjectConfig:
    """Configuration for a single Roboflow project."""
    project: str           # Project name/ID
    is_prediction: bool = True  # True = needs approval


@dataclass
class RoboflowWorkspaceConfig:
    """Configuration for a single Roboflow workspace."""
    workspace: str         # Workspace name
    api_key: str = ""      # API key for this workspace (each workspace can have its own key)
    projects: List[RoboflowProjectConfig] = field(default_factory=list)
    
    def __post_init__(self):
        # Convert raw dicts to RoboflowProjectConfig if needed
        if self.projects and isinstance(self.projects[0], dict):
            self.projects = [RoboflowProjectConfig(**p) for p in self.projects]


@dataclass
class RoboflowConfig:
    """Roboflow upload configuration with multi-workspace support."""
    enabled: bool
    batch_upload_size: int
    upload_workers: int
    retry_attempts: int
    retry_delay: int
    
    # Multi-workspace support (new)
    workspaces: List[RoboflowWorkspaceConfig] = field(default_factory=list)
    
    # Backward compatibility (legacy single workspace/project)
    api_key: str = ""  # Legacy global API key (used with legacy workspace/project)
    workspace: Optional[str] = None
    project: Optional[str] = None
    is_prediction: bool = True  # True = needs approval, False = ground truth
    upload_neither: bool = False  # Upload 'neither' folder (default: false for manual review)
    
    def __post_init__(self):
        # Convert raw dicts to RoboflowWorkspaceConfig if needed
        if self.workspaces and isinstance(self.workspaces[0], dict):
            self.workspaces = [RoboflowWorkspaceConfig(**w) for w in self.workspaces]
        
        # Backward compatibility: convert legacy single workspace/project to workspaces list
        if not self.workspaces and self.workspace and self.project:
            self.workspaces = [
                RoboflowWorkspaceConfig(
                    workspace=self.workspace,
                    api_key=self.api_key,  # Use legacy api_key for backward compat
                    projects=[RoboflowProjectConfig(project=self.project, is_prediction=self.is_prediction)]
                )
            ]
    
    def get_all_targets(self) -> List[tuple]:
        """Get all (api_key, workspace, project, is_prediction) tuples for uploads."""
        targets = []
        for ws in self.workspaces:
            for proj in ws.projects:
                targets.append((ws.api_key, ws.workspace, proj.project, proj.is_prediction))
        return targets


@dataclass
class PostProcessingConfig:
    """Post-processing configuration for handling overlapping annotations."""
    enabled: bool = True
    iou_threshold: float = 0.5  # IoU threshold for detecting overlaps
    strategy: str = "confidence"  # confidence, area, class_priority, soft_nms
    class_priority: List[str] = field(default_factory=lambda: ["teacher", "student"])
    soft_nms_sigma: float = 0.5  # Sigma for Soft-NMS
    min_confidence_after_decay: float = 0.1


@dataclass
class Config:
    """Main configuration container."""
    pipeline: PipelineConfig
    model: ModelConfig
    split: SplitConfig
    progress: ProgressConfig
    roboflow: RoboflowConfig
    post_processing: Optional[PostProcessingConfig] = None


def _expand_env_vars(obj: Any) -> Any:
    """Recursively expand ${VAR} patterns in config values."""
    if isinstance(obj, str):
        if obj.startswith("${") and obj.endswith("}"):
            var_name = obj[2:-1]
            return os.environ.get(var_name, "")
        return obj
    elif isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_expand_env_vars(v) for v in obj]
    return obj


def _dict_to_dataclass(data: Dict, cls: type) -> Any:
    """Convert dictionary to dataclass, handling nested structures."""
    field_names = {f.name for f in cls.__dataclass_fields__.values()}
    filtered_data = {k: v for k, v in data.items() if k in field_names}
    return cls(**filtered_data)


def load_config(config_path: str) -> Config:
    """
    Load and validate configuration from YAML file.
    
    Args:
        config_path: Path to the configuration YAML file
        
    Returns:
        Config object with all settings
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    
    # Expand environment variables
    raw = _expand_env_vars(raw)
    
    # Validate required sections
    required_sections = ['pipeline', 'model', 'split', 'progress', 'roboflow']
    for section in required_sections:
        if section not in raw:
            raise ValueError(f"Missing required config section: {section}")
    
    # Construct config objects
    try:
        # Handle optional post_processing config
        post_processing = None
        if 'post_processing' in raw:
            post_processing = _dict_to_dataclass(raw['post_processing'], PostProcessingConfig)
        
        config = Config(
            pipeline=_dict_to_dataclass(raw['pipeline'], PipelineConfig),
            model=_dict_to_dataclass(raw['model'], ModelConfig),
            split=_dict_to_dataclass(raw['split'], SplitConfig),
            progress=_dict_to_dataclass(raw['progress'], ProgressConfig),
            roboflow=_dict_to_dataclass(raw['roboflow'], RoboflowConfig),
            post_processing=post_processing
        )
    except TypeError as e:
        raise ValueError(f"Invalid configuration: {e}")
    
    # Validate model path exists
    if not config.model.path.exists():
        raise FileNotFoundError(f"Model file not found: {config.model.path}")
    
    return config


def validate_config(config: Config) -> List[str]:
    """
    Validate configuration and return list of warnings.
    
    Args:
        config: Configuration object to validate
        
    Returns:
        List of warning messages (empty if all good)
    """
    warnings = []
    
    # Check input directory
    if not config.pipeline.input_dir.exists():
        warnings.append(f"Input directory does not exist: {config.pipeline.input_dir}")
    
    # Check resolution
    if config.pipeline.resolution not in [640, 1024]:
        warnings.append(f"Unusual resolution {config.pipeline.resolution}, standard is 640 or 1024")
    
    # Check Roboflow API key
    if config.roboflow.enabled and not config.roboflow.api_key:
        warnings.append("Roboflow enabled but API key is empty")
    
    # Check batch size
    if config.roboflow.batch_upload_size < 100:
        warnings.append(f"Small batch size {config.roboflow.batch_upload_size} may be inefficient")
    
    return warnings


def load_config_from_dict(raw: Dict[str, Any]) -> Config:
    """
    Load configuration from a dictionary (for worker processes).
    
    Args:
        raw: Configuration dictionary
        
    Returns:
        Config object
    """
    try:
        config = Config(
            pipeline=_dict_to_dataclass(raw['pipeline'], PipelineConfig),
            model=_dict_to_dataclass(raw['model'], ModelConfig),
            split=_dict_to_dataclass(raw['split'], SplitConfig),
            progress=_dict_to_dataclass(raw['progress'], ProgressConfig),
            roboflow=_dict_to_dataclass(raw['roboflow'], RoboflowConfig)
        )
    except TypeError as e:
        raise ValueError(f"Invalid configuration: {e}")
    
    return config
