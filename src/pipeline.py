"""
Main pipeline orchestrator coordinating all components.

Author: Ahmed Hany ElBamby
Date: 06-02-2026
"""
import random
import time
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from tqdm import tqdm

from .config_manager import Config, load_config
from .preprocessor import ImagePreprocessor
from .progress_tracker import ProgressTracker
from .roboflow_uploader import DistributedUploader
from .dataset_cache import DatasetCache
from .parallel_processor import create_processor
from .utils import format_duration, estimate_eta

logger = logging.getLogger(__name__)


class SegmentationPipeline:
    """
    Main pipeline orchestrating the segmentation workflow.
    
    Coordinates:
    - Image scanning and preprocessing
    - Progress tracking and resume
    - SAM 3 segmentation (parallel or sequential)
    - Result filtering (moves no-detection images to 'neither' folder)
    - Annotation writing
    - Roboflow uploads
    """
    
    def __init__(self, config: Config):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Initialize components
        logger.info("Initializing pipeline components...")
        
        self.preprocessor = ImagePreprocessor(config)
        self.preprocessor.set_fast_scan(True)  # Skip cv2.imread during scan for speed
        self.tracker = ProgressTracker(Path(config.progress.db_path))
        self.uploader = DistributedUploader(config, self.tracker)
        self.cache = DatasetCache()  # Cache for faster rescans
        
        # Create parallel or sequential processor based on config
        num_workers = getattr(config.model, 'parallel_workers', 1)
        self.processor = create_processor(config)
        
        # Keep writer and filter for final operations (data.yaml and manifest)
        from .annotation_writer import AnnotationWriter
        from .result_filter import ResultFilter
        self.writer = AnnotationWriter(config)
        self.filter = ResultFilter(config)
        
        self.batch_size = config.roboflow.batch_upload_size
        self.checkpoint_interval = config.progress.checkpoint_interval
        
        logger.info(f"Pipeline initialized (parallel_workers={num_workers})")
    
    def _assign_splits(self, image_paths: List[Path]) -> List[str]:
        """
        Assign train/val/test splits to images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of split assignments
        """
        random.seed(self.config.split.seed)
        
        splits = []
        train_threshold = self.config.split.train
        val_threshold = train_threshold + self.config.split.valid
        
        for _ in image_paths:
            r = random.random()
            if r < train_threshold:
                splits.append('train')
            elif r < val_threshold:
                splits.append('valid')
            else:
                splits.append('test')
        
        # Log split distribution
        train_count = splits.count('train')
        valid_count = splits.count('valid')
        test_count = splits.count('test')
        logger.info(f"Split distribution - Train: {train_count}, Valid: {valid_count}, Test: {test_count}")
        
        return splits
    
    def run(self, job_name: str, resume: bool = False) -> Dict[str, Any]:
        """
        Run the segmentation pipeline.
        
        Args:
            job_name: Unique name for this processing job
            resume: Whether to resume an existing job
            
        Returns:
            Dictionary with final statistics
        """
        start_time = time.time()
        input_dir = Path(self.config.pipeline.input_dir)
        input_mode = getattr(self.config.pipeline, 'input_mode', 'flat')
        
        # Get per-split percentages
        split_percentages = {
            'train': getattr(self.config.pipeline, 'train_percent', 100),
            'valid': getattr(self.config.pipeline, 'valid_percent', 100),
            'test': getattr(self.config.pipeline, 'test_percent', 100),
        }
        
        # Initialize or resume job
        if not resume:
            logger.info(f"Starting new job: {job_name}")
            logger.info(f"Input mode: {input_mode}")
            logger.info(f"Split sampling: train={split_percentages['train']}%, valid={split_percentages['valid']}%, test={split_percentages['test']}%")
            
            # Check for validation cache first - if missing images are cached for this job, use them
            from .validator import Validator
            validator = Validator(self.config)
            cached_images = validator.get_cached_missing_images(job_name, unprocessed_only=True)
            validator.close()
            
            if cached_images:
                # Use cached missing images from validator
                logger.info(f"Found {len(cached_images)} cached missing images for job '{job_name}'")
                print(f"\n📋 Using {len(cached_images)} cached missing images from validation")
                
                image_paths = [path for path, split in cached_images]
                splits = [split for path, split in cached_images]
                
                # Log split distribution
                from collections import Counter
                split_counts = Counter(splits)
                logger.info(f"Cached images by split - Train: {split_counts.get('train', 0)}, "
                           f"Valid: {split_counts.get('valid', 0)}, Test: {split_counts.get('test', 0)}")
                print(f"   train: {split_counts.get('train', 0)}, valid: {split_counts.get('valid', 0)}, test: {split_counts.get('test', 0)}\n")
            
            # Normal scanning if no cached images
            elif input_mode == "pre-split":
                splits_list = ['train', 'valid', 'test']
                
                # Check cache first
                cache_valid, cached_files, cache_reason = self.cache.check_cache(input_dir, splits_list)
                logger.info(f"Cache status: {cache_reason}")
                
                if cache_valid and cached_files:
                    # Use cached file list
                    split_images = {
                        split: [Path(p) for p in cached_files.get(split, [])]
                        for split in splits_list
                    }
                    logger.info("Using cached scan results (no changes detected)")
                else:
                    # Full scan required
                    logger.info(f"Scanning {input_dir} for images...")
                    split_images = self.preprocessor.scan_presplit_directory(input_dir)
                    
                    # Save to cache for next time
                    self.cache.save_cache(input_dir, split_images, splits_list)
                
                # Apply per-split percentages
                import random
                random.seed(self.config.split.seed)  # Reproducible sampling
                sampled_images = {}
                
                for split_name, paths in split_images.items():
                    percent = split_percentages.get(split_name, 100)
                    
                    if percent == 0:
                        # Skip this split entirely
                        sampled_images[split_name] = []
                        logger.info(f"Skipping {split_name} (0%)")
                    elif percent >= 100:
                        # Use all images
                        sampled_images[split_name] = paths
                        logger.info(f"Using all {split_name} images: {len(paths)}")
                    else:
                        # Sample percentage
                        sample_count = max(1, int(len(paths) * percent / 100))
                        sampled_images[split_name] = random.sample(paths, sample_count) if len(paths) > sample_count else paths
                        logger.info(f"Sampled {percent}% of {split_name}: {len(sampled_images[split_name])}/{len(paths)} images")
                
                split_images = sampled_images
                
                # Flatten into lists while preserving split info
                image_paths = []
                splits = []
                for split_name, paths in split_images.items():
                    image_paths.extend(paths)
                    splits.extend([split_name] * len(paths))
                
                if not image_paths:
                    raise ValueError(f"No images selected! Check your train/valid/test_percent values.")
                
                logger.info(f"Pre-split distribution - Train: {len(split_images.get('train', []))}, "
                           f"Valid: {len(split_images.get('valid', []))}, Test: {len(split_images.get('test', []))}")
            else:
                # Flat mode - scan all images and assign random splits
                logger.info(f"Scanning {input_dir} for images...")
                image_paths = self.preprocessor.scan_directory(input_dir)
                
                if not image_paths:
                    raise ValueError(f"No valid images found in {input_dir}")
                
                # For flat mode, use average of percentages for overall sampling
                avg_percent = (split_percentages['train'] + split_percentages['valid'] + split_percentages['test']) / 3
                if avg_percent < 100:
                    import random
                    random.seed(self.config.split.seed)
                    sample_count = max(1, int(len(image_paths) * avg_percent / 100))
                    image_paths = random.sample(image_paths, sample_count)
                    logger.info(f"Sampled {avg_percent:.0f}% (avg): {len(image_paths)} images")
                else:
                    logger.info(f"Found {len(image_paths)} valid images")
                
                # Assign random splits
                splits = self._assign_splits(image_paths)
            
            # Create job in tracker
            job_id = self.tracker.create_job(job_name, image_paths, splits)
        else:
            logger.info(f"Resuming job: {job_name}")
            
            job_id = self.tracker.get_job_id(job_name)
            if job_id is None:
                raise ValueError(f"Job '{job_name}' not found. Cannot resume.")
            
            # Reset any stuck images
            self.tracker.reset_processing_images(job_id)
        
        # Get progress
        progress = self.tracker.get_progress(job_id)
        total_images = progress.get('total_images', 0)
        already_processed = progress.get('processed_count', 0)
        
        if resume and already_processed > 0:
            logger.info(f"Resuming from {already_processed}/{total_images} images")
        
        # Processing loop
        processed_count = 0
        error_count = 0
        batch_num = len(self.tracker.get_uploaded_batches(job_id))
        batch_images_processed = 0
        
        # Start the processor
        self.processor.start()
        
        with tqdm(
            total=total_images,
            initial=already_processed,
            desc="Processing images",
            unit="img"
        ) as pbar:
            
            while True:
                # Get batch of pending images
                pending = self.tracker.get_pending_images(job_id, limit=100)
                
                if not pending:
                    break
                
                # Mark as processing
                image_ids = [p[0] for p in pending]
                self.tracker.mark_processing(image_ids)
                
                # Prepare tasks for parallel processing
                tasks = [(image_id, str(image_path), split) for image_id, image_path, split in pending]
                
                # Process batch (parallel or sequential based on config)
                results = self.processor.process_batch(tasks)
                
                # Count results in this batch
                batch_success = 0
                batch_detections = 0
                
                # Handle results
                for image_id, image_path, split, success, has_detections, error_msg in results:
                    if success:
                        self.tracker.mark_completed(image_id)
                        processed_count += 1
                        batch_success += 1
                        if has_detections:
                            batch_images_processed += 1
                            batch_detections += 1
                    else:
                        self.tracker.mark_error(image_id, error_msg or "Unknown error")
                        error_count += 1
                        logger.error(f"Error processing {Path(image_path).name}: {error_msg}")
                    
                    pbar.update(1)
                
                # Update progress bar description with stats
                pbar.set_postfix({
                    'detections': batch_images_processed,
                    'errors': error_count
                })
                
                # Log progress every 100 images
                if processed_count % 100 < len(results):
                    logger.info(f"Progress: {already_processed + processed_count}/{total_images} "
                               f"({(already_processed + processed_count)/total_images*100:.1f}%) | "
                               f"Detections: {batch_images_processed} | Errors: {error_count}")
                
                # Checkpoint periodically
                if (processed_count + error_count) % self.checkpoint_interval == 0:
                    self.tracker.checkpoint(job_id)
                
                # Queue batch for upload when ready
                if batch_images_processed >= self.batch_size:
                    batch_num += 1
                    batch_id = self.tracker.create_batch(
                        job_id, 
                        batch_num, 
                        batch_images_processed
                    )
                    self.uploader.queue_batch(
                        self.config.pipeline.output_dir,
                        batch_id
                    )
                    batch_images_processed = 0
                    logger.info(f"Queued batch {batch_num} for upload")
        
        # Upload final partial batch
        if batch_images_processed > 0:
            batch_num += 1
            batch_id = self.tracker.create_batch(job_id, batch_num, batch_images_processed)
            self.uploader.queue_batch(self.config.pipeline.output_dir, batch_id)
            logger.info(f"Queued final batch {batch_num} ({batch_images_processed} images)")
        
        # Generate data.yaml
        self.writer.write_data_yaml()
        
        # Write neither manifest (for filtered images)
        self.filter.write_neither_manifest()
        
        # Wait for main uploads
        print("\n" + "="*60)
        print("📤 UPLOAD PHASE: Uploading to Roboflow...")
        print("="*60)
        logger.info("Waiting for Roboflow uploads to complete...")
        self.uploader.wait_for_uploads()
        
        # Upload neither folder if enabled in config
        # By default, neither folder is preserved for manual review and NOT uploaded
        neither_dir = self.config.pipeline.neither_dir
        if self.uploader.should_upload_neither():
            logger.info("Uploading neither folder (upload_neither: true)...")
            self.uploader.upload_neither_folder(neither_dir)
        else:
            neither_count = self.filter.get_neither_count()
            if neither_count > 0:
                logger.info(f"Neither folder preserved for manual review: {neither_dir} ({neither_count} images)")
        
        # Final checkpoint
        self.tracker.checkpoint(job_id)
        
        # Calculate statistics
        elapsed = time.time() - start_time
        final_progress = self.tracker.get_progress(job_id)
        upload_stats = self.uploader.get_stats()
        annotation_stats = self.writer.get_stats()
        filter_stats = self.filter.get_stats()
        
        stats = {
            'job_name': job_name,
            'total_images': final_progress.get('total_images', 0),
            'processed': final_progress.get('processed_count', 0),
            'errors': final_progress.get('error_count', 0),
            'duration': format_duration(elapsed),
            'duration_seconds': elapsed,
            'uploads': upload_stats,
            'annotations': annotation_stats,
            'filtered': filter_stats
        }
        
        logger.info(f"Pipeline complete! Processed {stats['processed']} images in {stats['duration']}")
        logger.info(f"Filtering: {filter_stats['with_detections']} with detections, {filter_stats['no_detections']} moved to 'neither' ({filter_stats['detection_rate']})")
        
        return stats
    
    def get_status(self, job_name: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a job.
        
        Args:
            job_name: Job name to check
            
        Returns:
            Status dictionary or None if job not found
        """
        job_id = self.tracker.get_job_id(job_name)
        
        if job_id is None:
            return None
        
        progress = self.tracker.get_progress(job_id)
        split_progress = self.tracker.get_progress_by_split(job_id)
        pending_batches = self.tracker.get_pending_batches(job_id)
        uploaded_batches = self.tracker.get_uploaded_batches(job_id)
        
        return {
            'job_name': job_name,
            'job_id': job_id,
            'progress': progress,
            'by_split': split_progress,
            'pending_uploads': len(pending_batches),
            'completed_uploads': len(uploaded_batches)
        }
    
    def cleanup(self):
        """Release resources."""
        self.processor.shutdown()
        self.uploader.shutdown()
        self.tracker.close()
        logger.info("Pipeline resources released")
