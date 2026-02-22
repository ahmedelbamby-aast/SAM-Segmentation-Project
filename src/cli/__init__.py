"""
CLI entry points for SAM 3 Segmentation Pipeline.

Each sub-module provides a standalone command for one pipeline stage:
    sam3-preprocess   — validate and resize images
    sam3-segment      — run SAM 3 inference
    sam3-postprocess  — apply NMS post-processing
    sam3-filter       — categorise images with/without detections
    sam3-annotate     — write YOLOv11 annotation files
    sam3-validate     — compare input/output datasets
    sam3-upload       — upload batches to Roboflow
    sam3-download     — download SAM 3 model weights
    sam3-pipeline     — run the full pipeline end-to-end
    sam3-progress     — inspect job progress

Author: Ahmed Hany ElBamby
Date: 23-02-2026
"""
