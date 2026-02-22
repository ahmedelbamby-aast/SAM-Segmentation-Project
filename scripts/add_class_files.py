#!/usr/bin/env python3
"""
Add class name files to existing output dataset for Roboflow.

Author: Ahmed Hany ElBamby
Date: 06-02-2026
"""
import sys
import yaml
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def add_class_files(output_dir: Path, class_names: list):
    """Add _classes.txt files to each split folder and fix data.yaml."""
    output_dir = Path(output_dir)
    
    # Write to each split folder
    for split in ['train', 'valid', 'test']:
        split_dir = output_dir / split
        if split_dir.exists():
            classes_file = split_dir / '_classes.txt'
            with open(classes_file, 'w', encoding='utf-8') as f:
                for class_name in class_names:
                    f.write(f"{class_name}\n")
            print(f"Created: {classes_file}")
    
    # Write root level classes.txt
    root_classes = output_dir / 'classes.txt'
    with open(root_classes, 'w', encoding='utf-8') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    print(f"Created: {root_classes}")
    
    # Fix data.yaml with proper dict format for names
    data_yaml = output_dir / 'data.yaml'
    names_dict = {i: name for i, name in enumerate(class_names)}
    
    data = {
        'path': str(output_dir.absolute()),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': len(class_names),
        'names': names_dict  # Dict format: {0: 'teacher', 1: 'student'}
    }
    
    with open(data_yaml, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"Fixed: {data_yaml}")
    print(f"  names: {names_dict}")
    
    print("\n✅ Class files added! Roboflow will now recognize class names.")


if __name__ == "__main__":
    # Default class names from config
    class_names = ["teacher", "student"]
    
    # Default output directory
    output_dir = PROJECT_ROOT / "output" / "dataset"
    
    # Allow override from command line
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])
    
    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}")
        sys.exit(1)
    
    print(f"Adding class files to: {output_dir}")
    print(f"Class names: {class_names}")
    print()
    
    add_class_files(output_dir, class_names)
