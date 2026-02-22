"""
Setup script for SAM 3 Segmentation Pipeline.

Author: Ahmed Hany ElBamby
Date: 06-02-2026
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding='utf-8') if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = [
        line.strip() 
        for line in requirements_path.read_text().splitlines()
        if line.strip() and not line.startswith('#') and not line.startswith('git+')
    ]

setup(
    name="sam3-segmentation-pipeline",
    version="1.0.0",
    author="Ahmed Hany ElBamby",
    author_email="ahmed.elbamby@example.com",
    description="Automated image segmentation pipeline using SAM 3 for teacher/student detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/sam3-segmentation-pipeline",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sam3-pipeline=scripts.run_pipeline:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml"],
    },
)
