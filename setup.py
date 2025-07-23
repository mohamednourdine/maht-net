#!/usr/bin/env python3
"""
MAHT-Net: Multi-Stage Attention-enhanced Hybrid Transformer Network
for Cephalometric Landmark Detection

Setup script for installing the MAHT-Net package.
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    with open(readme_path, 'r', encoding='utf-8') as f:
        return f.read()

# Read requirements
def read_requirements(filename):
    requirements_path = os.path.join(os.path.dirname(__file__), filename)
    with open(requirements_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f 
                if line.strip() and not line.startswith('#')]

setup(
    name="maht-net",
    version="1.0.0",
    author="Mohamed Nourdine",
    author_email="mohamednjikam25@hotmail.com", 
    description="Multi-Stage Attention-enhanced Hybrid Transformer Network for Cephalometric Landmark Detection",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/mohamednourdine/maht-net",
    project_urls={
        "Bug Tracker": "https://github.com/mohamednourdine/maht-net/issues",
        "Documentation": "https://github.com/mohamednourdine/maht-net/tree/main/documentation",
        "Source Code": "https://github.com/mohamednourdine/maht-net",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": read_requirements("requirements-dev.txt"),
        "aws": [
            "boto3>=1.29.0",
            "awscli>=1.29.0",
            "s3fs>=2023.9.0",
        ],
        "viz": [
            "plotly>=5.17.0",
            "streamlit>=1.28.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "maht-train=src.training.trainer:main",
            "maht-eval=evaluate:main",
            "maht-predict=src.models.maht_net:predict_cli",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.md"],
    },
    zip_safe=False,
    keywords=[
        "cephalometric",
        "landmark detection",
        "medical imaging",
        "computer vision",
        "deep learning",
        "transformer",
        "attention",
        "orthodontics",
        "medical AI",
    ],
)
