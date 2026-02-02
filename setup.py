#!/usr/bin/env python3
"""Setup configuration for the urban-scene-cnn package."""

from setuptools import setup, find_packages

setup(
    name="urban-scene-cnn",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "scikit-learn>=1.3.0",
        "seaborn>=0.12.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    entry_points={
        "console_scripts": [
            "train-cnn=src.main:main",
        ],
    },
    author="Muhammad Hassan Naeem",
    author_email="hassan@example.com",
    description="CNN for Urban Scene Classification using MIT Places Dataset",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Hassan-Naeem-code/Computer-Vision-Week-03",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
