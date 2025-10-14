# -*- coding: utf-8 -*-
from pathlib import Path
from setuptools import find_packages, setup

ROOT = Path(__file__).parent
README = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""

setup(
    name="samri",
    version="0.0.1",
    description="SAMRI: Segment Anything Model for MRI — decoder-only fine-tuning on precomputed MRI embeddings.",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Zhao Wang",
    author_email="zhao.wang1@uq.edu.au",
    url="https://github.com/zhaowangmxy/SAMRI",
    license="Apache-2.0",
    packages=find_packages(exclude=("notebooks", "tests", "docs", "examples")),
    include_package_data=True,
    python_requires=">=3.10",
    # Keep torch out so users choose their CUDA/ROCm wheel
    install_requires=[
        "numpy",
        "scipy",
        "scikit-image",
        "scikit-learn",
        "nibabel",
        "SimpleITK>=2.2.1",
        "opencv-python",
        "matplotlib",
        "tqdm",
        "pandas",
        "einops",
        "h5py"
    ],
    extras_require={
        # Optional: for COCO tools (Linux/macOS typically)
        "coco": ["pycocotools; platform_system != 'Windows'"],
        # Optional: for notebooks and interactive plots
        "notebooks": ["jupyterlab", "ipympl", "ipywidgets"],
        # Dev tooling
        "dev": ["flake8", "isort", "black", "mypy"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],
    keywords=["MRI", "segmentation", "medical-imaging", "deep-learning", "SAM"],
    zip_safe=False,
)
