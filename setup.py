# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import setuptools
from docs.conf import find_version


def fetch_requirements():
    with open("requirements.txt") as f:
        reqs = f.read().strip().split("\n")
    return reqs


setuptools.setup(
    name="HolisticTraceAnalysis",
    description="A python library for analyzing PyTorch Profiler traces",
    version=find_version("hta/version.py"),
    url="https://github.com/facebookresearch/HolisticTraceAnalysis",
    python_requires=">=3.8",
    author="Meta Platforms Inc.",
    author_email="todo@meta.com",
    license="MIT",
    install_requires=fetch_requirements(),
    include_package_data=True,
    packages=setuptools.find_packages(include=["hta*"]),  # Only include code within hta.
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: BSD License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],
)
