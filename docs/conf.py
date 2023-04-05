# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import pathlib
import re
import sys

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.


sys.path.insert(0, os.path.abspath(".."))


def find_version(version_file_path) -> str:
    with open(version_file_path) as version_file:
        version_match = re.search(
            r"^__version_tuple__ = (.*)", version_file.read(), re.M
        )
        if version_match:
            ver_tup = eval(version_match.group(1))
            ver_str = ".".join([str(x) for x in ver_tup])
            return ver_str
        raise RuntimeError("Unable to find version tuple.")


# -- Project information -----------------------------------------------------

project = "Holistic Trace Analysis"
copyright = "2022, Meta Platforms Inc."
author = "Meta Platforms Inc."
repo_root_path = pathlib.Path(__file__).absolute().parents[1]
release = find_version(str(repo_root_path) + "/hta/version.py")

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",  # support NumPy and Google style docstrings
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.doctest",
    "sphinx.ext.ifconfig",
]

source_suffix = {".rst": "restructuredtext"}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["source/_static"]
