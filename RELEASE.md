# Steps to make a release
1. Pull the most recent tags: `git fetch --all --tags`.
1. Create a new branch from main, say `release-vX.Y.Z` where X.Y.Z is the new release number.
1. Bump the version in `hta/version.py`. Versions must adhere to [Semantic Versioning](https://semver.org/).
1. Make a tag, `git tag vX.Y.Z`.
1. Merge the release branch along **with the tag** into main through a PR `git push --tags -u origin
   [branch_name]`
1. Install `twine` and `build` locally: `pip install --upgrade twine build`.
1. Build the source distribution and wheel files: `python3 -m build`.
1. Verify the new package can be installed using pip:
    1. In a new conda environment execute: `pip install dist/HolisticTraceAnalysis-X.Y.Z.tar.gz`
    1. Verify version of the new package: `python -c 'import hta; print(hta.__version__)'`
1. Upload the release to PyPI: `twine upload dist/*` (requires PyPI account).
1. Create a new release on [this
  page](https://github.com/facebookresearch/HolisticTraceAnalysis/releases) on Github.
