# Build and view documentation locally

1. Install HTA and required packages from the root of the repo.

    ```
    pip install -e .
    pip install -r docs/requirements.txt
    ```

1. Build the documentation (from the root of the repo)

    ```
    sphinx-build -a docs docs/build
    ```

    This will generate the documentation locally in the `docs/build` folder
    of the repo.

1. Open `index.html` in the `build` folder in a browser.
