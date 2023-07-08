---
layout: post
title: "How to Use Poetry with Conda for Package Management on a Specific Python Version"
date: 2022-07-08 12:55:00 +0000
description: 
tags: [poetry, conda, python-packaging]
comments: true
---

# How to Use Poetry with Conda for Package Management on a Specific Python Version

<br>

**TL/DR**:  Both Conda and Poetry can be used for Python package management, however Poetry additionally helps one build a Python package.  Poetry is more modern and provides many more tools out-of-the-box for better reproducability and package builds.  It is not common to use both together, however if we want a specific Python version we can get that with Conda and then manage our dependencies and package with Poetry.

# Setup Poetry in a Conda Environment

Here we will use `conda` for setting up with a specific Python version (3.9) and `poetry` for all package management. Note, we will not be using `conda` for package management to avoid getting the tools out of sync.  Always use `poetry` to install and update packages.

## Prerequisites

1. Anaconda3 or Miniconda3 installed ([Installation guide](https://docs.anaconda.com/anaconda/install/index.html)).

2. Updated `conda`.  To update `conda`, run the following from the `base` enviroment.  Elevated privileges may be necessary (e.g., an Admin console or `sudo`) if a permission error occurs.
    
    ```
    conda update -n base -c defaults conda
    ```

## Instructions

### Setup Poetry and Conda

1. Install `poetry` by following this [Installation guide](https://python-poetry.org/docs/#installation).

2. Configure `poetry` to use `conda` environment.

    ```
    poetry config virtualenvs.path $CONDA_ENV_PATH
    poetry config virtualenvs.create false
    ```

Where `$CONDA_ENV_PATH` is the path to the base `envs` folder (e.g., `/Users/myuser/anaconda3/envs`).

3. Create a Python 3.9 `conda` environment (here called `awesomeness`).
    
    ```
    conda create -n awesomeness python=3.9
    ```

4. Activate the `conda` environment.

    ```
    conda activate awesomeness
    ```

If you have a `pyproject.toml`, go to your project directory and install from the `pyproject.toml` package specification file.

    poetry install
    

> IMPORTANT:  Always use poetry to update package versions and install new packages and NOT `conda` otherwise the local environment and `poetry` (and thus the project's `pyproject.toml`) will be out of sync.

### Building the package

The `pyproject.toml` is also used to build packages.  To create it see the [Poetry documentation on the `pyproject.toml`](https://python-poetry.org/docs/pyproject/).

If a `pyproject.toml` already exists and the source code for the package, then the package can be built to be installed as standalone or pushed to PyPI for others to use.  To build a package:

1. Build the package by following [Packaging](https://python-poetry.org/docs/libraries/#packaging) (run from the base of the repo as it uses `pyproject.toml` to specify the package contents).  This places the package files into a folder called `dist`.
2. Activate the conda environment if it's not already and pip install the Python wheel (`.whl`) from the `dist` folder as follows:
    ```bash
    conda activate awesomeness
    pip install dist/awesome_package-0.1.0-py3-none-any.whl
    ```
   
### Updating the environment

To add packages with `poetry` follow this [official guide](https://python-poetry.org/docs/basic-usage/#installing-dependencies) (go to Installing with `poetry.lock`).  The `poetry.lock` file should not be commited to this project repo.

Note on JupyterLab: to get JupyterLab to work with Python 3.6 and Poetry, install as follows.

    poetry add traitlets@4.3.3
    poetry add jupyterlab@2.2.8
    

## References

1. [GitHub issue comment on installing Poetry and Conda to use together](https://github.com/python-poetry/poetry/issues/105#issuecomment-470829436).