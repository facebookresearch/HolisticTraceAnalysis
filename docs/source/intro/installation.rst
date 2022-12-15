Installation
============

We recommend using a Conda environment to install HTA. To install Anaconda, see
`here <https://docs.anaconda.com/anaconda/install/index.html>`_. Holistic Trace
Analysis runs on Linux and Mac with Python >= 3.8.

**Get the HTA source code**

.. code-block::

  git clone https://github.com/facebookresearch/HolisticTraceAnalysis.git

**Using a Conda environment**

.. code-block::

  # create the environment env_name
  conda create -n env_name

  # activate the environment
  conda activate env_name

  # deactivate the environment
  conda deactivate

**Installing Holistic Trace Analysis**

Execute the command below from the root of the repo

.. code-block::

   pip install -e .
