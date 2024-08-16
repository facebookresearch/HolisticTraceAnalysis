Installation
============

We recommend using a Conda environment to install HTA. To install Anaconda, see
`here <https://docs.anaconda.com/anaconda/install/index.html>`_. Holistic Trace
Analysis runs on Linux and Mac with Python >= 3.8.


**Setup a Conda environment**

.. code-block::

  # create the environment env_name
  conda create -n env_name

  # activate the environment
  conda activate env_name

  # deactivate the environment
  conda deactivate

**Installing Holistic Trace Analysis**

Install using pip

.. code-block::

   pip install HolisticTraceAnalysis

Install from source

.. code-block::

  # get the source code
  git clone https://github.com/facebookresearch/HolisticTraceAnalysis

  # move into the cloned directory
  cd HolisticTraceAnalysis

  # execute the command below from the root of the repo
  pip install -e .
