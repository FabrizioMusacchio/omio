Installation
====================

OMIO targets Python 3.12 and higher and builds on the standard scientific Python stack
commonly used in microscopy and large scale image processing workflows. Core
dependencies include NumPy, tifffile, zarr, dask, napari, and related libraries
for metadata handling and image I/O.

Recommended Installation Method
--------------------------------- 

The recommended way to install OMIO for end users is via 
the `Python Package Index (PyPI) <https://pypi.org/project/omio-microscopy/>`_:

.. code-block:: python

   conda create -n omio python=3.12 -y
   conda activate omio
   pip install omio-microscopy


For Developers
-----------------------------------

For development work or reproducible analysis pipelines, it is often convenient
to install OMIO from source:

.. code-block:: python

   git clone https://github.com/FabrizioMusacchio/OMIO.git
   cd OMIO
   pip install .

Alternatively, OMIO can be installed directly from `GitHub <https://github.com/FabrizioMusacchio/OMIO>`_ 
without cloning the repository:

.. code-block:: python

   pip install git+https://github.com/FabrizioMusacchio/OMIO.git

If you plan to modify the code, use an editable installation:

.. code-block:: python

   pip install -e .

or, to include development dependencies such as testing and documentation tools:

.. code-block:: python

   pip install -e .[dev]

Avoid mixing local source folders and installed packages with the same name in
the same working directory, as this can lead to confusing import behavior and
unexpected imports during development.

