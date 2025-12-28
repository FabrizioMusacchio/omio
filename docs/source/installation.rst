Installation
====================

OMIO targets Python 3.12 and higher and builds on the standard scientific Python stack
commonly used in microscopy and large scale image processing workflows. Core
dependencies include NumPy, tifffile, zarr, dask, napari, and related libraries
for metadata handling and image I/O.

Recommended installation method
--------------------------------- 

The recommended way to install OMIO for end users is via the Python Package Index
(PyPI):

.. code-block:: bash

   conda create -n omio python=3.12 -y
   conda activate omio
   pip install omio-microscopy


For Developers
-----------------------------------

For development work or reproducible analysis pipelines, it is often convenient
to install OMIO from source:

.. code-block:: bash

   git clone https://github.com/FabrizioMusacchio/OMIO.git
   cd OMIO
   pip install .

Alternatively, OMIO can be installed directly from GitHub without cloning the
repository:

.. code-block:: bash

   pip install git+https://github.com/FabrizioMusacchio/OMIO.git

If you plan to modify the code, use an editable installation:

.. code-block:: bash

   pip install -e .

Avoid mixing local source folders and installed packages with the same name in
the same working directory, as this can lead to confusing import behavior and
unexpected imports during development.

