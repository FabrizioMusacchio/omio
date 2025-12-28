OMIO Documentation
==================

`OMIO (Open Microscopy Image I/O) <https://github.com/FabrizioMusacchio/omio>`_ is a policy-driven Python library 
for reading, organizing, merging, visualizing, and exporting multidimensional 
microscopy image data under explicit OME-compliant axis and metadata semantics.

OMIO is designed as an infrastructure layer between heterogeneous microscopy 
file formats and downstream analysis or visualization workflows. It provides 
a unified I/O interface that enforces consistent axis ordering, metadata 
normalization, and memory-aware data handling across NumPy, Zarr, Dask, 
napari, and OME-TIFF.

**NOTE:** OMIO is **currently under active development**. The API and 
feature set may change in future releases. We also welcome feedback, feature 
requests, and contributions via `GitHub issues <https://github.com/FabrizioMusacchio/omio/issues>`_. 
Please report any bugs or inconsistencies you encounter.


.. toctree::
   :maxdepth: 3
   :caption: Contents:

   overview
   installation
   usage
   api
   changelog
   contributing

`OMIO <https://github.com/FabrizioMusacchio/omio>`_ is `free and open-source software (FOSS) <https://en.wikipedia.org/wiki/Free_and_open-source_software>`_ 
distributed under the :ref:`GPL-3.0 license <license>`.

