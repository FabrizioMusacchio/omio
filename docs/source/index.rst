.. omio documentation master file, created by
   sphinx-quickstart on Tue Dec 23 22:22:00 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

OMIO Documentation
==================

`**OMIO** <https://github.com/FabrizioMusacchio/omio>`_ (Open Microscopy Image I/O) is a policy-driven Python library 
for reading, organizing, merging, visualizing, and exporting multidimensional 
microscopy image data under explicit OME-compliant axis and metadata semantics.

OMIO is designed as an infrastructure layer between heterogeneous microscopy 
file formats and downstream analysis or visualization workflows. It provides 
a unified I/O interface that enforces consistent axis ordering, metadata 
normalization, and memory-aware data handling across NumPy, Zarr, Dask, 
napari, and OME-TIFF.

**NOTE:** OMIO is **currently under active development**. The API and 
feature set may change in future releases.

This documentation provides:

* an overview of the pipeline and its scientific context,
* installation and basic usage instructions,
* a detailed description of the processing steps and parameters,
* an automatically generated API reference.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   overview
   general_usage
   real_world_workflows
   api
   contributing

MotilA is `free and open-source software (FOSS) <https://en.wikipedia.org/wiki/Free_and_open-source_software>`_ 
distributed under the :ref:`GPL-3.0 license <license>`.

