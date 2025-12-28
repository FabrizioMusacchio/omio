OMIO Documentation
==================

.. image:: https://badgen.net/badge/icon/GitHub%20repository?icon=github&label
   :target: https://github.com/FabrizioMusacchio/omio/
   :alt: GitHub Repository

.. image:: https://img.shields.io/github/v/release/FabrizioMusacchio/omio
   :alt: GitHub Release

.. image:: https://img.shields.io/pypi/v/omio-microscopy.svg
   :target: https://pypi.org/project/omio-microscopy/
   :alt: PyPI version

.. image:: https://img.shields.io/badge/License-GPL%20v3-green.svg
   :target: https://omio.readthedocs.io/en/latest/overview.html#license
   :alt: GPLv3 License

.. image:: https://github.com/FabrizioMusacchio/omio/actions/workflows/omio_tests.yml/badge.svg
   :alt: Tests

.. image:: https://img.shields.io/github/last-commit/FabrizioMusacchio/omio
   :target: https://github.com/FabrizioMusacchio/omio/commits/main/
   :alt: GitHub last commit

.. image:: https://img.shields.io/codecov/c/github/FabrizioMusacchio/omio?logo=codecov
   :target: https://codecov.io/gh/fabriziomusacchio/omio
   :alt: codecov

.. image:: https://img.shields.io/github/issues/FabrizioMusacchio/omio
   :target: https://github.com/FabrizioMusacchio/omio/issues
   :alt: GitHub Issues Open

.. image:: https://img.shields.io/github/issues-closed/FabrizioMusacchio/omio?color=53c92e
   :target: https://github.com/FabrizioMusacchio/omio/issues?q=is%3Aissue%20state%3Aclosed
   :alt: GitHub Issues Closed

.. image:: https://img.shields.io/github/issues-pr/FabrizioMusacchio/omio
   :target: https://github.com/FabrizioMusacchio/omio/pulls
   :alt: GitHub Issues or Pull Requests

.. image:: https://readthedocs.org/projects/omio/badge/?version=latest
   :target: https://omio.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://img.shields.io/github/languages/code-size/fabriziomusacchio/omio
   :alt: GitHub code size in bytes

.. image:: https://img.shields.io/pypi/dm/omio-microscopy?logo=pypy&label=PiPY%20downloads&color=blue
   :target: https://pypistats.org/packages/omio-microscopy
   :alt: PyPI Downloads

.. image:: https://static.pepy.tech/personalized-badge/omio-microscopy?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=BLUE&left_text=PiPY+total+downloads
   :target: https://pepy.tech/projects/omio-microscopy
   :alt: PyPI Total Downloads

.. image:: https://img.shields.io/badge/Example%20Datasets-10.5281%2Fzenodo.18078231-blue
   :target: https://doi.org/10.5281/zenodo.18078231
   :alt: Example Datasets on Zenodo

.. image:: https://img.shields.io/badge/Zenodo%20Archive-10.5281%2Fzenodo.18030883-blue
   :target: https://doi.org/10.5281/zenodo.18030883
   :alt: Zenodo Archive

`OMIO (Open Microscopy Image I/O) <https://github.com/FabrizioMusacchio/omio>`_ is a policy-driven Python library 
for reading, organizing, merging, visualizing, and exporting multidimensional 
microscopy image data under explicit OME-compliant axis and metadata semantics.

OMIO is designed as an infrastructure layer between heterogeneous microscopy 
file formats and downstream analysis or visualization workflows. It provides 
a unified I/O interface that enforces consistent axis ordering, metadata 
normalization, and memory-aware data handling across NumPy, Zarr, Dask, 
napari, and OME-TIFF.

.. note::
   **NOTE:** OMIO is **currently under active development**. The API and 
   feature set may change in future releases. We also welcome feedback, feature 
   requests, and contributions via `GitHub issues <https://github.com/FabrizioMusacchio/omio/issues>`_. 
   Please report any bugs or inconsistencies you encounter.


.. toctree::
   :maxdepth: 3
   :caption: Contents

   overview
   installation
   usage
   api
   changelog
   contributing

`OMIO <https://github.com/FabrizioMusacchio/omio>`_ is `free and open-source software (FOSS) <https://en.wikipedia.org/wiki/Free_and_open-source_software>`_ 
distributed under the :ref:`GPL-3.0 license <license>`.

