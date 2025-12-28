Example Data Sets
=====================

In order to follow along with the examples provided in this documentation,
you may download the following example data sets from Zenodo:

.. image:: https://img.shields.io/badge/Example%20Datasets-10.5281%2Fzenodo.18078231-blue
   :target: https://doi.org/10.5281/zenodo.18078231
   :alt: Example Datasets on Zenodo


These example data sets are provided to facilitate testing, validation, and
demonstration of OMIO's image reading, metadata parsing, axis normalization,
and batch handling capabilities.

The Zenodo record contains two categories of data:

* **OMIO-generated dummy data**, consisting of small, artificially generated
  TIFF, OME-TIFF, and RAW files designed specifically for software testing.
  These files do not represent real microscopy measurements and must not be
  used for scientific analysis.
* **Selected third-party example files** obtained from public repositories.
  These files are included to test OMIO against realistic file formats and
  metadata structures encountered in practice (for example CZI, LSM, and
  multi-channel TIFF files).

Licensing and attribution
----------------------------

The Zenodo record is a collection. Individual files and subfolders may be
licensed differently:

* OMIO-generated dummy data and the accompanying generation scripts are released
  under the license specified for the Zenodo record.
* Third-party example files are redistributed under their original licenses
  (for example CC BY 4.0 or CC0 1.0).

Each third-party file is stored in its own subfolder within the Zenodo archive
together with a dedicated README file documenting the original source,
license, citation requirements, and modification status.

If you use any third-party file beyond basic OMIO input/output testing,
please cite the original source listed in the corresponding README file.