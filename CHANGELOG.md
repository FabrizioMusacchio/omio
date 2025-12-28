## OMIO Changelog

See here for a detailed list of changes made in each release of OMIO.
Please, also refer to the Repository [Releases page](https://github.com/FabrizioMusacchio/omio/releases).

Each release is also archived on Zenodo for long-term preservation and citation purposes:

[![Zenodo Archive](https://img.shields.io/badge/Zenodo%20Archive-10.5281%2Fzenodo.18030883-blue)](https://doi.org/10.5281/zenodo.18030883)

--- 

## 🚀 OMIO v0.1.4

This release focuses on improving documentation and usability.

### Summary of Changes
#### 📚 Citation and Archiving
* OMIO releases are now linked to [Zenodo](https://zenodo.org/records/18030883), enabling long-term archiving and versioned software snapshots.
* A Zenodo DOI ([10.5281/zenodo.18030883](https://zenodo.org/records/18030883)) is associated with the project, making OMIO formally citable in scientific publications.
* Citation metadata has been added to the repository to document the preferred citation form.

#### 📖 Documentation Updates
* The README has been revised to correct and clarify several example usage snippets.
* Example code now reflects the current public API and recommended usage patterns more accurately.

#### 🔎 Notes
This release focuses on establishing a stable citation and archiving workflow and on improving the reliability of user-facing documentation. No changes to the core API or reader behavior were introduced.

--- 

## 🚀 OMIO v0.1.3

is just a dummy release for connecting the repository to Zenodo.

---

## 🚀 OMIO v0.1.2

This release is a small maintenance update.

### Summary of Changes
#### 🧩 Fixed
* Correctly resolve the installed package version at runtime when OMIO is distributed under the PyPI name **omio-microscopy** while being imported as `omio`.
* Ensure the reported OMIO version now matches the version defined in `pyproject.toml`.

#### 🧪 Quality
* All existing tests pass with the corrected version handling.
* No API or behavior changes for users beyond the version fix.

This release prepares OMIO for stable use via `pip install omio-microscopy` while keeping the familiar `import omio` interface.


---

## 🚀 OMIO v0.1.1

This is the first public release of **OMIO (Open Microscopy Image I/O)**, providing a unified, reproducible, and OME-compliant image loading layer for bioimaging and microscopy data.

### Summary of Changes
#### ✨ Highlights
OMIO v0.1.1 establishes the core design principles of the project: a single, canonical in-memory representation for microscopy images and metadata, explicit handling of OME axes, and robust support for large datasets via Zarr.

#### 🧠 Core Functionality
* Unified image reading interface for common microscopy formats, including TIFF, OME-TIFF, LSM, CZI, and Thorlabs RAW.
* Canonical internal image representation using the OME axis order **TZCYX**.
* Automatic axis normalization, validation, and correction based on file metadata.
* Consistent metadata handling aligned with OME concepts, including physical pixel sizes, time increments, and axis annotations.
* Explicit provenance tracking of original filenames, file types, and metadata sources.

#### 🔬 Thorlabs RAW Support
* Native reading of Thorlabs RAW files using accompanying XML metadata.
* YAML metadata fallback when XML metadata is unavailable, enabling reproducible interpretation of legacy or incomplete datasets.
* Automatic correction of Z dimension inconsistencies based on RAW file size.
* Optional memory-efficient Zarr output for large RAW datasets, with slice-wise copying to limit peak RAM usage.

#### 📦 Zarr Integration
* Optional output as NumPy arrays or Zarr arrays (in-memory or on-disk).
* Automatic chunk size computation based on image shape and axis order.
* Incremental writing strategies to support large files and interactive environments.

#### 👁️ Napari Integration
* Built-in Napari viewer utilities for interactive inspection of OMIO-loaded images.
* Automatic handling of OME axes and dimensionality for Napari display.
* Support for efficient visualization of large Zarr-backed datasets without full materialization in memory.

#### 🔗 Merging and Utilities
* Concatenation of compatible 5D image stacks along selected OME axes.
* Optional zero-padding to merge datasets with mismatched non-merge dimensions.
* Robust handling of filename collisions and metadata provenance during merge operations.
* Helper utilities for Zarr group inspection, metadata recovery, and axis consistency checks.

#### 🧪 Testing and Robustness
* Extensive automated test coverage across readers, edge cases, and failure modes.
* Synthetic test data for RAW and TIFF paths, complemented by small CC BY 4.0 test images for CZI and LSM formats.
* Clear warning and error behavior for incomplete metadata, unsupported configurations, and inconsistent inputs.

#### 📦 Packaging
* First PyPI release under the distribution name **omio-microscopy**.
* Importable Python package name remains **omio**.
* Python 3.12 or newer required.

#### 🔭 Scope and Outlook
This release focuses on correctness, transparency, and reproducibility rather than maximal format coverage. OMIO is designed as a stable foundation for downstream analysis pipelines, where consistent axis semantics and metadata integrity are critical.

Future releases will expand format support, refine metadata policies, and further improve performance and interoperability with downstream bioimaging tools.