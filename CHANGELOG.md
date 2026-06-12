## OMIO Changelog

See here for a detailed list of changes made in each release of OMIO.
Please, also refer to the Repository [Releases page](https://github.com/FabrizioMusacchio/omio/releases).

Each release is also archived on Zenodo for long-term preservation and citation purposes:

[![Zenodo Archive](https://img.shields.io/badge/Zenodo%20Archive-10.5281%2Fzenodo.18030883-blue)](https://doi.org/10.5281/zenodo.18030883)


---

## 🚀 OMIO v0.2.2

June 12, 2026

This maintenance release adds validated reuse of OMIO's on-disk Zarr caches and stores enough OMIO metadata inside each cache to reopen compatible datasets without rereading the original microscopy file.

### 📃 Changes
#### ✨ Added
* Added opt-in disk-cache reuse for `imread(..., zarr_store="disk", reuse_disk_cache=True)`.
* Compatible existing `.omio_cache/<basename>.zarr` stores can now be reopened directly instead of being rebuilt from the source file.
* Added persisted OMIO cache manifests for disk-backed TIFF, CZI, and Thorlabs RAW reads.
* Added persisted OMIO metadata payloads for disk-backed TIFF, CZI, and Thorlabs RAW reads, enabling cache reuse without reopening the original source image for metadata extraction.

#### 🧩 Changed
* OMIO now stores disk-cache metadata and cache validation info directly in Zarr attributes, which are persisted in the store's `zarr.json` under the current Zarr v3 layout.
* Disk-cache reuse is validated against source path, file size, modification time, OMIO version, reader/backend identity, pixel unit, and any explicit physical-size overrides before reuse is allowed.
* If a disk cache is missing OMIO metadata, missing cache info, stale, or otherwise incompatible, OMIO now falls back automatically to a full rebuild and refreshes the cache in place.

#### 🧪 Testing and robustness
* Added regression tests covering persisted disk-cache metadata payloads for TIFF-backed Zarr caches.
* Added regression tests verifying that TIFF, CZI, and Thorlabs RAW readers can reopen compatible disk caches without rereading the original source files.
* Added regression tests covering fallback behavior when a stale or incomplete disk cache is encountered.

#### 📚 Documentation and citation
* OMIO is now described in a dedicated preprint that can be cited in academic work:
  * DOI: [10.64898/2026.06.09.731118](https://doi.org/10.64898/2026.06.09.731118)


## 🚀 OMIO v0.2.1

May 13, 2026

This maintenance release restores compatibility with recent `czifile`, `tifffile`, and `zarr`/Dask dependency revisions and keeps OMIO's CZI, TIFF, conversion, and napari-cache workflows working across both legacy and current APIs.

### 📃 Changes
#### 🧩 Fixed
* Restored CZI reader compatibility with `czifile` `2026.4.30` and newer.
* `read_czi` now resolves axes via scene metadata when `CziFile.axes` is no longer available.
* `read_czi` now retrieves structured CZI metadata via `metadata(asdict=True)` when the older `metadata(raw=False)` API is unavailable.
* Compatibility is implemented via feature detection instead of a hard dependency-version check, preserving support for older `czifile` releases.
* Restored TIFF axis handling for files where recent `tifffile` revisions collapse singleton dimensions in `series[0].axes` but still preserve the full shape in `shaped_metadata`.
* `read_tif` now uses `shaped_metadata` as a fallback axis source when `tifffile` drops singleton axes such as `T=1` or `Z=1`.
* Restored compatibility with current `dask.array.to_zarr` / `zarr` behavior by switching napari-cache writes away from the deprecated `zarr_read_kwargs` path and to the current `mode="w"` API.

#### 🧪 Testing and robustness
* Added regression tests covering both legacy and current `czifile` CZI-reader APIs.
* Added regression tests covering TIFF axis fallback from `shaped_metadata` when singleton dimensions are omitted from `series[0].axes`.
* Full `pytest` suite passes again under the current dependency set.

#### 🎨 Project identity
* OMIO has its first own logo now! (December 30, 2025) That's cool, but does not affect functionality or justify a dedicated release on its own.



--- 

## 🚀 OMIO v0.2.0

December 29, 2025

This release introduces a more consistent public API, improves TIFF and OME-TIFF handling (including multi-file OME-TIFF series and paginated stacks), strengthens napari visualization robustness, and significantly expands documentation and example data.

### 📃 Changes
#### ✨ Highlights

* API consolidation: `write_ometiff` has been renamed to `imwrite` to align with `imread` and `imconvert`.
* Improved TIFF family robustness: better physical pixel size handling, clearer container policies, and correct behavior for multi-file OME-TIFF series.
* More robust napari visualization: clearer viewer summaries and safeguards against accidental loss of spatial axes.
* Major documentation expansion and a Zenodo-hosted example dataset for tutorials and testing.

#### ⚠️ Breaking changes
* `write_ometiff` → `imwrite`
  * Rationale: improves naming consistency across the core API (compared to `imread` and `imconvert`).
  * Migration: replace `write_ometiff(...)` with `imwrite(...)`.


#### 🧬 TIFF and LSM reading improvements
* `read_tif` now emits explicit warnings when `PhysicalSizeX`, `PhysicalSizeY`, or `PhysicalSizeZ` cannot be read from metadata and default or user-provided values are used instead.
* Improved fallback extraction of physical pixel sizes from TIFF tags when `imagej_metadata` is incomplete.
  * `_standardize_imagej_metadata` has been extended accordingly.
* Metadata inspection logic refined.
  * `shaped_metadata` is now ignored in "not yet implemented metadata types" checks, as it typically contains only shape information.
* README and `read_tif` docstrings now explicitly document support for multi-file OME-TIFF series.
  * Passing the path of a single file is sufficient, as OMIO reconstructs the full logical dataset via OME-XML references.

#### 📁 Folder reading and OME-TIFF series detection
* `imread` now correctly detects multi-file OME-TIFF series when a folder path is provided.
  * Previous behavior could incorrectly treat all TIFF files in a folder as independent images. This is now fixed.
* The same fix propagates to `imconvert` and `bids_batch_convert`.


#### 🆕 New utility function
* Added `create_thorlabs_raw_yaml`.
  * Allows users to generate an empty `experiment.yaml` template for Thorlabs RAW folders when `Experiment.xml` is missing.

#### 👁️ Napari visualization updates
* Improved the final status message of the napari opener.
  * Now prints a concise summary including layer names, scales, and shapes.
* Added internal safety checks to prevent spatial axes `X` and `Y` from being squeezed away when their dimension equals 1.

#### 🛠️ Utilities and tests
* `test_all_readers_with_dummy_data.py` now generates more informative dummy data.
  * Dummy TIFF files include text annotations.
  * Additional folder structures are created to demonstrate batch processing and folder handling behavior.

#### 📚 Documentation
Expanded and reorganized documentation, including:

* A Core Workflow guide covering reading, inspecting, viewing, and writing images, with examples for `imread`, `imwrite`, and `imconvert`.
* A detailed overview of supported formats (LSM, CZI, Thorlabs RAW) with usage examples.
* Clarified TIFF container policies:
  * multi-series TIFF stacks
  * paginated TIFF stacks
  * multi-file OME-TIFF series
* Documentation of folder reading semantics in `imread`, including tagged folders and folder stacks.
* Guidance on large file handling using Zarr-backed lazy loading and memory mapping, including Dask-based napari visualization.
* A section on creating empty images and metadata with utilities for OME-compliant structures.
* A new section on batch conversion over a BIDS-like tree using `bids_batch_convert`.


#### 🧪 Example dataset
* Added a Zenodo-hosted example dataset containing artificially generated toy data and selected publicly available real-world microscopy data for tutorials and testing.
  * DOI: [10.5281/zenodo.18078231](https://doi.org/10.5281/zenodo.18078231)

#### 📝 Notes for maintainers
* Verify that all documentation and examples consistently use `imwrite`.
* Ensure that references to `write_ometiff` are removed or updated.
* Highlight the API rename prominently in upgrade notes and downstream documentation.


---

## 🚀 OMIO v0.1.4

Dec 23, 2025

This release focuses on improving documentation and usability.

### 📃 Changes
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

Dec 23, 2025

This is just a dummy release for connecting the repository to Zenodo.

---

## 🚀 OMIO v0.1.2

Dec 23, 2025

This release is a small maintenance update.

### 📃 Changes
#### 🧩 Fixed
* Correctly resolve the installed package version at runtime when OMIO is distributed under the PyPI name **omio-microscopy** while being imported as `omio`.
* Ensure the reported OMIO version now matches the version defined in `pyproject.toml`.

#### 🧪 Quality
* All existing tests pass with the corrected version handling.
* No API or behavior changes for users beyond the version fix.

This release prepares OMIO for stable use via `pip install omio-microscopy` while keeping the familiar `import omio` interface.


---

## 🚀 OMIO v0.1.1

Dec 23, 2025

This is the first public release of **OMIO (Open Microscopy Image I/O)**, providing a unified, reproducible, and OME-compliant image loading layer for bioimaging and microscopy data.

### 📃 Changes
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
