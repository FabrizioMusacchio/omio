# OMIO – Open Microscopy Image I/O

![GitHub Release](https://img.shields.io/github/v/release/FabrizioMusacchio/omio) [![PyPI version](https://img.shields.io/pypi/v/omio.svg)](https://pypi.org/project/omio/) [![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-green.svg)](https://omio.readthedocs.io/en/latest/overview.html#license) ![Tests](https://github.com/FabrizioMusacchio/omio/actions/workflows/omio_tests.yml/badge.svg) [![GitHub last commit](https://img.shields.io/github/last-commit/FabrizioMusacchio/omio)](https://github.com/FabrizioMusacchio/omio/commits/main/)  [![codecov](https://img.shields.io/codecov/c/github/FabrizioMusacchio/omio?logo=codecov)](https://codecov.io/gh/fabriziomusacchio/omio)  [![GitHub Issues Open](https://img.shields.io/github/issues/FabrizioMusacchio/omio)](https://github.com/FabrizioMusacchio/omio/issues) [![GitHub Issues Closed](https://img.shields.io/github/issues-closed/FabrizioMusacchio/omio?color=53c92e)](https://github.com/FabrizioMusacchio/omio/issues?q=is%3Aissue%20state%3Aclosed) [![GitHub Issues or Pull Requests](https://img.shields.io/github/issues-pr/FabrizioMusacchio/omio)](https://github.com/FabrizioMusacchio/omio/pulls)  [![Documentation Status](https://readthedocs.org/projects/omio/badge/?version=latest)](https://omio.readthedocs.io/en/latest/?badge=latest) ![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/fabriziomusacchio/omio) [![PyPI - Downloads](https://img.shields.io/pypi/dm/omio?logo=pypy&label=PiPY%20downloads&color=blue)](https://pypistats.org/packages/omio) [![PyPI Total Downloads](https://static.pepy.tech/personalized-badge/omio?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=BLUE&left_text=PiPY+total+downloads)](https://pepy.tech/projects/omio) [![Read the docs](https://badgen.net/badge/rtd/Documentation)](https://omio.readthedocs.io)  


**OMIO** (Open Microscopy Image I/O) is a policy-driven Python library for reading, organizing, merging, visualizing, and exporting multidimensional microscopy image data under explicit OME-compliant axis and metadata semantics.

OMIO is designed as an infrastructure layer between heterogeneous microscopy file formats and downstream analysis or visualization workflows. It provides a unified I/O interface that enforces consistent axis ordering, metadata normalization, and memory-aware data handling across NumPy, Zarr, Dask, napari, and OME-TIFF.

**NOTE 1:** OMIO is **currently under active development**. The API and feature set may change in future releases.

**NOTE 2:** A **documentation website** is being prepared and will be made available soon.


## What is OMIO good for your microscopy data?
Modern microscopy workflows face a recurring and largely unsolved problem: While image acquisition formats are diverse, downstream analysis and visualization pipelines implicitly assume consistent data semantics.

In practice, this leads to:

* ambiguous or undocumented axis conventions,
* silent shape mismatches across files,
* inconsistent or partially missing physical metadata,
* ad-hoc merge scripts that fail for large data,
* format-specific reader logic leaking into analysis code,
* and brittle visualization workflows for large volumetric or time-series data.

OME-TIFF and OME-XML define a powerful metadata standard, but **most real-world microscopy data do not arrive in a clean, OME-conform form**. Instead, users are left to bridge the gap manually, often repeatedly and inconsistently.

OMIO addresses this gap by acting as a **semantic I/O layer** rather than a format converter.

It provides:

* explicit and enforced axis semantics,
* structured metadata normalization,
* controlled merge and padding policies,
* memory-efficient handling of large datasets,
* and a consistent interface for both batch conversion and interactive visualization.


## Design principles
OMIO is built around the following principles:

### Explicit axis semantics
All image data are handled internally using explicit axis strings (default: `TZCYX`).  
Axis order is never implicit and never guessed silently.

### OME-aware, but not OME-exclusive
OME semantics are used as the **internal reference model**, but OMIO is not restricted to OME-TIFF input or output.  
OME-TIFF is treated as one well-defined sink among several possible representations.

### Policy-driven behavior
Operations such as merging, padding, and metadata reconciliation are governed by explicit, documented policies rather than hidden heuristics.

### Memory-aware by construction
Large datasets can be processed via Zarr and Dask without loading entire volumes into RAM.  
Chunk-aligned copying and cache-based workflows are first-class concepts and allow both memory-mapped and out-of-core processing as well as memory-efficient visualization in napari.

### Separation of concerns
Reading, merging, visualization, and writing are distinct stages that can be composed but are not entangled.

## Core features

### Unified image reading
OMIO provides a single entry point for reading microscopy image data. At the moment, supported formats include:

* TIFF / OME-TIFF
* LSM
* CZI
* Thorlabs RAW

All readers return:

* an image object (NumPy array or Zarr array),
* a normalized metadata dictionary,
* and an explicit axis specification.


### Metadata normalization and enforcement
OMIO normalizes metadata into a consistent dictionary that includes:

* axis string (`axes`)
* full 5D shape (`TZCYX`)
* physical pixel sizes (`PhysicalSizeX/Y/Z`)
* temporal resolution (`TimeIncrement`)
* units
* structured provenance stored via OME MapAnnotations

Non-OME metadata are preserved and stored explicitly inside annotations rather than discarded.

### Controlled merging along semantic axes
OMIO supports concatenation and merging along semantic axes:

* Time (`T`)
* Depth (`Z`)
* Channel (`C`)

Merge behavior is configurable:

* strict compatibility checks, or
* zero-padding of non-merge axes to maximal extents.

All merges propagate provenance information into metadata annotations.

### Image compression
OMIO supports reading and writing compressed image data. When writing OME-TIFF files, users can choose a zlib compression level between 0 (no compression) and 9 (maximum compression). I.e, OMIO can be used to shrink large uncompressed TIFF files into compressed OME-TIFF files, which facilitates significant disk space savings, faster I/O performance, and better transfer speeds.


### Folder-based and BIDS-like workflows
OMIO supports structured folder traversal for large projects, including:

* reading all files in a folder,
* merging multiple files within a folder,
* merging structured "folder stacks" (e.g. repeated acquisitions),
* batch processing of BIDS-like directory hierarchies.

These workflows are designed to reflect **how microscopy data are actually organized**, not how idealized examples look.


### OME-TIFF export
OMIO can write OME-TIFF files with:

* correct axis order,
* correct physical and temporal metadata,
* optional BigTIFF handling for large datasets,
* embedded MapAnnotations for provenance and custom metadata.

Filename handling is collision-safe and metadata-aware.


### Napari integration
OMIO provides direct integration with napari, supporting:

* NumPy-based visualization for small data,
* Zarr-backed visualization for large data,
* optional Dask-based slicing and caching,
* correct spatial scaling and channel handling.

Axis squeezing and cache generation are performed explicitly and transparently.

## Expected project structure (BIDS-like)
OMIO supports batch processing of projects organized in a BIDS-like manner.  
An abstract example is shown below:

```bash
project_name/
├─ sub-01/
│  ├─ exp_TP001/
│  │  ├─ image.tif
│  │
│  ├─ exp_TP002/
│  │  ├─ TAG_01/
│  │  │  ├─ image1.czi
│  │  ├─ TAG_02/
│  │  │  ├─ image2.czi
│  │
│  ├─ exp_TP003/
│  │  ├─ image1.tif
│  │  ├─ image2.tif
│
├─ sub-02/
│  ├─ exp_TP001/
│  │  ├─ image.tif
```

OMIO detects subjects, experiments, and optional tag folders via user-defined matching rules and applies consistent read and merge policies across the project.


## Typical usage patterns

### Read a single file

```python
img, metadata = omio.imread("image.czi")
```

### Read all images in a folder

```python
images, metadatas = omio.imread("experiment_folder", return_list=True)
```

### Merge multiple files along time

```python
merged_img, merged_md = omio.imread(
    "experiment_folder",
    merge_multiple_files_in_folder=True,
    merge_along_axis="T")
```

### Convert to OME-TIFF

```python
omio.convert_to_ometiff("experiment_folder")
```

### Batch conversion over a BIDS-like project

```python
omio.convert_bids_batch_to_ometiff(
    fname="project_root",
    sub="sub-",
    exp="TP",
    tagfolder="TAG_",
    merge_tagfolders=True)
```


## Scope and non-goals
OMIO intentionally does **not**:

* perform image processing or analysis,
* infer or guess missing metadata silently,
* replace domain-specific analysis pipelines.

Its purpose is to provide a **reliable, explicit, and reproducible I/O layer** on which such pipelines can be built.


## License
OMIO is released under the GNU General Public License v3.0 (GPLv3). Please refer to the [LICENSE](LICENSE) file for details.


## Citation
If you use OMIO in scientific work, please cite it appropriately. (CITATION.cff will be provided.)


## Acknowledgments
OMIO was developed to support real-world microscopy workflows where data heterogeneity, scale, and metadata inconsistencies are the norm rather than the exception.

