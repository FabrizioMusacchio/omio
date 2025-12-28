Overview
========

OMIO (Open Microscopy Image I/O) is a policy driven Python library for reading,
organizing, merging, visualizing, and exporting multidimensional microscopy image
data under explicit OME compliant axis and metadata semantics.

OMIO is designed as an infrastructure layer between heterogeneous microscopy file
formats and downstream analysis or visualization workflows. It provides a unified
I/O interface that enforces consistent axis ordering, metadata normalization, and
memory aware data handling across NumPy, Zarr, Dask, napari, and OME TIFF.

The project is under active development. While the core concepts are stable, the
public API and feature set may evolve over time.


Motivation and problem statement
--------------------------------

Modern microscopy workflows face a recurring and largely unsolved problem. While
image acquisition formats are diverse and vendor specific, downstream analysis
and visualization pipelines implicitly assume consistent data semantics.

In practice, this leads to:

* ambiguous or undocumented axis conventions
* silent shape mismatches across files
* inconsistent or partially missing physical metadata
* ad hoc merge scripts that fail for large data
* format specific reader logic leaking into analysis code
* brittle visualization workflows for large volumetric or time series data

OME TIFF and OME XML define a powerful metadata standard, but most real world
microscopy data do not arrive in a clean OME conform form. Instead, users are left
to bridge the gap manually, often repeatedly and inconsistently.

OMIO addresses this gap by acting as a semantic I/O layer rather than a simple
format converter.


Design principles
-----------------

Explicit axis semantics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All image data handled by OMIO carry an explicit axis string, with the default
internal convention being ``TZCYX``. Axis order is never implicit and never guessed
silently.

OME aware, but not OME exclusive
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

OME semantics serve as the internal reference model for OMIO, but the library is
not restricted to OME TIFF input or output. OME TIFF is treated as one well defined
sink among several possible representations.

Policy driven behavior
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Operations such as merging, padding, and metadata reconciliation are governed by
explicit and documented policies rather than hidden heuristics.

Memory aware by construction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Large datasets can be processed via Zarr and Dask without loading entire volumes
into memory. Chunk aligned copying, cache based workflows, and memory mapped
access are first class concepts and allow both out of core processing and
interactive visualization in napari.

Separation of concerns
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Reading, merging, visualization, and writing are treated as distinct stages that
can be composed flexibly but are not entangled.


Core functionality
------------------

Unified image reading
^^^^^^^^^^^^^^^^^^^^^^^^^^

OMIO provides a single entry point for reading microscopy image data from files or
folders. Supported formats include common TIFF based formats, Zeiss LSM and CZI,
and Thorlabs RAW files.

All readers return:

* an image object, either a NumPy array or a Zarr array
* a normalized metadata dictionary
* an explicit axis specification

Metadata normalization and enforcement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Metadata are normalized into a structured dictionary that includes axis
information, full five dimensional shape, physical pixel sizes, temporal
resolution, units, and structured provenance stored via OME MapAnnotations.

Non OME metadata are preserved and stored explicitly rather than discarded.

Controlled merging along semantic axes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

OMIO supports concatenation and merging along semantic axes such as time, depth,
and channel. Merge behavior is configurable and can enforce strict compatibility
or allow zero padding of non merge axes to maximal extents. All merges propagate
provenance information into metadata annotations.

Folder based and BIDS like workflows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

OMIO supports structured folder traversal for large projects, including reading
all files in a folder, merging multiple files within a folder, merging structured
folder stacks, and batch processing of BIDS like directory hierarchies. These
workflows reflect how microscopy data are commonly organized in practice.

OME TIFF export
^^^^^^^^^^^^^^^^^^^^^

OMIO can write OME TIFF files with correct axis order, physical and temporal
metadata, optional BigTIFF handling for large datasets, and embedded
MapAnnotations for provenance and custom metadata.

Napari integration
^^^^^^^^^^^^^^^^^^^^^^^^

OMIO integrates directly with napari, supporting NumPy based visualization for
small data, Zarr backed visualization for large data, and correct spatial scaling
and channel handling. Axis squeezing and cache generation are performed explicitly
and transparently.

Creating metadata and image templates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

OMIO provides utility functions to create empty metadata and image templates
that can be populated programmatically or used as blueprints for new datasets.


Typical usage patterns
----------------------

OMIO exposes a small set of core functions that cover most workflows:

* ``imread()`` for reading images from files or folders
* ``imconvert()`` for converting images to OME TIFF
* ``imwrite()`` for writing images to OME TIFF
* ``bids_batch_convert()`` for batch conversion of BIDS like projects
* ``open_in_napari()`` for interactive visualization
* utility functions for metadata and image template handling

These functions are designed to be composable and to keep I/O concerns separate
from downstream analysis logic.


Scope and non goals
-------------------

OMIO intentionally does not perform image processing or analysis, does not infer
or guess missing metadata silently, and does not replace domain specific analysis
pipelines.

Its purpose is to provide a reliable, explicit, and reproducible I/O layer on
which such pipelines can be built.


Installation
------------

Please refer to the `installation instructions <installation>`_ for details on how to
install OMIO and its dependencies.


Further reading
---------------

Detailed usage examples, API documentation, and contribution guidelines are
available in the remaining sections of this documentation. 


.. _license:

License
-------

OMIO is distributed under the terms of the `GNU General Public License v3.0 (GPL-3.0) <https://github.com/FabrizioMusacchio/omio?tab=GPL-3.0-1-ov-file>`_.

In summary, users are permitted to

* **use** the software for any purpose  
* **modify** the source code and adapt it to their needs
* **redistribute** the original or modified code

Under the following conditions:

* **Copyleft** applies. Modifications must be released under the same GPL-3.0 license.  
* The **original copyright notice and license** must be preserved.

Not permitted:

* Use of OMIO in **proprietary or closed-source** applications  
* Redistribution of modified versions under more restrictive terms  

OMIO is provided **without any warranty**, including implied warranties of merchantability
or fitness for a particular purpose.

For full license terms, see the ``LICENSE`` file in the `repository <https://github.com/FabrizioMusacchio/omio?tab=GPL-3.0-1-ov-file>`_ or  
`https://www.gnu.org/licenses/gpl-3.0.html <https://www.gnu.org/licenses/gpl-3.0.html>`_.


Citation
--------

If you use OMIO in your research, please cite it as:

   Fabrizio Musacchio. (2025). OMIO: A Python library for unified 
   I/O of multidimensional microscopy images. Zenodo. 
   `https://doi.org/10.5281/zenodo.18030883 <https://doi.org/10.5281/zenodo.18030883>`_

On Zenodo, you can select other citation formats as needed as well 
as a DOI for dedicated OMIO software versions.

Acknowledgements
----------------

OMIO was developed to support real-world microscopy workflows where data 
heterogeneity, scale, and metadata inconsistencies are the norm rather 
than the exception.

For questions, suggestions or bug reports, please refer to the
`GitHub issue tracker <https://github.com/FabrizioMusacchio/OMIO/issues>`_ of 
the `OMIO repository <https://github.com/FabrizioMusacchio/OMIO>`_ or contact the maintainer 
directly:

| **Fabrizio Musacchio**: `Email <mailto:fabrizio.musacchio@dzne.de>`_ | `GitHub <https://github.com/FabrizioMusacchio>`_ | `Website <https://www.fabriziomusacchio.com>`_

