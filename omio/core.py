"""
OMIO – Open Microscopy Image I/O and OME-TIFF conversion utilities.

OMIO, acronym for Open Microscopy Image I/O, is a lightweight, research-oriented 
Python module that provides a unified interface for reading, normalizing, merging, 
visualizing, and writing multi-dimensional microscopy image data in OME-compliant 
formats. It is designed as a practical glue layer between heterogeneous microscopy 
file formats and downstream analysis or visualization tools, with a strong emphasis 
on reproducible axis semantics, metadata integrity, and memory-aware workflows.

Scope and design goals
----------------------
OMIO addresses common pain points in microscopy data handling:

* Reading heterogeneous microscopy formats (TIFF, OME-TIFF, LSM, CZI, Thorlabs RAW)
  through a single entry point.
* Enforcing a strict, explicit OME axis convention (TZCYX) internally, without
  silently repairing incompatible data.
* Normalizing and validating metadata so that physical pixel sizes, time
  increments, and axis sizes remain consistent and explicit.
* Providing controlled merge operations along selected axes (T, Z, or C), with
  well-defined policies for strict compatibility checks versus zero-padding.
* Supporting both NumPy-based in-memory workflows and Zarr-based, chunked,
  memory-efficient workflows for large datasets.
* Enabling direct visualization in napari, including scale-aware display and
  channel handling.
* Writing standards-compliant OME-TIFF output suitable for ImageJ, Fiji, napari,
  and downstream quantitative pipelines.

OMIO deliberately does not aim to replace format-specific libraries. Instead, it
orchestrates them under a consistent policy layer that makes assumptions explicit
and reproducible.

Core functionality overview
---------------------------
The module is structured around a small set of high-level entry points, supported
by internal helper utilities:

* imread
    Universal reader that accepts files, folders, or folder stacks and returns
    NumPy or Zarr arrays together with validated OME-style metadata. Supports
    optional merging across files or folder stacks along a user-defined axis.

* imwrite
    OME-TIFF writer that enforces axis order, handles BigTIFF decisions, embeds
    physical scale metadata, and preserves provenance via OME MapAnnotations.

* imconvert
    End-to-end converter that combines imread and imwrite to transform
    arbitrary supported input data into OME-TIFF with minimal boilerplate.

* bids_batch_convert
    Batch-level converter operating on a BIDS-like directory hierarchy, supporting
    subject and experiment discovery, optional tagfolder logic, and controlled
    merging policies.

* open_in_napari
    Convenience interface for opening OMIO-handled data directly in napari,
    supporting NumPy, Zarr, and Zarr+Dask backends with correct spatial scaling.

Axis and metadata policy
------------------------
Internally, OMIO assumes a strict five-dimensional axis order:

    T Z C Y X

All merge, validation, and write operations rely on this convention. Axes are not
implicitly inferred or repaired beyond explicit user requests. Metadata fields
such as PhysicalSizeX/Y/Z and TimeIncrement are treated as first-class quantities
and are validated and propagated consistently across merges and conversions.

Merging semantics are intentionally conservative: incompatible inputs trigger
warnings and abort the merge unless zero-padding is explicitly enabled.

Intended audience and use cases
-------------------------------
OMIO is intended for researchers working with multi-dimensional microscopy data
who need a transparent and scriptable way to:

* Convert legacy or vendor-specific formats into OME-TIFF.
* Assemble time series, z-stacks, or channel stacks from multiple acquisitions.
* Prepare large datasets for downstream analysis without exceeding memory limits.
* Maintain explicit provenance and metadata across preprocessing steps.

The module favors clarity and explicit policy over aggressive automation, and is
therefore best suited for controlled analysis pipelines rather than black-box
end-user tools.

Author
-------
Author: Fabrizio Musacchio  
First version: December 2025
Ported to modularized structure: August 2026

This module is part of the OMIO project and is developed in the context of
scientific microscopy data processing workflows.
"""
# %% IMPORTS
import os, re
import hashlib
from copy import deepcopy

from importlib.metadata import version, PackageNotFoundError, packages_distributions

import glob
from tabnanny import verbose
import warnings
from typing import Any, Dict, List, Tuple, Union

import shutil
import xml.etree.ElementTree as ET
import numpy as np
import napari
import tifffile
import czifile as czi
import datetime
import zlib
import xml.etree.ElementTree as ET
import zarr
from tqdm import tqdm
import dask.array as da
import yaml
# %% MODULE-SCOPE GLOBALS
def _resolve_omio_version() -> str:
    # primary: known PyPI distribution name
    try:
        return version("omio-microscopy")
    except PackageNotFoundError:
        pass

    # fallback: map import package -> installed distribution(s)
    try:
        dist_names = packages_distributions().get("omio", [])
        for dist in dist_names:
            try:
                return version(dist)
            except PackageNotFoundError:
                continue
    except Exception:
        pass

    return "0.0.0+unknown"
_OMIO_VERSION = _resolve_omio_version()

_OME_AXES = "TZCYX" # this is the canonical OME axes order. DO NOT CHANGE!
_AXIS_TO_INDEX = {"T": 0, "Z": 1, "C": 2, "Y": 3, "X": 4} # DO NOT CHANGE!
_ALLOWED_MERGE_AXES = {"T", "Z", "C"}
_CACHE_SCHEMA_VERSION = 1

# make current _OMIO_VERSION available as 'version' attribute outside the module:
version = _OMIO_VERSION
# %% HELPER FUNCTIONS FOR READERS

def hello_world():
    """
    Print a simple sanity-check message including the current OMIO version.

    This function is intended as a minimal diagnostic utility to verify that
    the OMIO package can be imported correctly, that external dependencies are
    resolved, and that the module-level version variable is accessible at
    runtime. It has no return value and produces output only via standard
    output.

    Side effects
    ------------
    Prints a message of the form:
        "Hello from omio.py! OMIO version: <version>"
    """
    print("Hello from omio.py! OMIO version:", _OMIO_VERSION)

def _reorder_numpy(arr, axes_string, OME_axes, OME_axes_order):
    """
    Reorder a NumPy array into OME-compliant axis order (TZCYX).

    This helper performs the minimal, strictly in-RAM axis-normalization step used
    in the NumPy branch of `_correct_for_OME_axes_order`. It takes an input array
    together with its declared axis string and returns a new NumPy array where:

    * all OME axes (T, Z, C, Y, X) are present,
    * any missing axes are appended as singleton dimensions in the order defined
      by the global OME axes sequence,
    * the array is then permuted into the canonical OME axis order TZCYX.

    The function assumes that `OME_axes` and `OME_axes_order` are defined in the
    surrounding module scope. It does not alter metadata; it operates purely on
    the numerical array representation.

    Parameters
    ----------
    arr : np.ndarray
        The image array whose axes are described by `axes_string`.
    axes_string : str
        Axis declaration for `arr`, using characters from {T, Z, C, Y, X}.
        Its length must match `arr.ndim`. Axes missing from this declaration
        will be created as singleton dimensions and appended at the end.

    Returns
    -------
    np.ndarray
        A NumPy array with all OME axes present and ordered as TZCYX.

    Notes
    -----
    * This function is intended only for cases where the full array fits in RAM.
      For Zarr-backed arrays or large images, use the streaming variant inside
      `_correct_for_OME_axes_order` instead.
    * The returned array is a fully materialized NumPy array, even if the input
      originated from a lazy source.
    """
    curr_image = np.asarray(arr)
    curr_axes_full = axes_string
    for ax in OME_axes:
        if ax not in curr_axes_full:
            curr_image = np.expand_dims(curr_image, axis=-1)
            curr_axes_full += ax
    permute_from = np.arange(len(curr_axes_full), dtype=int)
    permute_to   = [OME_axes_order[ax] for ax in curr_axes_full]
    curr_image   = np.moveaxis(curr_image, permute_from, permute_to)
    return curr_image
def _correct_for_OME_axes_order(image: Union[np.ndarray, zarr.core.array.Array],
                                metadata: Dict[str, Any],
                                memap_large_file: bool =False,
                                verbose: bool =True) -> Tuple[Union[np.ndarray, zarr.core.array.Array], tuple, str]:
    """
    Normalize an image array to canonical OME axis order (TZCYX).

    This internal helper ensures that image data and its associated axis metadata
    are brought into the canonical OME axis convention TZCYX. It supports both
    in-memory NumPy arrays and Zarr-backed arrays and selects the appropriate
    strategy depending on input type and memory constraints.

    Three execution paths are distinguished:

    * NumPy input:
    The array is fully reordered in RAM and returned as a NumPy array.

    * Zarr input with memap_large_file=False:
    The full Zarr array is read once into RAM, reordered as a NumPy array, and
    then written back to a newly created Zarr store at the original location.

    * Zarr input with memap_large_file=True:
    The data are copied slice-wise into a temporary Zarr store on disk, iterating
    over all non-spatial axes while streaming full (Y, X) planes. This mode avoids
    loading the entire dataset into memory and is intended for large files.

    Missing OME axes are inserted as singleton dimensions, and existing axes are
    permuted into the canonical order. The function operates purely on array data
    and axis ordering; it does not modify or regenerate higher-level metadata.

    Parameters
    ----------
    image : np.ndarray or zarr.core.array.Array
        Input image data. Either a fully materialized NumPy array or a Zarr array.
    metadata : dict
        Metadata dictionary containing at least the key ``"axes"``, which declares
        the current axis order of the input image using characters from
        {T, Z, C, Y, X}. Optional entries such as ``"SizeY"`` and ``"SizeX"`` are
        used to determine optimal chunk sizes when creating Zarr outputs.
    memap_large_file : bool, optional
        If True and the input is a Zarr array, reorder the data via slice-wise,
        on-disk copying to avoid loading the full dataset into RAM. If False,
        the Zarr array is fully read into memory before reordering. Default is False.

    Returns
    -------
    image_out : np.ndarray or zarr.core.array.Array
        The reordered image data in canonical OME axis order TZCYX. The return type
        matches the chosen execution path.
    shape_out : tuple
        Shape of the reordered image array.
    axes_out : str
        The canonical OME axis string, equal to ``_OME_AXES`` (typically "TZCYX").

    Raises
    ------
    ValueError
        If the length of ``metadata["axes"]`` does not match ``image.ndim``.

    Notes
    -----
    * The canonical axis mapping and axis sequence are taken from the module-level
    constants ``_AXIS_TO_INDEX`` and ``_OME_AXES``.
    * For Zarr inputs, the original store is replaced on disk by the reordered
    version. Temporary stores are removed once the operation completes.
    * When no persistent Zarr store path is available, the function falls back to
    returning a fully materialized NumPy array.
    """ 
    if verbose:
        print("  Correcting for OME axes order...")
    
    # canonical OME axes: TZCYX
    #OME_axes_order = {"T": 0, "Z": 1, "C": 2, "Y": 3, "X": 4}
    #OME_axes = "TZCYX"
    OME_axes_order = _AXIS_TO_INDEX
    OME_axes = _OME_AXES

    curr_axes  = metadata["axes"]
    curr_shape = image.shape

    if len(curr_axes) != len(curr_shape):
        raise ValueError(
            f"Metadata axes '{curr_axes}' (len={len(curr_axes)}) does not match "
            f"image.ndim={len(curr_shape)}")

    # branch 1: pure NumPy arrays:
    if not isinstance(image, zarr.core.array.Array):
        if verbose:
            print("    Got NumPy array as input. Will return reordered NumPy array.")
        curr_image = _reorder_numpy(image, curr_axes, OME_axes, OME_axes_order)
        return curr_image, curr_image.shape, OME_axes

    # branch 2: Zarr array w/o streaming (full read in RAM):
    if verbose:
        print("    Got Zarr array as input...")
    src = image

    if not memap_large_file:
        # in this case, in this case the Zarr source is fully read into RAM once:
        if verbose:
            print("    memap_large_file=False: Reading full Zarr into RAM for reordering...")
        curr_image = _reorder_numpy(src[...], curr_axes, OME_axes, OME_axes_order)

        try:
            src_path = str(src.store_path).replace("file://", "")
        except AttributeError:
            # no path available, return NumPy array directly
            if verbose:
                print("    While memap_large_file=False, no store_path available, returning NumPy array.")
            return curr_image, curr_image.shape, OME_axes

        if os.path.exists(src_path):
            shutil.rmtree(src_path)

        size_y = metadata.get("SizeY", curr_image.shape[OME_axes_order["Y"]])
        size_x = metadata.get("SizeX", curr_image.shape[OME_axes_order["X"]])
        # 5D chunks: (T, Z, C, Y, X)
        target_chunks = (1, 1, 1, size_y, size_x)

        dst = zarr.open(
            src_path,
            mode="w",
            shape=curr_image.shape,
            dtype=curr_image.dtype,
            chunks=target_chunks)
        if verbose:
            print("    Writing reordered data back to Zarr store...")
        dst[...] = curr_image

        image_out = zarr.open(src_path, mode="r+")
        return image_out, image_out.shape, OME_axes

    # branch 3: memory-mapped large file, streaming copy in (Y, X):
    if verbose:
        print("    memap_large_file=True: Copying data slice-wise into Zarr array on disk (will take some time)...")

    # target shape in TZCYX; fill missing axes with singleton dimensions:
    full_shape = [1] * len(OME_axes)  # T, Z, C, Y, X
    for i, ax in enumerate(curr_axes):
        full_shape[OME_axes_order[ax]] = curr_shape[i]
    full_shape = tuple(full_shape)

    iy = OME_axes_order["Y"]
    ix = OME_axes_order["X"]
    outer_axes_idx = [k for k in range(len(OME_axes)) if k not in (iy, ix)]
    outer_shape = tuple(full_shape[k] for k in outer_axes_idx)
    total_outer = int(np.prod(outer_shape)) if outer_shape else 1
    # "total_outer" is 1 if only Y and X are present; it actually counts the number of
    # iterations needed over all non-spatial axes.

    try:
        src_path = str(src.store_path).replace("file://", "")
    except AttributeError:
        # fallback: when no path exists, read once into RAM:
        if verbose:
            print("    While memap_large_file=True, no store_path available, returning NumPy array.")
        curr_image = _reorder_numpy(src[...], curr_axes, OME_axes, OME_axes_order)
        return curr_image, curr_image.shape, OME_axes

    tmp_path = src_path + "_ome_tmp"
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)

    size_y = metadata.get("SizeY", full_shape[OME_axes_order["Y"]])
    size_x = metadata.get("SizeX", full_shape[OME_axes_order["X"]])
    # 5D chunks: (T, Z, C, Y, X)
    target_chunks = (1, 1, 1, size_y, size_x)

    dst = zarr.open(
        tmp_path,
        mode="w",
        shape=full_shape,
        dtype=src.dtype,
        chunks=target_chunks)

    if total_outer == 1:
        if verbose:
            print("    Only Y and X axes present, copying full data at once...")
        # dst is of shape (1,1,1,Y,X) and we need to copy src with shape (Y,X):
        dst[0,0,0,...] = src[...]
        #dst[...] = src[...]
    else:
        iterator = tqdm(
            np.ndindex(*outer_shape),
            total=total_outer,
            desc="    Reordering axes to TZCYX and copying to temporary Zarr store"
        )

        for outer_idx in iterator:
            dest_index = [None] * len(OME_axes)
            o_pos = 0
            for k in range(len(OME_axes)):
                if k in (iy, ix):
                    dest_index[k] = slice(None)
                else:
                    dest_index[k] = outer_idx[o_pos]
                    o_pos += 1

            src_index = []
            for i, ax in enumerate(curr_axes):
                if ax in ("Y", "X"):
                    src_index.append(slice(None))
                else:
                    j = OME_axes_order[ax]
                    src_index.append(dest_index[j])

            dst[tuple(dest_index)] = src[tuple(src_index)]

    if os.path.exists(src_path):
        shutil.rmtree(src_path)
    os.rename(tmp_path, src_path)

    image_out = zarr.open(src_path, mode="r+")
    return image_out, image_out.shape, OME_axes
def _batch_correct_for_OME_axes_order(images: List[Union[np.ndarray, zarr.core.array.Array]],
                                      metadatas: List[Dict[str, Any]],
                                      memap_large_file: bool =False,
                                      verbose: bool =True
                                      ) -> Tuple[List[Union[np.ndarray, zarr.core.array.Array]], List[Dict[str, Any]]]:
    """
    Apply OME axis normalization to a batch of images.

    This function is a thin batch wrapper around `_correct_for_OME_axes_order`. It
    iterates over a list of images and their corresponding metadata dictionaries
    and normalizes each image to the canonical OME axis order TZCYX.

    Each image is processed independently using the same logic as in the single-image
    function, including the choice between in-RAM reordering and slice-wise,
    on-disk copying for Zarr arrays depending on `memap_large_file`.

    The input lists are modified in place: both the image objects and the associated
    metadata entries (``"shape"`` and ``"axes"``) are updated for each element.

    Parameters
    ----------
    images : list of np.ndarray or zarr.core.array.Array
        List of input images to be reordered.
    metadatas : list of dict
        List of metadata dictionaries corresponding to `images`. Each dictionary
        must contain the key ``"axes"`` describing the current axis order of the
        associated image.
    memap_large_file : bool, optional
        Forwarded to `_correct_for_OME_axes_order`. If True, Zarr inputs are
        reordered via slice-wise on-disk copying to limit memory usage. Default is
        False.

    Returns
    -------
    images_out : list of np.ndarray or zarr.core.array.Array
        List of reordered images in canonical OME axis order TZCYX. Elements may be
        NumPy arrays or Zarr arrays, depending on input type and processing mode.
    metadatas_out : list of dict
        The updated metadata dictionaries. For each entry, ``"shape"`` and
        ``"axes"`` reflect the reordered image.

    Notes
    -----
    * Processing is performed sequentially; no parallelism is introduced.
    * This function mutates its inputs in place.
    """
    
    # ensure that both lists have the same length:
    if len(images) != len(metadatas):
        if verbose:
            print("Error: In _batch_correct_for_OME_axes_order, images and metadatas have different lengths!")
            print(f"  len(images) = {len(images)}, len(metadatas) = {len(metadatas)}. Returning unmodified inputs.")
        return images, metadatas
    
    for image_i in range(len(images)):
        images[image_i], metadatas[image_i]["shape"], metadatas[image_i]["axes"] = \
            _correct_for_OME_axes_order(images[image_i], metadatas[image_i], memap_large_file=memap_large_file,
                                        verbose=verbose)
    return images, metadatas

# filter-function for removing non-OME-conform axes from CZI files:
def _filter_image_data_for_ome_tif(imagedata, axes):
    """
    Filter image data to retain only OME-relevant axes.

    This helper removes non-OME axes from an image array by selecting the first
    index along any axis that is not part of the canonical OME axis set. The
    resulting array contains only axes from the OME convention, while preserving
    their original relative order.

    The operation is purely index-based: non-OME axes are collapsed via integer
    indexing, and no resampling or data modification beyond slicing is performed.

    Parameters
    ----------
    imagedata : np.ndarray or array-like
        Input image data array.
    axes : str
        Axis declaration for `imagedata`. Its length must match
        ``imagedata.ndim``. Axes not present in the canonical OME axis set are
        removed by slicing.

    Returns
    -------
    filtered_data : np.ndarray
        The image data restricted to OME-relevant axes.
    filtered_axes : str
        Axis string corresponding to `filtered_data`, containing only axes from
        the canonical OME axis set and in the same relative order as in `axes`.

    Notes
    -----
    * The canonical OME axis set is taken from the module-level constant
    ``_OME_AXES``.
    * Non-OME axes are reduced by taking index 0 along that dimension, which
    implicitly assumes that these axes are either singleton or that only the
    first element is of interest.
    * This function performs no validation of axis semantics beyond string
    membership.
    """
    # imagedata = CZI_image     # for testing
    # axes = metadata["axes"]   # for testing
    
    # define desired axes:
    #desired_axes = 'TZCYX'
    desired_axes = _OME_AXES
    
    # determine the slices for the desired axes:
    slices = [slice(None) if axes[i] in desired_axes else 0 for i in range(imagedata.ndim)]
    
    # apply the slices to filter the data:
    filtered_data = imagedata[tuple(slices)]
    
    # filter the axis string:
    filtered_axes = ''.join([axis for axis in axes if axis in desired_axes])
    
    return filtered_data, filtered_axes

# extract the SizeX, SizeY, SizeZ, SizeC, SizeT, SizeS from the metadata:
def _get_ome_image_sizes(imageshape, metadata):
    """
    Populate OME size fields from an image shape and axis declaration.

    This helper derives the standard OME size entries (``SizeT``, ``SizeZ``,
    ``SizeC``, ``SizeY``, ``SizeX``) from the provided image shape and axis string.
    All OME size fields are first initialized to 1 and then updated for axes that
    are present in the image.

    The function operates on a shallow copy of the input metadata dictionary and
    does not modify the original object.

    Parameters
    ----------
    imageshape : tuple
        Shape of the image array. Its length must match the length of
        ``metadata["axes"]``.
    metadata : dict
        Metadata dictionary containing an ``"axes"`` entry that declares the axis
        order of the image using characters from {T, Z, C, Y, X}.

    Returns
    -------
    metadata_update : dict
        A copy of the input metadata with OME-compliant size entries added or
        updated. For each axis in the canonical OME axis set, a corresponding
        ``Size<axis>`` key is present.

    Notes
    -----
    * The canonical OME axis set is taken from the module-level constant
    ``_OME_AXES``.
    * Axes not present in ``metadata["axes"]`` remain with size 1, consistent with
    OME conventions for singleton dimensions.
    * No validation is performed beyond positional correspondence between
    `imageshape` and ``metadata["axes"]``.
    """
    metadata_update = metadata.copy()
    #default_OME_axes = 'TZCYX'
    default_OME_axes = _OME_AXES
    
    # initialize size metadata:
    for axis in default_OME_axes:
        metadata_update[f"Size{axis}"] = 1
    # update size metadata:
    for axis_i, axis in enumerate(metadata_update["axes"]):
        metadata_update[f"Size{axis}"] = imageshape[axis_i]
        
    return metadata_update

# function to dynamically extract namespace:
def _get_namespace(xml_root):
    """
    Extract the XML namespace from an ElementTree root element.

    This helper inspects the tag of an XML root element and extracts the namespace
    URI if the tag is namespace-qualified. ElementTree represents such tags in the
    form ``"{namespace}tagname"``. If no namespace is present, an empty string is
    returned.

    Parameters
    ----------
    xml_root : xml.etree.ElementTree.Element
        Root element of an XML document.

    Returns
    -------
    namespace : str
        The namespace URI extracted from ``xml_root.tag``, or an empty string if
        the element is not namespace-qualified.

    Notes
    -----
    * The function relies on a simple regular expression match and does not
    validate the namespace URI.
    * This helper is typically used when parsing OME-XML or similar
    namespace-qualified XML formats.
    """
    match = re.match(r'\{(.*)\}', xml_root.tag)
    return match.group(1) if match else ''

# function to parse OME-XML metadata into human readable format:
def _parse_ome_metadata(ome_xml):
    """
    Parse OME-XML metadata and extract commonly used fields into a plain dictionary.

    This helper parses an OME-XML string and extracts a subset of pixel and
    acquisition metadata into a Python dictionary with simple scalar values.
    It is designed to be tolerant to missing attributes and to handle OME-XML
    documents that use arbitrary XML namespaces.

    The function focuses at the moment on two groups of information (and can be
    extended in the future):

    * The ``Pixels`` element:
    Extracts image dimensions (``SizeX``, ``SizeY``, ``SizeZ``, ``SizeC``,
    ``SizeT``), physical voxel sizes (``PhysicalSizeX``, ``PhysicalSizeY``,
    ``PhysicalSizeZ``) including their units, and the temporal sampling
    (``TimeIncrement`` and its unit). Additionally, it counts the number of
    ``Channel`` elements found under ``Pixels``.

    * ``MapAnnotation`` elements:
    Extracts key value pairs from ``MapAnnotation/Value/M`` entries and stores
    them under ``metadata["Annotations"]``. The ``Namespace`` attribute of the
    MapAnnotation is recorded if present.

    Missing or malformed numeric attributes are left at default values, and unit
    fields fall back to standard defaults.

    Parameters
    ----------
    ome_xml : str
        OME-XML metadata as a string.

    Returns
    -------
    metadata : dict
        Dictionary containing extracted metadata fields. Keys include:

        * ``SizeX``, ``SizeY``, ``SizeZ``, ``SizeC``, ``SizeT`` (int)
        * ``PhysicalSizeX``, ``PhysicalSizeY``, ``PhysicalSizeZ`` (float)
        * ``PhysicalSizeXUnit``, ``PhysicalSizeYUnit``, ``PhysicalSizeZUnit`` (str)
        * ``TimeIncrement`` (float), ``TimeIncrementUnit`` (str)
        * ``Channel_Count`` (int)
        * ``Annotations`` (dict), present even if empty

    Notes
    -----
    * XML parsing is performed via ``xml.etree.ElementTree``.
    * Namespace handling is based on `_get_namespace`, and tags are queried through
    a namespace mapping under the prefix ``"ome"``.
    * The function is intentionally permissive: it does not raise on missing fields
    and does not validate consistency across reported sizes and actual image data.
    * The returned annotation dictionary is a flat mapping of keys to strings.
    If multiple MapAnnotations contain identical keys, later entries will
    overwrite earlier ones.
    """
    
    # parse the XML content:
    root = ET.fromstring(ome_xml)
    namespace = _get_namespace(root)
    ns = {'ome': namespace}  # Namespace dictionary

    # initialize metadata dictionary with default values:
    metadata = {
        'SizeX': 0,
        'SizeY': 0,
        'SizeZ': 0,
        'SizeC': 0,
        'SizeT': 0,
        'PhysicalSizeX': 1.0,
        'PhysicalSizeY': 1.0,
        'PhysicalSizeZ': 1.0,
        'PhysicalSizeXUnit': 'micron',
        'PhysicalSizeYUnit': 'micron',
        'PhysicalSizeZUnit': 'micron',
        'TimeIncrement': 0.0,
        'TimeIncrementUnit': 'seconds',
        'Channel_Count': 0}

    try:
        # find the 'Pixels' element:
        pixels = root.find('.//ome:Pixels', ns)
        if pixels is not None:
            # extract metadata with try-except for each attribute:
            
            # SizeX:
            try:
                metadata['SizeX'] = int(pixels.attrib['SizeX'])
            except (KeyError, ValueError):
                pass
            # SizeY:
            try:
                metadata['SizeY'] = int(pixels.attrib['SizeY'])
            except (KeyError, ValueError):
                pass
            # SizeZ:
            try:
                metadata['SizeZ'] = int(pixels.attrib['SizeZ'])
            except (KeyError, ValueError):
                pass
            # SizeC:
            try:
                metadata['SizeC'] = int(pixels.attrib['SizeC'])
            except (KeyError, ValueError):
                pass
            # SizeT:
            try:
                metadata['SizeT'] = int(pixels.attrib['SizeT'])
            except (KeyError, ValueError):
                pass
            # PhysicalSizeX:
            try:
                metadata['PhysicalSizeX'] = float(pixels.attrib['PhysicalSizeX'])
            except (KeyError, ValueError):
                pass
            # PhysicalSizeY:
            try:
                metadata['PhysicalSizeY'] = float(pixels.attrib['PhysicalSizeY'])
            except (KeyError, ValueError):
                pass
            # PhysicalSizeZ:
            try:
                metadata['PhysicalSizeZ'] = float(pixels.attrib['PhysicalSizeZ'])
            except (KeyError, ValueError):
                pass

            metadata['PhysicalSizeXUnit'] = pixels.attrib.get('PhysicalSizeXUnit', 'micron')
            metadata['PhysicalSizeYUnit'] = pixels.attrib.get('PhysicalSizeYUnit', 'micron')
            metadata['PhysicalSizeZUnit'] = pixels.attrib.get('PhysicalSizeZUnit', 'micron')

            try:
                metadata['TimeIncrement'] = float(pixels.attrib['TimeIncrement'])
            except (KeyError, ValueError):
                pass

            metadata['TimeIncrementUnit'] = pixels.attrib.get('TimeIncrementUnit', 'seconds')

            # count channels:
            channels = pixels.findall('.//ome:Channel', ns)
            metadata['Channel_Count'] = len(channels)
    except ET.ParseError:
        print("Error: Invalid XML content. Could not extract Pixels metadata from OME-XML.")

    # find 'MapAnnotation's:
    try:
        # collect all Map Annotations in a separate sub-dictionary:
        metadata['Annotations'] = {}

        # there COULD be multiple MapAnnotations, so we loop over them:
        for ma in root.findall('.//ome:MapAnnotation', ns):
            # ma = root.findall('.//ome:MapAnnotation', ns)[0]  # for testing
            
            # extract Namespace attribute:
            try: 
                ns_attr = ma.get('Namespace', '')
            except:
                ns_attr = 'unknown'
            metadata['Annotations']['Namespace'] = ns_attr

            # check whether there is a <Value> element, otherwise skip:
            value_elem = ma.find('ome:Value', ns)
            if value_elem is None:
                continue

            # read all <M K="...">value</M> elements:
            for m in value_elem.findall('ome:M', ns):
                key = m.get('K')
                if not key:
                    continue
                val = (m.text or '').strip()

                metadata['Annotations'][key] = val       
    except ET.ParseError:
        print("Could not extract MapAnnotation from OME-XML.")

    return metadata


# function to standardize read imagej_metadata:
def _rational_to_float(r):
    """ 
    Convert a TIFF rational value to a float.
    Parameters
    ----------
    r : tuple, list, or float
        The rational value, typically as (numerator, denominator) or a float.
    Returns
    -------
    float or None
        The converted float value, or None if conversion fails.
    Notes
    -----
    * TIFF rationals are often stored as (num, den) tuples. If the denominator is zero,
      None is returned to avoid division errors.
    * If `r` is already a float or can be directly converted, that value is returned.
    * If `r` is None or cannot be converted, the function returns None.
    """
    # TIFF rationals often come as (num, den):
    if r is None:
        return None
    if isinstance(r, (tuple, list)) and len(r) == 2:
        num, den = r
        num = float(num)
        den = float(den)
        if den == 0:
            return None
        return num / den
    try:
        return float(r)
    except Exception:
        return None
def _unit_to_um_factor_from_resolutionunit(v):
    """ 
    Convert a TIFF ResolutionUnit value to a micron scaling factor.
    
    Parameters
    ----------
    v : int or str
        The TIFF ResolutionUnit value, either as an integer code or a descriptive string.   
    Returns
    -------
    float or None
        The scaling factor to convert from the specified unit to microns, or None if
        the unit is unrecognized.
    Notes
    -----
    * Standard TIFF ResolutionUnit codes are:
        - 1: None (interpreted here as microns)
        - 2: Inches (1 inch = 25400 microns)
        - 3: Centimeter (1 cm = 10000 microns)
    * Descriptive strings such as "inch", "centimeter", "millimeter", "micron", and "meter"
      are also recognized in a case-insensitive manner.
    * If `v` is None or does not match any known unit, the function returns None.
    """
    # TIFF ResolutionUnit: usually int codes or strings.
    # Standard: 2=inches, 3=centimeter.
    # Set by default in OMIO: 1=None (actually; in OMIO, we interprete this as microns)
    if v is None:
        return None
    if isinstance(v, int):
        if v == 2:
            return 25400.0
        if v == 3:
            return 10000.0
        if v == 1:
            return 1.0
        return None
    s = str(v).strip().lower()
    if "inch" in s:
        return 25400.0
    if "centimeter" in s or s == "cm":
        return 10000.0
    if "millimeter" in s or s == "mm":
        return 1000.0
    if "micron" in s or s == "µm" or s == "um":
        return 1.0
    if "meter" in s or s == "m":
        return 1e6
    return None
def _standardize_imagej_metadata(imagej_metadata: Dict[str, Any],
                                 tags: Union[list, None] = None,
                                 verbose: bool = False
                                 ) -> Dict[str, Any]:
    """
    Standardize ImageJ metadata keys and recover physical pixel sizes when possible.

    This helper normalizes the key casing of ImageJ metadata to a consistent
    OME-like naming scheme (for example ``sizex`` to ``SizeX`` and
    ``physicalsizex`` to ``PhysicalSizeX``) while leaving unknown keys unchanged.
    It additionally attempts to recover missing physical pixel size fields from
    common ImageJ encodings.

    If ``PhysicalSizeX`` is absent but an ``Info`` field is present, the function
    parses the ``Info`` string line-by-line and looks for entries of the form
    ``Scaling|Distance|...``. When found, it converts the stored scaling values into
    micron-based physical sizes and populates ``PhysicalSizeX``, ``PhysicalSizeY``,
    and ``PhysicalSizeZ`` accordingly. If ``PhysicalSizeZ`` is still missing after
    this step, the function falls back to ImageJ's ``spacing`` field if available.

    Parameters
    ----------
    imagej_metadata : dict
        ImageJ metadata dictionary. The mapping table assumes keys are already
        lowercased, but any keys are accepted. Values are preserved as-is.

    Returns
    -------
    standardized_metadata : dict
        New dictionary containing standardized keys. Non-standard keys are carried
        over unchanged. Physical size entries may be added if they can be inferred
        from ``Info`` or ``spacing``.

    Notes
    -----
    * Key standardization is performed via a fixed mapping table and is therefore
    conservative: only known keys are renamed.
    * The ``Info`` parsing logic is heuristic and depends on ImageJ writing a
    flattened scaling structure using keys such as ``Scaling|Distance|Id #1``,
    ``Scaling|Distance|Value #1``, and ``Scaling|Distance|DefaultUnitFormat #1``.
    * Physical size reconstruction from ``Info`` is best-effort. Failures are caught
    and reported via printing, and missing values are left unset.
    * If both reconstructed ``PhysicalSizeZ`` and ``spacing`` are present, the
    reconstructed value takes precedence.
    """
    # key mapping: lowercase keys to their standardized letter case:
    key_mapping = {
        'axes': 'axes',
        'shape': 'shape',
        'sizex': 'SizeX',
        'sizey': 'SizeY',
        'sizec': 'SizeC',
        'sizet': 'SizeT',
        'sizes': 'SizeZ',
        'physicalsizex': 'PhysicalSizeX',
        'physicalsizey': 'PhysicalSizeY',
        'physicalsizez': 'PhysicalSizeZ',
        'unit': 'unit',
        'physicalsizexunit': 'PhysicalSizeXUnit',
        'physicalsizeyunit': 'PhysicalSizeYUnit',
        'timeincrement': 'TimeIncrement',
        'timeincrementunit': 'TimeIncrementUnit',
        'frame_rate': 'frame_rate',
        'structuredannotations': 'StructuredAnnotations'}

    # initialize new dictionary to hold standardized metadata:
    standardized_metadata = {}

    # process each key in the input dictionary:
    for key, value in imagej_metadata.items():
        # if the key is in the mapping, use the standardized key:
        standardized_key = key_mapping.get(key, key)
        standardized_metadata[standardized_key] = value

    """ 
    In some imagej metadata, PhysicalSizeX and PhysicalSizeY are written into a collapsed
    XML/JSON structure under "Info", where relevant infos are stored under:
    
        Scaling|Distance|DefaultUnitFormat #1 = µm
        Scaling|Distance|DefaultUnitFormat #2 = µm
        Scaling|Distance|DefaultUnitFormat #3 = µm
        Scaling|Distance|Id #1 = X
        Scaling|Distance|Id #2 = Y
        Scaling|Distance|Id #3 = Z
        Scaling|Distance|Value #1 = 1.135E-07
        Scaling|Distance|Value #2 = 1.135E-07
        Scaling|Distance|Value #3 = 5E-07
        
    Since DefaultUnitFormat is, e.g., here 'µm', 'Scaling|Distance|Value' is the actual dispersion
    which needs to be converted into micron units:
    PhysicalSizeX = Scaling|Distance|Value #1 * factor to convert DefaultUnitFormat to micron
    ... 
    """

    unit_map_info = {'µm': 1e6, 'nm': 1e4, 'mm': 1e3, 'cm': 1e-3, 'm': 1.0}
    #unit_map_info = {'µm': 1.0,'um': 1.0,'nm': 1e-3,'mm': 1e3,'cm': 1e4,'m':  1e6,}
    unit_map_tags = {'inch': 25400.0, 'centimeter': 10000.0, 'millimeter': 1000.0, 'micron': 1.0, 'meter': 1e6}

    if "PhysicalSizeX" not in standardized_metadata or "PhysicalSizeY" not in standardized_metadata:
        # we do not also check for PhysicalSizeY/Z here, since they often come/miss together.
        # check whether standardized_metadata contains 'Info' key:
        if "Info" in standardized_metadata:
            info_str = standardized_metadata["Info"]
            # info_str is a string of form "' BitsPerPixel = 14\n DimensionOrder = XYCZT\n IsInterleaved = false\n IsRGB = false\n ...",
            # thus we need to parse it line by line:
            info_lines = info_str.split('\n')
            scaling_distance = {}
            for line in info_lines:
                line = line.strip()
                if line.startswith("Scaling|Distance|"):
                    parts = line.split(' = ')
                    if len(parts) == 2:
                        key_part = parts[0].replace("Scaling|Distance|", "")
                        value_part = parts[1]
                        scaling_distance[key_part] = value_part
            # now extract PhysicalSizeX, PhysicalSizeY, PhysicalSizeZ:
            try:
                for i in range(1, 4):
                    id_key = f"Id #{i}"
                    value_key = f"Value #{i}"
                    unit_key = f"DefaultUnitFormat #{i}"
                    if id_key in scaling_distance and value_key in scaling_distance and unit_key in scaling_distance:
                        axis_id = scaling_distance[id_key]
                        axis_value = float(scaling_distance[value_key])
                        axis_unit = scaling_distance[unit_key]
                        if axis_unit in unit_map_info:
                            physical_size = axis_value * unit_map_info[axis_unit]
                            if axis_id == 'X':
                                standardized_metadata["PhysicalSizeX"] = physical_size
                            elif axis_id == 'Y':
                                standardized_metadata["PhysicalSizeY"] = physical_size
                            elif axis_id == 'Z':
                                standardized_metadata["PhysicalSizeZ"] = physical_size
            except Exception as e:
                print(f"  Error while extracting PhysicalSize from Info: {e}")
                print(f"  Leaving PhysicalSize entries empty.")
        
        # PhysicalSizeX/Y could now still be missing; try to extract from tags:
        if "PhysicalSizeX" not in standardized_metadata or "PhysicalSizeY" not in standardized_metadata:
            if tags is not None:
                # sometimes, the tags list contains 'XResolution' and 'YResolution' entries:
                try:
                    # at the moment, we only consider tags[0], but there could be multiple tags
                    # (otherwise run the following loop additionally for all tags in tags, for tag in tags:):
                    tag0 = tags[0] if isinstance(tags, list) and len(tags) > 0 else tags

                    XRes = None
                    YRes = None
                    ResUnit = None

                    for _, t in tag0.items():
                        name = getattr(t, "name", None)
                        if name == "XResolution":
                            XRes = getattr(t, "value", None)
                            if verbose:
                                print(f"    Found XResolution tag with value: {XRes}")
                        elif name == "YResolution":
                            YRes = getattr(t, "value", None)
                            if verbose:
                                print(f"    Found YResolution tag with value: {YRes}")
                        elif name == "ResolutionUnit":
                            ResUnit = getattr(t, "value", None)
                            if verbose:
                                print(f"    Found ResolutionUnit tag with value: {ResUnit}")
                    x_pixels_per_unit = _rational_to_float(XRes)
                    y_pixels_per_unit = _rational_to_float(YRes)
                    factor_um = _unit_to_um_factor_from_resolutionunit(ResUnit)

                    # pixels_per_unit must be > 0 to avoid division by zero:
                    if (x_pixels_per_unit is not None and x_pixels_per_unit > 0 and
                        y_pixels_per_unit is not None and y_pixels_per_unit > 0 and
                        factor_um is not None):

                        standardized_metadata["PhysicalSizeX"] = factor_um / x_pixels_per_unit
                        standardized_metadata["PhysicalSizeY"] = factor_um / y_pixels_per_unit
                        standardized_metadata.setdefault("PhysicalSizeXUnit", "micron")
                        standardized_metadata.setdefault("PhysicalSizeYUnit", "micron")
                        
                        if verbose:
                            print(f"      Calculated PhysicalSizeX = {standardized_metadata['PhysicalSizeX']} micron")
                            print(f"      Calculated PhysicalSizeY = {standardized_metadata['PhysicalSizeY']} micron")
                    else:
                        if verbose:
                            print("    Could not extract PhysicalSizeX/Y from tags due to missing or invalid values.")
                    

                except Exception as e:
                    print(f"  Error while extracting PhysicalSize from tags: {e}")
                    print(f"  Leaving PhysicalSizeX/Y entries empty.")
            

    # handle missing PhysicalSizeZ by checking 'spacing' key:
    if "PhysicalSizeZ" not in standardized_metadata:
        if "spacing" in imagej_metadata:
            standardized_metadata["PhysicalSizeZ"] = imagej_metadata["spacing"]
            if verbose:
                print(f"    Extracted PhysicalSizeZ from 'spacing': {standardized_metadata['PhysicalSizeZ']}")
            
            if 'unit' in standardized_metadata:
                standardized_metadata["PhysicalSizeZUnit"] = standardized_metadata['unit']
                # convert to PhysicalSizeZ in micron:
                unit = standardized_metadata['unit'].lower()
                if unit in unit_map_tags:
                    factor = unit_map_tags[unit]
                    standardized_metadata["PhysicalSizeZ"] = standardized_metadata["PhysicalSizeZ"] * factor
                    standardized_metadata["PhysicalSizeZUnit"] = "micron"
                    if verbose:
                        print(f"      Converted PhysicalSizeZ to micron: {standardized_metadata['PhysicalSizeZ']} micron")

    return standardized_metadata

# function to standardize read lsm_metadata:
def _standardize_lsm_metadata(lsm_metadata):
    """
    Standardize Zeiss LSM metadata to an OME and ImageJ-compatible key scheme.

    This helper converts selected keys from Zeiss LSM metadata into a standardized
    naming convention aligned with the keys used for ImageJ and OME metadata. Only
    fields with a clear semantic correspondence are mapped; all other entries are
    copied verbatim.

    The function operates on a new dictionary and does not modify the input
    metadata object.

    Parameters
    ----------
    lsm_metadata : dict
        Metadata dictionary as returned by ``tifffile.lsm_metadata``.

    Returns
    -------
    standardized_metadata : dict
        Metadata dictionary with standardized keys. Dimension and voxel size fields
        are renamed to OME-style ``Size*`` and ``PhysicalSize*`` entries, and
        temporal sampling is mapped to ``TimeIncrement``.

    Notes
    -----
    * Zeiss LSM uses the non-standard spelling ``TimeIntervall``; this key is
    explicitly mapped to ``TimeIncrement``.
    * No unit conversion is performed. Values are transferred as-is and are
    assumed to be expressed in the units provided by the original LSM metadata.
    * Keys without an explicit mapping are preserved unchanged.
    """

    # mapping LSM → standardized ImageJ-like terminology:
    key_mapping = {
        'DimensionX': 'SizeX',
        'DimensionY': 'SizeY',
        'DimensionZ': 'SizeZ',
        'DimensionChannels': 'SizeC',
        'DimensionTime': 'SizeT',

        'VoxelSizeX': 'PhysicalSizeX',
        'VoxelSizeY': 'PhysicalSizeY',
        'VoxelSizeZ': 'PhysicalSizeZ',

        # Zeiss uses "TimeIntervall" (typo in original format)
        'TimeIntervall': 'TimeIncrement'
    }

    standardized_metadata = {}

    for key, value in lsm_metadata.items():
        # apply mapping if available, otherwise preserve key
        standardized_key = key_mapping.get(key, key)
        standardized_metadata[standardized_key] = value

    return standardized_metadata

# function to add file properties to metadata:
def _add_file_properties_to_metadata(metadata, fname, original_metadata_type="N/A"):
    """
    Augment a metadata dictionary with file-level provenance information.

    This helper ensures that a set of standard file-related metadata fields is
    present in the provided metadata dictionary. Missing entries are populated
    from the file system using the supplied file path. Existing keys are preserved
    and not overwritten.

    The added fields capture basic provenance information such as the original
    file name, file type, parent directory, metadata source, and a timestamp
    derived from the file system.

    Parameters
    ----------
    metadata : dict or None
        Metadata dictionary to be updated. If None, a new dictionary is created.
    fname : str
        Full path to the source file.
    original_metadata_type : str, optional
        Identifier describing the origin or format of the original metadata
        (for example ``"OME_XML"``, ``"ImageJ"``, or ``"LSM"``). Default is ``"N/A"``.

    Returns
    -------
    metadata : dict
        The updated metadata dictionary containing file provenance fields.

    Notes
    -----
    * File properties are added only if the corresponding keys are not already
    present in the dictionary.
    * The file type is derived from the filename extension without the leading
    dot.
    * The timestamp is obtained via ``os.path.getctime`` and expressed in UTC using
    an ISO-like string format. On some platforms, this value may represent the
    last metadata change time rather than true file creation time.
    * If file system access fails, the creation or change date is set to ``"N/A"``.
    """
    # ensure metadata dictionary exists:
    if metadata is None:
        metadata = {}

    # file path and name properties:
    folder_path = os.path.dirname(fname)
    fname_base, fname_extension = os.path.splitext(os.path.basename(fname))

    # add missing keys with derived values:
    metadata.setdefault("original_filetype", fname_extension[1:])  # remove leading '.'
    metadata.setdefault("original_filename", fname_base + fname_extension)
    metadata.setdefault("original_parentfolder", folder_path)
    metadata.setdefault("original_metadata_type", original_metadata_type)
    
    # add creation or change date:
    try:
        creation_date = datetime.datetime.fromtimestamp(
            os.path.getctime(fname), datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
        metadata.setdefault("original_creation_or_change_date", creation_date)
    except Exception:
        metadata.setdefault("original_creation_or_change_date", "N/A")

    return metadata

# function to check and update metadata units:
def _metadata_units_check(metadata, pixelunit="micron"):
    """
    Normalize unit fields in a metadata dictionary.

    This helper ensures that physical size unit entries are present and expressed
    using a consistent textual representation. Missing unit fields are populated
    with a default unit, and the commonly used symbol ``"µm"`` is normalized to the
    string ``"micron"``.

    The function operates in place on the provided metadata dictionary.

    Parameters
    ----------
    metadata : dict
        Metadata dictionary to be checked and updated.
    pixelunit : str, optional
        Default unit string to assign when a unit field is missing. Default is
        ``"micron"``.

    Returns
    -------
    metadata : dict
        The updated metadata dictionary with normalized unit entries.

    Notes
    -----
    * The following keys are checked: ``PhysicalSizeXUnit``, ``PhysicalSizeYUnit``,
    ``PhysicalSizeZUnit``, and ``unit``.
    * Only a simple string substitution is performed; no numerical unit conversion
    of the corresponding physical size values is applied.
    * The function mutates the input dictionary and also returns it for convenience.
    """
    # define the keys to check and their default value:
    unit_keys = [
        'PhysicalSizeXUnit',
        'PhysicalSizeYUnit',
        'PhysicalSizeZUnit',
        'unit']

    # loop over each key and check/update:
    for key in unit_keys:
        # add key with default value if missing:
        if key not in metadata:
            metadata[key] = pixelunit
        
        # convert 'µm' to 'micron' if present:
        elif metadata[key] == 'µm':
            metadata[key] = 'micron'

    # "unit" 

    return metadata

def _normalize_tiff_axes_string(reference_axes: str) -> str:
    """
    Normalize TIFF axis labels to OMIO's expected conventions.
    """
    if not isinstance(reference_axes, str):
        return reference_axes

    # in some weird tifs, an "I" is put instead of "T", so we correct for that:
    reference_axes = reference_axes.replace('I', 'T')

    # if reference_axes=="YXS", we assume we got a RGB image and thus we convert S to C:
    if reference_axes == "YXS":
        reference_axes = "YXC"

    # if there is a "Q" in reference_axes, we convert it to "C", "T" or "Z" (depending
    # on what is missing and in this order):
    if 'Q' in reference_axes:
        if 'C' not in reference_axes:
            reference_axes = reference_axes.replace('Q', 'C')
        elif 'T' not in reference_axes:
            reference_axes = reference_axes.replace('Q', 'T')
        elif 'Z' not in reference_axes:
            reference_axes = reference_axes.replace('Q', 'Z')
        elif 'P' not in reference_axes:
            reference_axes = reference_axes.replace('Q', 'P')
        else:
            raise ValueError(
                "Error: Unable to map axis 'Q' to C, T, Z or P, as all are already present in reference axes."
            )

    return reference_axes

def _get_axes_from_shaped_metadata(shaped_metadata):
    """
    Extract an axis string from tifffile's shaped metadata if available.
    """
    if isinstance(shaped_metadata, dict):
        candidates = [shaped_metadata]
    elif isinstance(shaped_metadata, (list, tuple)):
        candidates = [item for item in shaped_metadata if isinstance(item, dict)]
    else:
        candidates = []

    for item in candidates:
        axes = item.get("axes")
        if isinstance(axes, str) and axes:
            return axes

    return None

# function to check and update metadata axes and its correct order from reading:
def _ensure_axes_in_metadata(metadata, tif):
    """
    Ensure that axis metadata matches the axis order reported by a TIFF file.

    This helper verifies that the ``"axes"`` entry in a metadata dictionary is
    present and consistent with the axis declaration provided by
    ``tif.series[0].axes``. If the key is missing or inconsistent, it is updated
    to match the TIFF reference.

    A known non-standard convention in some TIFF files, where the time axis is
    encoded as ``"I"`` instead of ``"T"``, is explicitly corrected.

    Parameters
    ----------
    metadata : dict
        Metadata dictionary to be updated.
    tif : tifffile.TiffFile
        Opened TIFF file object from which the reference axis order is obtained.

    Returns
    -------
    metadata : dict
        The updated metadata dictionary with a validated ``"axes"`` entry.

    Notes
    -----
    * The function attempts to read ``tif.series[0].axes`` and falls back to the
    string ``"unknown"`` if this fails.
    * If an ``"axes"`` entry already exists and differs from the TIFF reference,
    it is overwritten and a diagnostic message is printed.
    * The input dictionary is modified in place and also returned for convenience.
    """
    try:
        # reference axes from tif.series[0]:
        reference_axes = tif.series[0].axes
    except (IndexError, AttributeError):
        print("Error: Unable to extract axes from tif.series[0]. Setting to 'unknown'.")
        reference_axes = 'unknown'

    reference_axes = _normalize_tiff_axes_string(reference_axes)
    shaped_axes = _normalize_tiff_axes_string(
        _get_axes_from_shaped_metadata(getattr(tif, "shaped_metadata", None))
    )
    target_ndim = len(metadata.get("shape", ())) if metadata.get("shape") is not None else 0

    if shaped_axes and target_ndim and len(reference_axes) != target_ndim and len(shaped_axes) == target_ndim:
        reference_axes = shaped_axes

    existing_axes = metadata.get("axes")
    if (
        isinstance(existing_axes, str)
        and target_ndim
        and len(existing_axes) == target_ndim
        and len(reference_axes) != target_ndim
    ):
        reference_axes = _normalize_tiff_axes_string(existing_axes)

    if 'axes' in metadata:
        # overwrite if the existing axes do not match:
        if metadata['axes'] != reference_axes:
            print(f"Mismatch found: existing axes '{metadata['axes']}' does not match reference axes '{reference_axes}'. Overwriting.")
            metadata['axes'] = reference_axes
    else:
        # add the 'axes' key if it is missing:
        metadata['axes'] = reference_axes

    return metadata

# function to ensure shape in metadata:
def _ensure_shape_in_metadata(metadata, image_shape):
    """
    Ensure that shape metadata matches the actual image array shape.

    This helper verifies that the ``"shape"`` entry in a metadata dictionary is
    present and consistent with the provided image shape. If the key is missing or
    contains a different value, it is updated to reflect the true shape of the
    image array.

    Differences between the stored metadata shape and the actual array shape can
    occur when readers collapse singleton dimensions. Such mismatches are corrected
    and reported via diagnostic messages.

    Parameters
    ----------
    metadata : dict
        Metadata dictionary to be updated.
    image_shape : tuple
        Actual shape of the image array.

    Returns
    -------
    metadata : dict
        The updated metadata dictionary with a validated ``"shape"`` entry.

    Notes
    -----
    * If a mismatch is detected, the metadata value is overwritten and a diagnostic
    message is printed.
    * The input dictionary is modified in place and also returned for convenience.
    """
    if 'shape' in metadata:
        # overwrite if the existing shape does not match:
        if metadata['shape'] != image_shape:
            print(f"  Info: Mismatch found between actual image shape {image_shape} and shape {metadata['shape']}")
            print(f"        read from its metadata. Correcting metadata entry. This is nothing to worry about, as")
            print(f"        the tifffile reader either squashed singleton dimensions in the shape or OMIO folded S into C.")
            metadata['shape'] = image_shape
    else:
        # add the 'shape' key if it is missing:
        metadata['shape'] = image_shape
    
    return metadata

# function to fold sample axis 'S' into channel axis 'C':
def _fold_samples_axis_into_channel(image,
                                    axes: str,
                                    zarr_store: str | None = None,
                                    cache_folder: str | None = None,
                                    base_name: str = "omio",
                                    verbose: bool = True):
    """
    Fold tifffile sample axis 'S' (e.g. RGB samples per pixel) into channel axis 'C'.

    Behavior
    * If 'S' not in axes: return unchanged.
    * If no 'C' exists: rename S -> C (no folding, just renaming).
    * If both 'C' and 'S' exist: fold into a single channel axis: C_new = C_old * S.
      For Zarr inputs, this creates a new Zarr array and copies slice-wise.

    Parameters
    ----------
    image : np.ndarray or zarr.core.array.Array
    axes : str
    zarr_store : {None, "memory", "disk"}
        If image is Zarr and zarr_store is not None, keep result as Zarr.
        If None, Zarr input will be materialized to NumPy.
    cache_folder : str or None
        Required for zarr_store="disk". Folder where a new .zarr store is created.
    base_name : str
        Used to name disk stores.
    """

    if "S" not in axes:
        return image, axes

    s_idx = axes.index("S")

    # case A: no channel axis exists, typical RGB: YXS -> YXC:
    if "C" not in axes:
        if verbose:
            print("  Info: Found sample axis 'S' without channel axis. Renaming S->C.")
        return image, axes.replace("S", "C")

    c_idx = axes.index("C")

    # For simplicity and predictability, enforce that C is before S.
    # If not, we will treat it logically anyway.
    axes_out = axes.replace("S", "")

    # NumPy path:
    if not isinstance(image, zarr.core.array.Array):
        if verbose:
            print("  Info: Found sample axis 'S' and channel axis 'C'. Folding S into C (NumPy).")

        arr = np.asarray(image)

        # move S next to C (right after C) if needed:
        if s_idx != c_idx + 1:
            arr = np.moveaxis(arr, s_idx, c_idx + 1)

            axes_list = list(axes)
            s_char = axes_list.pop(s_idx)
            axes_list.insert(c_idx + 1, s_char)
            axes = "".join(axes_list)
            s_idx = c_idx + 1

        c_size = arr.shape[c_idx]
        s_size = arr.shape[s_idx]
        new_c = int(c_size) * int(s_size)

        new_shape = list(arr.shape)
        new_shape[c_idx] = new_c
        new_shape.pop(s_idx)

        arr = arr.reshape(tuple(new_shape))
        return arr, axes_out

    # zarr path:
    if zarr_store not in (None, "memory", "disk"):
        raise ValueError(f"_fold_samples_axis_into_channel: invalid zarr_store={zarr_store!r}")

    if zarr_store is None:
        # policy: if caller did not request Zarr persistence, we materialize
        if verbose:
            print("  Info: Zarr input but zarr_store=None. Materializing to NumPy for S->C folding.")
        arr = np.asarray(image[...])
        return _fold_samples_axis_into_channel(arr, axes, zarr_store=None, verbose=verbose)

    if verbose:
        print("  Info: Found sample axis 'S' and channel axis 'C'. Folding S into C (Zarr, slice-wise).")

    # build output shape by replacing C with C*S and dropping S:
    src = image
    src_shape = src.shape

    c_size = int(src_shape[c_idx])
    s_size = int(src_shape[s_idx])
    new_c = c_size * s_size

    out_shape = list(src_shape)
    out_shape[c_idx] = new_c
    out_shape.pop(s_idx)
    out_shape = tuple(out_shape)

    # determine output chunks based on axes_out and out_shape:
    out_chunks = compute_default_chunks(out_shape, axes_out)

    # create output Zarr array:
    if zarr_store == "memory":
        store = zarr.storage.MemoryStore()
        dst = zarr.open(store=store, mode="w", shape=out_shape, dtype=src.dtype, chunks=out_chunks)
    else:
        if cache_folder is None:
            raise ValueError("_fold_samples_axis_into_channel: cache_folder must be provided for zarr_store='disk'")
        os.makedirs(cache_folder, exist_ok=True)
        out_path = os.path.join(cache_folder, f"{base_name}_Sfold.zarr")
        if os.path.exists(out_path):
            shutil.rmtree(out_path)
        dst = zarr.open(out_path, mode="w", shape=out_shape, dtype=src.dtype, chunks=out_chunks)

    # copy slice-wise:
    # we copy per outer index over all dims except (C, S, Y, X), and for each (c, s)
    # write one (Y, X) plane into the correct folded channel.
    iy = axes.index("Y")
    ix = axes.index("X")

    outer_axes = [k for k in range(len(axes)) if k not in (c_idx, s_idx, iy, ix)]
    outer_shape = tuple(src_shape[k] for k in outer_axes)
    total_outer = int(np.prod(outer_shape)) if outer_shape else 1

    iterator = tqdm(np.ndindex(*outer_shape), total=total_outer, desc="    Folding S into C")
    for outer_idx in iterator:
        # build a template index for src of length src.ndim:
        src_index = [slice(None)] * len(axes)
        pos = 0
        for k in outer_axes:
            src_index[k] = outer_idx[pos]
            pos += 1

        # now loop channels and samples and copy planes:
        for c in range(c_size):
            for s in range(s_size):
                src_index[c_idx] = c
                src_index[s_idx] = s

                # dest index is like src but without S, and C is folded:
                dst_index = []
                for k in range(len(axes)):
                    if k == s_idx:
                        continue
                    if k == c_idx:
                        dst_index.append(c * s_size + s)
                    else:
                        dst_index.append(src_index[k])

                dst[tuple(dst_index)] = src[tuple(src_index)]

    return dst, axes_out

# function to pick first array from zarr group according OMIO multi-series policy:
def _zarr_pick_first_array(z, prefer_keys=("0",), verbose=True):
    """
    Return a Zarr array from a Zarr object that might be a Group.
    Policy: prefer common full-resolution keys ("0"), otherwise take the first array-like entry.
    """
    # already an array-like object:
    if hasattr(z, "shape") and hasattr(z, "dtype"):
        return z

    # group-like: try to find arrays:
    keys = []
    try:
        # zarr Group has keys() in both zarr2 and zarr3:
        keys = list(z.keys())
    except Exception:
        keys = []

    # 1) prefer known keys:
    for k in prefer_keys:
        if k in keys:
            cand = z[k]
            if hasattr(cand, "shape") and hasattr(cand, "dtype"):
                if verbose:
                    print(f"  Info: Zarr Group detected. Using array key '{k}' with shape {cand.shape}.")
                return cand

    # 2) otherwise take the first array-like entry in sorted key order:
    for k in sorted(keys):
        cand = z[k]
        if hasattr(cand, "shape") and hasattr(cand, "dtype"):
            if verbose:
                print(f"  Info: Zarr Group detected. Using first array-like key '{k}' with shape {cand.shape}.")
            return cand

    raise TypeError(
        "read_tif: aszarr=True returned a Zarr Group, but no array-like entries were found.")

# helper-function to copy large arrays in (Y,X) slices memory-friendly into Zarr:
def _copy_to_zarr_in_xy_slices(src, dst, desc="slice-wise copying to Zarr"):
    """
    Copy an array to a Zarr destination by streaming (Y, X) slices.

    This helper performs a memory-friendly copy from `src` to `dst` by iterating
    over all outer dimensions and copying one full spatial plane at a time. It is
    intended for large arrays where copying the entire dataset into RAM would be
    undesirable.

    The function assumes that the last two axes of `src` and `dst` correspond to
    the spatial dimensions (Y, X). For arrays with two or fewer dimensions, the
    copy is performed in a single assignment.

    Parameters
    ----------
    src : array-like
        Source array supporting NumPy-style slicing. Typically a Zarr array or a
        NumPy array.
    dst : zarr.core.array.Array or array-like
        Destination array supporting NumPy-style slicing and assignment. Typically
        a Zarr array that has the same shape as `src`.
    desc : str, optional
        Description passed to the progress bar. Default is
        ``"slice-wise copying to Zarr"``.

    Returns
    -------
    None

    Notes
    -----
    * The copy is performed slice-wise over all indices of ``src.shape[:-2]`` and
    transfers full ``(:, :)`` planes for the last two dimensions.
    * The function does not perform shape or dtype validation; callers are expected
    to ensure compatibility between `src` and `dst`.
    * Progress reporting is provided via ``tqdm``.
    """
    src_shape = src.shape

    # trivial case: 0D, 1D or 2D -> copy in one go:
    if len(src_shape) <= 2:
        dst[...] = src[...]
        return

    outer_shape = src_shape[:-2]
    
    # determine number of slices to process for tqdm:
    total = int(np.prod(outer_shape))

    for outer_idx in tqdm(np.ndindex(*outer_shape), total=total, desc=desc):
        # build full index: (i0, i1, ..., i_{n-3}, :, :)
        idx = outer_idx + (slice(None), slice(None))
        dst[idx] = src[idx]

def _split_paginated_tiff_stack(image,
                                metadata: Dict[str, Any],
                                fname: str,
                                zarr_store: str | None,
                                zarr_store_path: Union[None, str, os.PathLike] = None,
                                verbose: bool = True) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """
    Split a paginated TIFF/LSM stack (axis ``P``) into per-page OMIO images.

    This helper is shared between the normal TIFF reader path and disk-cache
    reuse, so it only depends on the already prepared image array and metadata.
    """
    axis_to_use = "P"
    if verbose:
        print(f"  Detected paginated TIFF/LSM (axis '{axis_to_use}'); splitting into individual pages.")

    metadata = metadata.copy()
    metadata["original_metadata_type"] = "paginated_tif/lsm"
    metadata["spacing"] = metadata["PhysicalSizeZ"]
    metadata["PhysicalSizeXUnit"] = metadata["unit"]
    metadata["PhysicalSizeYUnit"] = metadata["unit"]
    metadata["PhysicalSizeZUnit"] = metadata["unit"]
    metadata["OMIO_VERSION"] = _OMIO_VERSION

    p_index = metadata["axes"].index(axis_to_use)
    nP = image.shape[p_index]
    axes_wo_P = metadata["axes"][:p_index] + metadata["axes"][p_index+1:]

    images = []
    metadatas = []
    for p in range(nP):
        slicer = [slice(None)] * image.ndim
        slicer[p_index] = p
        page_data = image[tuple(slicer)]

        if page_data.ndim == image.ndim:
            page_data = np.squeeze(page_data, axis=p_index)

        page_md = metadata.copy()
        page_md["axes"] = axes_wo_P
        page_md["shape"] = page_data.shape

        if zarr_store is None:
            images.append(np.asarray(page_data))
        else:
            page_shape = page_data.shape
            chunks = compute_default_chunks(page_shape, axes_wo_P)
            if verbose:
                print(f"    Page {p}: using chunks {chunks}")

            if zarr_store == "memory":
                store = zarr.storage.MemoryStore()
                page_zarr = zarr.open(
                    store=store,
                    mode="w",
                    shape=page_shape,
                    dtype=page_data.dtype,
                    chunks=chunks)
            else:
                page_path = _get_disk_cache_path(
                    fname,
                    suffix=f"_P{p}",
                    zarr_store_path=zarr_store_path)
                os.makedirs(os.path.dirname(page_path), exist_ok=True)
                if os.path.exists(page_path):
                    shutil.rmtree(page_path)
                page_zarr = zarr.open(
                    page_path,
                    mode="w",
                    shape=page_shape,
                    dtype=page_data.dtype,
                    chunks=chunks)

            _copy_to_zarr_in_xy_slices(page_data, page_zarr, desc=f"    Copying page {p} to Zarr")
            if zarr_store == "disk":
                page_md = _annotate_disk_cache_metadata(
                    page_md,
                    fname=fname,
                    zarr_path=page_path,
                    zarr_store_path=zarr_store_path)
            images.append(page_zarr)

        page_md = OME_metadata_checkup(page_md, verbose=verbose)
        metadatas.append(page_md)

    memap_large_file = (zarr_store == "disk")
    images, metadatas = _batch_correct_for_OME_axes_order(images, metadatas, memap_large_file, verbose=verbose)

    if verbose:
        print(f"  Finished splitting paginated TIFF into {nP} pages.")
        print("Reading paginated TIFF completed.")
    return images, metadatas

# function to compute default chunking for Zarr arrays out of image shape and axes:
def compute_default_chunks(shape, axes, max_xy_chunk=1024): 
    """
    Compute a default chunk pattern for Zarr arrays given a shape and axis string.

    Policy:
    - All non-spatial axes (e.g. T, Z, C) are chunked with size 1.
    - Spatial axes Y and X get chunk sizes up to `max_xy_chunk`,
      limited by the actual dimension size.
    - The order of chunk sizes follows `shape` and `axes` one-to-one.

    Parameters
    ----------
    shape : tuple of int
        Full array shape, e.g. (T, Z, C, Y, X).
    axes : str
        Axis string describing the layout, e.g. "TZCYX".
    max_xy_chunk : int, optional
        Maximum chunk size along Y and X. Defaults to 1024.

    Returns
    -------
    tuple of int
        Chunk sizes for each axis, same length as `shape`.
    """
    if len(shape) != len(axes):
        raise ValueError(
            f"Shape {shape} and axes '{axes}' have different lengths "
            f"({len(shape)} vs {len(axes)}).")

    chunks = [1] * len(shape)
    axis_to_index = {ax: i for i, ax in enumerate(axes)}

    # Y chunk:
    if "Y" in axis_to_index:
        iy = axis_to_index["Y"]
        chunks[iy] = min(shape[iy], max_xy_chunk)

    # X chunk:
    if "X" in axis_to_index:
        ix = axis_to_index["X"]
        chunks[ix] = min(shape[ix], max_xy_chunk)

    return tuple(chunks)

def _check_for_not_covered_metadata(tif, yet_covered_metadata, ignore_metadata=None):
    """
    Report metadata entries provided by tifffile that are not yet handled.

    This helper inspects a ``tifffile.TiffFile`` object for available ``*_metadata``
    attributes beyond those that are already covered by the current implementation.
    For each uncovered metadata entry that is present and non-null, a diagnostic
    message is printed to inform the user that additional metadata types exist but
    are not yet supported.

    The function is intended as a developer and user-facing diagnostic to highlight
    potentially relevant metadata formats and to encourage reporting of unsupported
    cases.

    Parameters
    ----------
    tif : tifffile.TiffFile
        Opened TIFF file object to be inspected for available metadata attributes.
    yet_covered_metadata : iterable of str
        Collection of metadata attribute names that are already handled and should
        be ignored during inspection.
    ignore_metadata : iterable of str or None, optional
        Additional metadata attribute names to be ignored during inspection.

    Returns
    -------
    None

    Notes
    -----
    * The function looks for attributes whose names end with ``"_metadata"``.
    * Metadata attributes listed in ``yet_covered_metadata`` are explicitly skipped.
    * Only metadata attributes that exist and return a non-``None`` value are
    reported.
    * The function produces output via printing and does not return structured
    information.
    """
    available_methods = dir(tif)
    available_metadata = []
    for method_name in available_methods:
        # we do not add imagej_metadata, ome_metadata or lsm_metadata again:
        if method_name in yet_covered_metadata:
            continue
        if method_name.endswith("_metadata"):
            try:
                #metadata_value = getattr(tif, method_name)
                available_metadata.append(method_name)
            except Exception as e:
                print(f"  Could not read metadata '{method_name}': {e}")
    #print("Available metadata entries in tifffile:", available_metadata.keys())
    # loop through available_metadata and check, which tif.available_metadata[i] is not None:
    not_readables = []
    for metadata_name in available_metadata:
        try: 
            metadata_value = getattr(tif, metadata_name)
            if metadata_value is not None and (ignore_metadata is None or metadata_name not in ignore_metadata):
                print(f"  Found available metadata '{metadata_name}' which is not yet implemented. Please contact")
                print(f"    the developers at https://github.com/FabrizioMusacchio/omio/issues and provide")
                print(f"    details and an example file. Please refer to the documentation for more information.")
        except Exception as e:
            not_readables.append(metadata_name)
            # print(f"  _check_for_not_covered_metadata: Could not read metadata '{metadata_name}': {e}")
    """ if len(not_readables) > 0:
        print(f"\n  _check_for_not_covered_metadata couldn't check all available metadata due to errors:\n    {not_readables}") """

# function for post-hoc shifting non-reserved OME-metadata into Annotations:
def OME_metadata_checkup(metadata: dict, 
                         namespace: str ="omio:metadata",
                         verbose: bool = True) -> dict:
    """
    Normalize metadata by collecting non-core entries into an OME Annotations block.

    This function performs a post-hoc cleanup of a metadata dictionary by separating
    core OME-compatible fields from auxiliary or tool-specific metadata. All
    non-core keys that are not explicitly retained at the top level are moved into
    a single ``"Annotations"`` dictionary, which is suitable for serialization as
    an OME ``MapAnnotation`` block.

    The input metadata dictionary is not modified in place; all operations are
    performed on a shallow copy.

    Parameters
    ----------
    metadata : dict
        Input metadata dictionary.
    namespace : str, optional
        Namespace identifier to be stored under ``Annotations["Namespace"]``.
        Default is ``"omio:metadata"``.

    Returns
    -------
    md : dict
        Normalized metadata dictionary in which auxiliary fields have been moved
        into an ``"Annotations"`` entry.

    Notes
    -----
    * Core OME-like keys (for example physical sizes, time increment, and axis
      declarations) remain at the top level.
    * Selected non-OME but operationally useful keys (such as ``Size*`` entries,
      ``shape``, and ``Channel_Count``) are explicitly retained at the top level.
    * All remaining keys are transferred into ``Annotations``.
    * Existing annotations are preserved and extended. The namespace is always
      set or overwritten with the provided value.
    * Keys starting with ``"original_"`` in an existing ``Annotations`` block are
      protected from being overwritten.
    """

    # define truly OME-like core keys that correspond to real OME attributes:
    core_keys = {
        "axes",
        "PhysicalSizeX", "PhysicalSizeY", "PhysicalSizeZ",
        "PhysicalSizeXUnit", "PhysicalSizeYUnit", "PhysicalSizeZUnit",
        "Description",
        "TimeIncrement", "TimeIncrementUnit"}

    # keys that are useful for downstream processing but are not written
    # into OME XML; they will be re-read/computed by Fiji or OMIO on load
    # anyways, and therefore stay at top-level:
    keep_keys = {
        "Annotations",           # handled explicitly
        "SizeX", "SizeY", "SizeZ", "SizeC", "SizeT",
        "Channel_Count", "shape", # "spacing", "unit",
        # note: key starting with original_*  are intentionally NOT in 
        # keep_keys, so that they are moved into Annotations
    }

    # work on a copy to avoid modifying the input in-place:
    md = dict(metadata)

    # start from any existing Annotations block if present:
    existing_annotations = md.get("Annotations", {})
    if not isinstance(existing_annotations, dict):
        existing_annotations = {}

    # copy existing annotations and FORCE our namespace:
    annotations = dict(existing_annotations)
    annotations["Namespace"] = namespace

    # collect all non-core, non-keep keys and move them into Annotations
    # while removing them from the metadata top-level:
    extra_keys = {}
    for key, value in list(md.items()):
        # skip core keys and keys we explicitly want to keep at top-level:
        if key in core_keys or key in keep_keys:
            continue
        extra_keys[key] = value
        del md[key]

    # now merge extra_keys into annotations:
    for key, value in extra_keys.items():
        # never overwrite existing "original_*" entries in Annotations:
        if key in annotations and key.startswith("original_"):
            if verbose:
                print(f"    Info: Skipping overwrite of original metadata key '{key}' in Annotations.")
            continue
        annotations[key] = value

    # write back the assembled annotations block
    md["Annotations"] = annotations

    return md

# %% ALL
__all__ = [name for name in globals() if not name.startswith("__")]
# %% END
