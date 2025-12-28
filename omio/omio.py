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

Author and provenance
---------------------
Author: Fabrizio Musacchio  
First version: December 21, 2025

This module is part of the OMIO project and is developed in the context of
scientific microscopy data processing workflows.
"""
# %% IMPORTS
import os, re
import hashlib

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

# make current _OMIO_VERSION available as 'version' attribute outside the module:
version = _OMIO_VERSION
# %% HELPER FUNCTIONS FOR READERS

# a simple hello world function (sanity check for external imports):
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

# function for correcting the axes order to OME-conform:
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
            # if C, T and Z are already present, we give a raise error:
            raise ValueError("Error: Unable to map axis 'Q' to C, T, Z or P, as all are already present in reference axes.")
            

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

# function to find a single yaml file in a folder (used for Thorlabs RAW metadata):
def _find_single_yaml(folder):
    """
    Locate a single YAML metadata file in a directory.

    This helper scans a directory for files with ``.yaml`` or ``.yml`` extensions
    and returns the path to a YAML file if present. It is primarily used to locate
    Thorlabs RAW metadata stored alongside image data.

    If no YAML files are found, the function returns ``None``. If multiple YAML
    files are present, a warning is issued and the first file encountered is
    returned.

    Parameters
    ----------
    folder : str
        Path to the directory to be searched.

    Returns
    -------
    yaml_path : str or None
        Full path to the YAML file if at least one is found, otherwise ``None``.

    Notes
    -----
    * When multiple YAML files are detected, the function does not attempt to
    disambiguate them beyond issuing a warning.
    * The order in which files are inspected follows ``os.listdir`` and is
    therefore platform-dependent.
    """
    yamls = [f for f in os.listdir(folder) if f.lower().endswith((".yaml", ".yml"))]
    if len(yamls) == 0:
        return None
    if len(yamls) > 1:
        warnings.warn(
            f"Multiple YAML metadata files found\n    in {folder}: \n    {yamls}\n"
            "    Please keep exactly one .yaml/.yml file for Thorlabs RAW metadata.\n"
            "    Will now take the first one found.")
    return os.path.join(folder, yamls[0])

# function to load yaml metadata (used for Thorlabs RAW metadata):
def _load_yaml_metadata(yaml_path):
    """
    Load YAML metadata from a file into a dictionary.

    This helper reads a YAML file from disk and parses its contents into a Python
    dictionary. It is intended for loading auxiliary metadata, such as Thorlabs RAW
    metadata stored alongside image data.

    The function requires PyYAML to be installed and uses ``yaml.safe_load`` for
    parsing. Empty YAML files are treated as empty dictionaries.

    Parameters
    ----------
    yaml_path : str
        Path to the YAML metadata file.

    Returns
    -------
    data : dict
        Dictionary containing the parsed YAML metadata. If the file is empty, an
        empty dictionary is returned.

    Raises
    ------
    ImportError
        If PyYAML is not installed.
    ValueError
        If the top-level YAML object is not a mapping/dictionary.

    Notes
    -----
    * Parsing is performed using ``yaml.safe_load`` to avoid execution of arbitrary
    code.
    * The function assumes UTF-8 encoding when reading the file.
    """
    if yaml is None:
        raise ImportError(
            "PyYAML is not installed, but a YAML metadata file was found. "
            "Install with: pip install pyyaml")
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML file {yaml_path} must contain a mapping/dict at top-level.")
    return data

# function that creates a dummy YAML files at fname's folder with the required keys:
def create_thorlabs_raw_yaml(fname: str,
                             T: int = 1, Z: int = 1, C: int = 1, Y: int = 1024, X: int=1024, bits: int = 16,
                             physicalsize_xyz: Union[tuple, list, None] = None, 
                             pixelunit: str = "micron",
                             time_increment: Union[float, None] = None, time_increment_unit: Union[str, None] = None,
                             annotations: Union[dict, None] = None, verbose: bool = True):
    """
    Create a dummy YAML file with the required keys for Thorlabs RAW metadata.
    This utility generates a YAML file in the same folder as the specified RAW file
    (`fname`) containing the necessary keys for reading the RAW file with
    `read_thorlabs_raw`. The generated YAML file serves as a metadata source when
    no XML metadata is available.
    Parameters
    ----------
    fname : str
        Path to the Thorlabs RAW file. The YAML file will be created in the same
        folder.
    T : int
        Number of time points. Default is 1.
    Z : int
        Number of Z slices. Default is 1.
    C : int
        Number of channels. Default is 1.
    Y : int
        Image height in pixels. Default is 1024.
    X : int
        Image width in pixels. Default is 1024.
    bits : int
        Bit depth per pixel (e.g., 8, 16, 32). Default is 16.
    physicalsize_xyz : tuple of float or None, optional
        Voxel sizes in the order ``(PhysicalSizeX, PhysicalSizeY, PhysicalSizeZ)``.
        Default is None.    
    pixelunit : str, optional
        Unit string for pixel sizes. Default is ``"micron"``.
    time_increment : float or None, optional
        Time increment between frames. Default is None.
    time_increment_unit : str or None, optional
        Unit for the time increment. Default is None.
    annotations : dict or None, optional
        Additional key-value pairs to include in the YAML file. Default is None.
    verbose : bool, optional
        If True, print diagnostic messages. Default is True.
    Returns
    -------
    None
    Raises
    ------
    IOError
        If the YAML file cannot be written. 
        
    Notes
    -----
    * The generated YAML file includes the required keys for Thorlabs RAW reading.
    * Additional annotations can be included via the `annotations` parameter.
    """ 
    
    folder = os.path.dirname(fname)
    fname_base, _ = os.path.splitext(os.path.basename(fname))
    yaml_path = os.path.join(folder, fname_base + "_metadata.yaml")
    ymd = {
        "T": T,
        "Z": Z,
        "C": C,
        "Y": Y,
        "X": X,
        "bits": bits,
    }
    if physicalsize_xyz is not None:
        ymd["PhysicalSizeX"] = physicalsize_xyz[0]
        ymd["PhysicalSizeY"] = physicalsize_xyz[1]
        ymd["PhysicalSizeZ"] = physicalsize_xyz[2]
    if pixelunit is not None:
        ymd["PixelUnit"] = pixelunit
    if time_increment is not None:
        ymd["TimeIncrement"] = time_increment
    if time_increment_unit is not None:
        ymd["TimeIncrementUnit"] = time_increment_unit
    if annotations is not None:
        ymd.update(annotations)

    with open(yaml_path, "w") as f:
        yaml.dump(ymd, f)

    if verbose:
        print(f"Created dummy YAML metadata file at {yaml_path}")

# function to require integer from dictionary (for housekeeping):
def _require_int(d, key):
    """
    Retrieve and cast a dictionary value to an integer.

    This helper enforces the presence of a specific key in a dictionary and returns
    its value cast to an integer. It is intended for simple validation and
    housekeeping tasks where integer-valued entries are required.

    Parameters
    ----------
    d : dict
        Dictionary from which the value is retrieved.
    key : hashable
        Key that must be present in the dictionary.

    Returns
    -------
    value : int
        Integer value associated with `key`.

    Raises
    ------
    KeyError
        If `key` is not present in the dictionary.
    ValueError
        If the value associated with `key` cannot be converted to an integer.
    """
    if key not in d:
        raise KeyError(key)
    return int(d[key])

# function to check for not yet covered metadata in tifffile:
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

# %% READER FUNCTIONS

# tif or lsm file reader (including series and paginated files):
def read_tif(fname, physicalsize_xyz=None, pixelunit="micron", 
             zarr_store=None, return_list=False, verbose=True):
    """
    Read TIFF family files into OMIO's canonical representation.

    This function reads TIFF, OME-TIFF, multi file OME-TIFF series, and 
    Zeiss LSM files using `tifffile`, extracts available metadata (OME-XML, ImageJ 
    metadata, and LSM metadata), standardizes metadata keys, and normalizes axis 
    handling to canonical OME order TZCYX. Depending on configuration, the returned 
    image is either a NumPy array in RAM or a Zarr array backed by an in-memory or 
    on-disk store.

    If the input is a paginated TIFF or LSM (axis "P"), OMIO splits the dataset into
    individual pages and returns a list of images together with a list of matching
    metadata dictionaries. In that case, lists are returned regardless of
    `return_list`, because a single object return would be semantically ambiguous.

    Parameters
    ----------
    fname : str
        Path to the input file. Note: read_tif is the core function
        for TIF and LSM file reading; omio.read() dispatches to this function when
        encountering a .tif or .lsm file. read_tif can only handle TIF and LSM files 
        but no folder paths (for this, please use read_thorlabs_raw_folder).
    physicalsize_xyz : tuple of float or None, optional
        Manual override for voxel sizes in the order
        ``(PhysicalSizeX, PhysicalSizeY, PhysicalSizeZ)``. If provided, these values
        override metadata-derived sizes. If None, missing sizes fall back to 1.0.
        Default is None.
    pixelunit : str, optional
        Unit string used for pixel size fields and unit normalization. Default is
        ``"micron"``.
    zarr_store : {None, "memory", "disk"}, optional
        Controls the representation of the returned image data.

        * None: load fully into RAM and return a NumPy array
        * "memory": return a Zarr array backed by an in-memory store
        * "disk": return a Zarr array stored in the cache folder
          ``{parent}/.omio_cache/<basename>.zarr``

        Default is None.
    return_list : bool, optional
        If True, force backward-compatible list return for non-paginated inputs by
        returning ``[image]`` and ``[metadata]``. Default is False.
    verbose : bool, optional
        If True, print diagnostic progress messages. Default is True.

    Returns
    -------
    image : np.ndarray or zarr.core.array.Array or list
        Image data in canonical OME axis order TZCYX. For paginated inputs, a list
        of per-page arrays is returned.
    metadata : dict or list
        Metadata dictionary aligned with the returned image. For paginated inputs,
        a list of per-page metadata dictionaries is returned.

    Raises
    ------
    ValueError
        If `zarr_store` is not one of {None, "memory", "disk"}.

    Notes
    -----
    * Metadata sources are merged in the order they are read. Missing essentials
      are filled from the image shape and default values.
    * Unit normalization updates unit fields only. Numerical unit conversion is not
      performed except for specific paginated LSM cases where Zeiss voxel sizes are
      converted from meters to micrometers.
    * If `zarr_store` is not None, tifffile's ``aszarr=True`` path is used and then
      materialized into a concrete Zarr store to ensure predictable downstream
      behavior. Data transfer uses slice-wise copying over the last two spatial
      dimensions to limit peak memory use.
    * Axis normalization to TZCYX may insert singleton dimensions for missing OME
      axes and may reorder existing axes. The updated axis string is stored in the
      returned metadata.
    * When `zarr_store="disk"`, the function may create and overwrite paths under
      ``.omio_cache``.
    * Multi-file OME-TIFF series are supported. In this layout, individual OME-TIFF
      files each store subsets of the full dataset (e.g. single time points,
      channels, or z-slices). OMIO/tifffile reconstructs the complete logical image by
      following the OME-XML metadata references across files. It is therefore
      sufficient to pass the path of a single file belonging to the series; all
      referenced files are discovered and read implicitly. The resulting image is
      returned as a contiguous and complete stack in canonical OME axis order.
      
    General note on series and pages
    --------------------------------
    TIFF family containers can store data in two different structural layers that are 
    easy to confuse:

    * Series are top level image datasets within a container. Each series can have its 
      own dimensionality, axis semantics, pixel type, and metadata context. In tifffile, 
      these are exposed via `tif.series`.
    * Pages are the lower level IFD entries that physically store image planes or tiles. 
      Depending on the file layout, pages can represent planes along Z, C, or T, pyramid 
      levels, tiles, or other internal subdivisions. In tifffile, these are exposed via 
      `tif.pages`.

    In many microscopy TIFF variants, tifffile reconstructs a logical N dimensional array 
    for a series by reading and stacking its pages. The exact mapping depends on the file 
    and on tifffile’s internal interpretation of the container structure. OMIO therefore 
    treats `tif.series` as the authoritative high level grouping and applies explicit, 
    deterministic policies where the container structure could otherwise lead to ambiguous 
    outcomes.

    OMIO behavior for paginated files
    ----------------------------------
    Some TIFF and LSM files are stored as paginated stacks and expose an explicit pagination 
    axis `P` in the inferred axis string. OMIO treats pagination as a semantic split into 
    independent image stacks:

    * If the input is detected as paginated (axis `P` present), OMIO splits the dataset into 
      per page images and returns `images` and `metadatas` as lists with matching length.
    * Lists are returned regardless of `return_list`, because a single object return would be 
      semantically ambiguous once pagination is present.
    * Each returned metadata dictionary corresponds to exactly one page and reflects the page 
      specific axis string with the pagination axis removed.
    * If `zarr_store` is set, each page is materialized into its own Zarr array according to 
      the selected backend (memory or disk).
    * After splitting, OMIO applies axis normalization to each page so that each page is 
      returned in canonical OME axis order.
    
    OMIO restrictions for multi-series TIFF/LSM files
    -------------------------------------------------
    TIFF and LSM containers may store multiple datasets ("series") in a single file.
    While tifffile exposes these as `tif.series`, OMIO enforces a strict and predictable
    policy to avoid ambiguous interpretations:

    * If a file contains exactly one series (`len(tif.series) == 1`), OMIO guarantees
      correct reading and normalization to canonical OME axis order (TZCYX).
    * If a file contains multiple series (`len(tif.series) > 1`), OMIO will process
      **only the first series (series 0)** and ignore all others.
    * A warning is emitted in this case, and the policy decision is recorded in the
      returned metadata.
    * OMIO does not attempt to infer relationships between multiple series, does not
      concatenate them, and does not inspect their shapes, axes, or photometric
      interpretation beyond series 0.

    This policy is intentional and favors reproducibility and explicit behavior over
    heuristic reconstruction of complex TIFF layouts.
    """
    
    # validate zarr_store parameter:
    if zarr_store not in (None, "memory", "disk"):
        raise ValueError(
            "read_tif: zarr_store must be one of None, 'memory', or 'disk'. "
            f"Got: {zarr_store!r}")
    
    # check, whether the user wants to set the pixel size manually:  
    if not physicalsize_xyz:
        physicalsize_xyz_ext = (1.0,1.0,1.0)
        set_input_pixelsize = False
    else:
        physicalsize_xyz_ext = tuple(float(v) for v in physicalsize_xyz)
        set_input_pixelsize = True

    # read the tif file:
    with tifffile.TiffFile(fname) as tif:
        # find out, how many series/pages exist:
        nseries = len(tif.series)
        npages  = len(tif.pages)
        # OMIO multi-series policy:
        if nseries > 1:
            if verbose:
                print(
                    f"WARNING: OMIO detected a multi-series TIFF/LSM file with {nseries} series.\n"
                    f"         OMIO currently processes only the first series (series 0).\n"
                    f"         All additional series are ignored.")
            # record policy decision in metadata later:
            series_shapes = []
            series_axes = []
            series_photometric = []

            for i in range(nseries):
                try:
                    series_shapes.append(list(tif.series[i].shape))
                except Exception:
                    series_shapes.append(None)

                try:
                    series_axes.append(str(tif.series[i].axes))
                except Exception:
                    series_axes.append(None)
                try:
                    series_photometric.append(str(tif.series[i].pages[0].photometric.name))
                except Exception:
                    series_photometric.append(None)
            multi_series_info = {"OMIO_MultiSeriesDetected": True,
                                 "OMIO_TotalSeries": nseries,
                                 "OMIO_ProcessedSeries": 0,
                                 "OMIO_MultiSeriesPolicy": "only_series_0",
                                 "OMIO_MultiSeriesShapes": series_shapes,
                                 "OMIO_MultiSeriesAxes": series_axes,
                                 "OMIO_MultiSeriesPhotometric": series_photometric}
        else:
            multi_series_info = {"OMIO_MultiSeriesDetected": False}
        
        """ 
        The difference between series and pages:
            A TIFF file can contain multiple SERIES, each representing a distinct
            image dataset with its own dimensions and metadata. Each series can be
            composed of multiple PAGES, where each page corresponds to a single image
            plane or slice within that series. Thus, series are higher-level groupings
            of related image data, while pages are the individual components that make
            up those datasets.
            
            However, "pages" in tifffile can also refer to channels or slices within
            a single series, depending on the context. This dual usage can lead to confusion.
            
            Furthermore, tifffile sometimes reads paginated tiffs as an array of image series
            in paginated images, but sometimes it only reads the first series and skips the 
            rest. Thus, we would need to check whether a single image is read, but nseries > 1
            exist. This is complicated at the moment, as I do not know how tifffile decides 
            along which axes it concatenates SERIES into a single array and when it does not.
            I.e., I can not simply check whether len(image) == nseries, and if not, try to
            loop over tif.series to read all series separately. Thus, for now, we need to 
            restrict OMIO's tif reader to only allow cases where either a single series exists
            (single stack case) or where the tif is paginated (for me, this seems only to be
            the case for paginated LSM files so far). 
            
            In lsm files, what I figured out so far, is, that the series are sets of different
            image scales of the same data (e.g., downsampled versions) + some photographed image
            description sheets. Thus, if tifffile fetches in multiple series only the first, 
            multi-layered image series, that seems to be okay.
            
            Update: I think, I figured it out that tifffile reads a multi-series tiff/lsm 
            into a single array only if all series have the same shape and axes. And this is
            what we accept for now in OMIO, i.e., we do not guarantee to read other mixed
            multi-series shapes.
        """
        
        # read image data either fully into RAM or as Zarr;
        # first, NumPy array in RAM:
        if zarr_store is None:
            if verbose:
                print("Reading TIFF fully into RAM...")
            image = tif.asarray()
            """ print(f"len(tif.series): {len(tif.series)}, nseries: {nseries}, len(image): {len(image)}")
            print(f"image.shape: {image.shape}")
            for series in range(len(tif.series)):
                print(f"tif.series[series].axes: {tif.series[series].axes}, tif.series[series].shape: {tif.series[series].shape}")
            tags = []
            for tag in range(len(tif.pages)):
                tags.append(tif.pages[tag].tags)
            for tag in tags:
                for key in tag.keys():
                    print(key, tag[key]) 
                print("-----") """
            """ DRAFT for multi-series handling; see comments above and herein:
            print(f"len(tif.series): {len(tif.series)}, nseries: {nseries}, len(image): {len(image)}")
            for series in range(len(tif.series)):
                print(tif.series[series].axes, tif.series[series].shape)
                #print(tif.series[series].pages.shape)
            if len(tif.series) > 1 and len(tif.series[0].shape) == 3:
                # len(tif.series[0].shape) == 3 ensures that we get a true RGB YXS image
                
                # try to read all series separately into a list:
                image_list = []
                image_list.append(tifffile.imread(fname, series=0)) # read first series
                image0_shape = tif.series[0].shape
                image0_axes  = tif.series[0].axes
                for series in range(1, len(tif.series)):
                    if tif.series[series].axes == image0_axes and tif.series[series].shape == image0_shape:
                        image_list.append(tifffile.imread(fname, series=series))
                # after all series are read, concatenate all arrays in the list...but along which axis?
                # For now, I can't resolve this, so this if-block is disabled and the restrict OMIO to only
                # guarantee single-series or lsm paginated tiffs with non-complex axis/series/pages layouts.
                
                # UPDATE: We do it like FIJI: We concatenate in T so that we get a TZCYX array in the end.
                
                # create an empty array with the final shape:
                T_N = len(image_list)
                final_shape = (T_N,) + image0_shape
                image = np.zeros(final_shape, dtype=image_list[0].dtype)
                for t in range(T_N):
                    image[t, ...] = image_list[t]
                
            else:
                image = tif.asarray() 
            """
        else:
            if verbose:
                print("Reading TIFF as Zarr...")
            src_store = tifffile.imread(fname, aszarr=True)
            src = zarr.open(src_store, mode="r")
            
            # IMPORTANT: OME-TIFF and pyramidal TIFFs may open as a Zarr Group, not an Array.
            # OMIO policy: only use one dataset, deterministically.
            src_array = _zarr_pick_first_array(src, prefer_keys=("0",), verbose=verbose)

            image = src_array  # from here on, we require array semantics (shape, dtype, slicing)

            # create target Zarr (memory or disk):
            fname_base, _ = os.path.splitext(os.path.basename(fname))

            chunks = getattr(src_array, "chunks", None)
            # If chunks are not known, compute them from shape/axes later after metadata exists.
            # For now, keep a placeholder and compute after _ensure_axes_in_metadata().
            target = None

            #image = src  # temporary; may be replaced by target after we know axes/chunks

        """ DRAFT warning for multi-series handling; see comments above and herein:
        # I cannot do the following check here, as an RGB is read like YXS and thus
        # len(image) equal the size of Y, which is not what we want to check here.
        
        # warn user if we have multi-series tif but only a single image read:
        if len(tif.series)>1 and len(image) == 1:
            print(f"WARNING: read_tif: Encountered multi-series TIFF with {len(tif.series)} series,")
            print(f"         but only a single image array was read with shape {image.shape}.")
            print(f"         OMIO currently only guarantees correct reading of single-series")
            print(f"         TIFF files or paginated LSM files. Please report this issue to")
            print(f"         the developers at https://github.com/FabrizioMusacchio/omio/issues.") 
        """

        image_shape = image.shape
        
        # try to extract metadata from tag pages (if any):
        try:
            tags = []
            for tag in range(len(tif.pages)):
                tags.append(tif.pages[tag].tags)
        except Exception:
            tags = None
        
        """
        for tag in tags:
            for key in tag.keys():
                print(key, tag[key]) 
            print("-----")
        
        tags = tif.pages[0].tags
        for key in tags.keys():
            print(key, tags[key]) 
        """
        imagej_metadata = tif.imagej_metadata
        ome_metadata    = tif.ome_metadata
        lsm_metadata    = tif.lsm_metadata
        #shaped_metadata  = tif.shaped_metadata
        
        # check for not yet covered metadata and give feedback to user (if any):
        yet_covered_metadata = ["imagej_metadata", "ome_metadata", "lsm_metadata"]
        ignore_metadata = ["shaped_metadata"]  # empirically, shaped_metadata this always contains 
                                               # just the image shape, so we ignore it for now
        _check_for_not_covered_metadata(tif, yet_covered_metadata, ignore_metadata)
        
        metadata = {}
        if ome_metadata is not None:
            md_ome = _parse_ome_metadata(ome_metadata)
            metadata.update(md_ome)
            #metadata = _parse_ome_metadata(ome_metadata) # extract relevant fields from OME-XML
            metadata = _add_file_properties_to_metadata(metadata, fname, original_metadata_type="OME_XML")
            #metadata["axes"], metadata["shape"] = _extract_axes_from_ome(ome_metadata) # this is actually obsolete, as we overwrite it later
        if imagej_metadata is not None:
            md_ij = _standardize_imagej_metadata(imagej_metadata, tags=tags, verbose=verbose)
            metadata.update(md_ij)
            metadata = _add_file_properties_to_metadata(metadata, fname, original_metadata_type="imagej_metadata")
        if lsm_metadata is not None:
            md_lsm = _standardize_lsm_metadata(lsm_metadata)
            metadata.update(md_lsm)
            #metadata = _standardize_lsm_metadata(lsm_metadata) # correct lsm keys
            metadata = _add_file_properties_to_metadata(metadata, fname, original_metadata_type="lsm_metadata")
        # let's check whether metadata is empty; if so, we create a minimal default
        # description based only on image shape and a unit-less pixel grid:
        if not metadata:
            # populate metadata with the default keys from _standardize_imagej_metadata; put as
            # PhysicalSizeX/Y/Z -> 1.0 and SizeX/Y/Z -> image.shape accordingly:
            
            # First, we need to check whether we read an RGB tif; in this case, the axes order
            # differs from {T/C/Z}YX to YX{T/C/Z/S}; this we can find out via a key in tags[0]',
            # that looks like: 262 TiffTag 262 PhotometricInterpretation @58 SHORT @66 = RGB.
            # We only take into account tags[0], i.e., the first page's tags, as we assume that
            # all pages have the same PhotometricInterpretation. OMIO cannot handle multi-page
            # tif with mixed photometric interpretations at the moment. Therefore:
            # NOTE: Under OMIO policy, only series 0 is considered. RGB detection via the first
            # page is therefore sufficient and INTENTIONALLY limited in scope.
            try:
                photometric = tags[0].get("PhotometricInterpretation", None).value
            except Exception:
                photometric = None
            if photometric is not None and photometric == photometric.RGB:
                # RGB tif; we need to address the axes differently; 
                # photometric == photometric.MINISBLACK would be grayscale tif and we would thus
                # have default axes {T/C/Z}YX handling.
                if len(image_shape) == 3:
                    # with our current knowledge of RGB tif file structures, we can assume that the
                    # shape is (SizeY, SizeX, SizeC), and, thus, we can only have 3 axes:
                    
                    # extract SizeX, SizeY, SizeC from shape correctly:
                    sizey = image_shape[-3]
                    sizex = image_shape[-2]
                    sizec = image_shape[-1]
                    sizez = 1
                    
                    metadata = {
                    "SizeX": sizex,
                    "SizeY": sizey,
                    "SizeZ": sizez,
                    "SizeC": sizec,
                    "PhysicalSizeX": 1.0,
                    "PhysicalSizeY": 1.0,
                    "PhysicalSizeZ": 1.0,
                    "unit": pixelunit,
                    "PhysicalSizeXUnit": pixelunit,
                    "PhysicalSizeYUnit": pixelunit,
                    "PhysicalSizeZUnit": pixelunit,
                    'original_metadata_type': 'multipage RGB TIFF'}
                else:
                    # unexpected shape for RGB tif:
                    raise ValueError(
                        f"read_tif: Encountered RGB TIFF with unexpected shape {image_shape}. "
                        "Expected shape (SizeY, SizeX, SizeC). Please report this issue "
                        "to the developers at https://github.com/FabrizioMusacchio/omio/issues.")
            else:
                metadata = {
                    "SizeX": image.shape[-1] if len(image.shape)>=1 else 1,
                    "SizeY": image.shape[-2] if len(image.shape)>=2 else 1,
                    "SizeZ": image.shape[-3] if len(image.shape)>=3 else 1,
                    "PhysicalSizeX": 1.0,
                    "PhysicalSizeY": 1.0,
                    "PhysicalSizeZ": 1.0,
                    "unit": pixelunit,
                    "PhysicalSizeXUnit": pixelunit,
                    "PhysicalSizeYUnit": pixelunit,
                    "PhysicalSizeZUnit": pixelunit}
            metadata = _add_file_properties_to_metadata(metadata, fname, original_metadata_type="N/A")
        # fallback if SizeX/Y/Z are missing:
        if "SizeX" not in metadata:
            metadata["SizeX"] = image.shape[-1] if len(image.shape)>=1 else 1
        if "SizeY" not in metadata:
            metadata["SizeY"] = image.shape[-2] if len(image.shape)>=2 else 1
        if "SizeZ" not in metadata:
            metadata["SizeZ"] = image.shape[-3] if len(image.shape)>=3 else 1
            
        # tiffwriter has problems with the µ-symbol, thus we replace it by "micron":
        # UPDATE: this is OBSOLETE as we use OME-XML for writing metadata now!
        metadata = _metadata_units_check(metadata, pixelunit=pixelunit)
        
        # fallback/ensure basic physical sizes exist:
        if "PhysicalSizeX" not in metadata:
            print(f"WARNING: PhysicalSizeX missing in metadata; setting to default or user-provided value: {physicalsize_xyz_ext[0]}")
            metadata["PhysicalSizeX"] = physicalsize_xyz_ext[0]
        if "PhysicalSizeY" not in metadata:
            print(f"WARNING: PhysicalSizeY missing in metadata; setting to default or user-provided value: {physicalsize_xyz_ext[1]}")
            metadata["PhysicalSizeY"] = physicalsize_xyz_ext[1]
        if "PhysicalSizeZ" not in metadata:
            print(f"WARNING: PhysicalSizeZ missing in metadata; setting to default or user-provided value: {physicalsize_xyz_ext[2]}")
            metadata["PhysicalSizeZ"] = physicalsize_xyz_ext[2]
        
        # annotate OMIO multi-series policy in metadata
        if "multi_series_info" in locals():
            metadata.update(multi_series_info)
        
        # ensure shape correctness in metadata:
        metadata = _ensure_shape_in_metadata(metadata, image_shape)
        
        # ensure axes correctness in metadata:
        metadata = _ensure_axes_in_metadata(metadata, tif)
        
        # conversion factor from meter to micrometer:
        conv_um = 10 ** 6
        
        # sanity check for read Zarr array existence:
        if zarr_store is not None and not isinstance(image, zarr.core.array.Array):
            # This branch should not happen: image is either np.ndarray (None) or zarr.Array (aszarr path)
            pass
  
        # materialize from tifffile's aszarr-backed array into a real Zarr store (if Zarr):
        if zarr_store is not None:
            if verbose:
                print(f"  zarr_store requested: {zarr_store}")
                print(f"  Preparing target Zarr array on/in {zarr_store}...")
                
            # get fname base for cache path:
            fname_base, _ = os.path.splitext(os.path.basename(fname))

            # compute robust chunks using our helper (preferred over tifffile's internal chunking):
            chunks = compute_default_chunks(image.shape, metadata["axes"])
            if verbose:
                print(f"  Using chunks: {chunks} (image shape is {image.shape}, axes are '{metadata['axes']}')")

            if zarr_store == "memory":
                store = zarr.storage.MemoryStore()
                zarr_array = zarr.open(
                    store=store,
                    mode="w",
                    shape=image.shape,
                    dtype=image.dtype,
                    chunks=chunks)
            else:
                zarr_cache_folder = os.path.join(os.path.dirname(fname), ".omio_cache")
                os.makedirs(zarr_cache_folder, exist_ok=True)
                zarr_cache_path = os.path.join(zarr_cache_folder, fname_base + ".zarr")
                if os.path.exists(zarr_cache_path):
                    shutil.rmtree(zarr_cache_path)

                zarr_array = zarr.open(
                    zarr_cache_path,
                    mode="w",
                    shape=image.shape,
                    dtype=image.dtype,
                    chunks=chunks)

            # Copy strategy: for TIFF, the source is already lazy and chunked; slice-wise XY copy is still safe.
            if verbose:
                print("  Copying TIFF data into Zarr...")
            _copy_to_zarr_in_xy_slices(image, zarr_array, desc="    Slice-wise copying TIFF to Zarr")
            image = zarr_array  # from now on, downstream uses Zarr

        # fold sample axis 'S' into channel axis 'C' while keeping Zarr (if requested)
        if "S" in metadata["axes"]:
            fname_base, _ = os.path.splitext(os.path.basename(fname))
            if zarr_store == "disk":
                cache_folder = os.path.join(os.path.dirname(fname), ".omio_cache")
            else:
                cache_folder = None

            image, metadata["axes"] = _fold_samples_axis_into_channel(
                image,
                metadata["axes"],
                zarr_store=zarr_store,
                cache_folder=cache_folder,
                base_name=fname_base,
                verbose=verbose)
            image_shape = image.shape
            metadata = _ensure_shape_in_metadata(metadata, image_shape)
        
        # handle paginated TIFFs (axis 'P'):
        if "P" in metadata["axes"]:
            axis_to_use = "P"
            """ if "P" in metadata["axes"]:
                axis_to_use = "P"
            elif "S" in metadata["axes"]:
                axis_to_use = "S"
                # rename S to P in axes string for consistency:
                #metadata["axes"] = metadata["axes"].replace("S", "P") """
            if verbose:
                print(f"  Detected paginated TIFF/LSM (axis '{axis_to_use}'); splitting into individual pages.")

            p_index = metadata["axes"].index(axis_to_use)
            metadata["original_metadata_type"] = "paginated_tif/lsm"

            try:
                multi_page_metadata = tif.pages[0].tags["CZ_LSMINFO"].value
                metadata["PhysicalSizeX"] = multi_page_metadata["VoxelSizeX"] * conv_um
                metadata["PhysicalSizeY"] = multi_page_metadata["VoxelSizeY"] * conv_um
                metadata["PhysicalSizeZ"] = multi_page_metadata["VoxelSizeZ"] * conv_um
                metadata["original_metadata_type"] = "CZ_LSMINFO"
            except Exception:
                metadata["PhysicalSizeX"] = physicalsize_xyz_ext[0]
                metadata["PhysicalSizeY"] = physicalsize_xyz_ext[1]
                metadata["PhysicalSizeZ"] = physicalsize_xyz_ext[2]
                metadata["original_metadata_type"] = "N/A"

            if set_input_pixelsize:
                metadata["PhysicalSizeX"] = physicalsize_xyz_ext[0]
                metadata["PhysicalSizeY"] = physicalsize_xyz_ext[1]
                metadata["PhysicalSizeZ"] = physicalsize_xyz_ext[2]

            metadata["spacing"] = metadata["PhysicalSizeZ"]
            metadata["PhysicalSizeXUnit"] = metadata["unit"]
            metadata["PhysicalSizeYUnit"] = metadata["unit"]
            metadata["OMIO_VERSION"] = _OMIO_VERSION

            nP = image.shape[p_index]
            #axes_wo_P = metadata["axes"].replace(axis_to_use, "")
            p_index = metadata["axes"].index(axis_to_use)
            axes_wo_P = metadata["axes"][:p_index] + metadata["axes"][p_index+1:]

            images = []
            metadatas = []

            # For paginated stacks:
            #   if zarr_store is None: return NumPy pages
            #   if zarr_store is "disk": write per-page Zarr to disk (memory-friendly)
            #   if zarr_store is "memory": create per-page MemoryStore Zarr arrays
            for p in range(nP):
                # p = 0  # for testing only
                slicer = [slice(None)] * image.ndim
                slicer[p_index] = p
                page_data = image[tuple(slicer)]

                # Remove singleton P/S-axis if it still exists
                # (Depending on backend, indexing may keep dims)
                if page_data.ndim == image.ndim:
                    page_data = np.squeeze(page_data, axis=p_index)

                page_md = metadata.copy()
                page_md["axes"] = axes_wo_P
                page_md["shape"] = page_data.shape

                if zarr_store is None:
                    images.append(np.asarray(page_data))
                else:
                    fname_base, _ = os.path.splitext(os.path.basename(fname))
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
                        zarr_cache_folder = os.path.join(os.path.dirname(fname), ".omio_cache")
                        os.makedirs(zarr_cache_folder, exist_ok=True)
                        page_name = f"{fname_base}_P{p}.zarr"
                        page_path = os.path.join(zarr_cache_folder, page_name)
                        if os.path.exists(page_path):
                            shutil.rmtree(page_path)
                        page_zarr = zarr.open(
                            page_path,
                            mode="w",
                            shape=page_shape,
                            dtype=page_data.dtype,
                            chunks=chunks)

                    # Copy page data
                    _copy_to_zarr_in_xy_slices(page_data, page_zarr, desc=f"    Copying page {p} to Zarr")
                    images.append(page_zarr)

                # Post-hoc OME metadata checkup
                page_md = OME_metadata_checkup(page_md, verbose=verbose)
                metadatas.append(page_md)

            # reorder to OME axes (batch):
            memap_large_file = False
            if zarr_store=="disk":
                memap_large_file = True
            images, metadatas = _batch_correct_for_OME_axes_order(images, metadatas, memap_large_file, verbose=verbose)

            # for paginated inputs, always return lists, because splitting is the point (OMIO policy):
            if verbose:
                print(f"  Finished splitting paginated TIFF into {nP} pages.")
                print("Reading paginated TIFF completed.")
            return images, metadatas

        # normal single-stack TIFF handling:
        metadata = _get_ome_image_sizes(image.shape, metadata)

        # external pixel size override:
        if set_input_pixelsize:
            metadata["PhysicalSizeX"] = physicalsize_xyz_ext[0]
            metadata["PhysicalSizeY"] = physicalsize_xyz_ext[1]
            metadata["PhysicalSizeZ"] = physicalsize_xyz_ext[2]

        # sanity fallback if physically unreasonable:
        if metadata["PhysicalSizeX"] <= 0:
            metadata["PhysicalSizeX"] = 1
        if metadata["PhysicalSizeY"] <= 0:
            metadata["PhysicalSizeY"] = 1
        if metadata["PhysicalSizeZ"] <= 0:
            metadata["PhysicalSizeZ"] = 1

        metadata["spacing"] = metadata["PhysicalSizeZ"]
        if metadata["PhysicalSizeXUnit"] is None:
            metadata["PhysicalSizeXUnit"] = metadata["unit"]
        if metadata["PhysicalSizeYUnit"] is None:
            metadata["PhysicalSizeYUnit"] = metadata["unit"]
        if metadata["PhysicalSizeZUnit"] is None:
            metadata["PhysicalSizeZUnit"] = metadata["unit"]
        if metadata["PhysicalSizeXUnit"] =="inch" or metadata["PhysicalSizeYUnit"] =="inch" or metadata["PhysicalSizeZUnit"] =="inch":
            # print a warning, as inch is not a typical unit for microscopy images:
            print("WARNING: read_tif detected pixel unit 'inch', which is unusual for microscopy images.")
            print("         This can happen when ImageJ metadata is missing, could not be read correctly, or")
            print("         old metadata conventions were used. Please verify the returned physical pixel")
            print("          sizes in the original metadata.")
        metadata["OMIO_VERSION"] = _OMIO_VERSION

        # correct for OME axes order:
        memap_large_file = False
        if zarr_store=="disk":
            memap_large_file = True
        image, _, metadata["axes"] = _correct_for_OME_axes_order(image, metadata, memap_large_file, verbose=verbose)
        
        # shape may have changed after axes reordering:
        metadata["shape"] = image.shape

        # post-hoc OME metadata checkup and correction;
        metadata = OME_metadata_checkup(metadata, verbose=verbose)
        
        if verbose:
            print("Finished reading TIFF.")
            
        if return_list:
            return [image], [metadata]
        else:
            return image, metadata

# CZI file reader:
def read_czi(fname, physicalsize_xyz=None, pixelunit="micron", zarr_store=None, 
             return_list=False, verbose=True):
    """
    Read Zeiss CZI files into OMIO's canonical representation.

    This function reads a Zeiss CZI file using `czifile`, extracts basic acquisition
    metadata, filters and normalizes axes to the canonical OME axis convention
    TZCYX, and optionally materializes the result as a Zarr array backed by an
    in-memory store or an on-disk cache.

    CZI pixel data are always read fully into RAM first, because lazy, memory-mapped
    reading is not supported in this code path. Optional Zarr export therefore
    represents an explicit post-read materialization step for downstream workflows
    that benefit from chunked access or reduced peak RAM usage in later stages.

    Parameters
    ----------
    fname : str
        Path to the CZI file. Note: read_czi is the core function
        for Zeiss CZI file reading; omio.read() dispatches to this function when
        encountering a .czi file. read_czi can only handle RAW files but no
        folder paths (for this, please use read_thorlabs_raw_folder).
    physicalsize_xyz : tuple of float or None, optional
        Manual override for voxel sizes in the order
        ``(PhysicalSizeX, PhysicalSizeY, PhysicalSizeZ)``. If provided, these values
        override metadata-derived sizes. If None, missing or invalid sizes fall back
        to 1.0. Default is None.
    pixelunit : str, optional
        Unit string used for pixel size fields and unit normalization. Default is
        ``"micron"``.
    zarr_store : {None, "memory", "disk"}, optional
        Controls the representation of the returned image data.

        * None: return a NumPy array in RAM
        * "memory": return a Zarr array backed by an in-memory store
        * "disk": return a Zarr array stored in the cache folder
          ``{parent}/.omio_cache/<basename>.zarr``

        Existing on-disk stores at that location are replaced. Default is None.
    return_list : bool, optional
        If True, return ``[image]`` and ``[metadata]`` for backward compatibility.
        Default is False.
    verbose : bool, optional
        If True, print diagnostic progress messages. Default is True.

    Returns
    -------
    image : np.ndarray or zarr.core.array.Array
        Image data in canonical OME axis order TZCYX. If `zarr_store` is not None,
        the returned object is a Zarr array.
    metadata : dict
        Metadata dictionary aligned with the returned image, including axis and size
        information and an ``Annotations`` block for non-core fields.

    Raises
    ------
    ValueError
        If `zarr_store` is not one of {None, "memory", "disk"}.

    Notes
    -----
    * Non-OME axes present in CZI files (for example B, V, or trailing singleton
      axes) are collapsed by indexing at 0 so that only OME-relevant axes remain.
      The resulting axis string is updated accordingly.
    * Physical voxel sizes are extracted from the CZI scaling metadata and converted
      to micrometer units using a fixed conversion factor. If values are missing or
      non-positive, they fall back to 1.0.
    * Axis reordering to TZCYX may insert singleton dimensions for missing OME axes
      and may permute existing axes. The updated axis declaration is stored in the
      returned metadata.
    * When `zarr_store="disk"`, the function may create and overwrite paths under
      ``.omio_cache``.
    """

    # validate zarr_store parameter
    if zarr_store not in (None, "memory", "disk"):
        raise ValueError(
            "read_czi: zarr_store must be one of None, 'memory', or 'disk'. "
            f"Got: {zarr_store!r}")

    # determine whether pixel sizes were set manually
    if not physicalsize_xyz:
        physicalsize_xyz_ext = (1.0, 1.0, 1.0)
        set_input_pixelsize = False
    else:
        physicalsize_xyz_ext = tuple(float(v) for v in physicalsize_xyz)
        set_input_pixelsize = True

    # read CZI into memory (no memory mapping possible)
    if verbose:
        print("Reading CZI fully into RAM...")
    CZI_image = czi.imread(fname)
    CZI_metadata_obj = czi.CziFile(fname)

    # initialize metadata:
    metadata = {}
    fname_base, fname_extension = os.path.splitext(os.path.basename(fname))
    metadata["original_filetype"] = fname_extension[1:]
    metadata["original_filename"] = fname_base + fname_extension
    metadata["original_parentfolder"] = os.path.dirname(fname)
    metadata["original_metadata_type"] = "czi_metadata"

    try:
        metadata["original_creation_or_change_date"] = datetime.datetime.fromtimestamp(
            os.path.getctime(fname), datetime.UTC).strftime('%Y-%m-%dT%H:%M:%S')
    except Exception:
        metadata["original_creation_or_change_date"] = "N/A"

    # extract CZI axes (e.g. BVCTZYX0)
    metadata["axes"] = CZI_metadata_obj.axes

    # filter unwanted non-OME axes (keep only TZCYX):
    CZI_image, metadata["axes"] = _filter_image_data_for_ome_tif(CZI_image, metadata["axes"])

    # extract scaling metadata:
    CZImetadata_xyz = (
        CZI_metadata_obj.metadata(raw=False)['ImageDocument']['Metadata']
        ['Scaling']['Items']['Distance'])
    conv_um = 10 ** 6

    for item in CZImetadata_xyz:
        if item['Id'] == 'X':
            metadata["PhysicalSizeX"] = item['Value'] * conv_um
        elif item['Id'] == 'Y':
            metadata["PhysicalSizeY"] = item['Value'] * conv_um
        elif item['Id'] == 'Z':
            metadata["PhysicalSizeZ"] = item['Value'] * conv_um

    metadata["shape"] = CZI_image.shape
    metadata["unit"] = pixelunit

    # overwrite pixel sizes if provided externally
    if set_input_pixelsize:
        metadata["PhysicalSizeX"] = physicalsize_xyz_ext[0]
        metadata["PhysicalSizeY"] = physicalsize_xyz_ext[1]
        metadata["PhysicalSizeZ"] = physicalsize_xyz_ext[2]

    # fallback if metadata not usable:
    if metadata.get("PhysicalSizeX", 0) <= 0:
        metadata["PhysicalSizeX"] = 1
    if metadata.get("PhysicalSizeY", 0) <= 0:
        metadata["PhysicalSizeY"] = 1
    if metadata.get("PhysicalSizeZ", 0) <= 0:
        metadata["PhysicalSizeZ"] = 1

    # imagej compatibility (no µ symbol) ⟵ Actually, now obsolete as we write ome-tif only!
    if metadata["unit"] == "µm":
        metadata["unit"] = "micron"

    metadata["spacing"] = metadata["PhysicalSizeZ"]
    metadata["PhysicalSizeXUnit"] = metadata["unit"]
    metadata["PhysicalSizeYUnit"] = metadata["unit"]
    metadata["OMIO_VERSION"] = _OMIO_VERSION

    # ensure SizeT, SizeZ, SizeC, SizeY, SizeX are consistent with current CZI_image
    metadata = _get_ome_image_sizes(CZI_image.shape, metadata)

    # OME axis reordering: NumPy path or streaming-Zarr path; as the stack still sits fully
    # in RAM, we use _correct_for_OME_axes_order w/o memap_large_file logic:
    CZI_image, metadata["shape"], metadata["axes"] = _correct_for_OME_axes_order(
                CZI_image, metadata, memap_large_file=False, verbose=verbose)

    
    # Optional Zarr-export: write the CZI array into .omio_cache ("disk") or into RAM ("memory")
    if zarr_store is not None:
        # compute suitable chunk sizes:
        chunks = compute_default_chunks(CZI_image.shape, metadata["axes"], max_xy_chunk=1024)
        
        if verbose:
            print(f"  writing CZI array with shape {CZI_image.shape} into Zarr store on/in {zarr_store} with chunks {chunks}...")

        if zarr_store == "memory":
            # write into in-memory Zarr store:
            store = zarr.storage.MemoryStore()
            z = zarr.open(
                store=store,
                mode="w",
                shape=CZI_image.shape,
                dtype=CZI_image.dtype,
                chunks=chunks)
            z[:] = CZI_image[:]
            del CZI_image
            CZI_image = z
        elif zarr_store == "disk":
            # write into on-disk Zarr store in .omio_cache folder:
            zarr_cache_folder = os.path.join(metadata["original_parentfolder"], ".omio_cache")
            os.makedirs(zarr_cache_folder, exist_ok=True)

            zarr_path = os.path.join(zarr_cache_folder, fname_base + ".zarr")
            if os.path.exists(zarr_path):
                shutil.rmtree(zarr_path)

            z = zarr.open(
                zarr_path,
                mode="w",
                shape=CZI_image.shape,
                dtype=CZI_image.dtype,
                chunks=chunks,
            )
            # direct copy (array is fully in RAM)
            z[:] = CZI_image[:]
            del CZI_image     # free RAM
            CZI_image = z     # continue working with Zarr array

    # post-hoc OME metadata checkup and correction:
    metadata = OME_metadata_checkup(metadata, verbose=verbose)

    if verbose:
        print("Finished reading CZI.")

    if return_list:
        return [CZI_image], [metadata]
    else:
        return CZI_image, metadata

# Thorlabs RAW file reader:
def read_thorlabs_raw(fname, physicalsize_xyz=None, pixelunit="micron",
                      zarr_store=None, return_list=False, verbose=True):
    """
    Read Thorlabs RAW files into OMIO's canonical representation.

    This function reads a Thorlabs RAW file and constructs an image array together
    with an OMIO metadata dictionary that follows the canonical OME axis convention
    TZCYX. Dimensions and acquisition metadata are obtained from an accompanying XML
    file in the same folder. If no XML is present, the function falls back to a
    single YAML metadata file located in the same folder.

    The RAW payload is interpreted as a contiguous raster of pixel values that must
    be reshaped into a 5D stack ``(T, Z, C, Y, X)``. If requested, the data are
    materialized into a Zarr array either in memory or on disk. For Zarr output,
    copying is performed slice-wise over the last two spatial dimensions to limit
    peak RAM usage.
    
    YAML fallback in case of missing XML
    --------------------------------------
    In case no XML metadata file is found, the function looks for a YAML file
    in the same folder. If found, it extracts the necessary dimensions and pixel
    size information from the YAML keys ``T``, ``Z``, ``C``, ``Y``, ``X``, ``bits``,
    ``PhysicalSizeX``, ``PhysicalSizeY``, ``PhysicalSizeZ``, and ``pixelunit``.
    
    The YAML file is not generated automatically by OMIO; it must be created
    manually if no XML is available.
    
    An example YAML file might look like this:
    .. code-block:: yaml
    
        T: 1
        Z: 10
        C: 3
        Y: 512
        X: 512
        bits: 16
        PhysicalSizeX: 0.65
        PhysicalSizeY: 0.65
        PhysicalSizeZ: 2.0
        pixelunit: micron
        
    Saved as e.g. ``image_metadata.yaml`` in the same folder as the RAW file,
    this file allows read_thorlabs_raw to successfully interpret the RAW pixel.
    
    OMIO offers a utility function to help create such YAML files:
    ``omio.utilities.create_thorlabs_raw_yaml()``, which prompts the user for
    the necessary parameters and writes the YAML file (or takes defaults).
    
    Note: The values entered in the YAML file must match the actual RAW data size.
    I.e., the user must know the correct dimensions and bit depth in advance.

    If neither XML nor YAML metadata is available, the function does not raise an
    exception. Instead, it emits a warning and returns ``(None, None)`` or
    ``([None], [None])`` depending on `return_list`.

    Parameters
    ----------
    fname : str
        Path to the RAW file. Note: the function expects an XML or YAML metadata file 
        to be present in the same folder. Also: read_thorlabs_raw is the core function
        for Thorlabs RAW reading; omio.read() dispatches to this function when
        encountering a .raw file. read_thorlabs_raw can only handle RAW files but no
        folder paths (for this, please use read_thorlabs_raw_folder).
    physicalsize_xyz : tuple of float or None, optional
        Manual override for voxel sizes in the order
        ``(PhysicalSizeX, PhysicalSizeY, PhysicalSizeZ)``. If provided, these values
        override XML or YAML values. Default is None.
    pixelunit : str, optional
        Default unit string used when neither XML nor YAML provides a unit.
        Default is ``"micron"``.
    zarr_store : {None, "memory", "disk"}, optional
        Controls the representation of the returned image data.

        * None: read and return a NumPy array in RAM
        * "memory": return a Zarr array backed by an in-memory store
        * "disk": return a Zarr array stored in the cache folder
          ``{parent}/.omio_cache/<basename>.zarr``

        Existing on-disk stores at that location are replaced. Default is None.
    return_list : bool, optional
        If True, return ``[image]`` and ``[metadata]`` for backward compatibility.
        Default is False.
    verbose : bool, optional
        If True, print diagnostic progress messages. Default is True.

    Returns
    -------
    image : np.ndarray or zarr.core.array.Array or None
        Image data in canonical OME axis order TZCYX, or None if dimensions cannot
        be inferred from XML or YAML.
    metadata : dict or None
        Metadata dictionary aligned with the returned image, or None if metadata is
        unavailable. The dictionary includes an ``Annotations`` block for auxiliary
        fields when reading succeeds.

    Raises
    ------
    ValueError
        If `zarr_store` is not one of {None, "memory", "disk"}, or if an XML file is
        present but incomplete or inconsistent.
    FileNotFoundError
        If `fname` does not exist.
    ImportError
        If `zarr_store` is "memory" or "disk" but Zarr support is unavailable.

    Notes
    -----
    * RAW reading requires the dimensions T, Z, C, Y, X and a bit depth to infer the
      dtype and reshape the pixel stream. XML metadata is preferred. YAML is used
      only if XML is absent.
    * YAML fallback expects at minimum the keys ``T``, ``Z``, ``C``, ``Y``, ``X``,
      and ``bits``. Additional keys such as ``pixelunit``, ``PhysicalSizeX/Y/Z``,
      and ``TimeIncrement`` are optional.
    * For `zarr_store` not None, the function uses ``numpy.memmap`` and slice-wise
      copying to avoid loading the full RAW into RAM before writing.
    * Axis normalization to TZCYX is applied at the end and may insert singleton
      dimensions or reorder axes. The updated axis string and shape are stored in
      the returned metadata.
    * When `zarr_store="disk"`, the function may create and overwrite paths under
      ``.omio_cache``.
    """

    if zarr_store not in (None, "memory", "disk"):
        raise ValueError("read_thorlabs_raw: zarr_store must be one of None, 'memory', or 'disk'. "
                         f"Got: {zarr_store!r}")

    if verbose:
        print(f"Reading Thorlabs RAW file: {fname}")

    if not os.path.exists(fname):
        raise FileNotFoundError(f"The Thorlabs RAW file {fname} does not exist.")

    if zarr_store in ("memory", "disk") and zarr is None:
        raise ImportError("zarr is required for zarr_store='memory' or 'disk'.")

    folder = os.path.dirname(fname)
    fname_base, fname_extension = os.path.splitext(os.path.basename(fname))

    
    # initialize metadata with provenance and placeholders:
    metadata = {}
    metadata["OMIO_VERSION"] = _OMIO_VERSION
    metadata["original_filetype"] = fname_extension[1:]
    metadata["original_filename"] = fname_base + fname_extension
    metadata["original_parentfolder"] = folder
    metadata["original_metadata_type"] = "thorlabs_metadata"
    try:
        metadata["original_creation_or_change_date"] = datetime.datetime.fromtimestamp(
            os.path.getctime(fname), datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S")
    except Exception:
        metadata["original_creation_or_change_date"] = "N/A"

    metadata["axes"] = "TZCYX"
    metadata["shape"] = 0

    # these must be resolved from XML or YAML, otherwise we cannot read the RAW:
    dims = None  # dict with keys T,Z,C,Y,X,bits
    unit_from_meta = None

    
    # preferred: XML metadata in same folder
    xml_files = [f for f in os.listdir(folder) if f.lower().endswith(".xml")]
    xml_path = None
    if xml_files:
        xml_path = os.path.join(folder, xml_files[0])
        if verbose:
            print(f"  Found XML file: {xml_files[0]}. Will use it for metadata extraction...")

        tree = ET.parse(xml_path)
        root = tree.getroot()

        try:
            lsm_node = root.find(".//LSM")
            if lsm_node is None:
                raise ValueError(f"The XML file {xml_path} is missing the LSM node.")

            # dimensions X, Y:
            X = int(lsm_node.get("pixelX"))
            Y = int(lsm_node.get("pixelY"))

            # channels C:
            C = 1
            wavelengths_node = root.find(".//Wavelengths")
            if wavelengths_node is not None:
                wavelengths_n = wavelengths_node.findall(".//Wavelength")
                if wavelengths_n:
                    C = len(wavelengths_n)
                else:
                    C = int(lsm_node.get("channel"))
            else:
                C = int(lsm_node.get("channel", C))

            # time T:
            T_node = root.find(".//Timelapse")
            if T_node is not None:
                T = int(T_node.get("timepoints"))
                T_step_size = float(T_node.get("intervalSec"))
            else:
                T = 1
                T_step_size = 1.0

            # Bits and dtype
            bits = 16
            cam_node = root.find(".//Camera")
            if cam_node is not None:
                bits = int(cam_node.get("bitsPerPixel", bits))

            if bits == 32:
                dtype = np.float32
            elif bits > 8:
                dtype = np.uint16
            else:
                dtype = np.uint8
            bytes_per_pixel = np.dtype(dtype).itemsize

            # Z estimate and step size:
            Z_node = root.find(".//ZStage")
            Z_streaming = root.find(".//Streaming")
            if Z_node is not None and Z_streaming is not None and bool(int(Z_streaming.get("zFastEnable", "0"))):
                Z = int(Z_node.get("steps"))
                Z_stepSize = float(Z_node.get("stepSizeUM"))
            else:
                Z = 1
                Z_stepSize = 1.0

            # correct Z from file size (flyback frames etc.):
            file_size = os.path.getsize(fname)
            denom = X * Y * C * T * bytes_per_pixel
            if denom <= 0:
                raise ValueError("Invalid dimension product for file size check.")

            if file_size % denom != 0:
                if verbose:
                    print(f"  WARNING: RAW file size {file_size} is not an integer multiple of\n"
                        f"           X*Y*C*T*bytes_per_pixel={denom}. Z_from_file_size will be truncated.")
            Z_from_file_size = file_size // denom
            if Z != Z_from_file_size:
                if verbose:
                    print(f"    Info: Z from XML ({Z}) does not match file size calculation ({Z_from_file_size}).\n"
                        "    Using file size derived Z.")
                Z = Z_from_file_size

            dims = {"T": T, "Z": Z, "C": C, "Y": Y, "X": X, "bits": bits}

            # OME like metadata:
            metadata["SizeX"] = X
            metadata["SizeY"] = Y
            metadata["SizeC"] = C
            metadata["SizeT"] = T
            metadata["SizeZ"] = Z

            px_um = float(lsm_node.get("pixelSizeUM"))
            metadata["PhysicalSizeX"] = px_um
            metadata["PhysicalSizeY"] = px_um
            metadata["PhysicalSizeZ"] = Z_stepSize

            unit_from_meta = "micron"
            metadata["unit"] = unit_from_meta
            metadata["PhysicalSizeXUnit"] = unit_from_meta
            metadata["PhysicalSizeYUnit"] = unit_from_meta
            metadata["PhysicalSizeZUnit"] = unit_from_meta

            metadata["TimeIncrement"] = float(T_step_size)
            metadata["TimeIncrementUnit"] = "seconds"
            
            metadata["bits_per_pixel"] = bits

            try:
                metadata["frame_rate"] = float(lsm_node.get("frameRate", 0.0))
            except Exception:
                metadata["frame_rate"] = 0.0

            # Optional: date from XML
            date_node = root.find(".//Date")
            if date_node is not None:
                date_str = date_node.get("date")
                local_time = None
                try:
                    local_time = datetime.datetime.strptime(date_str, "%m/%d/%Y %H:%M:%S")
                except Exception:
                    local_time = None

                if local_time is not None:
                    creation_date_utc = local_time.replace(tzinfo=datetime.UTC)
                    metadata["original_creation_or_change_date"] = creation_date_utc.strftime("%Y-%m-%dT%H:%M:%S")

        except Exception as e:
            raise ValueError(f"The XML file {xml_path} is incomplete or inconsistent: {e}")

    
    # fallback: YAML metadata in same folder if XML missing:
    if dims is None:
        yaml_path = _find_single_yaml(folder)
        if yaml_path is not None:
            if verbose:
                print(f"  No XML file found. Found YAML metadata file: {os.path.basename(yaml_path)}.")
            ymd = _load_yaml_metadata(yaml_path)

            # required keys to read RAW:
            try:
                T = _require_int(ymd, "T")
                Z = _require_int(ymd, "Z")
                C = _require_int(ymd, "C")
                Y = _require_int(ymd, "Y")
                X = _require_int(ymd, "X")
                bits = _require_int(ymd, "bits")
            except KeyError as e:
                warnings.warn(
                    f"YAML metadata file {yaml_path} is missing required key {e}. "
                    "Cannot read RAW file. Please add the missing keys.")
                if return_list:
                    return [None], [None]
                return None, None

            dims = {"T": T, "Z": Z, "C": C, "Y": Y, "X": X, "bits": bits}

            metadata["SizeX"] = X
            metadata["SizeY"] = Y
            metadata["SizeC"] = C
            metadata["SizeT"] = T
            metadata["SizeZ"] = Z

            # Unit and physical sizes are optional in YAML
            unit_from_meta = ymd.get("pixelunit", None)
            if unit_from_meta is not None:
                metadata["unit"] = str(unit_from_meta)

            for k in ("PhysicalSizeX", "PhysicalSizeY", "PhysicalSizeZ"):
                if k in ymd:
                    try:
                        metadata[k] = float(ymd[k])
                    except Exception:
                        pass

            if "TimeIncrement" in ymd:
                try:
                    metadata["TimeIncrement"] = float(ymd["TimeIncrement"])
                except Exception:
                    pass
            if "TimeIncrementUnit" in ymd:
                metadata["TimeIncrementUnit"] = str(ymd["TimeIncrementUnit"])

            metadata["original_metadata_type"] = "thorlabs_yaml_metadata"
        """ else:
            print("  No XML or YAML metadata file found or multiple YAML files in the folder. Will return None.")
            if return_list:
                return [None], [None]
            return None, None """
    
    # if neither XML nor YAML provided dimensions, do not abort. Warn and return None:
    if dims is None:
        print("WARNING: No Thorlabs XML metadata and no YAML fallback found.\n"
              "         Cannot infer RAW dimensions (T, Z, C, Y, X, bits). Create a YAML file in the same folder as the RAW\n"
              "         file with keys: T, Z, C, Y, X, bits (and optionally pixelunit, PhysicalSizeX/Y/Z, TimeIncrement,\n"
              "         TimeIncrementUnit). Please refer to the documentation for details.\n"
              "         You may also use the utility function create_thorlabs_raw_yaml(fname) to create an empty YAML file\n"
              "         template that you can fill in manually. It will be created in the same folder as the RAW file.\n")
        print("         Example YAML content (save as, e.g., Experiment.yaml into the same folder as the RAW file):\n\n           T: 1\n           Z: 1\n           C: 1\n           Y: 512\n           X: 512\n           bits: 16\n           pixelunit: micron\n           PhysicalSizeX: 0.5\n           PhysicalSizeY: 0.5\n           PhysicalSizeZ: 1.0\n           TimeIncrement: 1.0\n           TimeIncrementUnit: seconds\n")
        print("         You may also use omio.create_thorlabs_raw_yaml(fname) to generate such a file interactively.\n")
        if return_list:
            return [None], [None]
        return None, None

    
    # final unit handling and external overrides:
    # apply unit fallback if not set by XML or YAML:
    if "unit" not in metadata or metadata["unit"] is None:
        metadata["unit"] = pixelunit

    # apply external physical size override if provided:
    if physicalsize_xyz is not None:
        psx, psy, psz = (float(physicalsize_xyz[0]), float(physicalsize_xyz[1]), float(physicalsize_xyz[2]))
        metadata["PhysicalSizeX"] = psx
        metadata["PhysicalSizeY"] = psy
        metadata["PhysicalSizeZ"] = psz

    # ensure physical sizes exist as fallbacks (do not invent units beyond pixel grid):
    if "PhysicalSizeX" not in metadata or metadata["PhysicalSizeX"] is None:
        metadata["PhysicalSizeX"] = 1.0
    if "PhysicalSizeY" not in metadata or metadata["PhysicalSizeY"] is None:
        metadata["PhysicalSizeY"] = 1.0
    if "PhysicalSizeZ" not in metadata or metadata["PhysicalSizeZ"] is None:
        metadata["PhysicalSizeZ"] = 1.0

    metadata["PhysicalSizeXUnit"] = metadata.get("PhysicalSizeXUnit", metadata["unit"])
    metadata["PhysicalSizeYUnit"] = metadata.get("PhysicalSizeYUnit", metadata["unit"])
    metadata["PhysicalSizeZUnit"] = metadata.get("PhysicalSizeZUnit", metadata["unit"])

    
    # read RAW data and optionally materialize into Zarr:
    T = dims["T"]
    Z = dims["Z"]
    C = dims["C"]
    Y = dims["Y"]
    X = dims["X"]
    bits = dims["bits"]

    if bits == 32:
        dtype = np.float32
    elif bits > 8:
        dtype = np.uint16
    else:
        dtype = np.uint8

    expected_elements = T * Z * C * Y * X

    if zarr_store is None:
        if verbose:
            print("  Reading entire Thorlabs RAW file into RAM...")
        with open(fname, "rb") as f:
            raw_data = np.frombuffer(f.read(), dtype=dtype)

        if raw_data.size != expected_elements:
            warnings.warn(
                f"RAW data size mismatch: expected {expected_elements} elements, got {raw_data.size}. "
                "Check XML/YAML metadata.")
            if return_list:
                return [None], [None]
            return None, None

        image = raw_data.reshape((T, Z, C, Y, X))
        metadata["shape"] = image.shape

    else:
        if verbose:
            print("  Preparing Zarr representation (via memmap + slice-wise copy)...")
        raw_data = np.memmap(fname, dtype=dtype, mode="r")

        if raw_data.size != expected_elements:
            warnings.warn(
                f"RAW data size mismatch: expected {expected_elements} elements, got {raw_data.size}. "
                "Check XML/YAML metadata.")
            if return_list:
                return [None], [None]
            return None, None

        image_np = raw_data.reshape((T, Z, C, Y, X))
        metadata["shape"] = image_np.shape

        chunks = compute_default_chunks(image_np.shape, metadata["axes"])
        if verbose:
            print(f"  Computed Zarr chunks: {chunks} (shape: {image_np.shape})")

        if zarr_store == "memory":
            if verbose:
                print("  Writing into in-memory Zarr store...")
            store = zarr.storage.MemoryStore()
            zarr_array = zarr.open(
                store=store,
                mode="w",
                shape=image_np.shape,
                dtype=image_np.dtype,
                chunks=chunks)
        else:
            if verbose:
                print("  Writing into on-disk Zarr store for memory mapping...")
            zarr_cache_folder = os.path.join(folder, ".omio_cache")
            os.makedirs(zarr_cache_folder, exist_ok=True)
            zarr_cache_path = os.path.join(zarr_cache_folder, fname_base + ".zarr")
            if os.path.exists(zarr_cache_path):
                shutil.rmtree(zarr_cache_path)

            zarr_array = zarr.open(
                zarr_cache_path,
                mode="w",
                shape=image_np.shape,
                dtype=image_np.dtype,
                chunks=chunks)

        _copy_to_zarr_in_xy_slices(image_np, zarr_array,
                                  desc="    slice-wise copying Thorlabs RAW to Zarr")

        image = zarr_array

    
    # final normalization steps:
    memap_large_file_flag = (zarr_store == "disk")
    image, metadata["shape"], metadata["axes"] = _correct_for_OME_axes_order(
        image, metadata, memap_large_file=memap_large_file_flag, verbose=verbose)

    metadata = OME_metadata_checkup(metadata, verbose=verbose)

    if verbose:
        print("Finished reading Thorlabs RAW file.")

    if return_list:
        return [image], [metadata]
    return image, metadata

# %% OMIO_CACHE CLEANUP FUNCTION

def cleanup_omio_cache(fname, full_cleanup=False, verbose=True):
    """
    Remove OMIO-generated on-disk cache data under the `.omio_cache` folder.

    This utility deletes Zarr stores created by OMIO when reading files with
    ``zarr_store="disk"``. The cache is expected to live in a hidden subfolder
    ``.omio_cache`` within a dataset's parent directory.

    Two modes are supported:

    * Targeted cleanup:
      If ``fname`` is a file path and ``full_cleanup`` is False, only the corresponding
      cache store ``.omio_cache/<basename>.zarr`` is removed.

    * Full cleanup:
      If ``full_cleanup`` is True, or if ``fname`` points to a directory, the entire
      ``.omio_cache`` folder under that directory is removed.
    
    Parameters
    ----------
    fname : str
        Path to a file whose cache should be removed, or a directory containing an
        ``.omio_cache`` folder to be cleaned.
    full_cleanup : bool, optional
        If True, delete the entire ``.omio_cache`` folder. If False and ``fname`` is a
        file, delete only the cache store corresponding to that file's basename.
        Default is False.
    verbose : bool, optional
        If True, print diagnostic messages. Default is True.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If `fname` is neither an existing file nor an existing directory.

    Notes
    -----
    * Cache deletion is performed via recursive directory removal and is not
      reversible.
    * If no ``.omio_cache`` folder exists at the expected location, the function
      returns without error.
    """
    if os.path.isfile(fname):
        parent_folder = os.path.dirname(fname)
        base_name = os.path.splitext(os.path.basename(fname))[0]
    elif os.path.isdir(fname):
        parent_folder = fname
        base_name = None
    else:
        raise ValueError(f"cleanup_omio_cache: {fname} is neither a file nor a folder.")

    omio_cache_folder = os.path.join(parent_folder, ".omio_cache")
    if not os.path.exists(omio_cache_folder):
        if verbose:
            print(f"No .omio_cache folder found in {parent_folder}. Nothing to clean up.")
        return

    if full_cleanup or base_name is None:
        print(f"Performing full cleanup of .omio_cache folder: {omio_cache_folder}")
        shutil.rmtree(omio_cache_folder)
        print("Cleanup complete.")
    else:
        zarr_path = os.path.join(omio_cache_folder, base_name + ".zarr")
        if os.path.exists(zarr_path):
            print(f"Deleting Zarr store for {base_name}: {zarr_path}")
            shutil.rmtree(zarr_path)
            print("Deletion complete.")
        else:
            print(f"No Zarr store found for {base_name} in .omio_cache. Nothing to delete.")

# %% EMPTY IMAGE AND METADATA CREATORS

# function to create empty OME metadata dict with default values:
def create_empty_metadata(physicalsize_xyz: Union[tuple[float, float, float], None] = None,
                          pixelunit: str = "micron",
                          time_increment: Union[float, None] = None,
                          time_increment_unit: str = None,
                          shape: Union[tuple[int, int, int, int, int], None] = None,
                          annotations: dict | None = None,
                          input_metadata: dict | None = None,
                          verbose: bool = True) -> dict:
    """
    Create a new OMIO metadata dictionary populated with canonical default keys.

    This factory returns a metadata dictionary that follows OMIO's OME-oriented key
    conventions and provides a complete set of standard fields with safe default
    values. It is intended as a starting point for downstream routines that
    progressively refine metadata, for example by filling sizes from image data or
    merging acquisition metadata from files.

    The returned dictionary always includes:

    * canonical axis declaration under ``"axes"`` (typically TZCYX),
    * shape and per-axis size fields (``shape``, ``SizeT``, ``SizeZ``, ``SizeC``,
      ``SizeY``, ``SizeX``),
    * physical voxel sizes and time sampling (``PhysicalSize*``, ``TimeIncrement``,
      ``TimeIncrementUnit``),
    * a unit field (``unit``),
    * an ``Annotations`` mapping for auxiliary fields,
    * the current OMIO version identifier under ``_OMIO_VERSION``.

    User-provided values can be injected via `input_metadata`, overridden via
    dedicated arguments, and merged into the ``Annotations`` block. Finally, the
    metadata are normalized via `OME_metadata_checkup` to ensure that non-core
    entries are moved into ``Annotations`` and a namespace entry is present.

    Parameters
    ----------
    physicalsize_xyz : tuple of float or None, optional
        Optional voxel size override in the order
        ``(PhysicalSizeX, PhysicalSizeY, PhysicalSizeZ)``. If provided, these values
        overwrite the defaults and any corresponding entries from `input_metadata`.
    pixelunit : str, optional
        Unit string for pixel sizes. Common micrometer spellings are normalized to
        the symbol ``"µm"`` in the returned dictionary. Default is ``"micron"``.
    time_increment : float or None, optional
        Optional override for ``TimeIncrement``. If None, the default value is used.
    time_increment_unit : str or None, optional
        Optional override for ``TimeIncrementUnit``. If None, the default value is
        used.
    shape : tuple of int or None, optional
        Optional 5D shape tuple in canonical order ``(T, Z, C, Y, X)``. If provided,
        ``shape`` and the corresponding ``Size*`` fields are set consistently. If
        the tuple does not have length 5, a warning is issued and the shape is not
        set.
    annotations : dict or None, optional
        Additional key value pairs to merge into the ``Annotations`` block.
    input_metadata : dict or None, optional
        Existing metadata dictionary whose entries are merged into the returned
        dictionary prior to applying explicit overrides.
    verbose : bool, optional
        If True, enable diagnostic messages from downstream normalization steps.
        Default is True.

    Returns
    -------
    md : dict
        A normalized OMIO metadata dictionary containing canonical keys and user
        overrides, with auxiliary fields stored under ``Annotations``.

    Notes
    -----
    * The function constructs a new dictionary and does not modify `input_metadata`
      in place, but if `input_metadata["Annotations"]` is a dictionary it may be
      reused and updated during merging.
    * The default axis string is taken from the module-level constant ``_OME_AXES``,
      and size indices are derived from ``_AXIS_TO_INDEX``.
    * Final normalization is performed by `OME_metadata_checkup`, which may move
      non-core fields into ``Annotations`` and enforce an annotations namespace.
    """
    md = {
        "axes": _OME_AXES,      # "TZCYX"
        "shape": None,

        "SizeT": None,
        "SizeZ": None,
        "SizeC": None,
        "SizeY": None,
        "SizeX": None,

        "PhysicalSizeX": 1,
        "PhysicalSizeY": 1,
        "PhysicalSizeZ": 1,

        "TimeIncrement": 1,
        "TimeIncrementUnit": "s",

        "unit": "µm" if pixelunit in ("micron", "micrometer", "um", "µm") else pixelunit,
        "Annotations": {},
        "OMIO_VERSION": _OMIO_VERSION}

    # if input_metadata is provided, update md with it:
    if isinstance(input_metadata, dict):
        md.update(input_metadata)

    if physicalsize_xyz is not None:
        # overwrite physical sizes by given values:
        md["PhysicalSizeX"] = float(physicalsize_xyz[0])
        md["PhysicalSizeY"] = float(physicalsize_xyz[1])
        md["PhysicalSizeZ"] = float(physicalsize_xyz[2])

    if time_increment is not None:
        # overwrite time increment by given value:
        md["TimeIncrement"] = float(time_increment)
        
    if time_increment_unit is not None:
        # overwrite time increment unit by given value:
        md["TimeIncrementUnit"] = str(time_increment_unit)

    if shape is not None:
        if len(shape) != 5:
            warnings.warn("create_empty_metadata: shape must be a 5-tuple (T, Z, C, Y, X).\n"
                          f"  Got: {shape!r}. Cannot set user provided shape into metadata.")
        else:
            md["shape"] = tuple(int(v) for v in shape)
            md["SizeT"] = int(shape[_AXIS_TO_INDEX["T"]])
            md["SizeZ"] = int(shape[_AXIS_TO_INDEX["Z"]])
            md["SizeC"] = int(shape[_AXIS_TO_INDEX["C"]])
            md["SizeY"] = int(shape[_AXIS_TO_INDEX["Y"]])
            md["SizeX"] = int(shape[_AXIS_TO_INDEX["X"]])

    if isinstance(annotations, dict):
        if isinstance(input_metadata, dict):
            # if input_metadata already has Annotations, update them:
            existing_annotations = input_metadata.get("Annotations", {})
            if isinstance(existing_annotations, dict):
                existing_annotations.update(annotations)
                md["Annotations"] = existing_annotations
        else:
            md["Annotations"] = dict(annotations)

    # make md OME-compliant:
    md = OME_metadata_checkup(md, verbose=verbose)

    return md

# function to create empty OME ordered image with axes TZCYX:
def create_empty_image(shape: tuple[int, int, int, int, int] = (1, 1, 1, 1, 1),
                       dtype=np.uint16,
                       fill_value=0,
                       zarr_store: Union[None, str] = None,
                       zarr_store_path: Union[None, str] = None,
                       zarr_store_name: Union[None, str] = None,
                       return_metadata: bool = False,
                       input_metadata: Union[None, dict] = None,
                       verbose: bool = True
                       ) -> Union[None,
                                  np.ndarray,
                                  "zarr.core.array.Array",
                                  tuple[np.ndarray, dict],
                                  tuple["zarr.core.array.Array", dict]]:
    """
    Create an empty 5D image in canonical OME axis order TZCYX.

    This factory creates a new image container with shape ``(T, Z, C, Y, X)`` and a
    specified dtype, either as a NumPy array in RAM or as a Zarr array backed by an
    in-memory store or an on-disk cache. Optionally, it also returns a matching OMIO
    metadata dictionary consistent with the created image.

    For Zarr output, chunking is determined via `compute_default_chunks` using the
    canonical OME axes. When writing to disk, the array is created under a hidden
    cache folder ``.omio_cache`` located in the specified parent directory. Any
    existing store at the target path is replaced.

    Parameters
    ----------
    shape : tuple of int, optional
        Desired image shape as a 5-tuple ``(T, Z, C, Y, X)``. Default is
        ``(1, 1, 1, 1, 1)``. If `shape` is None or does not have length 5, a warning
        is issued and the function returns None (or ``(None, None)`` if
        `return_metadata` is True).
    dtype : numpy dtype, optional
        Data type of the created array. Default is ``np.uint16``.
    fill_value : scalar or None, optional
        Value used to initialize the array. If 0 and `zarr_store` is None, a
        zero-initialized NumPy array is created via `np.zeros`. If `fill_value` is
        None for Zarr output, the array is left uninitialized. Default is 0.
    zarr_store : {None, "memory", "disk"}, optional
        Storage backend for the created image.

        * None: return a NumPy array in RAM
        * "memory": return a Zarr array backed by a `zarr.storage.MemoryStore`
        * "disk": return a Zarr array stored under ``.omio_cache`` on disk

        Default is None.
    zarr_store_path : str or None, optional
        Path used to determine the parent directory for on-disk storage when
        `zarr_store="disk"`. If this is a directory, it is used directly. If it is
        a file path, its parent directory is used. Required for `zarr_store="disk"`.
    zarr_store_name : str or None, optional
        Basename used for the on-disk Zarr store when `zarr_store="disk"`. The final
        store path is ``<parent>/.omio_cache/<zarr_store_name>.zarr``. Required for
        `zarr_store="disk"`.
    return_metadata : bool, optional
        If True, return a tuple ``(image, metadata)`` where `metadata` is created by
        `create_empty_metadata` and is consistent with `shape`. Default is False.
    input_metadata : dict or None, optional
        Optional metadata dictionary merged into the generated metadata when
        `return_metadata` is True. Default is None.
    verbose : bool, optional
        If True, print diagnostic messages for some path handling cases. Default is
        True.

    Returns
    -------
    image : np.ndarray or zarr.core.array.Array or None
        The created image container. Returns None if validation fails.
    metadata : dict, optional
        Only returned when `return_metadata` is True. The metadata dictionary is
        consistent with the created image shape and canonical axes TZCYX.

    Notes
    -----
    * The function assumes canonical OME axes TZCYX as defined by the module-level
      constant ``_OME_AXES``.
    * For `zarr_store="disk"`, any existing store at the target location is removed
      before creating a new one.
    * Chunking is delegated to `compute_default_chunks`. For very small arrays,
      chunk sizes may match the full dimensions.
    """
    if shape is None or len(shape) != 5:
        print("WARNING create_empty_image: shape must be a 5-tuple (T, Z, C, Y, X).\n"
             f"        Got: {shape!r}. Will return None.")
        if return_metadata:
            return None, None
        else:
            return None

    if zarr_store is None:
        # numpy array in RAM:
        if fill_value == 0:
            if return_metadata:
                return np.zeros(shape, dtype=dtype), create_empty_metadata(shape=shape, 
                                                                           input_metadata=input_metadata,
                                                                           verbose=verbose)
            else:
                return np.zeros(shape, dtype=dtype)
        else:
            arr = np.empty(shape, dtype=dtype)
            arr[...] = fill_value
            if return_metadata:
                return arr, create_empty_metadata(shape=shape, input_metadata=input_metadata,
                                                  verbose=verbose)
            else: 
                return arr
    else:
        # zarr_store is not None:
        
        # sanity check whether fname is not None, otherwise print warning and return None:
        if zarr_store not in ("memory", "disk"):
            warnings.warn("create_empty_image: zarr_store must be 'memory', or 'disk'. "
                             f"Got: {zarr_store!r}")
            if return_metadata:
                return None, None
            else:
                return None
        
        # calculate chunks from shape:
        try:
            chunks = compute_default_chunks(shape, _OME_AXES, max_xy_chunk=1024)
        except TypeError:
            chunks = compute_default_chunks(shape, _OME_AXES)
        
        if zarr_store == "memory":
            store = zarr.storage.MemoryStore()
            z_out = zarr.open(store=store, mode="w", shape=shape, dtype=dtype, chunks=chunks)
        else:
            # disk:
            if zarr_store_path is None:
                warnings.warn("create_empty_image: for zarr_store='disk', a valid zarr_store_path must be provided.\n"
                              f"  Got: {zarr_store_path!r}")
                if return_metadata:
                    return None, None
                else:
                    return None
            if zarr_store_name is None:
                warnings.warn("create_empty_image: for zarr_store='disk', a valid zarr_store_name must be provided.\n"
                              f"  Got: {zarr_store_name!r}")
                if return_metadata:
                    return None, None
                else:
                    return None

            if os.path.isdir(zarr_store_path):
                parent_folder = zarr_store_path
            else:
                parent_folder = os.path.dirname(zarr_store_path) or "."
                if verbose:
                    print(f"    zarr_store_path is a file; taking its parent folder:")
                    print(f"    {parent_folder}")

            cache_folder = os.path.join(parent_folder, ".omio_cache")
            os.makedirs(cache_folder, exist_ok=True)

            zarr_path = os.path.join(cache_folder, zarr_store_name + ".zarr")
            if os.path.exists(zarr_path):
                shutil.rmtree(zarr_path)

            z_out = zarr.open(zarr_path, mode="w", shape=shape, dtype=dtype, chunks=chunks)

        # initialize with fill_value (optionally, leave as uninitialized if fill_value is None)
        if fill_value is not None:
            if fill_value == 0:
                z_out[:] = 0
            else:
                z_out[:] = np.asarray(fill_value, dtype=dtype)

        if return_metadata:
            return z_out, create_empty_metadata(shape=shape, input_metadata=input_metadata,
                                                verbose=verbose)
        else:
            return z_out

# function to update metadata from image shape and axes:
def update_metadata_from_image(metadata: dict, 
                               image: Union[np.ndarray, "zarr.core.array.Array"],
                               run_checkup: bool = True,
                               verbose: bool = True) -> dict:
    """
    Update size-related metadata fields from a 5D image in canonical OME order.

    This helper synchronizes a metadata dictionary with the shape of a provided
    image array. It enforces OMIO's canonical axis convention TZCYX, reads the image
    shape, stores it under ``"shape"``, and updates the corresponding ``Size*``
    fields (``SizeT``, ``SizeZ``, ``SizeC``, ``SizeY``, ``SizeX``).

    Optionally, the result is normalized via `OME_metadata_checkup`, which collects
    non-core fields into ``Annotations`` and enforces the annotations namespace.

    Parameters
    ----------
    metadata : dict
        Input metadata dictionary to update. If None, an empty dictionary is used.
    image : np.ndarray or zarr.core.array.Array
        Image array whose shape defines the updated metadata. The image must be 5D
        and already in canonical axis order TZCYX.
    run_checkup : bool, optional
        If True, run `OME_metadata_checkup` on the updated metadata. Default is True.
    verbose : bool, optional
        If True, enable diagnostic messages from the normalization step. Default is
        True.

    Returns
    -------
    md : dict
        Updated metadata dictionary with consistent ``axes``, ``shape``, and
        ``Size*`` fields.

    Raises
    ------
    ValueError
        If the provided image is not 5D, since OMIO expects canonical order TZCYX.

    Notes
    -----
    * The function enforces ``md["axes"] = _OME_AXES`` unconditionally. It does not
      attempt to infer axes from the input metadata.
    * The input dictionary is copied; updates are applied to a new dictionary and
      the original `metadata` is not modified in place.
    """
    if metadata is None:
        metadata = {}

    md = dict(metadata)

    # enforce axes
    md["axes"] = _OME_AXES

    # read shape
    shape = tuple(image.shape)
    if len(shape) != 5:
        raise ValueError(f"update_metadata: expected 5D image (TZCYX). Got shape={shape}.")

    md["shape"] = shape
    md["SizeT"] = int(shape[_AXIS_TO_INDEX["T"]])
    md["SizeZ"] = int(shape[_AXIS_TO_INDEX["Z"]])
    md["SizeC"] = int(shape[_AXIS_TO_INDEX["C"]])
    md["SizeY"] = int(shape[_AXIS_TO_INDEX["Y"]])
    md["SizeX"] = int(shape[_AXIS_TO_INDEX["X"]])

    if run_checkup:
        md = OME_metadata_checkup(md, verbose=verbose)

    return md

# %% OME-TIF WRITER

# function to estimate compressed size of an image:
def _estimate_compressed_size(image, sample_fraction=0.001, compression_level=3):
    """
    Estimate the compressed size of an image array using sampling and zlib.

    This helper provides a rough estimate of the compressed size of an image by
    compressing a small representative sample and extrapolating the resulting
    compression ratio to the full dataset. It supports both NumPy arrays and
    Zarr arrays.

    For NumPy inputs, a linear prefix of the flattened array is sampled according
    to `sample_fraction`. For Zarr inputs, a single spatial (Y, X) plane is
    extracted to avoid materializing large portions of the dataset.

    Parameters
    ----------
    image : np.ndarray or zarr.core.array.Array
        Image data whose compressed size is to be estimated.
    sample_fraction : float, optional
        Fraction of the total number of elements to sample for NumPy arrays.
        The minimum sample size is one element. Default is 0.001.
    compression_level : int, optional
        Compression level passed to ``zlib.compress``, between 0 (no compression)
        and 9 (maximum compression). Default is 3.

    Returns
    -------
    estimated_compressed_size : float
        Estimated compressed size of the full image in bytes.

    Notes
    -----
    * The estimate assumes that the sampled region is representative of the entire
    image. Strong spatial or temporal heterogeneity can lead to inaccurate
    estimates.
    * For Zarr inputs, only a single spatial slice is sampled, which may bias the
    estimate if compression characteristics vary across non-spatial axes.
    * The function does not account for container or metadata overhead associated
    with specific storage formats.
    """
    
    # Get a contiguous chunk of the image as a sample:
    is_zarr = isinstance(image, zarr.core.array.Array)
    if is_zarr:
        # if Zarr, first just get a small chunk, e.g., first time slice, z-slice etc.:
        slicer = [0] * (image.ndim - 2) + [slice(None), slice(None)]
        sample_block = np.asarray(image[tuple(slicer)])
        sample = sample_block.ravel()
    else:
        sample_size = max(1, int(np.prod(image.shape) * sample_fraction))
        sample = image.ravel()[:sample_size]

    # Compress the sample using specified compression level
    compressed_sample = zlib.compress(sample.tobytes(), level=compression_level)

    # Estimate compression ratio
    compression_ratio = len(compressed_sample) / sample.nbytes

    # Estimate compressed size of the entire image
    estimated_compressed_size = image.nbytes * compression_ratio

    return estimated_compressed_size
# function to check whether to use BigTIFF:
def _check_bigtiff(image, compression_level=3):
    """
    Determine whether BigTIFF should be used for writing an image.

    This helper decides whether an image exceeds the practical size limits of
    standard TIFF files and therefore requires the BigTIFF format. The decision is
    based first on the uncompressed in-memory size and, if that exceeds the limit,
    optionally refined using an estimate of the compressed size.

    The threshold used corresponds to the maximum addressable size of classic TIFF
    files, reduced by a safety margin.

    Parameters
    ----------
    image : np.ndarray or zarr.core.array.Array
        Image data to be evaluated.
    compression_level : int, optional
        Compression level passed to the internal compressed-size estimator.
        This value is forwarded to `_estimate_compressed_size` and should be in the
        range supported by zlib (0 to 9). Default is 3.

    Returns
    -------
    use_bigtiff : bool
        True if the image should be written as BigTIFF, False if standard TIFF is
        sufficient.

    Notes
    -----
    * The initial decision is based on the raw in-memory size ``image.nbytes``.
    * If the raw size exceeds the TIFF limit, a compressed-size estimate is used as
    a secondary check. If the estimated compressed size falls below the limit,
    BigTIFF is not required.
    * The compressed-size estimate is heuristic and may misclassify borderline
    cases depending on image content and compression behavior.
    """
    # (2**32 - 2**25)/1024**3  # in GB
    # estimated_size/1024**3   # in GB

    # check, whether image size is larger than 4GB:
    if image.nbytes  > 2**32 - 2**25:
        use_bigtiff = True
    else:
        use_bigtiff = False

    # check, whether the estimated size after compression is smaller than the maximum 
    # size of a normal tif file (if so, reset use_bigtiff to False):
    if use_bigtiff:
        estimated_size = _estimate_compressed_size(image, sample_fraction=0.001,compression_level=compression_level)
        if estimated_size  < 2**32 - 2**25:
            use_bigtiff = False
    
    return use_bigtiff
# function to check and modify output filename if it already exists:
def _check_fname_out(fname_out, overwrite):
    """
    Resolve output filename collisions by appending a numeric suffix.

    This helper checks whether an output filename already exists on disk. If it
    does and overwriting is not permitted, a numeric suffix is appended to the base
    filename before the ``.ome.tif`` extension. The suffix is incremented until a
    non-existing filename is found.

    Parameters
    ----------
    fname_out : str
        Proposed output filename, expected to end with ``.ome.tif``.
    overwrite : bool
        If True, allow overwriting an existing file and return `fname_out`
        unchanged. If False, generate a modified filename if needed.

    Returns
    -------
    fname_out_rev : str
        A filename that does not exist on disk, either the original `fname_out` or
        a suffixed variant.

    Notes
    -----
    * The suffix is inserted as a space followed by an integer, for example
    ``"image 1.ome.tif"``.
    * The function assumes the ``.ome.tif`` extension is present and does not
    attempt to generalize to other extensions.
    """
    """ fname_out_rev = fname_out
    if os.path.exists(fname_out) and not overwrite:
        i = 0
        while os.path.exists(fname_out_rev):
            i += 1
            fname_out_rev = fname_out.replace(".ome.tif", f" {i}.ome.tif")
    return fname_out_rev """
    if not fname_out.endswith(".ome.tif"):
        raise ValueError(
            "_check_fname_out: fname_out must end with '.ome.tif'. "
            f"Got: {fname_out!r}"
        )

    if overwrite or not os.path.exists(fname_out):
        return fname_out

    base = fname_out[:-len(".ome.tif")]
    i = 1
    while True:
        candidate = f"{base} {i}.ome.tif"
        if not os.path.exists(candidate):
            return candidate
        i += 1
# function to normalize axes and squeeze singleton S axis:
def _normalize_axes_for_ometiff(image, axes):
    """
    Normalize axes for OME-TIFF writing by removing trivial singleton dimensions.

    This helper prepares image data and its axis declaration for OME-TIFF output.
    It currently handles the special case of a singleton ``"S"`` axis by removing
    it when its corresponding dimension has size 1. The image array is squeezed
    accordingly, and the axis string is updated to remain consistent.

    After normalization, the function verifies that the axis string length matches
    the array dimensionality.

    Parameters
    ----------
    image : array-like
        Image data to be normalized. The input is converted to a NumPy array via
        ``np.asarray``.
    axes : str
        Axis declaration corresponding to `image`.

    Returns
    -------
    arr : np.ndarray
        Normalized NumPy array with trivial singleton axes removed.
    axes : str
        Updated axis string consistent with the returned array.

    Raises
    ------
    ValueError
        If the resulting axis string length does not match ``arr.ndim``.

    Notes
    -----
    * Only the ``"S"`` axis is handled explicitly. Other singleton dimensions are
    not modified.
    * The function is intended as a small preprocessing step before writing
    OME-TIFF files.
    """
    arr = np.asarray(image)
    if "S" in axes:
        s_idx = axes.index("S")
        if arr.shape[s_idx] == 1:
            arr = np.squeeze(arr, axis=s_idx)
            axes = axes.replace("S", "")
    if len(axes) != arr.ndim:
        raise ValueError(
            f"_normalize_axes_for_ometiff: axes '{axes}' (len={len(axes)}) "
            f"does not fit to arr.ndim={arr.ndim}"
        )
    return arr, axes
# function to extract original filename from metadata:
def _get_original_filename_from_metadata(metadata: dict) -> Union[None, str]:
    """
    Extract the original filename from an OMIO metadata dictionary.

    This helper attempts to recover the original filename stored inside the
    ``Annotations`` entry of a metadata dictionary. It supports both supported
    representations of annotations used within OMIO:

    * a single annotations dictionary
    * a list of annotation dictionaries

    Only the basename of the file is returned. Any directory components are
    stripped. If no valid filename can be found, the function returns ``None``.

    Parameters
    ----------
    metadata : dict
        Metadata dictionary that may contain an ``Annotations`` entry.

    Returns
    -------
    str or None
        The original filename (basename only) if present and non-empty, otherwise
        ``None``.

    Notes
    -----
    * The function looks specifically for the key ``"original_filename"`` inside
    ``metadata["Annotations"]``.
    * If ``Annotations`` is a list, the first valid occurrence is returned.
    * Invalid metadata structures or empty values are silently ignored.
    """
    if not isinstance(metadata, dict):
        return None

    anns = metadata.get("Annotations", None)

    # dict case
    if isinstance(anns, dict):
        fn = anns.get("original_filename", None)
        if isinstance(fn, str) and fn.strip():
            return os.path.basename(fn.strip())

    # list of dicts case
    if isinstance(anns, list):
        for a in anns:
            if not isinstance(a, dict):
                continue
            fn = a.get("original_filename", None)
            if isinstance(fn, str) and fn.strip():
                return os.path.basename(fn.strip())

    return None

# main OME-TIFF writer function:
def imwrite(fname: str, 
                  images: Union[np.ndarray, "zarr.core.array.Array", list[Union[np.ndarray, "zarr.core.array.Array"]]], 
                  metadatas: Union[dict, list[dict]],
                  compression_level: int = 3, 
                  relative_path: Union[None, str] = None, 
                  overwrite: bool = False, 
                  return_fnames: bool = False, 
                  indicate_merged_files: bool = False,
                  verbose: bool = True) -> Union[None, list[str]]:
    """
    Write image stacks as OME-TIFF with OMIO-normalized metadata.

    This function is OMIO's main OME-TIFF writer. It accepts either a single image
    and metadata dictionary or lists of images and metadatas. For each stack, it
    constructs an OME-XML metadata payload compatible with `tifffile.imwrite`,
    normalizes axes for OME-TIFF writing, decides whether BigTIFF is required, and
    writes a compressed OME-TIFF using zlib.

    Output naming follows a provenance-first policy:

    * If the metadata contain an original filename inside ``Annotations``, that
      basename is used as the output basename.
    * Otherwise, the basename is derived from `fname` (file stem) or from the
      directory name if `fname` is a directory.
    * Filename collisions are resolved by `_check_fname_out` unless `overwrite` is
      True.
    * If multiple stacks are written and no per-stack provenance name is available,
      a numeric suffix ``_NNN`` is appended to keep outputs distinct.

    If `relative_path` is provided, outputs are written into a subfolder relative to
    the chosen output parent directory.

    Parameters
    ----------
    fname : str
        Output anchor path. If `fname` is a directory, outputs are written into that
        directory (or into `relative_path` below it). If `fname` is a file path,
        outputs are written next to that file (or into `relative_path` below that
        parent directory).
    images : np.ndarray or zarr.core.array.Array or list of such arrays
        Image data to write. A single image is accepted and treated as a one-element
        list. Arrays are expected to represent OME-like dimensions; the function
        normalizes axes and permutes to the writer's target order internally.
    metadatas : dict or list of dict
        Metadata dictionary or list of dictionaries aligned with `images`. Each
        metadata dictionary should include at least ``axes`` and physical pixel sizes
        (``PhysicalSizeX``, ``PhysicalSizeY``) for correct resolution tagging.
    compression_level : int, optional
        zlib compression level passed to `tifffile.imwrite` via
        ``compressionargs={"level": ...}``. Typical values are 0 to 9. Default is 3.
    relative_path : str or None, optional
        If not None, outputs are written into ``<out_parent>/<relative_path>`` and
        the directory is created if needed. Default is None.
    overwrite : bool, optional
        If True, allow overwriting existing output files. If False, resolve name
        collisions by appending a numeric suffix. Default is False.
    return_fnames : bool, optional
        If True, return a list of written filenames. If False, return None. Default
        is False.
    indicate_merged_files : bool, optional
        If True, append ``"_merged"`` to the output basename for each written stack.
        This is intended to mark stacks that originate from prior merging steps.
        Default is False.
    verbose : bool, optional
        If True, print diagnostic messages about output naming and BigTIFF decisions.
        Default is True.

    Returns
    -------
    list of str or None
        If `return_fnames` is True, returns a list of full paths to the written
        OME-TIFF files in the order processed. Otherwise returns None.

    Raises
    ------
    ValueError
        If `images` and `metadatas` have different lengths.

    Notes
    -----
    * BigTIFF selection is determined by `_check_bigtiff`, using the uncompressed
      array size and, if needed, an estimated compressed size.
    * Axes are normalized by `_normalize_axes_for_ometiff` (currently removing a
      singleton ``"S"`` axis) and then permuted into the writer's target axis order
      before writing.
    * Physical pixel sizes are written both as OME physical size fields and as TIFF
      resolution tags using ``resolution=(1/PhysicalSizeY, 1/PhysicalSizeX)``.
    * Map annotations are written from ``metadata["Annotations"]``. If annotations
      are a dictionary, a single MapAnnotation is written. If annotations are a
      list of dictionaries, multiple MapAnnotations are written. A namespace entry
      is ensured if missing.
    * The function writes with ``photometric="minisblack"`` and disables ImageJ
      metadata blocks (``imagej=False``), relying on OME metadata for
      interoperability.
    
    """
    
    
    # check whether images and metadatas are lists:
    #images_was_list = isinstance(images, list) and len(images) > 1
    if not isinstance(images, list):
        images = [images]
    if not isinstance(metadatas, list):
        metadatas = [metadatas]
    if len(images) != len(metadatas):
        raise ValueError("imwrite: images and metadatas must have the same length.")
    
    # decide output parent directory:
    # * if fname is a directory: output next to that directory (or inside relative_path if set)
    # * if fname is a file: output next to the file (or inside relative_path if set)
    if os.path.isdir(fname):
        out_parent = fname
        fallback_base = os.path.basename(os.path.normpath(fname))
        """ # if name was a directory and images was not a list, writer received 
        # an image stack merged from multiple files; in this case, we append 
        # to the new filename "merged" to indicate this:
        if images_was_list==False:
            merged_files_appendix = "_merged" """
    else:
        out_parent = os.path.dirname(fname)
        fallback_base = os.path.splitext(os.path.basename(fname))[0]
        fallback_base = fallback_base.split(".")[0]  # strip dot-separated extra extensions
    
    # append "_merged" if requested:
    merged_files_appendix = ""
    if indicate_merged_files==True:
            merged_files_appendix = "_merged"
    
    # default output template uses fallback_base, but per-stack we may override via metadata provenance soon:
    fname_out = os.path.join(out_parent, fallback_base + ".ome.tif")
    #relative_path = "omio_outputs" # this will become a switch with None, "subfolder" or any relative path like or "../" "../subfolder"
    if relative_path is not None:
        out_parent = os.path.join(out_parent, relative_path)
        os.makedirs(out_parent, exist_ok=True)
        # refresh fname_out template (fallback)
        fname_out = os.path.join(out_parent, fallback_base + ".ome.tif")
        
    # we loop over images and metadatas:
    stack_n = len(images)
    stack_count = 0
    fnames_written = []
    for image, metadata in zip(images, metadatas):
        # image = images[0]
        # metadata = metadatas[0].copy()
        # check, whether bigtiff is necessary:
        use_bigtiff = _check_bigtiff(image, compression_level=compression_level)
        
        # build output filename base for this stack:
        orig_fn = _get_original_filename_from_metadata(metadata)
        if orig_fn is not None:
            base_i = os.path.splitext(orig_fn)[0]
            base_i = base_i.split(".")[0]
        else:
            base_i = fallback_base
        fname_out_i = os.path.join(out_parent, base_i + merged_files_appendix + ".ome.tif")
        # if multiple outputs, append index only if needed (collision-safe); We do NOT blindly 
        # append index, because original filenames are already unique in most cases:
        fname_out_stack = _check_fname_out(fname_out_i, overwrite)

        # if overwrite is False and _check_fname_out returns the same name but file exists,
        # _check_fname_out should already modify it.
        
        # if stack_n>1 and no provenance name exists, solve via adding numbering:
        if stack_n > 1 and orig_fn is None:
            stack_count += 1
            fname_out_i = os.path.join(out_parent, f"{base_i}_{stack_count:03d}.ome.tif")

        fname_out_stack = _check_fname_out(fname_out_i, overwrite)
        if verbose:
            print(f"Writing OME-TIFF to: {fname_out_stack} (bigtiff={use_bigtiff})")

        # reorder axes to OME standard TZCYX:
        axes_in = metadata.get("axes", "TZCYX")
        image_ome, axes_in = _normalize_axes_for_ometiff(image, axes_in)
        desired_axes = "TCZYX"
        if axes_in != desired_axes:
            idx = {ax: i for i, ax in enumerate(axes_in)}
            perm = [idx[ax] for ax in desired_axes]
            image_ome = np.moveaxis(image_ome, perm, range(len(perm)))
            axes_out = desired_axes
        else:
            axes_out = axes_in
        len_unit = metadata.get("unit", "µm")
        if len_unit in ("micron", "micrometer", "um"):
            len_unit = "µm"
        # check whether 
        
        ome_meta = {
            "axes": axes_out,
            "SizeX": metadata.get("SizeX", None),
            "SizeY": metadata.get("SizeY", None),
            "SizeZ": metadata.get("SizeZ", None),
            "SizeT": metadata.get("SizeT", None),
            "SizeC": metadata.get("SizeC", None),
            "PhysicalSizeX": metadata.get("PhysicalSizeX", None),
            "PhysicalSizeY": metadata.get("PhysicalSizeY", None),
            "PhysicalSizeZ": metadata.get("PhysicalSizeZ", None),
            "PhysicalSizeXUnit": len_unit,
            "PhysicalSizeYUnit": len_unit,
            "PhysicalSizeZUnit": len_unit,
            #'Description': 'A multi-dimensional, multi-resolution image',
            #'Channel': {'Name': ['Channel 1 fab', 'Channel 2 fab']},
            # 'MapAnnotation': {  
            #     'Namespace': 'omio:metadata',
            #     '_OMIO_VERSION': '0.1.0',
            #     'Experiment': 'MSD',
            #     'Experimenter': 'Fabrizio'},
            }
        # get the time increment if present:
        time_incr = metadata.get("TimeIncrement", None)
        if time_incr is not None:
            ome_meta["TimeIncrement"] = float(time_incr)
            tunit = metadata.get("TimeIncrementUnit", "s")
            if tunit in ("sec", "seconds"):
                tunit = "s"
            ome_meta["TimeIncrementUnit"] = tunit
        # get any MapAnnotations if present:
        annotations = metadata.get("Annotations", None)
        if isinstance(annotations, dict):
            ma = dict(annotations)
            if "Namespace" not in ma:
                ma["Namespace"] = "omio:metadata"
            ome_meta["MapAnnotation"] = ma
        elif isinstance(annotations, list):
            ma_list = []
            for ann in annotations:
                if not isinstance(ann, dict):
                    continue
                ma = dict(ann)
                if "Namespace" not in ma:
                    ma["Namespace"] = "omio:metadata"
                ma_list.append(ma)
            if ma_list:
                ome_meta["MapAnnotation"] = ma_list
        tifffile.imwrite(
            fname_out_stack,
            image_ome,
            ome=True,
            compression="zlib",
            compressionargs={"level": compression_level},
            resolution=(1/metadata["PhysicalSizeY"], 1/metadata["PhysicalSizeX"]),
            metadata=ome_meta,
            photometric="minisblack",
            imagej=False,
            bigtiff=use_bigtiff)
        fnames_written.append(fname_out_stack)
    if return_fnames:
        return fnames_written

# %% NAPARI-VIEWER CONVENIENCE FUNCTIONS

# function for squeezing a Zarr array for Napari visualization:
def _squeeze_zarr_to_napari_cache(src, fname, axes="TZCYXS", cache_folder_name=".omio_cache"):

    if not isinstance(src, zarr.core.array.Array):
        raise TypeError("_squeeze_zarr_to_napari_cache expects a zarr.core.Array as `src`.")

    src_shape = src.shape
    axes_list = list(axes)
    if len(axes_list) != len(src_shape):
        raise ValueError(f"axes length {len(axes_list)} does not match src.ndim {len(src_shape)}")

    # keep all non singleton axes, but never drop Y or X even if singleton
    keep_indices = [i for i, dim in enumerate(src_shape)
                    if (dim > 1) or (axes_list[i] in ("Y", "X"))]

    squeezed_axes = "".join(axes_list[i] for i in keep_indices)
    squeezed_shape = tuple(src_shape[i] for i in keep_indices)

    napari_zarr_path = fname
    if os.path.exists(napari_zarr_path):
        shutil.rmtree(napari_zarr_path)

    if src.chunks is not None:
        squeezed_chunks = tuple(src.chunks[i] for i in keep_indices)
    else:
        squeezed_chunks = None

    dst = zarr.open(
        napari_zarr_path,
        mode="w",
        shape=squeezed_shape,
        dtype=src.dtype,
        chunks=squeezed_chunks)

    # copy shortcut for 2D or less
    if len(squeezed_shape) <= 2:
        src_idx = []
        for i, dim in enumerate(src_shape):
            if i in keep_indices:
                src_idx.append(slice(None))
            else:
                src_idx.append(0)
        dst[...] = src[tuple(src_idx)]
        return dst, squeezed_axes

    # determine positions of spatial axes inside the squeezed representation
    y_pos = squeezed_axes.find("Y")
    x_pos = squeezed_axes.find("X")
    if y_pos < 0 or x_pos < 0:
        raise ValueError("Squeezed axes must contain Y and X.")

    # outer axes are all except Y and X
    outer_axes_positions = [i for i in range(len(squeezed_axes)) if i not in (y_pos, x_pos)]
    outer_shape = tuple(squeezed_shape[i] for i in outer_axes_positions)
    total_outer = int(np.prod(outer_shape)) if outer_shape else 1

    # build mapping from squeezed positions to original indices
    squeezed_to_orig = {sq_i: orig_i for sq_i, orig_i in enumerate(keep_indices)}

    for outer_idx in tqdm(
        np.ndindex(*outer_shape) if outer_shape else [()],
        total=total_outer,
        desc="creating Napari view Zarr (squeezed)"
    ):
        # build dst index in squeezed space
        dst_idx = [0] * len(squeezed_shape)

        # fill outer axes indices
        for pos, val in zip(outer_axes_positions, outer_idx):
            dst_idx[pos] = val

        # set Y and X to full slices
        dst_idx[y_pos] = slice(None)
        dst_idx[x_pos] = slice(None)

        # now build src index in original space
        src_idx = [0] * len(src_shape)
        for sq_pos in range(len(squeezed_axes)):
            orig_pos = squeezed_to_orig[sq_pos]
            ax = squeezed_axes[sq_pos]
            if ax in ("Y", "X"):
                src_idx[orig_pos] = slice(None)
            else:
                src_idx[orig_pos] = dst_idx[sq_pos]

        dst[tuple(dst_idx)] = src[tuple(src_idx)]

    return dst, squeezed_axes
# function to get channel axis from axes and shape:
def _get_channel_axis_from_axes_and_shape(axes, shape, target_axis="C"):
    """
    Return the index of a specific axis in a squeezed array.

    This helper determines the positional index of a given axis label within an
    axis string and its corresponding array shape. It is typically used after
    singleton dimensions have been removed, where the remaining axes define the
    layout of a reduced array.

    Parameters
    ----------
    axes : str
        Axis string describing the order of dimensions in the array, for example
        ``"ZCYX"``.
    shape : tuple
        Shape of the array corresponding to `axes`.
    target_axis : str, optional
        Axis label to locate. The default is ``"C"`` for the channel axis.

    Returns
    -------
    int or None
        Zero-based index of the requested axis in the array if present, otherwise
        ``None``.

    Raises
    ------
    ValueError
        If the length of `axes` does not match the length of `shape`.

    Notes
    -----
    * The function performs a simple linear scan over the axis string.
    * No validation of axis semantics is performed beyond matching the label.
    """
    if len(axes) != len(shape):
        raise ValueError("axes and shape must have the same length")
    for i, ax in enumerate(axes):
        if ax == target_axis:
            return i
    return None
# function to get scales from axes and metadata:
def _get_scales_from_axes_and_metadata(axes, metadata):
    """
    Construct Napari scale values from an axis string and OMIO metadata.

    This helper derives a tuple of scale factors suitable for passing to Napari’s
    ``scale`` argument. Spatial axes are mapped to their corresponding physical
    voxel sizes stored in the metadata, while non-spatial axes receive a unit scale
    of 1.0. The channel axis ``"C"`` is explicitly excluded, because when Napari’s
    ``channel_axis`` parameter is used, Napari expects the scale tuple to have
    length ``ndim - 1`` and to cover only non-channel axes.

    Axis handling
    -------------
    * ``Z`` → ``metadata["PhysicalSizeZ"]``
    * ``Y`` → ``metadata["PhysicalSizeY"]``
    * ``X`` → ``metadata["PhysicalSizeX"]``
    * ``C`` → skipped (no scale entry)
    * All other axes (for example ``T`` or ``S``) → scale ``1.0``

    Parameters
    ----------
    axes : str
        Axis string corresponding to the array passed to Napari, for example
        ``"TCYX"`` or ``"TZCYX"``.
    metadata : dict
        Metadata dictionary providing physical voxel sizes under the keys
        ``PhysicalSizeX``, ``PhysicalSizeY``, and ``PhysicalSizeZ``.

    Returns
    -------
    tuple of float
        Scale values for all non-channel axes, in the order in which those axes
        appear in `axes`.

    Notes
    -----
    * No unit conversion is performed. The returned values are assumed to already
    be in the units expected by Napari.
    * Missing physical size entries in `metadata` will raise a ``KeyError``.
    """
    scales = []
    for ax in axes:
        # Channel axis is handled via `channel_axis` in napari and
        # must not receive a separate scale entry.
        if ax == "C":
            continue
        if ax == "Z":
            scales.append(metadata["PhysicalSizeZ"])
        elif ax == "Y":
            scales.append(metadata["PhysicalSizeY"])
        elif ax == "X":
            scales.append(metadata["PhysicalSizeX"])
        else:
            # T, S, and all other non-spatial axes:
            scales.append(1.0)
    return tuple(scales)
# function for squeezing a Zarr array for Napari visualization using Dask:
def _squeeze_numpy_keep_yx(image_np: np.ndarray, axes_full: str) -> tuple[np.ndarray, str]:
    """ 
    Squeeze a NumPy array by removing singleton axes except for Y and X.
    
    This helper removes all singleton dimensions from a NumPy array while preserving
    the Y and X axes, even if they are singleton. The function also constructs an
    updated axis string that reflects the new shape of the array.
    
    Parameters
    ----------
    image_np : np.ndarray
        Input NumPy array to be squeezed.
    axes_full : str
        Full axis string corresponding to `image_np.shape`. This is typically an OME-like
        axis declaration such as ``"TZCYXS"``.
    Returns
    -------
    image_sq : np.ndarray
        Squeezed NumPy array with singleton axes removed (except Y and X).
    axes_sq : str
        Updated axis string corresponding to `image_sq`.
    """
    if len(image_np.shape) != len(axes_full):
        raise ValueError("NumPy image does not match expected OME axis length")

    squeeze_axes = [
        i for i, (ax, dim) in enumerate(zip(axes_full, image_np.shape))
        if (dim == 1) and (ax not in ("Y", "X"))
    ]

    if squeeze_axes:
        image_sq = np.squeeze(image_np, axis=tuple(squeeze_axes))
    else:
        image_sq = image_np

    axes_sq = "".join(
        ax for ax, dim in zip(axes_full, image_np.shape)
        if (dim > 1) or (ax in ("Y", "X"))
    )

    return image_sq, axes_sq
def _squeeze_zarr_to_napari_cache_dask(src, fname, axes, cache_folder_name=".omio_cache"):
    """
    Create a squeezed on-disk Zarr view for Napari using Dask.

    This helper constructs a derived Zarr store in which all singleton dimensions of
    a source Zarr array are removed. The computation is performed with Dask so that
    the source array is not materialized fully in RAM. Instead, Dask streams chunks
    from the input Zarr, applies ``squeeze`` lazily, and writes the result into a
    new Zarr store under an OMIO cache folder.

    The function also returns the corresponding squeezed axis string, obtained by
    dropping axis labels whose dimensions were of length 1.

    Parameters
    ----------
    src : zarr.core.array.Array
        Source Zarr array. The array is expected to be OME-like ordered according to
        `axes` (often ``"TZCYXS"``).
    fname : str
        Path used to derive the cache location. The squeezed Zarr store is written
        into ``<dirname(fname)>/<cache_folder_name>/`` and named
        ``<basename(fname)>_napari_squeezed.zarr``.
    axes : str
        Axis string corresponding to ``src.shape``.
    cache_folder_name : str, optional
        Name of the cache folder created alongside `fname`. Default is
        ``".omio_cache"``.

    Returns
    -------
    squeezed_zarr : zarr.core.array.Array
        Newly created Zarr array stored on disk with all singleton axes removed.
    squeezed_axes : str
        Axis string corresponding to `squeezed_zarr`.

    Notes
    -----
    * Any existing Zarr store at the target path is deleted and replaced.
    * The write is performed via Dask’s Zarr writer to allow chunk-wise computation
    and writing. This avoids reading the full source array into memory.
    * The computed list of singleton axis indices is used only to derive the
    returned axis string; the actual squeeze operation is performed by
    ``da.squeeze``.
    * This function creates a derived representation for visualization and does not
    modify the source Zarr store.
    """

    base_dir = os.path.dirname(fname)
    cache_dir = os.path.join(base_dir, cache_folder_name)
    os.makedirs(cache_dir, exist_ok=True)

    target_path = os.path.join(cache_dir, os.path.basename(fname) + "_napari_squeezed.zarr")
    if os.path.exists(target_path):
        shutil.rmtree(target_path)

    darr = da.from_zarr(src)

    # squeeze only singleton axes that are not Y or X:
    squeeze_axes = [i for i, (ax, dim) in enumerate(zip(axes, src.shape))
                    if (dim == 1) and (ax not in ("Y", "X"))]

    squeezed_axes = "".join(ax for ax, dim in zip(axes, src.shape)
                            if (dim > 1) or (ax in ("Y", "X")))

    if squeeze_axes:
        darr = da.squeeze(darr, axis=tuple(squeeze_axes))

    da.to_zarr(darr, target_path, zarr_read_kwargs={"mode": "w"})
    squeezed_zarr = zarr.open(target_path, mode="r")

    return squeezed_zarr, squeezed_axes

# main single-image handle for Napari visualization of image(s) as NumPy, Zarr, or Zarr + Dask:
def _single_image_open_in_napari(
        image: Union[np.ndarray, "zarr.core.array.Array"], 
        metadata: dict, 
        fname: str, 
        zarr_mode: str = "numpy",
        cache_folder_name: str = ".omio_cache", 
        axes_full: str = "TZCYX", 
        viewer=None,
        viewer_name: Union[None, str] = None, 
        verbose: bool = True
        ) -> tuple["napari.Viewer", "napari.layers.Image", Union[np.ndarray, "zarr.core.array.Array"], str]:
    """
    Open or extend a Napari viewer with a single OMIO image.

    This helper prepares an image in OMIO’s canonical OME axis convention and then
    adds it as a Napari image layer. It supports NumPy arrays and Zarr arrays, and
    for Zarr inputs it provides three strategies controlled by `zarr_mode`:

    * ``"numpy"``: fully materialize the Zarr array into RAM as a NumPy array,
    apply ``squeeze()``, and pass the result to Napari. This is fastest if the
    dataset fits comfortably in memory.
    * ``"zarr_nodask"``: create a new squeezed on-disk Zarr store under a cache
    folder by copying plane-wise. Napari reads from this derived store.
    * ``"zarr_dask"``: create the squeezed on-disk Zarr store using Dask for
    chunk-wise IO and parallelized writing, avoiding full materialization in RAM.

    For NumPy inputs, the array is squeezed in RAM and the axis string is reduced
    accordingly.

    The function attempts to reuse an existing viewer when possible: if `viewer` is
    provided it is used, otherwise ``napari.current_viewer()`` is tried, and if that
    fails a new viewer is created.

    Parameters
    ----------
    image : np.ndarray or zarr.core.array.Array or list or tuple
        Image data. If a list or tuple is provided, only the first element is used.
        The input is expected to be OME-normalized already (for example via
        ``_correct_for_OME_axes_order``) so that it matches `axes_full`.
    metadata : dict or list or tuple
        Metadata corresponding to `image`. If a list or tuple is provided, only the
        first element is used. The metadata should provide physical voxel sizes
        (``PhysicalSizeX``, ``PhysicalSizeY``, ``PhysicalSizeZ``) and optionally a
        length unit under ``unit``.
    fname : str
        Source filename used to derive the default layer name and cache locations.
    zarr_mode : {"numpy", "zarr_nodask", "zarr_dask"}, optional
        Strategy for handling Zarr inputs. Default is ``"numpy"``.
    cache_folder_name : str, optional
        Name of the hidden cache folder used to store derived Zarr stores created by
        the squeezing modes. Default is ``".omio_cache"``.
    axes_full : str, optional
        Axis string describing the full expected axis order of the input before
        squeezing. Default is ``"TZCYX"``. The implementation assumes that `image`
        is consistent with this declaration.
    viewer : napari.Viewer or None, optional
        Existing Napari viewer to reuse. If None, a current viewer is reused if
        available, otherwise a new viewer is created.
    viewer_name : str or None, optional
        Explicit layer name to use. If None, the basename of `fname` is used.
    verbose : bool, optional
        If True, print diagnostic progress messages. Default is True.

    Returns
    -------
    viewer : napari.Viewer
        The Napari viewer that was used or created.
    layer : napari.layers.Image
        The newly added image layer.
    napari_data : np.ndarray or dask.array.Array
        The data object passed to Napari. Zarr outputs are converted to a Dask array
        via ``da.from_zarr`` for better Napari behavior.
    napari_axes : str
        Axis string corresponding to `napari_data` after squeezing.

    Raises
    ------
    ValueError
        If `image` or `metadata` is an empty list or tuple.
    ValueError
        If the input array dimensionality does not match `axes_full`.
    ValueError
        If `zarr_mode` is not one of the supported values.

    Notes
    -----
    * The channel axis is inferred from the squeezed axis string via
    ``_get_channel_axis_from_axes_and_shape`` and passed to Napari as
    ``channel_axis`` when present.
    * Scale factors are computed from metadata via
    ``_get_scales_from_axes_and_metadata``. The channel axis is excluded from the
    scale tuple by design.
    * When `zarr_mode` produces a Zarr store, the store is written under the cache
    folder and may overwrite an existing derived store with the same name.
    """

    # fallback normalization: extract first element from lists/tuples
    if isinstance(image, (list, tuple)):
        if len(image) == 0:
            raise ValueError("  _single_image_open_in_napari: 'image' list is empty.")
        image = image[0]
    if isinstance(metadata, (list, tuple)):
        if len(metadata) == 0:
            raise ValueError("  _single_image_open_in_napari: 'metadata' list is empty.")
        metadata = metadata[0]

    
    # case 1: Zarr-array
    if isinstance(image, zarr.core.array.Array):
        if verbose:
            print("  Input is Zarr array.")
            print(f"  Preparing image for napari (zarr_mode='{zarr_mode}')...")
        if zarr_mode == "zarr_dask":
            # Zarr → squeezed Zarr w/ Dask:
            if verbose:
                print("  Using Dask for memory-efficient squeezing...")
            store_path = str(image.store).replace("file://", "")
            base_no_ext = store_path.replace(".zarr", "")

            squeezed_zarr, squeezed_axes = _squeeze_zarr_to_napari_cache_dask(src=image,
                                                fname=base_no_ext, axes=axes_full,
                                                cache_folder_name=cache_folder_name)
            napari_data = squeezed_zarr
            napari_axes = squeezed_axes

        elif zarr_mode == "zarr_nodask":
            # Zarr → squeezed Zarr w/o Dask:
            if verbose:
                print("  Memory-efficient squeezing Zarr without Dask...")
            store_path = str(image.store).replace("file://", "")
            base_no_ext = store_path.replace(".zarr", "")
            squeezed_zarr, squeezed_axes = _squeeze_zarr_to_napari_cache(src=image,
                                                fname=base_no_ext, axes=axes_full,
                                                cache_folder_name=cache_folder_name)
            napari_data = squeezed_zarr
            napari_axes = squeezed_axes
        elif zarr_mode == "numpy":
            # Zarr → NumPy into RAM, then squeeze:
            if verbose:
                print("  Loading full Zarr into RAM as NumPy array...")
            image_np = np.asarray(image)
            if len(image_np.shape) != len(axes_full):
                raise ValueError("NumPy image does not match expected OME axis length")
            #napari_data = image_np.squeeze()
            napari_data, napari_axes = _squeeze_numpy_keep_yx(image_np, axes_full)
            #napari_axes = "".join(ax for ax, dim in zip(axes_full, image_np.shape) if dim > 1)
        else:
            raise ValueError(
                f"  _single_image_open_in_napari: unknown zarr_mode='{zarr_mode}'. "
                f"  Use one of 'numpy', 'zarr_nodask', 'zarr_dask'.")

    # case 2: NumPy-array
    else:
        if verbose:
            print("  Input is NumPy array. Full loading into RAM (zarr_mode has no effect)...")
        image_np = np.asarray(image)
        if len(image_np.shape) != len(axes_full):
            raise ValueError("  NumPy image does not match expected OME axis length")
        #napari_data = image_np.squeeze()
        napari_data, napari_axes = _squeeze_numpy_keep_yx(image_np, axes_full)
        #napari_axes = "".join(ax for ax, dim in zip(axes_full, image_np.shape) if dim > 1)

    # determine channel axis:
    if len(napari_axes) != napari_data.ndim:
        raise ValueError(
            f"Internal error: napari_axes='{napari_axes}' (len={len(napari_axes)}) "
            f"does not match napari_data.shape={napari_data.shape} (ndim={napari_data.ndim}).")
    channel_axis = _get_channel_axis_from_axes_and_shape(axes=napari_axes, 
                                                        shape=napari_data.shape, 
                                                        target_axis="C")

    # get scales (C-axis is not scaled in _get_scales_from_axes_and_metadata):
    scales_array = _get_scales_from_axes_and_metadata(axes=napari_axes,metadata=metadata)

    # check whether a viewer is already given, create a new one otherwise:
    if viewer is None:
        try:
            viewer = napari.current_viewer()
        except Exception:
            viewer = None
        if viewer is None:
            viewer = napari.Viewer()

    # build layer name:
    if viewer_name is not None:
        layer_name = viewer_name
    else:
        layer_name = os.path.basename(fname)
    
    # convert napari_data into a dask-array if it's a Zarr (napari handles zarr dask arrays better):
    if isinstance(napari_data, zarr.core.array.Array):
        napari_data = da.from_zarr(napari_data)
    
    # add the new image layer:
    layer = viewer.add_image(napari_data, channel_axis=channel_axis, 
                             scale=scales_array, name=layer_name)
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = metadata.get("unit", "micron")

    return viewer, layer, napari_data, napari_axes
# main multi-image handler for Napari visualization of image(s) as NumPy, Zarr, or Zarr + Dask:
def open_in_napari(images: Union[np.ndarray, "zarr.core.array.Array", list[Union[np.ndarray, "zarr.core.array.Array"]]],
                   metadatas: Union[dict, list[dict]], 
                   fname: str, 
                   zarr_mode: str = "numpy", 
                   cache_folder_name: str = ".omio_cache", 
                   axes_full: str = "TZCYX", 
                   viewer: napari.Viewer = None, 
                   returns: bool=False, 
                   verbose: bool=True):
    """
    Open or extend a Napari viewer with one or multiple OMIO images.

    This is the main Napari convenience wrapper exposed to users. It accepts a
    single image or a sequence of images together with matching metadata objects,
    and adds each dataset as a Napari image layer by delegating per-image handling
    to ``_single_image_open_in_napari``.

    Input images may be NumPy arrays or Zarr arrays. For Zarr inputs, the behavior
    is controlled by `zarr_mode` and follows the same strategies implemented in
    ``_single_image_open_in_napari`` (full materialization to NumPy, creation of a
    squeezed cache Zarr without Dask, or creation of a squeezed cache Zarr with
    Dask). A single viewer instance is reused across all layers.

    Parameters
    ----------
    images : np.ndarray or zarr.core.array.Array or list of (np.ndarray or zarr.core.array.Array)
        Image data to visualize. If a single array is provided, it is treated as a
        one-element list. Each image is expected to be consistent with `axes_full`
        before squeezing (for example already normalized by
        ``_correct_for_OME_axes_order``).
    metadatas : dict or list of dict
        Metadata dictionaries corresponding to `images`. If a single dict is
        provided, it is treated as a one-element list. Each metadata dict should
        provide the physical voxel sizes used for Napari scaling (typically
        ``PhysicalSizeX``, ``PhysicalSizeY``, ``PhysicalSizeZ``) and optionally a
        unit string under ``unit``.
    fname : str
        Base name used for Napari layer naming and cache path construction. If
        multiple images are provided, an ``_idx{n}`` suffix is appended.
    zarr_mode : {"numpy", "zarr_nodask", "zarr_dask"}, optional
        Strategy for handling Zarr inputs, forwarded to
        ``_single_image_open_in_napari``. Default is ``"numpy"``.
    cache_folder_name : str, optional
        Name of the cache folder used for derived Zarr stores. Default is
        ``".omio_cache"``.
    axes_full : str, optional
        Full axis string describing the expected axis order of the input images
        before squeezing. Default is ``"TZCYX"``.
    viewer : napari.Viewer or None, optional
        Existing Napari viewer to reuse. If None, a current viewer is reused if
        available, otherwise a new viewer is created (via the single-image helper).
    returns : bool, optional
        If True, return detailed objects (viewer, layers, napari_datas, napari_axess).
        If False, the function returns None. Default is False.
    verbose : bool, optional
        If True, print diagnostic progress messages. Default is True.

    Returns
    -------
    viewer : napari.Viewer
        The Napari viewer that was used or created. Only returned if `returns=True`.
    layers : list of napari.layers.Image
        The image layers added to the viewer, one per input image. Only returned if
        `returns=True`.
    napari_datas : list of (np.ndarray or dask.array.Array)
        The data objects passed to Napari for each layer (Zarr inputs are typically
        converted to Dask arrays in the single-image helper). Only returned if
        `returns=True`.
    napari_axess : list of str
        Axis strings corresponding to each entry in `napari_datas` after squeezing.
        Only returned if `returns=True`.

    Raises
    ------
    ValueError
        If the number of images does not match the number of metadata dictionaries.

    Notes
    -----
    * This function does not perform axis normalization itself. It assumes that
      inputs already follow OMIO’s canonical axis convention as declared by
      ``axes_full``, and delegates squeezing, channel-axis inference, and scaling to
      ``_single_image_open_in_napari``.
    * When multiple images are opened, the layer name is derived from ``fname`` with a
      simple index suffix; if more informative naming is desired, pass a distinct
      ``fname`` per call or use the ``viewer_name`` mechanism in the single-image helper.
    """
    # check, whether images and metadatas are lists:
    if not isinstance(images, (list, tuple)):
        images = [images]
    if not isinstance(metadatas, (list, tuple)):
        metadatas = [metadatas]
    if len(images) != len(metadatas):
        raise ValueError("open_in_napari: images and metadatas must have the same length.")

    if verbose:
        print(f"Got {len(images)} image(s) to open in napari.")

    layers = []
    napari_datas = []
    napari_axess = []

    for idx, (img, md) in enumerate(zip(images, metadatas)):
        if verbose:
            print(f"Opening image {idx+1}/{len(images)} in napari...")
        
        # build layer name:
        layer_fname = fname if len(images) == 1 else f"{fname}_idx{idx}"
        
        # open in napari:
        v, layer, napari_data, napari_axes = _single_image_open_in_napari(
            image=img,
            metadata=md,
            fname=layer_fname,
            zarr_mode=zarr_mode,
            cache_folder_name=cache_folder_name,
            axes_full=axes_full,
            viewer=viewer,
            verbose=verbose)
        viewer = v
        layers.append(layer)
        napari_datas.append(napari_data)
        napari_axess.append(napari_axes)
    
    if verbose:
        print(f"Opened {len(images)} image(s) with scales:")
        if type(layers[0]) is list:
            layer_to_iterate = layers[0]
        else:
            layer_to_iterate = layers
        for i, layer in enumerate(layer_to_iterate):
            # i = 0
            # layer = layers[0][i]
            print(f"  Layer {i}: name='{layer.name}', scale={layer.scale}, shape={layer.data.shape}")
        #print("All images opened in napari.")
    if returns:
        return viewer, layers, napari_datas, napari_axess

# %% CONVENIENCE READER AND CONVERTER

# helper functions:

# function to normalize input filename(s) to a list of strings:
def _normalize_to_list(fname: Union[str, os.PathLike, List[Union[str, os.PathLike]]]) -> List[str]:
    """
    Normalize input filenames to a list of strings.

    This helper ensures that a filename argument is always represented as a list of
    string paths. It accepts a single path-like object or a sequence of such objects
    and converts all entries to their string representation.

    Parameters
    ----------
    fname : str or os.PathLike or list of (str or os.PathLike)
        Input filename or filenames to normalize.

    Returns
    -------
    list of str
        List of filename strings. A single input is wrapped into a one-element list.

    Notes
    -----
    * Path-like objects are converted using ``str(...)``.
    * Tuples are treated the same as lists and returned as a new list.
    """
    if isinstance(fname, (list, tuple)):
        return [str(f) for f in fname]
    return [str(fname)]
# function to check whether path is a directory:
def _is_dir(p: str) -> bool:
    """
    Check whether a path refers to an existing directory.

    This helper wraps ``os.path.isdir`` to provide a small, explicit predicate that
    tests whether the given path exists and is a directory.

    Parameters
    ----------
    p : str
        Path to test.

    Returns
    -------
    bool
        True if `p` exists and is a directory, False otherwise.
    """
    return os.path.isdir(p)
# function to check whether path is a file:
def _is_file(p: str) -> bool:
    """
    Check whether a path refers to an existing file.

    This helper wraps ``os.path.isfile`` to provide a small, explicit predicate that
    tests whether the given path exists and is a file.

    Parameters
    ----------
    p : str
        Path to test.

    Returns
    -------
    bool
        True if `p` exists and is a file, False otherwise.
    """
    return os.path.isfile(p)
# function to get lowercased file extension:
def _lower_ext(p: str) -> str:
    """
    Return the lowercased file extension of a path.

    This helper extracts the file extension from a path and normalizes it to
    lowercase. The returned string includes the leading dot. If the path has no
    extension, an empty string is returned.

    Parameters
    ----------
    p : str
        Path from which to extract the file extension.

    Returns
    -------
    str
        Lowercased file extension, including the leading dot, or an empty string if
        no extension is present.
    """
    return os.path.splitext(p)[1].lower()
# function to check whether path looks like an OME-TIFF:
def _looks_like_ome_tif(p: str) -> bool:
    """
    Check whether a path looks like an OME-TIFF filename.

    This helper performs a simple filename-based check to determine whether a path
    appears to refer to an OME-TIFF file by testing for the standard OME-TIFF
    extensions.

    Parameters
    ----------
    p : str
        Path or filename to check.

    Returns
    -------
    bool
        True if the path ends with ``.ome.tif`` or ``.ome.tiff`` (case-insensitive),
        False otherwise.

    Notes
    -----
    * This is a heuristic based solely on the filename extension and does not
    inspect file contents.
    """
    lp = p.lower()
    return lp.endswith(".ome.tif") or lp.endswith(".ome.tiff")
# function to list image files in a folder:
def _list_image_files_in_folder(folder: str,
                                allowed_ext: Union[None, set] = None,
                                recursive: bool = False) -> List[str]:
    """
    List image files in a folder matching supported extensions.

    This helper scans a directory for image files whose extensions match a set of
    allowed formats commonly handled by OMIO. It can operate either non-recursively
    on a single directory level or recursively across all subdirectories.

    OME-TIFF files are detected explicitly via their ``.ome.tif`` or ``.ome.tiff``
    suffixes and are always included when present.

    Parameters
    ----------
    folder : str
        Path to the directory to scan for image files.
    allowed_ext : set of str or None, optional
        Set of allowed lowercase file extensions (including the leading dot).
        If None, a default set is used:
        ``{".tif", ".tiff", ".lsm", ".czi", ".raw", ".ome.tif", ".ome.tiff"}``.
    recursive : bool, optional
        If True, search recursively through all subdirectories of `folder`.
        If False, only files directly inside `folder` are considered. Default is
        False.

    Returns
    -------
    list of str
        Sorted list of file paths matching the allowed extensions.

    Notes
    -----
    * Only regular files are included; directories are ignored.
    * Extension checks are case-insensitive.
    * The function does not validate file contents and relies solely on filename
    extensions.
    """
    if allowed_ext is None:
        allowed_ext = {".tif", ".tiff", ".lsm", ".czi", ".raw", ".ome.tif", ".ome.tiff"}

    patterns = []
    if recursive:
        patterns.append(os.path.join(folder, "**", "*"))
    else:
        patterns.append(os.path.join(folder, "*"))

    files = []
    for pat in patterns:
        for p in glob.glob(pat, recursive=recursive):
            if not os.path.isfile(p):
                continue
            lp = p.lower()
            if _looks_like_ome_tif(lp):
                files.append(p)
                continue
            ext = _lower_ext(lp)
            if ext in allowed_ext:
                files.append(p)

    files = sorted(files)
    return files
# function to get the first image file in a folder:
def _first_image_file_in_folder(folder: str,
                                allowed_ext: Union[None, set] = None) -> Union[None, str]:
    """
    Return the first image file found in a folder.

    This helper scans a directory for image files matching a set of allowed
    extensions and returns the first match according to the sorted order defined
    by ``_list_image_files_in_folder``. If no matching files are found, ``None`` is
    returned.

    Parameters
    ----------
    folder : str
        Path to the directory to scan for image files.
    allowed_ext : set of str or None, optional
        Set of allowed lowercase file extensions (including the leading dot). If
        None, the default extension set used by
        ``_list_image_files_in_folder`` is applied.

    Returns
    -------
    str or None
        Path to the first matching image file, or ``None`` if no image files are
        found.

    Notes
    -----
    * The search is non-recursive.
    * File ordering is determined by lexicographic sorting of the matched paths.
    * No validation of file contents is performed.
    """
    files = _list_image_files_in_folder(folder, allowed_ext=allowed_ext, recursive=False)
    if not files:
        return None
    return files[0]
# function to merge metadata sources:
def _merge_metadata_sources(sources: List[Dict[str, Any]],
                            namespace: str = "omio:merge",
                            keep_original_forever: bool = True) -> Dict[str, Any]:
    """
    Merge multiple metadata dictionaries originating from different image stacks
    into a single metadata dictionary with explicit provenance tracking.

    The merge policy is conservative and provenance focused:

    * Metadata from the first source (index 0) is taken as authoritative for
    physical scaling and timing fields.
    * PhysicalSizeX, PhysicalSizeY, PhysicalSizeZ, and TimeIncrement are compared
    across all sources. If inconsistencies are detected, a warning is issued and
    the value from source 0 is retained.
    * Image size related keys (SizeT, SizeZ, SizeC, SizeY, SizeX) are not recomputed
    here and are expected to be updated later from the merged image data.
    * Provenance information for each input source is collected and stored inside
    the Annotations block under a dedicated namespace.

    Parameters
    ----------
    sources : list of dict
        List of metadata dictionaries to be merged. Each entry is assumed to
        correspond to one image stack.
    namespace : str, optional
        Namespace prefix used for keys written into the Annotations block that
        describe the merge operation. Default is "omio:merge".
    keep_original_forever : bool, optional
        If True, existing original_* keys inside Annotations are preserved and not
        overwritten. Default is True.

    Returns
    -------
    dict
        A merged metadata dictionary based on the first source, extended with
        provenance and merge information stored in the Annotations field.

    Notes
    -----
    * This function does not modify the input dictionaries in place.
    * Provenance information includes original filename, parent folder, file type,
    metadata type, shape, and axes for each source stack.
    """
    if not sources:
        return {}

    md0 = dict(sources[0])

    def _get(md: Dict[str, Any], k: str, default=None):
        return md.get(k, default)

    # Compare physical sizes and time increment across sources and warn if inconsistent.
    keys_to_compare = ["PhysicalSizeX", "PhysicalSizeY", "PhysicalSizeZ", "TimeIncrement"]
    for k in keys_to_compare:
        v0 = _get(md0, k, None)
        for i, mdi in enumerate(sources[1:], start=1):
            vi = _get(mdi, k, None)
            if v0 is None or vi is None:
                continue
            try:
                if float(v0) != float(vi):
                    warnings.warn(
                        f"Metadata mismatch in '{k}' between stack 0 ({v0}) and stack {i} ({vi}). "
                        f"Using stack 0 value."
                    )
                    break
            except Exception:
                if v0 != vi:
                    warnings.warn(
                        f"Metadata mismatch in '{k}' between stack 0 ({v0}) and stack {i} ({vi}). "
                        f"Using stack 0 value."
                    )
                    break

    # Build provenance block.
    provenance = []
    for i, mdi in enumerate(sources):
        provenance.append({
            "index": i,
            "original_filename": mdi.get("original_filename", "N/A"),
            "original_parentfolder": mdi.get("original_parentfolder", "N/A"),
            "original_filetype": mdi.get("original_filetype", "N/A"),
            "original_metadata_type": mdi.get("original_metadata_type", "N/A"),
            "shape": mdi.get("shape", None),
            "axes": mdi.get("axes", None),
        })

    # Place provenance into Annotations under a single namespace.
    annotations = md0.get("Annotations", {})
    if not isinstance(annotations, dict):
        annotations = {}
    annotations = dict(annotations)

    # Preserve existing original_* keys inside annotations if requested.
    if keep_original_forever:
        pass

    # tifffile MapAnnotation is single namespace in your current policy, so keep it flat.
    # We store the merge info as JSON-like string to keep it simple and robust.
    # If you prefer, you can store it as multiple keys, but keep in mind Fiji display readability.
    annotations["Namespace"] = md0.get("Annotations", {}).get("Namespace", "omio:metadata")
    annotations[f"{namespace}:created_utc"] = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S")
    annotations[f"{namespace}:n_sources"] = str(len(sources))
    annotations[f"{namespace}:sources"] = str(provenance)

    md0["Annotations"] = annotations
    return md0
# function to compute merge target shapes:
def _compute_merge_target_shapes(images, merge_along_axis: str, context: str = "merge"):
    """
    Compute target shapes required for merging multiple 5D images along a given axis.

    This helper determines three shape descriptors used during merge operations:

    * max_shape:
    The maximum extent across all input images for every axis except the merge
    axis. This defines the required padding or broadcasting size for non-merged
    dimensions.

    * merged_shape:
    The final output shape after merging, where the merge axis length is the sum
    of the corresponding axis lengths across all inputs, and all other axes take
    their maximum extent.

    * shapes:
    The original shapes of all input images, preserved in input order.

    Parameters
    ----------
    images : list of array-like
        Sequence of input images. Each image must be 5-dimensional and follow the
        OMIO/OME axis convention.
    merge_along_axis : str
        Axis label along which the images will be concatenated (e.g. "T", "Z", "C").
        Must be a valid key in the global axis-to-index mapping.
    context : str, optional
        Short context string used to prefix warning messages. Default is "merge".

    Returns
    -------
    max_shape : tuple[int, int, int, int, int] or None
        Maximum shape across all non-merge axes. None if validation fails.
    merged_shape : tuple[int, int, int, int, int] or None
        Shape of the merged output image. None if validation fails.
    shapes : list of tuple[int, ...] or None
        List of original input shapes in the same order as `images`.
        None if validation fails.

    Notes
    -----
    * All input images are expected to be 5D. If any input violates this
    assumption, a warning is issued and the function returns (None, None, None).
    * This function performs no data allocation and no axis reordering. It only
    computes shape bookkeeping required for downstream merge logic.
    """
    axis_idx = _AXIS_TO_INDEX[merge_along_axis]

    shapes = []
    for i, img in enumerate(images):
        try:
            s = tuple(img.shape)
        except Exception:
            s = tuple(np.asarray(img).shape)
        if len(s) != 5:
            warnings.warn(f"{context}: expected 5D arrays. Got shape {s} at index {i}.")
            return None, None, None
        shapes.append(s)

    # max over non merge axes
    max_shape = list(shapes[0])
    for j in range(5):
        if j == axis_idx:
            continue
        max_shape[j] = max(s[j] for s in shapes)

    # merged shape: merge axis is sum, others max
    merged_shape = list(max_shape)
    merged_shape[axis_idx] = int(sum(s[axis_idx] for s in shapes))

    return tuple(max_shape), tuple(merged_shape), shapes
# function to validate merge inputs:
def _validate_merge_inputs_with_optional_padding(images, metadatas, merge_along_axis: str,
                                                zeropadding: bool,
                                                context: str = "merge"):
    """
    Validate inputs for a multi-stack merge operation, with optional zero-padding support.

    This function enforces OMIO's merge preconditions for a set of input images and
    their corresponding metadata entries. The validation is intentionally strict
    about axis semantics and dimensionality and provides two modes regarding shape
    compatibility for non-merge axes.

    Validation policy
    -----------------
    * The merge axis must be one of the allowed merge axes.
    * `images` and `metadatas` must be non-empty and have identical lengths.
    * Each metadata entry must declare canonical OME axes exactly as "TZCYX".
    No attempt is made to repair or normalize axes during validation.
    * Each image must be 5D and compatible with the canonical axis convention.

    Shape compatibility modes
    -------------------------
    * If `zeropadding` is False (strict mode):
    All non-merge axes must match exactly across all stacks. Only the merge axis
    is allowed to differ. Any mismatch aborts the merge.

    * If `zeropadding` is True (padding-permitted mode):
    Exact agreement on non-merge axes is not required. Only the 5D requirement is
    enforced, enabling later padding or broadcasting logic to harmonize shapes.

    Parameters
    ----------
    images : list of array-like
        Sequence of image arrays to be merged. Each image must be 5-dimensional and
        follow the OME axis convention implied by metadata axes "TZCYX".
    metadatas : list of dict
        Sequence of metadata dictionaries aligned with `images`. Each must contain
        an "axes" entry that equals "TZCYX".
    merge_along_axis : str
        Axis label along which the images are intended to be merged (e.g. "T", "Z", "C").
        Must be a member of `_ALLOWED_MERGE_AXES`.
    zeropadding : bool
        If True, allow shape mismatches on non-merge axes (while still requiring 5D).
        If False, require exact matching across all non-merge axes.
    context : str, optional
        Short context string used to prefix warning messages. Default is "merge".

    Returns
    -------
    bool
        True if validation passes under the selected policy and mode, otherwise False.

    Notes
    -----
    * The function emits warnings (rather than raising exceptions) to support
    higher-level workflows that may choose alternative merge strategies.
    * In strict mode, the first image (index 0) defines the reference shape for all
    non-merge axes.
    * This function performs no padding, concatenation, or data copying. It only
    checks preconditions for downstream merge logic.
    """
    if merge_along_axis not in _ALLOWED_MERGE_AXES:
        print(f"{context}: invalid merge_along_axis={merge_along_axis!r}.\n"
              f"    Allowed: {sorted(_ALLOWED_MERGE_AXES)}.")
        return False

    if not images or not metadatas or len(images) != len(metadatas):
        print(f"{context}: empty inputs or mismatched images/metadatas list lengths.")
        return False

    for i, md in enumerate(metadatas):
        ax = md.get("axes", None)
        if ax != _OME_AXES:
            print(f"{context}: axes mismatch at index {i}. Expected '{_OME_AXES}' but got {ax!r}.\n"
                "    Merge aborted.")
            return False

    # shape checks:
    axis_idx = _AXIS_TO_INDEX[merge_along_axis]
    try:
        shape0 = tuple(images[0].shape)
    except Exception:
        shape0 = tuple(np.asarray(images[0]).shape)

    if len(shape0) != 5:
        warnings.warn(f"{context}: expected 5D arrays (TZCYX). Got shape {shape0}. \n"
                      "    Merge aborted.")
        return False

    if zeropadding:
        # only need to ensure every input is 5D
        for i, img in enumerate(images):
            try:
                s = tuple(img.shape)
            except Exception:
                s = tuple(np.asarray(img).shape)
            if len(s) != 5:
                warnings.warn(
                    f"{context}: expected 5D arrays (TZCYX). Got shape {s} at index {i}. \n"
                    "    Merge aborted.")
                return False
        return True

    # strict mode: non merge axes must match
    must_match_axes = [a for a in _OME_AXES if a != merge_along_axis]
    for i, img in enumerate(images):
        try:
            shapei = tuple(img.shape)
        except Exception:
            shapei = tuple(np.asarray(img).shape)

        if len(shapei) != 5:
            warnings.warn(
                f"{context}: expected 5D arrays (TZCYX). Got shape {shapei} at index {i}. \n"
                "    Merge aborted.")
            return False

        for a in must_match_axes:
            j = _AXIS_TO_INDEX[a]
            if shapei[j] != shape0[j]:
                print(f"{context}: incompatible shapes for merge along '{merge_along_axis}'.\n"
                      f"    Mismatch in axis '{a}' between stack 0 ({shape0}) and stack {i} ({shapei}).\n"
                       "    Merge aborted.")
                return False

    return True
# function to open Zarr for merge output:
def _zarr_open_for_merge_output(zarr_store: str, folder: str, basename: str, shape, dtype, chunks):
    """
    Create and open a Zarr array to be used as the output target of a merge operation.

    This helper encapsulates OMIO’s policy for allocating the destination Zarr store
    used when merging multiple image stacks. The storage backend is selected via
    `zarr_store` and the resulting Zarr array is always opened in write mode,
    replacing any existing on-disk store if necessary.

    Storage modes
    -------------
    * zarr_store == "memory":
    Create a Zarr array backed by an in-memory `MemoryStore`. The data live only
    for the lifetime of the Python process.

    * zarr_store == "disk":
    Create a persistent Zarr array on disk at
    `{folder}/.omio_cache/<basename>.zarr`. If a Zarr store with the same name
    already exists, it is removed and recreated.

    Parameters
    ----------
    zarr_store : str
        Storage backend selector. Must be either "memory" or "disk".
    folder : str
        Parent folder used when creating an on-disk Zarr store.
    basename : str
        Base name (without extension) for the output Zarr directory.
    shape : tuple
        Shape of the output array.
    dtype : numpy.dtype
        Data type of the output array.
    chunks : tuple
        Chunk shape to use for the Zarr array.

    Returns
    -------
    zarr.core.array.Array
        An opened Zarr array ready to receive merged image data.

    Raises
    ------
    ValueError
        If `zarr_store` is not one of the supported values.

    Notes
    -----
    * This function performs no validation of `shape`, `dtype`, or `chunks`; it
    assumes these have already been computed and validated by the merge logic.
    * The `.omio_cache` folder is created automatically if it does not exist.
    """
    if zarr_store == "memory":
        store = zarr.storage.MemoryStore()
        return zarr.open(store=store, mode="w", shape=shape, dtype=dtype, chunks=chunks)

    if zarr_store == "disk":
        zarr_cache_folder = os.path.join(folder, ".omio_cache")
        os.makedirs(zarr_cache_folder, exist_ok=True)
        zarr_path = os.path.join(zarr_cache_folder, basename + ".zarr")
        if os.path.exists(zarr_path):
            shutil.rmtree(zarr_path)
        return zarr.open(zarr_path, mode="w", shape=shape, dtype=dtype, chunks=chunks)

    raise ValueError(f"_zarr_open_for_merge_output: invalid zarr_store={zarr_store!r}.")
# function to copy into zarr chunk-aligned:
def _copy_into_zarr_chunk_aligned(z_out, img, out_start: int, axis_idx: int):
    """
    Copy `img` into an output Zarr array `z_out`, writing blocks aligned to the
    output chunk grid along a specified merge axis.

    The copy is performed only along `axis_idx`, starting at the output offset
    `out_start`. All other axes are copied fully. To minimize overhead and to keep
    the copy compatible with interactive environments, the function iterates in
    contiguous blocks whose length matches `z_out.chunks[axis_idx]` whenever chunk
    information is available. If chunking is unknown or invalid, the function falls
    back to copying the full extent of `img` along the merge axis in a single block.

    A key implementation detail is that each block is materialized as a NumPy array
    via `np.asarray(img[...])` before assignment. This avoids assignment issues that
    can occur when attempting direct Zarr to Zarr writes in certain interactive
    (Jupyter or REPL) contexts, at the cost of temporarily holding the current block
    in RAM.

    Parameters
    ----------
    z_out : zarr.core.array.Array
        Destination Zarr array. Must be 5D and writable. Chunking is used to define
        block boundaries along `axis_idx` when available.
    img : array-like
        Source image data to copy. Can be a NumPy array or a Zarr array. Must be 5D
        and compatible with `z_out` on all non-merge axes.
    out_start : int
        Start index along `axis_idx` in `z_out` where the first element of `img`
        will be written.
    axis_idx : int
        Integer index of the axis along which the copy is offset and blockwise
        partitioned.

    Returns
    -------
    None
        The function writes into `z_out` in place.

    Notes
    -----
    * The function assumes both `z_out` and `img` are 5D (consistent with OMIO’s
    canonical TZCYX convention) and does not validate dimensionality beyond what
    is implicitly required by indexing.
    * Block boundaries are chosen to align with the destination chunk size along
    `axis_idx`, which is typically beneficial for write performance and reduces
    the chance of repeatedly touching the same chunks during sequential merges.
    * Memory usage is bounded by the size of a single block (full extents of the
    non-merge axes and `block` along the merge axis).
    """
    n = int(img.shape[axis_idx])

    # chunk length along merge axis in output
    chunk_len = int(z_out.chunks[axis_idx]) if getattr(z_out, "chunks", None) is not None else None
    if chunk_len is None or chunk_len <= 0:
        chunk_len = n  # fallback: one block

    src_pos = 0
    while src_pos < n:
        block = min(chunk_len, n - src_pos)

        out_slice = [slice(None)] * 5
        src_slice = [slice(None)] * 5

        out_slice[axis_idx] = slice(out_start + src_pos, out_start + src_pos + block)
        src_slice[axis_idx] = slice(src_pos, src_pos + block)

        # materialize only the block, not the whole img
        """ Note on memory efficiency (Dec 2025):
            When executed in Jupyter notebooks or Interactive Python environments,
            we would get an asynchronous assignment error if we would use 
            
                        z_out[tuple(slicer)] = img directly (img is Zarr!)
            
            Therefore, we convert to NumPy first, which puts the image slice-wise (!) 
            into RAM temporarily. This is the pill we have to swallow for now, i.e., 
            no further memory-efficient optimization is possible with current Zarr version 
            (as of 2025-12). """
        z_out[tuple(out_slice)] = np.asarray(img[tuple(src_slice)])

        src_pos += block
# function to copy into zarr with zero padding:
def _copy_into_zarr_with_padding(z_out, img, out_start: int, axis_idx: int,
                                 target_nonmerge_shape: tuple):
    """
    Copy a 5D source image `img` into a 5D output Zarr array `z_out` at a specified
    offset along a merge axis, while implicitly applying zero padding on all
    non-merge axes.

    The output array `z_out` is assumed to be pre-initialized with zeros and sized
    to the merge target shape. During copying, only the region that exists in the
    source is written: for every non-merge axis `j`, the function writes the slice
    `0:src_shape[j]` into `z_out`. Any remaining extent up to the non-merge target
    shape stays zero, thereby realizing padding without explicitly writing zeros.

    Copying is performed in contiguous blocks aligned to the destination chunk grid
    along the merge axis. If chunk information is unavailable or invalid, the
    function falls back to copying the full extent of `img` along the merge axis in
    a single block.

    Each written block is materialized as a NumPy array via `np.asarray(...)` before
    assignment. This avoids issues that can arise with direct Zarr to Zarr writes in
    interactive environments (for example Jupyter), at the cost of temporarily
    holding the current block in RAM.

    Parameters
    ----------
    z_out : zarr.core.array.Array
        Destination Zarr array. Must be writable and 5D. It should already be
        initialized with zeros so that unwritten regions represent padded zeros.
    img : array-like
        Source image data to copy. Can be a NumPy array or a Zarr array. Must be 5D.
    out_start : int
        Start index along the merge axis in `z_out` where the first element of `img`
        will be written.
    axis_idx : int
        Integer index of the merge axis (the axis along which stacking/concatenation
        occurs).
    target_nonmerge_shape : tuple
        A 5D shape defining the intended maximal extents on the non-merge axes for
        the merge operation. The merge axis length in this tuple is not used by this
        function; it is included for interface consistency with merge planning code.

    Returns
    -------
    None
        The function writes into `z_out` in place.

    Notes
    -----
    * The function assumes both `z_out` and `img` follow the 5D convention used in
    the merge pipeline (typically TZCYX) and does not perform full compatibility
    checks beyond what indexing requires.
    * Padding is implicit: only `0:src_shape[j]` is written for non-merge axes, and
    the remainder stays zero due to `z_out` initialization.
    * Memory usage is bounded by the size of one block: full extents of the source
    on non-merge axes and `block` elements along the merge axis.
    """
    src_shape = tuple(img.shape)
    n = int(src_shape[axis_idx])

    chunk_len = int(z_out.chunks[axis_idx]) if getattr(z_out, "chunks", None) is not None else None
    if chunk_len is None or chunk_len <= 0:
        chunk_len = n

    src_pos = 0
    while src_pos < n:
        block = min(chunk_len, n - src_pos)

        out_slice = [slice(None)] * 5
        src_slice = [slice(None)] * 5

        # merge axis placement:
        out_slice[axis_idx] = slice(out_start + src_pos, out_start + src_pos + block)
        src_slice[axis_idx] = slice(src_pos, src_pos + block)

        # non merge axes: only write the valid src region [0:src_shape[j]]:
        for j in range(5):
            if j == axis_idx:
                continue
            out_slice[j] = slice(0, src_shape[j])
            src_slice[j] = slice(0, src_shape[j])

        """ Note on memory efficiency (Dec 2025):
            When executed in Jupyter notebooks or Interactive Python environments,
            we would get an asynchronous assignment error if we would use 
            
                        z_out[tuple(slicer)] = img directly (img is Zarr!)
            
            Therefore, we convert to NumPy first, which puts the image slice-wise (!) 
            into RAM temporarily. This is the pill we have to swallow for now, i.e., 
            no further memory-efficient optimization is possible with current Zarr version 
            (as of 2025-12). """
        z_out[tuple(out_slice)] = np.asarray(img[tuple(src_slice)])
        src_pos += block
# function to merge images by concatenation along an axis:
def _merge_concat_along_axis(images, metadatas, merge_along_axis: str,
                             zarr_store: str,
                             namespace: str = "omio:merge",
                             zeropadding: bool = False,
                             verbose: bool = True):
    """
    Concatenate multiple 5D image stacks along a specified OME axis and return a
    merged image plus merged metadata, with optional zero padding and optional Zarr
    output.

    This routine implements OMIO's merge policy for images that are already in the
    canonical 5D OME order (typically TZCYX) and whose metadata explicitly declares
    `axes == "TZCYX"`. No axis repair or reshaping is attempted. The merge occurs by
    concatenation along `merge_along_axis`, where each input may contribute an
    arbitrary length greater than one on that axis.

    Two validation and shape policies are supported:

    Strict mode (zeropadding=False)
        All non-merge axes must match exactly across inputs. The output shape equals
        the common non-merge shape, and the merge axis length equals the sum of all
        input lengths along that axis.

    Zero padding mode (zeropadding=True)
        Non-merge axes may differ across inputs. The output non-merge extents are set
        to the per-axis maxima across all inputs. Each input is embedded into a
        zero-initialized target block by writing only its existing source region
        `0:src_shape[j]` on every non-merge axis. The merge axis is then concatenated
        as in strict mode.

    The merged metadata are created by combining `metadatas` according to
    `_merge_metadata_sources(...)` and then updated to reflect the merged image shape.
    Per-source provenance is recorded in `Annotations` under the provided `namespace`.

    Output representation is controlled by `zarr_store`:

    zarr_store is None
        The merge is performed in NumPy, returning a NumPy ndarray. In strict mode,
        inputs are concatenated directly. In zero padding mode, padded NumPy blocks
        are allocated per input before concatenation.

    zarr_store is "memory" or "disk"
        The merge target is created as a Zarr array (in-memory store or
        `{folder}/.omio_cache/<basename>.zarr`). Copying is performed incrementally
        into the destination to avoid loading all data at once. In strict mode, blocks
        are written in chunk-aligned slabs along the merge axis. In zero padding mode,
        the destination is zero-initialized and only the valid source region is written
        for each input, which implicitly leaves padded regions as zeros.

    Due to current Zarr behavior in interactive environments, Zarr-backed sources are
    materialized block-wise via `np.asarray(...)` during assignment into the output
    Zarr, trading small temporary RAM usage for robustness.

    Parameters
    ----------
    images : sequence of array-like
        Input image stacks. Each entry must be 5D and compatible with the declared
        OME axes order. Entries may be NumPy arrays or Zarr arrays.
    metadatas : sequence of dict
        Metadata dictionaries corresponding one-to-one with `images`. Each dict must
        declare `axes == "TZCYX"` (or the configured `_OME_AXES`) and should contain
        provenance fields used by the merge metadata policy.
    merge_along_axis : str
        Axis label along which to concatenate (must be in `_ALLOWED_MERGE_AXES` and
        present in `_OME_AXES`).
    zarr_store : {None, "memory", "disk"}
        Controls whether output is a NumPy array (None) or a Zarr array ("memory" or
        "disk").
    namespace : str, optional
        Namespace prefix used when writing merge provenance into `Annotations`.
        Default is "omio:merge".
    zeropadding : bool, optional
        If False, require exact non-merge axis matches. If True, allow mismatched
        non-merge axes and pad each input to the maxima before concatenation.
        Default is False.
    verbose : bool, optional
        If True, print diagnostic messages about shapes and progress.

    Returns
    -------
    merged : np.ndarray or zarr.core.array.Array or None
        The merged image. Returns None if validation fails or if Zarr output was
        requested but Zarr is unavailable.
    md_merged : dict or None
        The merged metadata dictionary aligned with `merged`. Returns None if the
        merge fails.

    Notes
    -----
    * Inputs must already be 5D and OME-ordered; this function does not reorder axes.
    * In Zarr mode, the output is written into an OMIO cache location when
    `zarr_store="disk"`. Existing stores at that path are replaced.
    * Zero padding is implemented by writing only existing source extents into a
    zero-initialized destination, leaving the remaining regions as zeros.
    """
    ok = _validate_merge_inputs_with_optional_padding(
        images, metadatas,
        merge_along_axis=merge_along_axis,
        zeropadding=zeropadding,
        context=f"merge_along_{merge_along_axis}")
    if not ok:
        return None, None

    axis_idx = _AXIS_TO_INDEX[merge_along_axis]

    if zeropadding:
        max_shape_nonmerge, merged_shape, _ = _compute_merge_target_shapes(
            images, merge_along_axis, context=f"merge_along_{merge_along_axis}")
        if verbose:
            print(f"Merging with zero padding along axis '{merge_along_axis}':")
            print(f"    max non-merge shape = {max_shape_nonmerge}")
            print(f"    merged shape        = {merged_shape}")
        if merged_shape is None:
            if verbose:
                print("Merge aborted due to shape computation failure.")
            return None, None
    else:
        shape0 = tuple(images[0].shape)
        merged_shape = list(shape0)
        merged_shape[axis_idx] = int(sum(int(img.shape[axis_idx]) for img in images))
        merged_shape = tuple(merged_shape)
        max_shape_nonmerge = shape0

    md_merged = _merge_metadata_sources(metadatas, namespace=namespace)
    md_merged["axes"] = _OME_AXES
    md_merged["shape"] = merged_shape
    md_merged["SizeT"] = int(merged_shape[_AXIS_TO_INDEX["T"]])
    md_merged["SizeZ"] = int(merged_shape[_AXIS_TO_INDEX["Z"]])
    md_merged["SizeC"] = int(merged_shape[_AXIS_TO_INDEX["C"]])
    md_merged["SizeY"] = int(merged_shape[_AXIS_TO_INDEX["Y"]])
    md_merged["SizeX"] = int(merged_shape[_AXIS_TO_INDEX["X"]])

    if zarr_store is None:
        # NumPy path:
        if not zeropadding:
            merged = np.concatenate([np.asarray(img) for img in images], axis=axis_idx)
            return merged, md_merged

        # zeropadding=True: build padded blocks then concatenate:
        padded = []
        for image_i, img in enumerate(images):
            src = np.asarray(img)

            # build per input target shape:
            out_shape = list(max_shape_nonmerge)
            out_shape[axis_idx] = src.shape[axis_idx]   # keep merge axis length per input

            if verbose:
                print(f"    Padding image {image_i} of shape {src.shape} to target shape {tuple(out_shape)}...")

            out = np.zeros(tuple(out_shape), dtype=src.dtype)

            sl = [slice(None)] * 5
            for j in range(5):
                sl[j] = slice(0, src.shape[j])

            out[tuple(sl)] = src
            padded.append(out)

        merged = np.concatenate(padded, axis=axis_idx)
        return merged, md_merged
        """ padded = []
        for image_i, img in enumerate(images):
            if verbose:
                print(f"    Padding image {image_i} of shape {tuple(img.shape)} to target non-merge shape {tuple(max_shape_nonmerge)}...")
            src = np.asarray(img)
            out = np.zeros(tuple(max_shape_nonmerge), dtype=src.dtype)
            sl = [slice(None)] * 5
            for j in range(5):
                sl[j] = slice(0, src.shape[j])
            
            # sanity check: src shape must fit into target non-merge shape
            out[tuple(sl)] = src
            padded.append(out)
        merged = np.concatenate(padded, axis=axis_idx)
        return merged, md_merged """

    # Zarr output requested:
    if zarr is None:
        warnings.warn("Merge: zarr_store was requested but zarr is not available. Merge aborted.")
        return None, None

    chunks = compute_default_chunks(merged_shape, _OME_AXES)
    folder0 = metadatas[0].get("original_parentfolder", ".")
    base0 = os.path.splitext(metadatas[0].get("original_filename", "merge"))[0]
    out_basename = f"{base0}_merged_{merge_along_axis}"

    z_out = _zarr_open_for_merge_output(
        zarr_store=zarr_store,
        folder=folder0,
        basename=out_basename,
        shape=merged_shape,
        dtype=images[0].dtype,
        chunks=chunks)
    
    """ start = 0
    for img in images:
        n = int(img.shape[axis_idx])
        slicer = [slice(None)] * 5
        slicer[axis_idx] = slice(start, start + n)
        # when executed in Jupyter notebooks or Interactive Python environments,
        # we get an asynchronous assignment error here with Zarr arrays if we 
        # try z_out[tuple(slicer)] = img directly. Therefore, we convert to NumPy first.
        # (can't be solved otherwise withe current Zarr version as of 2025-12)
        z_out[tuple(slicer)] = np.asarray(img)
        start += n """

    # z_out is zero initialized already, so "padding" is just writing the existing source region
    start = 0
    for img in images:
        if zeropadding:
            _copy_into_zarr_with_padding(z_out, img, out_start=start,
                                         axis_idx=axis_idx,
                                         target_nonmerge_shape=max_shape_nonmerge)
        else:
            _copy_into_zarr_chunk_aligned(z_out, img, out_start=start, axis_idx=axis_idx)
        start += int(img.shape[axis_idx])

    return z_out, md_merged
# function to merge folder-stacks with padding:
def _merge_folderstacks_with_padding(images, metadatas,
                                     merge_along_axis: str,
                                     zarr_store: str = None,
                                     zeropadding: bool = True,
                                     verbose: bool = True
                                     ) -> Tuple[Union[None, np.ndarray, "zarr.core.array.Array"], Union[None, dict]]:
    """
    Merge multiple 5D folder stacks by concatenating along a chosen OME axis, with an
    optional zero padding policy for mismatched non-merge dimensions and optional
    materialization into Zarr.

    This helper is intended for the common case where a folder contains multiple
    stacks that should be combined into a single canonical 5D array in OME axis
    order (TZCYX). The function enforces that all metadata declare `axes == "TZCYX"`
    and that all inputs are 5D. No axis repair, reordering, or dimensional inference
    is performed.

    Merge policy
    ------------
    * The output is constructed by concatenation along `merge_along_axis`.
    * Non-merge axes can be handled in two ways:

    zeropadding=False (strict)
        All non-merge axis lengths must match exactly across all inputs. If any
        mismatch is detected, the merge is aborted.

    zeropadding=True (padding)
        For each non-merge axis, the maximum size across all inputs is computed.
        Each input stack is then embedded into a zero-initialized target array of
        that padded shape by writing only the valid source region. Concatenation
        is performed on these padded arrays, so missing regions remain zero.

    Output materialization
    ----------------------
    * If `zarr_store is None`, the merged result is returned as a NumPy ndarray.
    * If `zarr_store` is not None, the merged NumPy result is written into a Zarr
    array created by `_zarr_open_for_merge_output(...)` and the returned image is
    that Zarr array.

    Practical note
    --------------
    This merge is primarily meaningful for `merge_along_axis="T"` in workflows where
    multiple time blocks belong to a single logical acquisition. Merging along "Z"
    or "C" is allowed but assumes that the remaining axes correspond to compatible
    acquisitions and that interpreting the concatenation as an extended Z stack or
    channel axis is semantically correct.

    Parameters
    ----------
    images : sequence of array-like
        Input image stacks. Each entry must be 5D (TZCYX). Entries may be NumPy
        arrays or Zarr arrays, but padding requires materialization via
        `np.asarray(...)`.
    metadatas : sequence of dict
        Metadata dictionaries corresponding one-to-one with `images`. Each must
        declare `axes == "TZCYX"` (or `_OME_AXES`).
    merge_along_axis : str
        Axis label along which to concatenate. Must be in `_ALLOWED_MERGE_AXES`.
    zarr_store : {None, "memory", "disk"}, optional
        If None, return a NumPy array. Otherwise, write the merged result to a Zarr
        store and return a Zarr array handle.
    zeropadding : bool, optional
        If True, pad mismatched non-merge axes to per-axis maxima using zeros before
        concatenation. If False, require exact non-merge axis matches.
    verbose : bool, optional
        If True, print diagnostic progress and merge mode information.

    Returns
    -------
    merged : np.ndarray or zarr.core.array.Array or None
        The merged image. Returns None if validation fails or if Zarr output was
        requested but Zarr is unavailable.
    md_merged : dict or None
        Metadata dictionary aligned with the returned merged image, including updated
        shape and SizeT/SizeZ/SizeC/SizeY/SizeX fields and merge provenance stored
        under the merge namespace.
    """
    if merge_along_axis not in _ALLOWED_MERGE_AXES:
        warnings.warn(
            f"merge_folder_stacks: invalid merge_along_axis={merge_along_axis!r}. "
            f"Allowed: {sorted(_ALLOWED_MERGE_AXES)}."
        )
        return None, None

    if not images:
        warnings.warn("merge_folder_stacks: no images to merge.")
        return None, None

    # path without zero-padding:
    if not zeropadding:
        # strict check: require identical sizes on all non merged axes
        if verbose:
            print(f"merge_folder_stacks: merging without zero-padding along axis '{merge_along_axis}'.")
        axis_idx = _AXIS_TO_INDEX[merge_along_axis]
        sh0 = tuple(images[0].shape)
        for i, img in enumerate(images):
            shi = tuple(img.shape)
            for j in range(5):
                if j == axis_idx:
                    continue
                if shi[j] != sh0[j]:
                    print( "WARNING: merge_folder_stacks: shape mismatch on non merged axis. \n"
                          f"         stack0={sh0}, stack{i}={shi}.\n"
                           "         Set zeropadding=True to allow padding merge. Merge aborted.")
                    return None, None

    # otherwise: path with zero-padding:
    if verbose:
        print(f"merge_folder_stacks: merging with zero-padding along axis '{merge_along_axis}'.")
    # require correct axes and 5D:
    for i, md in enumerate(metadatas):
        if md.get("axes", None) != _OME_AXES:
            warnings.warn(
                f"merge_folder_stacks: expected axes '{_OME_AXES}' but got {md.get('axes', None)!r} at index {i}.\n"
                "    Merge aborted.")
            return None, None
        if len(tuple(images[i].shape)) != 5:
            warnings.warn(
                f"merge_folder_stacks: expected 5D arrays (TZCYX) but got shape {tuple(images[i].shape)} at index {i}.\n"
                "    Merge aborted.")
            return None, None

    axis_idx = _AXIS_TO_INDEX[merge_along_axis]
    non_merge_idxs = [j for j in range(5) if j != axis_idx]

    # determine max sizes for non merged axes:
    max_sizes = list(images[0].shape)
    for j in non_merge_idxs:
        max_sizes[j] = max(int(img.shape[j]) for img in images)

    # Build padded arrays
    padded_arrays = []
    for img in images:
        src = np.asarray(img)  # padding requires NumPy materialization
        target_shape = list(src.shape)
        for j in non_merge_idxs:
            target_shape[j] = max_sizes[j]
        target_shape = tuple(target_shape)

        out = np.zeros(target_shape, dtype=src.dtype)

        slicer = [slice(None)] * 5
        for j in range(5):
            slicer[j] = slice(0, src.shape[j])
        out[tuple(slicer)] = src
        padded_arrays.append(out)

    # Now concat along merge axis
    merged_np = np.concatenate(padded_arrays, axis=axis_idx)

    md_merged = _merge_metadata_sources(metadatas, namespace="omio:merge_folderstacks")
    md_merged["axes"] = _OME_AXES
    md_merged["shape"] = merged_np.shape
    md_merged["SizeT"] = int(merged_np.shape[_AXIS_TO_INDEX["T"]])
    md_merged["SizeZ"] = int(merged_np.shape[_AXIS_TO_INDEX["Z"]])
    md_merged["SizeC"] = int(merged_np.shape[_AXIS_TO_INDEX["C"]])
    md_merged["SizeY"] = int(merged_np.shape[_AXIS_TO_INDEX["Y"]])
    md_merged["SizeX"] = int(merged_np.shape[_AXIS_TO_INDEX["X"]])

    if zarr_store is None:
        return merged_np, md_merged

    if zarr is None:
        warnings.warn("merge_folder_stacks: zarr_store was requested but zarr is not available. Merge aborted.")
        return None, None

    chunks = compute_default_chunks(merged_np.shape, _OME_AXES)
    folder0 = metadatas[0].get("original_parentfolder", ".")
    base0 = os.path.splitext(metadatas[0].get("original_filename", "merge"))[0]
    out_basename = f"{base0}_merged_folderstacks_{merge_along_axis}"

    z_out = _zarr_open_for_merge_output(
        zarr_store=zarr_store,
        folder=folder0,
        basename=out_basename,
        shape=merged_np.shape,
        dtype=merged_np.dtype,
        chunks=chunks,
    )
    z_out[:] = merged_np
    return z_out, md_merged
# function to dispatch to format-specific readers:
def _dispatch_read_file(path: str,
                        zarr_store: Union[None, str],
                        return_list: bool,
                        physicalsize_xyz: Union[None, Any],
                        pixelunit: str,
                        verbose: bool = True,
                        ) -> Tuple[Any, Dict[str, Any]]:
    """
    Dispatch a single microscopy file to the appropriate OMIO reader based on its
    filename extension and return the loaded image and metadata.

    This function selects one of OMIO's format specific readers and forwards common
    configuration parameters such as voxel size overrides, unit normalization, Zarr
    materialization mode, verbosity, and backward compatible list returns.

    Supported formats and dispatch rules
    ------------------------------------
    * TIFF family: OME TIFF (.ome.tif, .ome.tiff) and standard TIFF variants
    (.tif, .tiff, .lsm) are read via `read_tif(...)`.
    * Zeiss CZI: .czi is read via `read_czi(...)`.
    * Thorlabs RAW: .raw is read via `read_thorlabs_raw(...)`.

    Parameters
    ----------
    path : str
        Path to the input file to read.
    zarr_store : {None, "memory", "disk"}
        If None, the reader returns a NumPy array in RAM. If "memory" or "disk", the
        reader materializes the result as a Zarr array backed by an in memory store
        or an on disk cache store, respectively. The concrete behavior is determined
        by the called reader.
    return_list : bool
        Forwarded to the reader for backward compatibility. If True, readers may
        return `[image]` and `[metadata]` for non paginated inputs. Some readers may
        return lists regardless of this flag for semantically ambiguous cases
        (e.g. paginated TIFFs).
    physicalsize_xyz : Any or None
        Optional override for physical pixel sizes, forwarded to the reader. If
        provided, the reader uses these values instead of metadata derived sizes
        according to its own precedence policy.
    pixelunit : str
        Unit string forwarded to the reader for unit normalization and defaults.
    verbose : bool, optional
        If True, forward diagnostic progress output from the reader.

    Returns
    -------
    image : Any
        The loaded image, typically a NumPy ndarray or Zarr array, or a list of such
        objects if the reader returns multiple stacks.
    metadata : dict
        Metadata dictionary aligned with the returned image, or a list of dicts if
        the reader returns multiple stacks.

    Raises
    ------
    ValueError
        If the file extension is not supported by the dispatch rules.
    """
    
    lp = path.lower()

    if _looks_like_ome_tif(lp) or _lower_ext(lp) in {".tif", ".tiff", ".lsm"}:
        return read_tif(
            path,
            zarr_store=zarr_store,
            return_list=return_list,
            physicalsize_xyz=physicalsize_xyz,
            pixelunit=pixelunit,
            verbose=verbose)

    if _lower_ext(lp) == ".czi":
        return read_czi(
            path,
            zarr_store=zarr_store,
            return_list=return_list,
            physicalsize_xyz=physicalsize_xyz,
            pixelunit=pixelunit,
            verbose=verbose)

    if _lower_ext(lp) == ".raw":
        return read_thorlabs_raw(
            path,
            zarr_store=zarr_store,
            return_list=return_list,
            physicalsize_xyz=physicalsize_xyz,
            pixelunit=pixelunit,
            verbose=verbose)

    raise ValueError(f"Unsupported file extension '{_lower_ext(lp)}' for path: {path}")
# functions to detect and collapse OME multifile series:
_UUID_FILENAME_RE = re.compile(r'FileName="([^"]+)"')
def _ome_referenced_basenames(tif_path: str) -> list[str]:
    """
    Return list of basenames referenced via FileName="..." in OME-XML.
    Does not trigger multifile loading.
    """
    try:
        with tifffile.TiffFile(tif_path, _multifile=False) as tif:
            ome = tif.ome_metadata
    except Exception:
        return []
    if not ome:
        return []
    refs = _UUID_FILENAME_RE.findall(ome)
    return [os.path.basename(r) for r in refs]
class _UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra
def _collapse_ome_multifile_series(files: list[str], verbose: bool = True) -> list[str]:
    """
    Keep only one representative per OME multifile series.
    Groups files by OME-XML connectivity (connected components).
    Works if only some member files contain OME-XML and if refs are partial.
    """
    if not files:
        return []

    # Map basename -> all full paths seen (basename collisions are possible, keep list)
    base_to_paths: dict[str, list[str]] = {}
    for f in files:
        base_to_paths.setdefault(os.path.basename(f), []).append(f)

    uf = _UnionFind()

    # Build connectivity graph: file_basename <-> referenced_basename
    for f in files:
        b = os.path.basename(f)
        refs = _ome_referenced_basenames(f)
        if not refs:
            continue
        for r in refs:
            # Only union if the referenced file exists among discovered files
            if r in base_to_paths:
                uf.union(b, r)

    # Collect components
    comp: dict[str, set[str]] = {}
    for b in base_to_paths.keys():
        root = uf.find(b)
        comp.setdefault(root, set()).add(b)

    representatives: list[str] = []
    skipped = 0

    for root, members in comp.items():
        if len(members) == 1:
            # singletons: keep all their concrete paths (could be basename collisions)
            b = next(iter(members))
            representatives.extend(base_to_paths[b])
            continue

        # Multifile component: choose deterministic representative path
        # Pick lexicographically smallest basename, then lexicographically smallest full path for that basename
        members_sorted = sorted(members)
        rep_base = members_sorted[0]
        rep_path = sorted(base_to_paths[rep_base])[0]
        representatives.append(rep_path)

        # Skip all other members
        for b in members_sorted[1:]:
            skipped += len(base_to_paths[b])

        if verbose:
            print(
                f"Detected OME multifile series with {sum(len(base_to_paths[b]) for b in members_sorted)} files "
                f"({len(members_sorted)} unique basenames). Using representative: {os.path.basename(rep_path)}"
            )

    if verbose and skipped:
        print(f"Skipped {skipped} files that belong to already detected OME multifile series.")

    # Preserve original order as much as possible: sort representatives by their first occurrence in `files`
    pos = {p: i for i, p in enumerate(files)}
    representatives.sort(key=lambda p: pos.get(p, 10**12))

    return representatives

# OMIO's main universal image reader:
def imread(fname: Union[str, os.PathLike, List[Union[str, os.PathLike]]],
         zarr_store: Union[None, str] = None,
         return_list: bool = False,
         recursive: bool = False,
         folder_stacks: bool = False,
         merge_folder_stacks: bool = False,
         merge_multiple_files_in_folder: bool = False,
         merge_along_axis: str = "T",
         zeropadding: bool = True,
         physicalsize_xyz: Union[None, Any] = None,
         pixelunit: str = "micron",
         collapse_ome_multifile_series: bool = True,
         verbose: bool = True,
         ) -> Union[
             Tuple[Any, Dict[str, Any]],
             Tuple[List[Any], List[Dict[str, Any]]]]:
    """
    Read microscopy images and folders into OMIO's canonical representation, with optional
    folder stack handling and concatenation based merges.

    This is OMIO's high level entry point. It accepts a single file, a list of files, or a
    folder path. Supported input formats are TIFF family files (including OME TIFF and LSM),
    Zeiss CZI, and Thorlabs RAW. For each file, the corresponding format specific reader is
    selected automatically, metadata are standardized, and the returned image is normalized
    to OME axis order TZCYX.

    If `zarr_store` is set to "memory" or "disk", readers return a Zarr array instead of a
    NumPy array. For "disk", Zarr outputs are created in a hidden cache folder `.omio_cache`
    next to the source data. This is intended for large files where memory mapping and
    chunked access are required downstream.

    Folder input behavior
    ---------------------
    If `fname` resolves to a folder, OMIO lists all supported image files inside the folder
    (optionally recursive) and reads them in sorted order.

    If `folder_stacks=True` or `merge_folder_stacks=True`, the folder is interpreted as one
    member of a tagged folder stack family with names like `<TAG>_000`, `<TAG>_001`, etc.
    OMIO derives `<TAG>_` from the provided folder name, finds all co folders with the same
    tag in the parent directory, reads the first image file in each of these folders, and
    returns either the list of stacks or a merged stack.

    Merge behavior
    --------------
    Two merge modes are supported.

    * `merge_multiple_files_in_folder=True` merges all images found in a folder by
      concatenating along `merge_along_axis`. This is applied after reading all files from
      that folder.
    * `merge_folder_stacks=True` merges the tagged co folder stacks by concatenating along
      `merge_along_axis`.

    `merge_along_axis` must be one of {"T", "Z", "C"}. In merge modes, OMIO expects that all
    inputs are already in OME order and have 5 dimensions (TZCYX). If `zeropadding=False`,
    non merge axes must match exactly, otherwise the merge is aborted with a warning and a
    None result. If `zeropadding=True`, non merge axes are padded with zeros up to the
    maximum size across inputs before concatenation. The merge axis may have length greater
    than one in each input; OMIO concatenates the full segments in the discovered order.

    For merge outputs, metadata are merged with a provenance policy that records the inputs
    under the `Annotations` namespace and uses stack 0 as the reference for physical size and
    time increment fields.

    Parameters
    ----------
    fname : str, os.PathLike, or list of such
        File path, folder path, or list of file paths to read.
    zarr_store : {None, "memory", "disk"}, optional
        Controls whether images are returned as NumPy arrays (None) or as materialized Zarr
        arrays ("memory" or "disk"). Default is None.
    return_list : bool, optional
        If True, always return lists of images and metadata. If False, return a single image
        and metadata for single input cases, otherwise lists. Default is False.
    recursive : bool, optional
        If True and `fname` is a folder, search recursively for supported image files.
        Default is False.
    folder_stacks : bool, optional
        If True and `fname` is a folder, interpret it as a tagged folder stack member and
        read the first image file from each tagged co folder. Default is False.
    merge_folder_stacks : bool, optional
        If True, interpret tagged folder stacks and merge them along `merge_along_axis`.
        Default is False.
    merge_multiple_files_in_folder : bool, optional
        If True and `fname` is a folder, merge all files found in that folder along
        `merge_along_axis`. Default is False.
    merge_along_axis : {"T", "Z", "C"}, optional
        Axis along which concatenation is performed in merge modes. Default is "T".
    zeropadding : bool, optional
        If True, allow merges with mismatched non merge axes by zero padding to maxima. If
        False, require exact match on non merge axes. Default is True.
    physicalsize_xyz : Any or None, optional
        Optional voxel size override forwarded to the underlying readers. Default is None.
    pixelunit : str, optional
        Unit string forwarded to readers for unit normalization and defaults. Default is
        "micron".
    collapse_ome_multifile_series : bool, optional
        If True, detect OME multifile series and keep only one representative file per
        series to avoid duplicate loading. Default is True.
    verbose : bool, optional
        If True, print diagnostic progress messages. Default is True.

    Returns
    -------
    image, metadata : (Any, dict) or (list[Any], list[dict])
        For single non folder inputs and `return_list=False`, returns one image and one
        metadata dict. For multi file inputs, folder reads, or `return_list=True`, returns
        lists. Merge modes return a single merged image and merged metadata (or lists if
        `return_list=True`). If a requested merge fails validation, returns None results
        according to the calling branch.

    Raises
    ------
    ValueError
        If `merge_along_axis` is not one of {"T", "Z", "C"}.
    FileNotFoundError
        If a requested file path does not exist or is not a file.
    """
    if merge_along_axis not in _ALLOWED_MERGE_AXES:
        raise ValueError(f"read: merge_along_axis must be one of {sorted(_ALLOWED_MERGE_AXES)}. "
                         f"Got: {merge_along_axis!r}")

    allowed_ext = {".tif", ".tiff", ".lsm", ".czi", ".raw", ".ome.tif", ".ome.tiff"}
    # TODO: maybe we shift this variable to a module-level global later

    paths = _normalize_to_list(fname)

    # folder input cases:
    # sanity check:
    if merge_folder_stacks:
        if verbose:
            print(f"merge_folder_stacks={merge_folder_stacks} ⟶ will read and merge from tagged folder stacks.")
    if folder_stacks and not merge_folder_stacks:
        if verbose:
            print(f"folder_stacks={folder_stacks}, merge_folder_stacks={merge_folder_stacks} ⟶ will read from tagged folder stacks.")
    if len(paths) == 1 and _is_dir(paths[0]):
        folder = paths[0]

        if folder_stacks or merge_folder_stacks:
            # we expect folder to be one of the TAG_000 style folderstacks, thus, let's search for
            # the other TAG_XXX co-folders:
            folder_base = os.path.basename(folder)
            folder_path_to_base = os.path.dirname(folder)
            # first verify, that folder_base contains at least one underscore:
            if "_" not in folder_base:
                if verbose:
                    print(f"    Could not detect <TAG>_ from folder name: {folder_base!r}.")
                    print("    Abort merging.")
                return ([], []) if return_list else (None, {})
            # extract tag:
            tag = folder_base.split("_", 1)[0] + "_"
            if tag is None:
                if verbose:
                    print(f"    Could not detect <TAG>_ from folder name: {folder_base!r}.")
                    print("    Abort merging.")
                return ([], []) if return_list else (None, {})
            else:
                if verbose:
                    print(f"Detected folder stack tag: {tag!r}.")
            tagfolders = []
            for d in os.listdir(folder_path_to_base):
                d_full = os.path.join(folder_path_to_base, d)
                if not os.path.isdir(d_full):
                    continue
                if d.startswith(tag):
                    tagfolders.append(d)
            if not tagfolders:
                if verbose:
                    print(f"    folder_stacks={folder_stacks} or merge_folder_stacks={merge_folder_stacks} requested, but no co-folders with tag '{tag}' found.")
                    print("    Abort merging.")
                return ([], []) if return_list else (None, {})
            else:
                # sort:
                tagfolders = sorted(tagfolders)

            # prepend folder-path_to_base to tagfolders' entries:
            tagfolders_fullpaths = [os.path.join(folder_path_to_base, tf) for tf in tagfolders]

            images = []
            metadatas = []
            for sf in tagfolders_fullpaths:
                f0 = _first_image_file_in_folder(sf, allowed_ext=allowed_ext)
                if f0 is None:
                    if verbose:
                        print(f"    No valid image file found in folder stack: {sf!r}. Skipping.")
                    continue
                img, md = _dispatch_read_file(
                    f0,
                    zarr_store=zarr_store,
                    return_list=False,
                    physicalsize_xyz=physicalsize_xyz,
                    pixelunit=pixelunit,
                    verbose=verbose)
                
                # post-hoc OME metadata checkup and correction:
                md = OME_metadata_checkup(md, verbose=verbose)
                
                # update merged image stack and metadata lists:
                images.append(img)
                metadatas.append(md)

            if merge_folder_stacks:
                if not images:
                    if verbose:
                        print("    No valid images found in any of the folder stacks. Abort merging.")
                    return ([], []) if return_list else (None, {})

                merged_img, merged_md = _merge_folderstacks_with_padding(images, metadatas,
                                                        merge_along_axis=merge_along_axis,
                                                        zarr_store=zarr_store,
                                                        zeropadding=zeropadding,
                                                        verbose=verbose)
                # post-hoc OME metadata checkup and correction:
                if merged_md is not None:
                    merged_md = OME_metadata_checkup(merged_md, verbose=verbose)
                
                # return result:
                if return_list:
                    return [merged_img], [merged_md]
                return merged_img, merged_md

            # return results:
            if return_list:
                return images, metadatas
            if len(images) == 1:
                return images[0], metadatas[0]
            return images, metadatas

        # default folder behavior: read all image files in folder:
        files = _list_image_files_in_folder(folder, allowed_ext=allowed_ext, recursive=recursive)
        if collapse_ome_multifile_series:
            files = _collapse_ome_multifile_series(files, verbose=verbose)
        if not files:
            return ([], []) if return_list else (None, {})

        images = []
        metadatas = []
        for f in files:
            img, md = _dispatch_read_file(
                f,
                zarr_store=zarr_store,
                physicalsize_xyz=physicalsize_xyz,
                pixelunit=pixelunit,
                return_list=False,
                verbose=verbose)
            images.append(img)
            metadatas.append(md)

        if merge_multiple_files_in_folder:
            merged_img, merged_md = _merge_concat_along_axis(
                images, metadatas,
                merge_along_axis=merge_along_axis,
                zarr_store=zarr_store,
                namespace="omio:merge_multiple_files_in_folder",
                zeropadding=zeropadding,
                verbose=verbose)
            if merged_img is None:
                if return_list:
                    return [None], [None]
                return None, None

            if return_list:
                return [merged_img], [merged_md]
            return merged_img, merged_md

        if return_list:
            return images, metadatas
        if len(images) == 1:
            return images[0], metadatas[0]
        return images, metadatas

    # file input or list of files:
    images = []
    metadatas = []
    for p in paths:
        if not _is_file(p):
            raise FileNotFoundError(f"Path does not exist or is not a file: {p}")
        img, md = _dispatch_read_file(
            p,
            zarr_store=zarr_store,
            return_list=False,
            physicalsize_xyz=physicalsize_xyz,
            pixelunit=pixelunit,
            verbose=verbose)
        images.append(img)
        metadatas.append(md)

    if return_list:
        return images, metadatas

    if len(images) == 1:
        return images[0], metadatas[0]

    return images, metadatas

# OMIO'S universal converter (=imreader + imwrite):
def imconvert(fname: Union[str, os.PathLike, List[Union[str, os.PathLike]]],
         zarr_store: Union[None, str] = None,
         recursive: bool = False,
         folder_stacks: bool = False,
         merge_folder_stacks: bool = False,
         merge_multiple_files_in_folder: bool = False,
         merge_along_axis: str = "T",
         collapse_ome_multifile_series: bool = True,
         zeropadding: bool = True,
         physicalsize_xyz: Union[None, Any] = None,
         pixelunit: str = "micron",
         compression_level: int = 3, 
         relative_path: Union[None, str] = "omio_converted", 
         overwrite: bool = False, 
         return_fnames: bool = False,
         cleanup_cache: bool = True,
         verbose: bool = True) -> Union[None, List[str]]:
    """
    Convert microscopy image inputs to OME TIFF using OMIO's reader plus OME TIFF writer.

    This function is a convenience wrapper around `imread(...)` followed by
    `imwrite(...)`. It accepts a single file path, a list of file paths, or a
    folder path, reads the input data into OMIO's canonical representation (OME ordered
    axes TZCYX plus standardized metadata), and writes one OME TIFF per resulting image
    stack.

    Input path semantics (inherited from `imread(...)`)
    ---------------------------------------------------
    Input handling and optional merges follow the same semantics as `imread(...)`:
    folder reading can be recursive, tagged folder stacks can be interpreted as a sequence
    of co folders, and merge operations can concatenate multiple stacks along a chosen OME
    axis ("T", "Z", or "C"), optionally with zero padding on non merge axes.
    
    The behavior depends on the type and structure of `fname`:

    Single file path
        The file is read according to its extension (TIFF, OME TIFF, LSM, CZI, or RAW),
        converted to OMIO's internal representation, and written as a single OME TIFF.

    List of file paths
        Each file is read independently. By default, one OME TIFF per input file is
        written. If merge options are enabled (for example
        ``merge_multiple_files_in_folder``), files may be concatenated before writing.

    Folder path
        By default, all supported image files in the folder are read, optionally
        recursively if ``recursive=True``, and written as individual OME TIFF files.

        Additional folder specific modes are available:

        * ``folder_stacks=True``:
          The folder is interpreted as one element of a tagged folder stack
          (for example ``TAG_000``, ``TAG_001``). The first valid image file from each
          tagged folder is read and written as a separate OME TIFF.
        * ``merge_folder_stacks=True``:
          Tagged folder stacks are read as above, but the resulting stacks are
          concatenated along ``merge_along_axis`` and written as a single merged
          OME TIFF.
        * ``merge_multiple_files_in_folder=True``:
          All image files found in the folder are concatenated along
          ``merge_along_axis`` and written as a single merged OME TIFF.

    Merge behavior
    --------------
    Merge operations follow the same validation and padding rules as in `imread(...)`:
    
    * Allowed merge axes are "T", "Z", and "C".
    * If `zeropadding=False`, all non merge axes must match exactly.
    * If `zeropadding=True`, non merge axes are padded with zeros to the maximum size
      across inputs before concatenation.

    Output behavior
    ---------------
    The output location and naming follow `imwrite(...)`:
    
    * OME TIFFs are written next to the input file or inside the input folder.
    * If `relative_path` is provided, a subfolder is created under the chosen output
      parent directory.
    * When merge modes are used, output filenames may include an indicator suffix to
      reflect merged content.
    * If `overwrite=False`, existing files are not replaced and collision safe names
      are generated.

    Zarr handling and cache cleanup
    -------------------------------
    If `zarr_store` is "memory" or "disk", `imread(...)` may create Zarr arrays or
    materialize intermediate Zarr stores under a hidden `.omio_cache` directory.
    If `cleanup_cache=True`, this function removes the corresponding cache entries
    after writing. Cache cleanup is skipped when `zarr_store=None`.

    Parameters
    ----------
    fname : str, os.PathLike, or list of such
        File path, folder path, or list of file paths to convert.
    zarr_store : {None, "memory", "disk"}, optional
        Controls whether `imread(...)` returns NumPy arrays (None) or Zarr arrays
        ("memory" or "disk"). Default is None.
    recursive : bool, optional
        If True and `fname` is a folder, search recursively for supported image files.
        Default is False.
    folder_stacks : bool, optional
        Interpret a tagged folder as part of a folder stack and read one image per
        tagged subfolder. Default is False.
    merge_folder_stacks : bool, optional
        Merge tagged folder stacks along `merge_along_axis` and write a single OME TIFF.
        Default is False.
    merge_multiple_files_in_folder : bool, optional
        Merge all image files found in a folder along `merge_along_axis` and write a
        single OME TIFF. Default is False.
    merge_along_axis : {"T", "Z", "C"}, optional
        Axis along which concatenation is performed in merge modes. Default is "T".
    collapse_ome_multifile_series : bool, optional
        If True, detect OME multifile series and keep only one representative file per
        series to avoid duplicate loading. Default is True.
    zeropadding : bool, optional
        Allow padding of non merge axes during merges. Default is True.
    physicalsize_xyz : Any or None, optional
        Optional voxel size override forwarded to the underlying readers. Default is None.
    pixelunit : str, optional
        Unit string forwarded to readers for unit normalization. Default is "micron".
    compression_level : int, optional
        Zlib compression level passed to `imwrite(...)`. Default is 3.
    relative_path : str or None, optional
        Optional relative subfolder under the output parent directory where OME TIFFs
        are written. Default is "omio_converted".
    overwrite : bool, optional
        Control overwriting behavior for existing outputs. Default is False.
    return_fnames : bool, optional
        If True, return the list of written OME TIFF filenames. Default is False.
    cleanup_cache : bool, optional
        Remove `.omio_cache` entries after writing when Zarr output was used.
        Default is True.
    verbose : bool, optional
        Print diagnostic progress messages. Default is True.

    Returns
    -------
    None or list[str]
        If `return_fnames=True`, returns a list of output OME TIFF paths.
        Otherwise returns None.

    Raises
    ------
    ValueError
        If invalid merge options are provided.
    FileNotFoundError
        If an input file does not exist.
    Other exceptions
        Reader and writer errors may propagate during I O or metadata handling.
    """


    if verbose:
        print(f"Converting to OME-TIFF: {fname!r}")
    #print(f"Reading input...")
    images, metadatas = imread(
        fname=fname,
        zarr_store=zarr_store,
        recursive=recursive,
        folder_stacks=folder_stacks,
        merge_folder_stacks=merge_folder_stacks,
        merge_multiple_files_in_folder=merge_multiple_files_in_folder,
        merge_along_axis=merge_along_axis,
        collapse_ome_multifile_series=collapse_ome_multifile_series,
        zeropadding=zeropadding,
        physicalsize_xyz=physicalsize_xyz,
        pixelunit=pixelunit,
        verbose=verbose)

    #print(f"Writing OME-TIFF output...")
    if images is None or metadatas is None:
        if verbose:
            print("No images or metadata to write. Conversion aborted.")
        return None
    
    fnames_written = imwrite(
            fname=fname,
            images=images,
            metadatas=metadatas,
            compression_level=compression_level,
            relative_path=relative_path,
            overwrite=overwrite,
            indicate_merged_files=merge_multiple_files_in_folder or merge_folder_stacks,
            return_fnames=True,
            verbose=verbose)
    """ print(f"Written {len(fnames_written)} OME-TIFF files:")
    for f in fnames_written:
        print(f"    {f}") """
    if cleanup_cache:
        if zarr_store is not None:
            #cleanup_omio_cache(fname, full_cleanup=False, verbose=verbose)
            if os.path.isdir(str(fname)):
                cleanup_omio_cache(fname, full_cleanup=True, verbose=verbose)
            else:
                cleanup_omio_cache(fname, full_cleanup=False, verbose=verbose)
        else:
            if verbose:
                print(f"Skipping omio cache cleanup because zarr_store=None.")
    if return_fnames:
        return fnames_written

# %% BIDS BATCH CONVERTER

# helper function for name matching:
def _match_name(name: str, pattern: str, mode: str) -> bool:
    """
    Match a string against a pattern using a selectable matching mode.

    This helper provides a small, explicit abstraction over common name matching
    strategies used throughout OMIO, for example when selecting files, folders,
    or tagged stack components.

    Supported matching modes
    ------------------------
    * "startswith":
        Return True if `name` starts with `pattern`, equivalent to
        `name.startswith(pattern)`.

    * "exact":
        Return True if `name` and `pattern` are identical strings.

    * "regex":
        Interpret `pattern` as a regular expression and return True if
        `re.match(pattern, name)` succeeds. The match is anchored at the beginning
        of `name`, following Python's `re.match` semantics.

    Parameters
    ----------
    name : str
        The string to be tested, typically a filename or folder name.
    pattern : str
        The pattern to match against `name`. Interpreted according to `mode`.
    mode : {"startswith", "exact", "regex"}
        Matching strategy to use.

    Returns
    -------
    bool
        True if the match succeeds under the selected mode, False otherwise.

    Raises
    ------
    ValueError
        If `mode` is not one of the supported values {"startswith", "exact", "regex"}.

    Notes
    -----
    This function does not perform any normalization (such as lowercasing) of
    either `name` or `pattern`. Callers are responsible for ensuring consistent
    string preprocessing when required.
    """
    if mode == "startswith":
        return name.startswith(pattern)
    if mode == "exact":
        return name == pattern
    if mode == "regex":
        return re.match(pattern, name) is not None
    raise ValueError(f"_match_name: invalid mode={mode!r}. Allowed: 'startswith','exact','regex'.")

# OMIO's BIDS-like batch converter function:
def bids_batch_convert(
    fname: str, # must be a directory
    sub: str,   # e.g. "ID" (subject folder detection)
    exp: str,   # e.g. "TP000" (experiment folder detection)
    exp_match_mode: str = "startswith",      # "startswith" | "exact" | "regex"
    tagfolder: str | None = None,            # e.g. "TAG_" (if set: only tagged folders inside exp)
    merge_multiple_files_in_folder: bool = False,
    merge_tagfolders: bool = False,          # if tagfolder is not None: merge TAGFOLDER_01..N into one output
    merge_along_axis: str = "T",
    collapse_ome_multifile_series: bool = True,
    zeropadding: bool = True,
    zarr_store: str | None = None,
    recursive: bool = False,
    physicalsize_xyz=None,
    pixelunit: str = "micron",
    compression_level: int = 3,
    relative_path: str | None = "omio_converted",
    overwrite: bool = False,
    cleanup_cache: bool = True,
    return_fnames: bool = False,
    verbose: bool = True):
    """
    Batch converter for a BIDS-like directory tree.

    This function traverses a project root folder and converts image files found in a
    subject and experiment hierarchy into OME-TIFF using OMIO’s reader and writer.
    It supports two main discovery modes: direct conversion of image files located
    inside experiment folders, or conversion and optional merging of tagged
    subfolders (folder-stacks) inside experiment folders.
    
    Abstract expected folder scheme
    -------------------------------
    The converter expects a project root that contains subject folders, which in turn
    contain experiment folders. Depending on whether `tagfolder` is provided, an
    experiment folder either contains image files directly, or contains multiple
    tagfolders which contain the image files.

    The schematic below uses ``<...>`` as placeholders for your chosen naming policy::

        project_root (= fname)
        ├─ <sub*>
        │  ├─ <exp*>
        │  │  ├─ image_01.tif / image_01.ome.tif / image_01.lsm / image_01.czi / image_01.raw
        │  ├─ <exp*>
        │  │  ├─ image_01.tif / image_01.ome.tif / image_01.lsm / image_01.czi / image_01.raw
        │  │  ├─ image_02.tif / image_02.ome.tif / image_02.lsm / image_02.czi / image_02.raw
        │  │  └─ ...
        │  ├─ <exp*>
        │  │  ├─ <tagfolder*>01
        │  │  │  ├─ image_01.tif / image_01.czi / image_01.raw / ...
        │  │  │  └─ ...
        │  │  ├─ <tagfolder*>02
        │  │  │  ├─ image_02.tif / image_02.czi / image_02.raw / ...
        │  │  │  └─ ...
        │  │  └─ ...
        │  └─ ...
        └─ <sub*>
        └─ ...

    Where:
    
    * ``<sub*>`` are subject folders detected by prefix matching with ``sub``.
      For example, if ``sub="sub"``, then ``"sub-01"``, ``"sub01"``, ``"sub_01"``, and
      ``"sub-A"`` all match, because this function uses ``startswith(sub)`` only.
    * ``<exp*>`` are experiment folders detected within each subject folder via ``exp`` and
      ``exp_match_mode`` (``"startswith"``, ``"exact"``, or ``"regex"``).
    * ``<tagfolder*>`` are optional tagfolders detected within an experiment folder via
      prefix matching with ``tagfolder`` (for example ``"TAG_"``).
      If ``tagfolder`` is set, direct image files in ``<exp*>`` are ignored and only
      tagfolders are processed.
    

    Folder discovery and selection
    ------------------------------
    The input ``fname`` must be a directory and is treated as the project root.

    Subject detection:
    
    * Every immediate subdirectory of ``fname`` whose name starts with ``sub`` is treated
      as a subject folder. No additional validation is performed.

    Experiment detection:
    
    * Within each subject folder, every immediate subdirectory whose name matches ``exp``
      under ``exp_match_mode`` is treated as an experiment folder.
      Matching modes are:
      
      * ``"startswith"``: folder name starts with ``exp``
      * ``"exact"``: folder name equals ``exp``
      * ``"regex"``: ``re.match(exp, foldername)`` succeeds

    Conversion behavior inside each experiment folder
    -------------------------------------------------
    Two mutually exclusive modes exist depending on `tagfolder`.

    Mode A: tagfolder is None (direct file conversion):
    
    * The converter processes image files located directly in the experiment folder.
    * If ``merge_multiple_files_in_folder=False``, every supported image file is
      converted to its own OME-TIFF output.
    * If ``merge_multiple_files_in_folder=True``, all supported image files in the
      experiment folder are read and concatenated along ``merge_along_axis`` (with
      optional ``zeropadding`` on non-merge axes) into one merged output.
    
    Mode B: tagfolder is not None (tagged folder stacks):
    
    * Direct image files in the experiment folder are ignored.
    * The converter searches for tagfolders inside the experiment folder whose name
      starts with ``tagfolder`` (for example ``"TAG_"``).
    * If ``merge_tagfolders=False`` (default), each tagfolder is converted separately
      and produces its own OME-TIFF output.
    * If ``merge_tagfolders=True``, all tagfolders are read and merged into a single
      output by reusing OMIO’s folder-stack logic. To keep output naming stable and
      collision-free when provenance-driven naming is used, a synthetic provenance
      name is injected into ``metadata["Annotations"]["original_filename"]``.

    Input path semantics
    --------------------
    Only directory input is accepted:
    
    * ``fname`` must be an existing directory and is treated as the project root.
    * All outputs are written within the experiment scope determined by traversal.

    Output placement and naming
    ---------------------------
    Output placement follows OMIO’s writer conventions via ``imconvert()`` and
    ``imwrite()``:

    * If ``relative_path`` is not None, outputs are written into a subfolder named
      ``relative_path`` under the relevant experiment folder (or under the experiment
      folder when writing a merged tagfolder product).
    * If ``relative_path`` is None, outputs are written directly into the experiment
      folder.
    * Per-stack output basenames are preferably derived from metadata provenance via
      ``Annotations["original_filename"]`` when present. Otherwise, a fallback basename
      is derived from the corresponding folder name.
    * If ``overwrite=False``, name collisions are resolved by appending an incrementing
      suffix to the output filename.

    Merging semantics
    -----------------
    * ``merge_along_axis`` must be one of {"T","Z","C"}.
    * In merge operations, the merge axis segments are concatenated in discovery order.
    * If ``zeropadding=True``, non-merge axes may differ between inputs and will be
      padded with zeros to the maximum size across inputs before concatenation.
      
      If ``zeropadding=False``, non-merge axes must match exactly or the merge is aborted.

    Zarr and cache handling
    -----------------------
    * ``zarr_store`` controls whether intermediate data are represented as NumPy in RAM
      or as Zarr arrays ("memory" or "disk") during reading and merging.
    * If ``cleanup_cache=True`` and ``zarr_store`` is not None, the function removes the
      per-input `.omio_cache` artifacts created during conversion once outputs are written.

    Parameters
    ----------
    fname : str
        Project root directory (must exist).
    sub : str
        Prefix used to detect subject folders at the project root level.
    exp : str
        Pattern used to detect experiment folders within each subject.
    exp_match_mode : {"startswith","exact","regex"}
        Matching strategy for experiment folder selection.
    tagfolder : str or None
        If None, convert direct files in experiment folders. If set, only process
        tagged subfolders inside experiment folders whose names start with `tagfolder`.
    merge_multiple_files_in_folder : bool
        If tagfolder is None, optionally merge all image files in an experiment folder
        into a single output.
    merge_tagfolders : bool
        If tagfolder is set, optionally merge all detected tagfolders into a single output.
    merge_along_axis : {"T","Z","C"}
        Axis along which merges are performed.
    collapse_ome_multifile_series : bool
        If True, detect and collapse OME multifile series during reading to avoid
        duplicate loading. 
    zeropadding : bool
        If True, allow mismatched non-merge axes by padding with zeros before merging.
    zarr_store : {None,"memory","disk"}
        Intermediate representation for reading and merging.
    recursive : bool
        Passed through to the underlying folder readers for file discovery.
    physicalsize_xyz : tuple or None
        Optional override for physical voxel sizes.
    pixelunit : str
        Unit string for pixel size fields (default "micron").
    compression_level : int
        zlib compression level for OME-TIFF writing.
    relative_path : str or None
        Subfolder name for outputs under experiment folders. Default "omio_converted".
    overwrite : bool
        If True, existing output files may be overwritten. Otherwise, collision-safe
        suffixing is used.
    cleanup_cache : bool
        If True, remove `.omio_cache` artifacts created during conversion.
    return_fnames : bool
        If True, return a list of all written output filenames.
    verbose : bool
        If True, print progress and diagnostic messages.

    Returns
    -------
    list[str] or None
        If `return_fnames=True`, returns a list of written OME-TIFF file paths.
        Otherwise returns None. The list may be empty if nothing matched or all
        conversions failed.
    """
    if fname is None or not os.path.isdir(str(fname)):
        raise ValueError(f"bids_batch_convert: fname must be an existing directory. Got: {fname!r}\n"
                         "Conversion aborted.")

    if merge_along_axis not in _ALLOWED_MERGE_AXES:
        raise ValueError(
            f"bids_batch_convert: merge_along_axis must be one of {sorted(_ALLOWED_MERGE_AXES)}.\n"
            f"Got: {merge_along_axis!r}\n"
            "Conversion aborted.")

    project = str(fname)
    written_all = []

    # subject folders: startswith(sub) only; OMIO policy: OMIO will treat all folders
    # found here as subjects; thus, if the user is messy with their folder naming,
    # they may get unexpected results.
    subs = []
    subjects_list = []
    for d in sorted(os.listdir(project)):
        full = os.path.join(project, d)
        if os.path.isdir(full) and d.startswith(sub):
            subs.append(full)
            subjects_list.append(d)
    if verbose:
        print(f"OMIO batch processor received BIDS project named={os.path.basename(project)!r}")
        print(f"in given root path={os.path.dirname(project)!r}.")
        print(f"Detected subjects with provided subject tag={sub!r} are:")
        for s in subjects_list:
            print(f"   {s}")
        print(f"⟶ {len(subs)} subject(s)")
        print(f"Will now look for experiment folders matching {exp!r} with mode={exp_match_mode!r} inside each subject.")
        
    if not subs:
        warnings.warn(f"[OMIO batch] No subject folders found in {project!r} starting with {sub!r}.")
        if return_fnames:
            return written_all

    # loop over subjects:
    for sub_path in subs:
        # sub_path = subs[0] # for testing
        sub_name = os.path.basename(sub_path)
        if verbose:
            print(f"\nBatch processing subject {sub_name}...")

        # experiment folders inside subject:
        exp_folders = []
        for d in sorted(os.listdir(sub_path)):
            full = os.path.join(sub_path, d)
            if not os.path.isdir(full):
                if verbose:
                    print(f"  Not a directory: {full!r}. Skipping.")
                continue
            if _match_name(d, exp, exp_match_mode):
                exp_folders.append(full)

        if verbose:
            print(f"  {len(exp_folders)} matched experiment folder(s) with exp-tag {exp!r} found with mode={exp_match_mode!r}:")
            for ef in exp_folders:
                print(f"    {os.path.basename(ef)!r}")

        if not exp_folders:
            if verbose:
                print(f"  No exp folders matched {exp!r} with mode={exp_match_mode!r}. Skipping subject.")
            continue
        
        # loop over experiments:
        for exp_path in exp_folders:
            # exp_path = exp_folders[0]  # for testing
            exp_name = os.path.basename(exp_path)
            if verbose:
                print(f"  Processing '{exp_name}' exp folder...\n")

            # default relative path per case:
            rel_default = relative_path

            # -------------------------
            # Case A: no tagfolder -> direct files in exp_path
            # -------------------------
            if tagfolder is None:
                try:
                    fnames_written = imconvert(
                        fname=exp_path,
                        zarr_store=zarr_store,
                        recursive=recursive,
                        folder_stacks=False,
                        merge_folder_stacks=False,
                        merge_multiple_files_in_folder=merge_multiple_files_in_folder,
                        merge_along_axis=merge_along_axis,
                        zeropadding=zeropadding,
                        physicalsize_xyz=physicalsize_xyz,
                        pixelunit=pixelunit,
                        compression_level=compression_level,
                        relative_path=rel_default,
                        overwrite=overwrite,
                        return_fnames=True,
                        cleanup_cache=cleanup_cache,
                        verbose=verbose)
                    if verbose:
                        print("\n")
                    if isinstance(fnames_written, list):
                        written_all.extend(fnames_written)
                except Exception as e:
                    if verbose:
                        print(f"    Conversion failed (direct files). Are there any image files in {exp_path!r}?\n"
                          f"    Or did you forget to set tagfolder=?\n"
                          f"    Error: {type(e).__name__}: {e}")
                continue

            # -------------------------
            # Case B: tagfolder set -> only tagged folders inside exp_path
            # -------------------------
            tagfolders = []
            for d in sorted(os.listdir(exp_path)):
                full = os.path.join(exp_path, d)
                if os.path.isdir(full) and d.startswith(tagfolder):
                    tagfolders.append(full)

            if not tagfolders:
                if verbose:
                    print(f"    tagfolder={tagfolder!r} requested, but no tagfolders found. Skipping exp.")
                continue

            if verbose:
                print(f"    found {len(tagfolders)} tagfolder(s) starting with {tagfolder!r}")
            
            rel_tag = relative_path

            # -------------------------
            # B1: default = each tagfolder gets its own output
            # -------------------------
            if not merge_tagfolders:
                for tf in tagfolders:
                    tf_name = os.path.basename(tf)
                    if verbose:
                        print(f"      {tf_name}: converting tagfolder...\n")

                    try:
                        fnames_written = imconvert(
                            fname=tf,
                            zarr_store=zarr_store,
                            recursive=recursive,
                            folder_stacks=False,  # ⟵ important: we are already in a tagfolder!
                            merge_folder_stacks=False,
                            merge_multiple_files_in_folder=merge_multiple_files_in_folder,
                            merge_along_axis=merge_along_axis,
                            zeropadding=zeropadding,
                            physicalsize_xyz=physicalsize_xyz,
                            pixelunit=pixelunit,
                            compression_level=compression_level,
                            relative_path=rel_tag,
                            overwrite=overwrite,
                            return_fnames=True,
                            cleanup_cache=cleanup_cache)
                        if verbose:
                            print("\n")
                        if isinstance(fnames_written, list):
                            written_all.extend(fnames_written)
                    except Exception as e:
                        if verbose:
                            print(f"      conversion failed (tagfolder).\n"
                                  f"      Error: {type(e).__name__}: {e}")
                continue

            # -------------------------
            # B2: merge_tagfolders=True -> merge ALL tagfolders into ONE output
            #     Writer uses original_filename; for a merged product we inject a synthetic
            #     provenance name to avoid collisions and make output self-describing.
            # -------------------------
            try:
                # Read and merge by reusing my imread TAG-folder logic:
                merged_img, merged_md = imread(
                    fname=tagfolders[0],         # imread expects one of the tagfolders; it auto-detects the tag
                    zarr_store=zarr_store,
                    return_list=False,
                    recursive=recursive,
                    folder_stacks=True,
                    merge_folder_stacks=True,    # triggers reading of all tagfolders and merging
                    merge_multiple_files_in_folder=False,
                    merge_along_axis=merge_along_axis,
                    collapse_ome_multifile_series=collapse_ome_multifile_series,
                    zeropadding=zeropadding,
                    physicalsize_xyz=physicalsize_xyz,
                    pixelunit=pixelunit,
                    verbose=verbose)
                if verbose:
                    print("\n")

                if merged_img is None or merged_md is None:
                    if verbose:
                        print(f"    {exp_name}: merge_tagfolders produced None. Skipping.")
                    continue

                # Inject synthetic provenance name so writer can stay "original_filename-driven":
                # This avoids depending on fname basename or exp folder name.
                merged_md = dict(merged_md)
                ann = merged_md.get("Annotations", {})
                if not isinstance(ann, dict):
                    ann = {}
                ann = dict(ann)
                ann["original_filename"] = f"{sub_name}_{exp_name}_{tagfolder}merged.ome.tif"
                merged_md["Annotations"] = ann

                # Write merged output at exp level (not inside a tagfolder).
                # We call writer with fname=exp_path to place output in exp scope.
                fnames_written = imwrite(
                    fname=exp_path,
                    images=merged_img,
                    metadatas=merged_md,
                    compression_level=compression_level,
                    relative_path=relative_path if relative_path is not None else "merged",
                    overwrite=overwrite,
                    return_fnames=True,
                    verbose=verbose,
                    indicate_merged_files=True)
                if isinstance(fnames_written, list):
                    written_all.extend(fnames_written)

                if cleanup_cache and zarr_store is not None:
                    cleanup_omio_cache(exp_path, full_cleanup=False, verbose=verbose)

            except Exception as e:
                if verbose:
                    print(f"    {exp_name}: conversion failed (merge_tagfolders). "
                      f"    Error: {type(e).__name__}: {e}")

    if verbose:
        print(f"\nOMIO batch processing done. Written {len(written_all)} file(s).")
        for f in written_all:
            print(f"  {f}")

    if return_fnames:
        return written_all

# %% END
