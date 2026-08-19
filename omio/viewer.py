""" 
OMIO NAPARI VIEWER MODULE

This module provides functions to visualize microscopy 
images in Napari, including handling of Zarr arrays and automatic 
squeezing of singleton dimensions.

author: Fabrizio Musacchio  
first version: December 2025
ported to modularized structure: August 2026
"""
# %% IMPORTS
from __future__ import annotations

from .core import *
from .cache import *
# %% VIEWER FUNCTIONS
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

    da.to_zarr(darr, target_path, mode="w")
    squeezed_zarr = zarr.open(target_path, mode="r")

    return squeezed_zarr, squeezed_axes

def _strip_image_extension(name: str) -> str:
    """
    Return a display-oriented image name without common microscopy file suffixes.
    """
    name = os.path.basename(str(name))
    lname = name.lower()
    for suffix in (".ome.tiff", ".ome.tif", ".tiff", ".tif", ".czi", ".lsm", ".raw", ".zarr"):
        if lname.endswith(suffix):
            return name[:-len(suffix)]
    return os.path.splitext(name)[0]

def _metadata_lookup(metadata: dict, key: str, default=None):
    """
    Look up a metadata key at top level first and then inside Annotations.
    """
    if not isinstance(metadata, dict):
        return default
    if key in metadata:
        return metadata[key]
    annotations = metadata.get("Annotations", {})
    if isinstance(annotations, dict):
        return annotations.get(key, default)
    return default

def _derive_image_name_from_metadata(metadata: dict, fallback: Union[None, str] = None) -> str:
    """
    Derive a user-facing image name from OMIO metadata or a fallback path/name.
    """
    for key in ("original_filename", "filename", "Filename", "file_name"):
        value = _metadata_lookup(metadata, key)
        if value:
            return _strip_image_extension(str(value))
    if fallback:
        return _strip_image_extension(str(fallback))
    return "omio_image"

def _derive_napari_layer_name(metadata: dict, image_name: Union[None, str] = None) -> str:
    """
    Derive a Napari layer name, giving explicit user input priority over metadata.
    """
    if image_name is not None:
        return _strip_image_extension(str(image_name))
    return _derive_image_name_from_metadata(metadata)

def _prefix_napari_layer_names(image_base_name: str,
                               layer_names: Union[None, str, list[str], tuple[str, ...]] = None):
    """
    Prefix optional Napari layer/channel names with the image display name.
    """
    if layer_names is None:
        return image_base_name
    if isinstance(layer_names, str):
        return f"{image_base_name} {layer_names}"
    if isinstance(layer_names, (list, tuple)):
        return [f"{image_base_name} {name}" for name in layer_names]
    return layer_names

def _derive_napari_cache_anchor(image,
                                metadata: dict,
                                image_name: Union[None, str] = None,
                                cache_folder_name: str = ".omio_cache") -> str:
    """
    Derive a stable path anchor for Napari-derived Zarr caches.

    File-backed Zarr arrays use their own store path. In-memory arrays fall back to
    OMIO metadata paths, then to the current working directory.
    """
    if isinstance(image, zarr.core.array.Array):
        try:
            store_path = str(image.store).replace("file://", "")
            if store_path and not store_path.startswith("<") and "MemoryStore" not in store_path:
                return store_path[:-5] if store_path.endswith(".zarr") else store_path
        except Exception:
            pass

    zarr_path = _metadata_lookup(metadata, "omio_zarr_store_path")
    if zarr_path:
        zarr_path = str(zarr_path)
        return zarr_path[:-5] if zarr_path.endswith(".zarr") else zarr_path

    cache_folder = _metadata_lookup(metadata, "omio_cache_folder")
    if cache_folder:
        parent = os.path.dirname(os.path.normpath(str(cache_folder)))
        return os.path.join(parent, _derive_image_name_from_metadata(metadata, fallback=image_name))

    parent = _metadata_lookup(metadata, "original_parentfolder")
    if parent:
        return os.path.join(str(parent), _derive_image_name_from_metadata(metadata, fallback=image_name))

    return os.path.join(os.getcwd(), _derive_image_name_from_metadata(metadata, fallback=image_name))

# main single-image handle for Napari visualization of image(s) as NumPy, Zarr, or Zarr + Dask:
def _single_image_open_in_napari(
        image: Union[np.ndarray, "zarr.core.array.Array"], 
        metadata: dict, 
        image_name: Union[None, str] = None, 
        zarr_mode: str = "numpy",
        cache_folder_name: str = ".omio_cache", 
        axes_full: str = "TZCYX", 
        viewer=None,
        viewer_name: Union[None, str] = None,
        layer_names: Union[None, str, list[str]] = None,
        blending: str = "additive",
        fname: Union[None, str] = None,
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
    image_name : str or None, optional
        Display name or path used to derive the default layer name. If None, OMIO
        derives the name from metadata entries such as
        ``Annotations["original_filename"]``. Default is None.
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
        Explicit layer name to use. This legacy argument is kept for internal
        compatibility. If provided, it takes precedence over `image_name`.
    layer_names : str, list of str, or None, optional
        Optional layer or channel suffix name(s). The resolved image name is
        prepended automatically. For example, ``image_name="sample"`` and
        ``layer_names=["channel 0", "channel 1"]`` become
        ``["sample channel 0", "sample channel 1"]`` in Napari. Default is None.
    blending : str, optional
        Napari blending mode forwarded to ``viewer.add_image``. Default is
        ``"additive"``.
    fname : str or None, optional
        Deprecated alias for `image_name`, accepted for backward compatibility.
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
    if image_name is None and fname is not None:
        image_name = fname

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
            base_no_ext = _derive_napari_cache_anchor(
                image=image,
                metadata=metadata,
                image_name=image_name,
                cache_folder_name=cache_folder_name)

            squeezed_zarr, squeezed_axes = _squeeze_zarr_to_napari_cache_dask(src=image,
                                                fname=base_no_ext, axes=axes_full,
                                                cache_folder_name=cache_folder_name)
            napari_data = squeezed_zarr
            napari_axes = squeezed_axes

        elif zarr_mode == "zarr_nodask":
            # Zarr → squeezed Zarr w/o Dask:
            if verbose:
                print("  Memory-efficient squeezing Zarr without Dask...")
            base_no_ext = _derive_napari_cache_anchor(
                image=image,
                metadata=metadata,
                image_name=image_name,
                cache_folder_name=cache_folder_name)
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
        image_base_name = _derive_napari_layer_name(metadata, image_name=image_name)
        layer_name = _prefix_napari_layer_names(image_base_name, layer_names=layer_names)
    
    # convert napari_data into a dask-array if it's a Zarr (napari handles zarr dask arrays better):
    if isinstance(napari_data, zarr.core.array.Array):
        napari_data = da.from_zarr(napari_data)
    
    # add the new image layer:
    layer = viewer.add_image(napari_data, channel_axis=channel_axis, 
                             scale=scales_array, name=layer_name, blending=blending)
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = metadata.get("unit", "micron")

    return viewer, layer, napari_data, napari_axes
# main multi-image handler for Napari visualization of image(s) as NumPy, Zarr, or Zarr + Dask:
def open_in_napari(images: Union[np.ndarray, "zarr.core.array.Array", list[Union[np.ndarray, "zarr.core.array.Array"]]],
                   metadatas: Union[dict, list[dict]], 
                   image_name: Union[None, str] = None,
                   zarr_mode: str = "numpy", 
                   cache_folder_name: str = ".omio_cache", 
                   axes_full: str = "TZCYX", 
                   viewer: napari.Viewer = None, 
                   returns: bool=False,
                   layer_names: Union[None, str, list[str], list[list[str]]] = None,
                   blending: str = "additive",
                   fname: Union[None, str] = None,
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
    image_name : str or None, optional
        Display name or path used to derive default Napari layer names. If None,
        OMIO derives names from metadata entries such as
        ``Annotations["original_filename"]``. Default is None.
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
    layer_names : str, list of str, list of list of str, or None, optional
        Optional layer or channel suffix name(s). The resolved image name is
        prepended automatically. For one image with a channel axis, a list of
        strings names the channel layers below the image name. For multiple images,
        a list matching the number of images can provide one suffix per image.
        Default is None.
    blending : str, optional
        Napari blending mode forwarded to ``viewer.add_image``. Default is
        ``"additive"``.
    fname : str or None, optional
        Deprecated alias for `image_name`, accepted for backward compatibility.
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
    * The former ``fname`` argument is still accepted as an alias for
      `image_name`, so existing code using a third positional argument or
      ``fname=...`` continues to work.
    """
    if image_name is None and fname is not None:
        image_name = fname

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

    per_image_layer_names = [layer_names] * len(images)
    if isinstance(layer_names, str):
        if len(images) > 1:
            per_image_layer_names = [f"{layer_names}_idx{idx}" for idx in range(len(images))]
        else:
            per_image_layer_names = [layer_names]
    elif isinstance(layer_names, (list, tuple)) and len(images) > 1 and len(layer_names) == len(images):
        per_image_layer_names = list(layer_names)

    for idx, (img, md) in enumerate(zip(images, metadatas)):
        if verbose:
            print(f"Opening image {idx+1}/{len(images)} in napari...")
        
        # build image name fallback:
        if image_name is None:
            layer_image_name = None
        elif len(images) == 1:
            layer_image_name = image_name
        else:
            layer_image_name = f"{_strip_image_extension(str(image_name))}_idx{idx}"
        
        # open in napari:
        v, layer, napari_data, napari_axes = _single_image_open_in_napari(
            image=img,
            metadata=md,
            image_name=layer_image_name,
            zarr_mode=zarr_mode,
            cache_folder_name=cache_folder_name,
            axes_full=axes_full,
            viewer=viewer,
            layer_names=per_image_layer_names[idx],
            blending=blending,
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
# %% ALL
__all__ = [name for name in globals() if not name.startswith("__")]
# %% END
