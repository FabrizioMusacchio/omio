""" 
OMIO TEMPLATE MODULE
This module provides functions to create and manipulate OMIO 
metadata templates.

author: Fabrizio Musacchio  
first version: December 2025
ported to modularized structure: August 2026
"""
# %% IMPORTS
from .core import *
from .cache import *
# %% TEMPLATE FUNCTIONS
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
        store path is ``<parent>/.omio_cache/<zarr_store_name>.zarr``. If None,
        ``"empty_image"`` is used.
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
        consistent with the created image shape and canonical axes TZCYX. For
        `zarr_store="disk"`, it also contains ``"omio_cache_folder"``,
        ``"omio_zarr_store_path"``, and ``"omio_zarr_store_name"`` so users can
        inspect or clean up the generated cache later.

    Notes
    -----
    * The function assumes canonical OME axes TZCYX as defined by the module-level
      constant ``_OME_AXES``.
    * For `zarr_store="disk"`, any existing store at the target location is removed
      before creating a new one.
    * For disk-backed empty images, OMIO metadata and cache information are also
      stored in the Zarr attributes.
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
                zarr_store_name = "empty_image"
            zarr_store_name = str(zarr_store_name)
            if zarr_store_name.endswith(".zarr"):
                zarr_store_name = zarr_store_name[:-5]

            zarr_store_path = os.fspath(zarr_store_path)
            if os.path.isdir(zarr_store_path):
                parent_folder = zarr_store_path
            elif os.path.isfile(zarr_store_path):
                parent_folder = os.path.dirname(zarr_store_path) or "."
                if verbose:
                    print(f"    zarr_store_path is a file; taking its parent folder:")
                    print(f"    {parent_folder}")
            else:
                basename = os.path.basename(os.path.normpath(zarr_store_path))
                _, ext = os.path.splitext(basename)
                if ext:
                    parent_folder = os.path.dirname(zarr_store_path) or "."
                    if verbose:
                        print(f"    zarr_store_path looks like a file path; taking its parent folder:")
                        print(f"    {parent_folder}")
                else:
                    parent_folder = zarr_store_path

            cache_folder = os.path.join(parent_folder, ".omio_cache")
            os.makedirs(cache_folder, exist_ok=True)

            cache_folder = os.path.abspath(cache_folder)
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
            metadata = create_empty_metadata(shape=shape, input_metadata=input_metadata,
                                             verbose=verbose)
            if zarr_store == "disk":
                metadata["omio_cache_folder"] = cache_folder
                metadata["omio_zarr_store_path"] = zarr_path
                metadata["omio_zarr_store_name"] = zarr_store_name
                metadata["omio_zarr_store_type"] = "disk"

                cache_info = {
                    "schema_version": _CACHE_SCHEMA_VERSION,
                    "cache_kind": "empty_image",
                    "creator": "create_empty_image",
                    "omio_version": _OMIO_VERSION,
                    "omio_cache_folder": cache_folder,
                    "zarr_store_path": zarr_path,
                    "zarr_store_name": zarr_store_name,
                    "shape": [int(v) for v in shape],
                    "dtype": str(np.dtype(dtype)),
                    "chunks": [int(v) for v in chunks],
                    "fill_value": _jsonify_for_storage(fill_value),
                }
                _write_disk_cache_payload(z_out, metadata, cache_info, verbose=verbose)
            return z_out, metadata
        else:
            if zarr_store == "disk":
                metadata = create_empty_metadata(shape=shape, input_metadata=input_metadata,
                                                 verbose=verbose)
                metadata["omio_cache_folder"] = cache_folder
                metadata["omio_zarr_store_path"] = zarr_path
                metadata["omio_zarr_store_name"] = zarr_store_name
                metadata["omio_zarr_store_type"] = "disk"
                cache_info = {
                    "schema_version": _CACHE_SCHEMA_VERSION,
                    "cache_kind": "empty_image",
                    "creator": "create_empty_image",
                    "omio_version": _OMIO_VERSION,
                    "omio_cache_folder": cache_folder,
                    "zarr_store_path": zarr_path,
                    "zarr_store_name": zarr_store_name,
                    "shape": [int(v) for v in shape],
                    "dtype": str(np.dtype(dtype)),
                    "chunks": [int(v) for v in chunks],
                    "fill_value": _jsonify_for_storage(fill_value),
                }
                _write_disk_cache_payload(z_out, metadata, cache_info, verbose=verbose)
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
# %% ALL
__all__ = [name for name in globals() if not name.startswith("__")]
# %% END
