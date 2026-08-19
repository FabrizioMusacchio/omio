""" 
OMIO CZI READER

This module provides functions to read Zeiss CZI files into OMIO's 
canonical representation.

author: Fabrizio Musacchio  
first version: December 2025
ported to modularized structure: August 2026
"""
# %% IMPORTS
from ..core import *
from ..cache import *
# %% CZI FUNCTIONS
def _get_primary_czi_scene(czi_file):
    """
    Return the first available scene object from a ``czifile.CziFile``.

    Recent ``czifile`` revisions moved axis-related metadata from the file object
    to scene-level ``CziImage`` objects. This helper provides a version-tolerant
    way to obtain that primary scene.
    """
    scenes = getattr(czi_file, "scenes", None)
    if scenes is None:
        return None

    try:
        return scenes[0]
    except Exception:
        pass

    if hasattr(scenes, "values"):
        try:
            return next(iter(scenes.values()))
        except Exception:
            pass

    try:
        first = next(iter(scenes))
    except Exception:
        return None

    if hasattr(first, "axes") or hasattr(first, "dims"):
        return first

    try:
        return scenes[first]
    except Exception:
        return None

def _get_czi_axes(czi_file):
    """
    Resolve axis metadata across old and new ``czifile`` APIs.
    """
    axes = getattr(czi_file, "axes", None)
    if isinstance(axes, str) and axes:
        return axes

    scene = _get_primary_czi_scene(czi_file)
    if scene is not None:
        scene_axes = getattr(scene, "axes", None)
        if isinstance(scene_axes, str) and scene_axes:
            return scene_axes

        dims = getattr(scene, "dims", None)
        if dims:
            return "".join(str(axis) for axis in dims)

    raise AttributeError(
        "Could not determine CZI axes from czifile metadata. "
        "Neither CziFile.axes nor scene axes/dims were available."
    )

def _get_czi_metadata_dict(czi_file):
    """
    Return structured CZI metadata across old and new ``czifile`` APIs.
    """
    metadata_func = getattr(czi_file, "metadata", None)
    if not callable(metadata_func):
        raise AttributeError("czifile CziFile object does not provide metadata().")

    try:
        metadata = metadata_func(asdict=True)
    except TypeError:
        metadata = None

    if isinstance(metadata, dict):
        return metadata

    try:
        metadata = metadata_func(raw=False)
    except TypeError as exc:
        raise TypeError(
            "Could not retrieve structured CZI metadata from czifile. "
            "Expected either metadata(asdict=True) or metadata(raw=False)."
        ) from exc

    if isinstance(metadata, dict):
        return metadata

    raise TypeError(
        "czifile metadata() did not return a dictionary for CZI metadata extraction."
    )

def read_czi(fname, physicalsize_xyz=None, pixelunit="micron", zarr_store=None, 
             zarr_store_path=None, return_list=False, reuse_disk_cache=False,
             verbose=True):
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

        Existing on-disk stores at that location are replaced unless
        ``reuse_disk_cache=True`` and a validated OMIO cache is already present.
        Default is None.
    zarr_store_path : str, os.PathLike, or None, optional
        Parent directory in which OMIO creates ``.omio_cache`` when
        ``zarr_store="disk"``. If None, the cache is created next to the source
        file as before. Passing the ``.omio_cache`` folder itself is also
        accepted. Default is None.
    return_list : bool, optional
        If True, return ``[image]`` and ``[metadata]`` for backward compatibility.
        Default is False.
    reuse_disk_cache : bool, optional
        If True and ``zarr_store="disk"``, OMIO first checks for a compatible
        existing on-disk cache and reuses it instead of rebuilding the Zarr store.
        Validation uses the persisted OMIO cache manifest stored inside the Zarr
        attributes. Default is False.
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
      ``.omio_cache``. OMIO metadata and cache validation info are persisted in the
      Zarr attributes so the store can later be reopened without rereading the
      original file.
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
    cache_override = physicalsize_xyz_ext if set_input_pixelsize else None

    if zarr_store == "disk" and reuse_disk_cache:
        cached_image, cached_metadata = _try_reuse_disk_cache(
            fname=fname,
            reader_name="czi",
            pixelunit=pixelunit,
            physicalsize_xyz_override=cache_override,
            zarr_store_path=zarr_store_path,
            verbose=verbose,
        )
        if cached_image is not None:
            if verbose:
                print("Finished reading CZI from reused disk cache.")
            if return_list:
                return [cached_image], [cached_metadata]
            return cached_image, cached_metadata

    # read CZI into memory (no memory mapping possible)
    if verbose:
        print("Reading CZI fully into RAM...")
    CZI_image = czi.imread(fname)

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

    with czi.CziFile(fname) as CZI_metadata_obj:
        # extract CZI axes (e.g. BVCTZYX0)
        metadata["axes"] = _get_czi_axes(CZI_metadata_obj)

        # extract scaling metadata:
        czi_metadata_dict = _get_czi_metadata_dict(CZI_metadata_obj)

    # filter unwanted non-OME axes (keep only TZCYX):
    CZI_image, metadata["axes"] = _filter_image_data_for_ome_tif(CZI_image, metadata["axes"])

    CZImetadata_xyz = (
        czi_metadata_dict['ImageDocument']['Metadata']['Scaling']['Items']['Distance'])
    conv_um = 10 ** 6

    if isinstance(CZImetadata_xyz, dict):
        CZImetadata_xyz = [CZImetadata_xyz]

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
            zarr_path = _get_disk_cache_path(fname, zarr_store_path=zarr_store_path)
            os.makedirs(os.path.dirname(zarr_path), exist_ok=True)
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

    if zarr_store == "disk" and isinstance(CZI_image, zarr.core.array.Array):
        cache_info = _build_disk_cache_info(
            fname=fname,
            reader_name="czi",
            pixelunit=pixelunit,
            physicalsize_xyz_override=cache_override,
            cache_kind="primary",
        )
        metadata = _annotate_disk_cache_metadata(
            metadata,
            fname=fname,
            zarr_path=_get_zarr_array_store_path(
                CZI_image,
                _get_disk_cache_path(fname, zarr_store_path=zarr_store_path)),
            zarr_store_path=zarr_store_path)
        _write_disk_cache_payload(CZI_image, metadata, cache_info, verbose=verbose)

    if verbose:
        print("Finished reading CZI.")

    if return_list:
        return [CZI_image], [metadata]
    else:
        return CZI_image, metadata
# %% ALL
__all__ = [name for name in globals() if not name.startswith("__")]
# %% END
