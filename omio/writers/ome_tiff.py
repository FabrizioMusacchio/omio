""" 
OMIO OME-TIFF WRITER

This module provides functions to write images in the OME-TIFF format.

author: Fabrizio Musacchio  
first version: December 2025
ported to modularized structure: August 2026
"""
# %% IMPORTS
from ..core import *
from ..cache import *
# %% OME-TIFF WRITER FUNCTIONS
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

def _normalize_axes_shape_for_ometiff(shape, axes):
    """
    Normalize an axis declaration for OME-TIFF writing without reading image data.

    This is the shape-only counterpart to ``_normalize_axes_for_ometiff``. It is
    used for Zarr-backed inputs where converting the full array to NumPy would
    defeat the purpose of disk-backed workflows.
    """
    shape = tuple(shape)
    if len(axes) != len(shape):
        raise ValueError(
            f"_normalize_axes_shape_for_ometiff: axes '{axes}' (len={len(axes)}) "
            f"does not fit to shape ndim={len(shape)}"
        )

    keep_indices = list(range(len(shape)))
    if "S" in axes:
        s_idx = axes.index("S")
        if shape[s_idx] == 1:
            keep_indices.remove(s_idx)
            axes = axes.replace("S", "")
            shape = tuple(shape[i] for i in keep_indices)

    if len(axes) != len(shape):
        raise ValueError(
            f"_normalize_axes_shape_for_ometiff: axes '{axes}' (len={len(axes)}) "
            f"does not fit to normalized shape ndim={len(shape)}"
        )

    return shape, axes, keep_indices

def _iter_ometiff_planes_from_zarr(image, axes_in, desired_axes="TCZYX"):
    """
    Yield 2D planes from a Zarr array in the requested OME-TIFF axis order.

    The iterator keeps only a single YX plane in memory at a time while preserving
    the same logical axis permutation used by the in-memory writer path.
    """
    source_shape = tuple(image.shape)
    norm_shape, norm_axes, keep_indices = _normalize_axes_shape_for_ometiff(
        source_shape,
        axes_in)

    missing = [ax for ax in desired_axes if ax not in norm_axes]
    if missing:
        raise ValueError(
            "imwrite: Zarr-backed OME-TIFF writing requires axes "
            f"{desired_axes!r}; missing {missing!r} in metadata axes {norm_axes!r}."
        )
    extra = [ax for ax in norm_axes if ax not in desired_axes]
    if extra:
        raise ValueError(
            "imwrite: Zarr-backed OME-TIFF writing does not support extra non-OME "
            f"axes {extra!r} in metadata axes {norm_axes!r}."
        )

    target_shape = tuple(norm_shape[norm_axes.index(ax)] for ax in desired_axes)
    target_outer_axes = desired_axes[:-2]
    target_outer_shape = target_shape[:-2]

    norm_to_source = {norm_i: src_i for norm_i, src_i in enumerate(keep_indices)}
    axis_to_source = {
        ax: norm_to_source[norm_axes.index(ax)]
        for ax in norm_axes
    }

    source_spatial_axes = "".join(
        ax for ax in axes_in
        if ax in ("Y", "X") and ax in axis_to_source
    )
    if sorted(source_spatial_axes) != ["X", "Y"]:
        raise ValueError(
            "imwrite: Zarr-backed OME-TIFF writing requires exactly one Y and one X axis "
            f"after normalization; got spatial axes {source_spatial_axes!r}."
        )

    for outer_index in np.ndindex(*target_outer_shape):
        source_index = [0] * len(source_shape)

        for ax, value in zip(target_outer_axes, outer_index):
            source_index[axis_to_source[ax]] = value

        source_index[axis_to_source["Y"]] = slice(None)
        source_index[axis_to_source["X"]] = slice(None)

        plane = np.asarray(image[tuple(source_index)])
        if source_spatial_axes != "YX":
            plane = np.moveaxis(
                plane,
                [source_spatial_axes.index("Y"), source_spatial_axes.index("X")],
                [0, 1])

        yield plane

def _prepare_zarr_for_ometiff_write(image, axes_in, desired_axes="TCZYX"):
    """
    Prepare a Zarr array for plane-wise OME-TIFF writing.

    Returns an iterator, target shape, dtype, and output axes without materializing
    the full Zarr store.
    """
    norm_shape, norm_axes, _ = _normalize_axes_shape_for_ometiff(image.shape, axes_in)

    missing = [ax for ax in desired_axes if ax not in norm_axes]
    if missing:
        raise ValueError(
            "imwrite: Zarr-backed OME-TIFF writing requires axes "
            f"{desired_axes!r}; missing {missing!r} in metadata axes {norm_axes!r}."
        )
    extra = [ax for ax in norm_axes if ax not in desired_axes]
    if extra:
        raise ValueError(
            "imwrite: Zarr-backed OME-TIFF writing does not support extra non-OME "
            f"axes {extra!r} in metadata axes {norm_axes!r}."
        )

    target_shape = tuple(norm_shape[norm_axes.index(ax)] for ax in desired_axes)
    planes = _iter_ometiff_planes_from_zarr(image, axes_in, desired_axes=desired_axes)

    return planes, target_shape, image.dtype, desired_axes

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

    Zarr-backed inputs are written plane-wise to avoid materializing the full store
    in RAM. OMIO still writes the same OME-TIFF layout as the in-memory path, but
    only the currently written YX plane is converted to NumPy at a time.

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
        normalizes axes and permutes to the writer's target order internally. Zarr
        arrays are streamed plane-wise so large disk-backed arrays do not need to be
        loaded fully into RAM before OME-TIFF export.
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
    * NumPy-backed inputs are normalized by `_normalize_axes_for_ometiff` (currently
      removing a singleton ``"S"`` axis) and then permuted into the writer's target
      axis order before writing.
    * Zarr-backed inputs use an equivalent shape/index based normalization and are
      passed to `tifffile.imwrite` as a plane iterator together with explicit
      ``shape`` and ``dtype``.
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

        # reorder axes to the OME-TIFF writer target order without forcing Zarr
        # inputs into memory.
        axes_in = metadata.get("axes", "TZCYX")
        desired_axes = "TCZYX"
        if isinstance(image, zarr.core.array.Array):
            image_ome, image_ome_shape, image_ome_dtype, axes_out = _prepare_zarr_for_ometiff_write(
                image,
                axes_in,
                desired_axes=desired_axes)
        else:
            image_ome, axes_in = _normalize_axes_for_ometiff(image, axes_in)
            if axes_in != desired_axes:
                idx = {ax: i for i, ax in enumerate(axes_in)}
                perm = [idx[ax] for ax in desired_axes]
                image_ome = np.moveaxis(image_ome, perm, range(len(perm)))
                axes_out = desired_axes
            else:
                axes_out = axes_in
            image_ome_shape = None
            image_ome_dtype = None
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
        imwrite_kwargs = {
            "ome": True,
            "compression": "zlib",
            "compressionargs": {"level": compression_level},
            "resolution": (1/metadata["PhysicalSizeY"], 1/metadata["PhysicalSizeX"]),
            "metadata": ome_meta,
            "photometric": "minisblack",
            "imagej": False,
            "bigtiff": use_bigtiff,
        }
        if isinstance(image, zarr.core.array.Array):
            tifffile.imwrite(
                fname_out_stack,
                data=image_ome,
                shape=image_ome_shape,
                dtype=image_ome_dtype,
                **imwrite_kwargs)
        else:
            tifffile.imwrite(
                fname_out_stack,
                image_ome,
                **imwrite_kwargs)
        fnames_written.append(fname_out_stack)
    if return_fnames:
        return fnames_written
# %% ALL
__all__ = [name for name in globals() if not name.startswith("__")]
# %% END
