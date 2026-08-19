""" 
OMIO READ MODULE

This module provides functions to read microscopy image files into OMIO's
canonical representation.

author: Fabrizio Musacchio  
first version: December 2025
ported to modularized structure: August 2026
"""
# %% IMPORTS
from .core import *
from .cache import *
from .readers.tif import read_tif
from .readers.czi import read_czi
from .readers.thorlabs_raw import read_thorlabs_raw
# %% READ FUNCTIONS
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
                        zarr_store_path: Union[None, str, os.PathLike] = None,
                        return_list: bool = False,
                        physicalsize_xyz: Union[None, Any] = None,
                        pixelunit: str = "micron",
                        reuse_disk_cache: bool = False,
                        on_error: str = "raise",
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
    zarr_store_path : str, os.PathLike, or None
        Optional parent directory for ``.omio_cache`` when `zarr_store` is "disk".
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
    on_error : {"raise", "return_none"}, optional
        Error policy forwarded to readers that support recoverable batch-friendly
        failures. Currently used by the Thorlabs RAW reader for unrecoverable
        metadata problems.
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
            zarr_store_path=zarr_store_path,
            return_list=return_list,
            physicalsize_xyz=physicalsize_xyz,
            pixelunit=pixelunit,
            reuse_disk_cache=reuse_disk_cache,
            verbose=verbose)

    if _lower_ext(lp) == ".czi":
        return read_czi(
            path,
            zarr_store=zarr_store,
            zarr_store_path=zarr_store_path,
            return_list=return_list,
            physicalsize_xyz=physicalsize_xyz,
            pixelunit=pixelunit,
            reuse_disk_cache=reuse_disk_cache,
            verbose=verbose)

    if _lower_ext(lp) == ".raw":
        return read_thorlabs_raw(
            path,
            zarr_store=zarr_store,
            zarr_store_path=zarr_store_path,
            return_list=return_list,
            physicalsize_xyz=physicalsize_xyz,
            pixelunit=pixelunit,
            reuse_disk_cache=reuse_disk_cache,
            on_error=on_error,
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
         zarr_store_path: Union[None, str, os.PathLike] = None,
         reuse_disk_cache: bool = False,
         on_error: str = "raise",
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
    NumPy array. For "disk", Zarr outputs are created in a hidden cache folder `.omio_cache`.
    By default, this cache folder is created next to the source data. If
    `zarr_store_path` is provided, `.omio_cache` is created under that location instead.
    This is intended for large files where memory mapping and chunked access are required
    downstream, including workflows where source data live on a server but the Zarr cache
    should be kept on a local disk. Disk-backed caches also persist OMIO metadata and cache
    validation information directly in the Zarr store attributes so that later calls may
    safely reuse an existing cache.

    **Folder input behavior:**
    If `fname` resolves to a folder, OMIO lists all supported image files inside the folder
    (optionally recursive) and reads them in sorted order.

    If `folder_stacks=True` or `merge_folder_stacks=True`, the folder is interpreted as one
    member of a tagged folder stack family with names like `<TAG>_000`, `<TAG>_001`, etc.
    OMIO derives `<TAG>_` from the provided folder name, finds all co folders with the same
    tag in the parent directory, reads the first image file in each of these folders, and
    returns either the list of stacks or a merged stack.

    **Merge behavior:**
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
    zarr_store_path : str, os.PathLike, or None, optional
        Parent directory in which OMIO creates ``.omio_cache`` when
        ``zarr_store="disk"``. If None, caches are created next to each source file.
        Passing the ``.omio_cache`` folder itself is also accepted. When
        ``reuse_disk_cache=True`` and a custom cache location was used to create the
        cache, the same `zarr_store_path` should be provided again so OMIO knows
        where to look. Default is None.
    reuse_disk_cache : bool, optional
        If True and ``zarr_store="disk"``, OMIO first attempts to reuse a validated
        existing on-disk cache instead of rebuilding it from the original source
        file. Validation compares source path, file size, modification time,
        OMIO version, relevant backend versions, and applicable read overrides.
        Default is False.
    on_error : {"raise", "return_none"}, optional
        Error policy for format-specific reader failures that support explicit
        batch-friendly skipping. Currently this affects Thorlabs RAW files with
        unrecoverable XML/YAML metadata problems. ``"raise"`` preserves the
        default behavior. ``"return_none"`` returns ``None`` image/metadata pairs
        for those files so callers can skip them explicitly. Default is ``"raise"``.
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
    tuple
        Returns ``(image, metadata)`` for single non-folder inputs when
        `return_list=False`. For multi-file inputs, folder reads, or
        `return_list=True`, returns ``(images, metadatas)`` as lists. Merge modes
        return a single merged image and merged metadata, or lists if
        `return_list=True`. If a requested merge fails validation, None results may
        be returned according to the calling branch.

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
    if on_error not in ("raise", "return_none"):
        raise ValueError("read: on_error must be one of 'raise' or 'return_none'. "
                         f"Got: {on_error!r}")

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
                    zarr_store_path=zarr_store_path,
                    return_list=False,
                    physicalsize_xyz=physicalsize_xyz,
                    pixelunit=pixelunit,
                    reuse_disk_cache=reuse_disk_cache,
                    on_error=on_error,
                    verbose=verbose)

                if img is None or md is None:
                    if verbose:
                        print(f"    Reader returned None for folder stack file: {f0!r}. Skipping.")
                    continue
                
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
                zarr_store_path=zarr_store_path,
                physicalsize_xyz=physicalsize_xyz,
                pixelunit=pixelunit,
                reuse_disk_cache=reuse_disk_cache,
                return_list=False,
                on_error=on_error,
                verbose=verbose)
            if img is None or md is None:
                if verbose:
                    print(f"    Reader returned None for file: {f!r}. Skipping.")
                continue
            images.append(img)
            metadatas.append(md)

        if merge_multiple_files_in_folder:
            if not images:
                if verbose:
                    print("    No readable images found in folder. Abort merging.")
                if return_list:
                    return [None], [None]
                return None, None

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
            zarr_store_path=zarr_store_path,
            return_list=False,
            physicalsize_xyz=physicalsize_xyz,
            pixelunit=pixelunit,
            reuse_disk_cache=reuse_disk_cache,
            on_error=on_error,
            verbose=verbose)
        images.append(img)
        metadatas.append(md)

    if return_list:
        return images, metadatas

    if len(images) == 1:
        return images[0], metadatas[0]

    return images, metadatas
# %% ALL
__all__ = [name for name in globals() if not name.startswith("__")]
# %% END
