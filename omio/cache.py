""" 
OMIO DISK-CACHE MANAGEMENT

This module provides functions to manage OMIO's disk cache, including
serialization and deserialization of metadata, and resolution of cache paths.

author: Fabrizio Musacchio  
first version: December 2025
ported to modularized structure: August 2026
"""
# %% IMPORTS
from .core import *
# %% CACHE FUNCTIONS
def _jsonify_for_storage(obj):
    """
    Convert nested objects into JSON-compatible plain Python containers.

    This helper is used before persisting OMIO metadata and cache information into
    Zarr attributes. It recursively converts NumPy scalar types, tuples, lists,
    dicts, and paths into JSON-serializable primitives while leaving ordinary
    Python scalars unchanged.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return [_jsonify_for_storage(v) for v in obj]
    if isinstance(obj, list):
        return [_jsonify_for_storage(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonify_for_storage(v) for k, v in obj.items()}
    if isinstance(obj, os.PathLike):
        return os.fspath(obj)
    return obj

def _restore_cached_metadata(metadata: Dict[str, Any], shape) -> Dict[str, Any]:
    """
    Normalize metadata loaded back from a persisted disk-cache entry.

    Cached metadata are stored through JSON-like Zarr attributes, which means
    tuples may come back as lists. This helper restores the keys that OMIO relies
    on most strongly to their expected runtime representation.
    """
    md = deepcopy(metadata)
    md["shape"] = tuple(shape)
    if "axes" in md and md["axes"] is not None:
        md["axes"] = str(md["axes"])
    return md

def _resolve_zarr_store_parent(zarr_store_path: Union[None, str, os.PathLike]) -> Union[None, str]:
    """
    Resolve a user-provided disk-cache parent location.

    ``zarr_store_path`` points to the parent directory in which OMIO creates
    ``.omio_cache``. Passing the ``.omio_cache`` folder itself is also accepted.
    """
    if zarr_store_path is None:
        return None

    path = os.fspath(zarr_store_path)
    if os.path.basename(os.path.normpath(path)) == ".omio_cache":
        return os.path.dirname(os.path.normpath(path)) or "."
    if os.path.isfile(path):
        return os.path.dirname(path) or "."
    return path

def _get_disk_cache_folder(fname: str,
                           zarr_store_path: Union[None, str, os.PathLike] = None) -> str:
    """
    Return OMIO's disk-cache folder for a source file and optional cache parent.
    """
    cache_parent = _resolve_zarr_store_parent(zarr_store_path)
    if cache_parent is None:
        cache_parent = os.path.dirname(fname)
    return os.path.join(cache_parent, ".omio_cache")

def _get_disk_cache_path(fname: str,
                         suffix: str = "",
                         zarr_store_path: Union[None, str, os.PathLike] = None) -> str:
    """
    Return OMIO's canonical on-disk Zarr cache path for a source file.

    Parameters
    ----------
    fname : str
        Source file path.
    suffix : str, optional
        Optional suffix inserted before the ``.zarr`` extension. This is used for
        derived cache variants such as per-page paginated TIFF outputs.
    zarr_store_path : str, os.PathLike, or None, optional
        Optional parent directory in which OMIO creates ``.omio_cache``. If None,
        the source file's parent directory is used.
    """
    fname_base, _ = os.path.splitext(os.path.basename(fname))
    cache_folder = _get_disk_cache_folder(fname, zarr_store_path=zarr_store_path)
    return os.path.join(cache_folder, fname_base + suffix + ".zarr")

def _get_reader_backend_versions(reader_name: str) -> Dict[str, str | None]:
    """
    Collect backend version metadata relevant to a given OMIO reader.
    """
    versions = {
        "numpy": getattr(np, "__version__", None),
        "zarr": getattr(zarr, "__version__", None),
    }
    if reader_name == "tif":
        versions["tifffile"] = getattr(tifffile, "__version__", None)
    elif reader_name == "czi":
        versions["czifile"] = getattr(czi, "__version__", None)
    elif reader_name == "raw":
        versions["yaml"] = getattr(yaml, "__version__", None)
    return versions

def _build_disk_cache_info(fname: str,
                           reader_name: str,
                           pixelunit: str,
                           physicalsize_xyz_override: tuple[float, float, float] | None,
                           cache_kind: str = "primary") -> Dict[str, Any]:
    """
    Build a cache manifest describing the provenance and validity constraints of a
    disk-backed OMIO Zarr cache.
    """
    stat = os.stat(fname)
    return {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "cache_kind": cache_kind,
        "reader_name": reader_name,
        "source_path": os.path.abspath(fname),
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "pixelunit": pixelunit,
        "physicalsize_xyz_override": (
            [float(v) for v in physicalsize_xyz_override]
            if physicalsize_xyz_override is not None else None
        ),
        "omio_version": _OMIO_VERSION,
        "backend_versions": _get_reader_backend_versions(reader_name),
    }

def _annotate_disk_cache_metadata(metadata: Dict[str, Any],
                                  fname: str,
                                  zarr_path: str,
                                  zarr_store_path: Union[None, str, os.PathLike] = None) -> Dict[str, Any]:
    """
    Record OMIO disk-cache locations in a metadata dictionary.
    """
    metadata["omio_cache_folder"] = os.path.dirname(zarr_path)
    metadata["omio_zarr_store_path"] = zarr_path
    metadata["omio_zarr_store_name"] = os.path.basename(zarr_path)[:-5] if zarr_path.endswith(".zarr") else os.path.basename(zarr_path)
    metadata["omio_zarr_store_type"] = "disk"
    if zarr_store_path is not None:
        metadata["omio_zarr_store_parent"] = _resolve_zarr_store_parent(zarr_store_path)
    return metadata

def _get_zarr_array_store_path(zarr_array: "zarr.core.array.Array",
                               fallback: str) -> str:
    """
    Return the persistent store path for a Zarr array when available.
    """
    try:
        path = str(zarr_array.store_path).replace("file://", "")
        if path:
            return path
    except Exception:
        pass
    return fallback

def _write_disk_cache_payload(zarr_array: "zarr.core.array.Array",
                              metadata: Dict[str, Any],
                              cache_info: Dict[str, Any],
                              verbose: bool = False) -> None:
    """
    Persist OMIO metadata and cache validation info directly into a Zarr store.

    OMIO stores both payloads as Zarr attributes. In current Zarr v3 layouts these
    attributes are serialized into the store's ``zarr.json`` file, which keeps the
    cache self-contained and avoids maintaining a second metadata sidecar file.
    """
    zarr_array.attrs["omio_metadata"] = _jsonify_for_storage(metadata)
    zarr_array.attrs["omio_cache_info"] = _jsonify_for_storage(cache_info)
    if verbose:
        print("  Stored OMIO metadata and cache info in Zarr attrs.")

def _validate_disk_cache_info(cache_info: Dict[str, Any],
                              fname: str,
                              reader_name: str,
                              pixelunit: str,
                              physicalsize_xyz_override: tuple[float, float, float] | None) -> tuple[bool, str]:
    """
    Validate whether a persisted disk-cache manifest matches the current read
    request closely enough for safe reuse.
    """
    try:
        stat = os.stat(fname)
    except Exception as exc:
        return False, f"source stat failed: {exc}"

    expected_override = (
        [float(v) for v in physicalsize_xyz_override]
        if physicalsize_xyz_override is not None else None
    )
    expected = {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "reader_name": reader_name,
        "source_path": os.path.abspath(fname),
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "pixelunit": pixelunit,
        "physicalsize_xyz_override": expected_override,
        "omio_version": _OMIO_VERSION,
        "backend_versions": _get_reader_backend_versions(reader_name),
    }

    for key, expected_value in expected.items():
        actual_value = cache_info.get(key)
        if actual_value != expected_value:
            return False, f"cache manifest mismatch for '{key}'"
    return True, "ok"

def _try_reuse_disk_cache(fname: str,
                          reader_name: str,
                          pixelunit: str,
                          physicalsize_xyz_override: tuple[float, float, float] | None,
                          zarr_store_path: Union[None, str, os.PathLike] = None,
                          verbose: bool = False) -> tuple[Union["zarr.core.array.Array", None], Union[Dict[str, Any], None]]:
    """
    Attempt to reopen and validate an existing OMIO disk cache for a source file.

    Returns ``(None, None)`` when reuse is not possible or not safe.
    """
    zarr_path = _get_disk_cache_path(fname, zarr_store_path=zarr_store_path)
    if not os.path.exists(zarr_path):
        return None, None

    try:
        image = zarr.open(zarr_path, mode="r")
    except Exception as exc:
        if verbose:
            print(f"  Existing disk cache could not be opened. Rebuilding cache. Reason: {exc}")
        return None, None

    if not isinstance(image, zarr.core.array.Array):
        if verbose:
            print("  Existing disk cache is not a Zarr array. Rebuilding cache.")
        return None, None

    cache_metadata = image.attrs.get("omio_metadata")
    cache_info = image.attrs.get("omio_cache_info")
    if cache_metadata is None or cache_info is None:
        if verbose:
            print("  Existing disk cache has no OMIO metadata/cache info. Rebuilding cache.")
        return None, None

    valid, reason = _validate_disk_cache_info(
        cache_info=cache_info,
        fname=fname,
        reader_name=reader_name,
        pixelunit=pixelunit,
        physicalsize_xyz_override=physicalsize_xyz_override,
    )
    if not valid:
        if verbose:
            print(f"  Existing disk cache is stale or incompatible. Rebuilding cache. Reason: {reason}")
        return None, None

    metadata = _restore_cached_metadata(cache_metadata, image.shape)
    if "axes" not in metadata:
        if verbose:
            print("  Existing disk cache metadata are incomplete. Rebuilding cache.")
        return None, None

    if verbose:
        print(f"  Reusing existing OMIO disk cache: {zarr_path}")
    return image, metadata

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
      ``.omio_cache`` folder under that directory is removed. Passing the
      ``.omio_cache`` folder itself is also supported.
    
    Parameters
    ----------
    fname : str
        Path to a file whose cache should be removed, a directory containing an
        ``.omio_cache`` folder to be cleaned, or the ``.omio_cache`` folder itself.
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
        fname_norm = os.path.normpath(fname)
        if os.path.basename(fname_norm) == ".omio_cache":
            parent_folder = os.path.dirname(fname_norm)
            omio_cache_folder = fname_norm
        else:
            parent_folder = fname
            omio_cache_folder = os.path.join(parent_folder, ".omio_cache")
        base_name = None
    else:
        raise ValueError(f"cleanup_omio_cache: {fname} is neither a file nor a folder.")

    if os.path.isfile(fname):
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
# %% ALL
__all__ = [name for name in globals() if not name.startswith("__")]
# %% END
