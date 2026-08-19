""" 
OMIO CONVERT MODULE
This module provides functions to convert microscopy image 
files to OME-TIFF format.

author: Fabrizio Musacchio  
first version: December 2025
ported to modularized structure: August 2026
"""
# %% IMPORTS
from .core import *
from .cache import cleanup_omio_cache
from .read import imread
from .writers.ome_tiff import imwrite
# %% CONVERT FUNCTIONS
def imconvert(fname: Union[str, os.PathLike, List[Union[str, os.PathLike]]],
         zarr_store: Union[None, str] = None,
         reuse_disk_cache: bool = False,
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

    **Input path semantics (inherited from `imread(...)`):**
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

    **Merge behavior:**
    Merge operations follow the same validation and padding rules as in `imread(...)`:
    
    * Allowed merge axes are "T", "Z", and "C".
    * If `zeropadding=False`, all non merge axes must match exactly.
    * If `zeropadding=True`, non merge axes are padded with zeros to the maximum size
      across inputs before concatenation.

    **Output behavior:**
    The output location and naming follow `imwrite(...)`:
    
    * OME TIFFs are written next to the input file or inside the input folder.
    * If `relative_path` is provided, a subfolder is created under the chosen output
      parent directory.
    * When merge modes are used, output filenames may include an indicator suffix to
      reflect merged content.
    * If `overwrite=False`, existing files are not replaced and collision safe names
      are generated.

    **Zarr handling and cache cleanup:**
    If `zarr_store` is "memory" or "disk", `imread(...)` may create Zarr arrays or
    materialize intermediate Zarr stores under a hidden `.omio_cache` directory.
    If `reuse_disk_cache=True` together with ``zarr_store="disk"``, existing validated
    OMIO disk caches may be reopened instead of rebuilt from the original source files.
    If `cleanup_cache=True`, this function removes the corresponding cache entries
    after writing. Cache cleanup is skipped when `zarr_store=None`.

    Parameters
    ----------
    fname : str, os.PathLike, or list of such
        File path, folder path, or list of file paths to convert.
    zarr_store : {None, "memory", "disk"}, optional
        Controls whether `imread(...)` returns NumPy arrays (None) or Zarr arrays
        ("memory" or "disk"). Default is None.
    reuse_disk_cache : bool, optional
        Forwarded to `imread(...)`. If True and ``zarr_store="disk"``, existing
        validated OMIO disk caches may be reused instead of rebuilt. Those caches
        persist OMIO metadata and cache manifests directly in the Zarr store.
        Default is False.
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
    list[str] or None
        If `return_fnames` is True, returns a list of output OME-TIFF paths in the
        order processed. Otherwise returns None.

    Raises
    ------
    ValueError
        If invalid merge options are provided.
    FileNotFoundError
        If an input file does not exist.
    Exception
        Reader and writer errors may propagate during I/O or metadata handling.
    """


    if verbose:
        print(f"Converting to OME-TIFF: {fname!r}")
    #print(f"Reading input...")
    images, metadatas = imread(
        fname=fname,
        zarr_store=zarr_store,
        reuse_disk_cache=reuse_disk_cache,
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
# %% ALL
__all__ = [name for name in globals() if not name.startswith("__")]
# %% END
