""" 
OMIO BATCH PROCESSOR

This module provides functions to perform batch processing of image files
using OMIO's readers and writers.

author: Fabrizio Musacchio  
first version: December 2025
ported to modularized structure: August 2026
"""
# %% IMPORTS
from .core import *
from .cache import cleanup_omio_cache
from .read import imread
from .writers.ome_tiff import imwrite
from .convert import imconvert
# %% BATCH FUNCTIONS
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
    reuse_disk_cache: bool = False,
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
    
    **Abstract expected folder scheme:**
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
    

    **Folder discovery and selection:**
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

    **Conversion behavior inside each experiment folder:**
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

    **Input path semantics:**
    Only directory input is accepted:
    
    * ``fname`` must be an existing directory and is treated as the project root.
    * All outputs are written within the experiment scope determined by traversal.

    **Output placement and naming:**
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

    **Merging semantics:**
    * ``merge_along_axis`` must be one of {"T","Z","C"}.
    * In merge operations, the merge axis segments are concatenated in discovery order.
    * If ``zeropadding=True``, non-merge axes may differ between inputs and will be
      padded with zeros to the maximum size across inputs before concatenation.
      
      If ``zeropadding=False``, non-merge axes must match exactly or the merge is aborted.

    **Zarr and cache handling:**
    * ``zarr_store`` controls whether intermediate data are represented as NumPy in RAM
      or as Zarr arrays ("memory" or "disk") during reading and merging.
    * If ``reuse_disk_cache=True`` together with ``zarr_store="disk"``, OMIO may reuse
      an already existing validated disk cache instead of rebuilding it from the
      original image file.
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
        If `return_fnames` is True, returns a list of written OME-TIFF file paths.
        Otherwise returns None. The list may be empty if nothing matched or all
        conversions failed.

    Raises
    ------
    ValueError
        If `fname` is not an existing directory, or if `merge_along_axis` is not one
        of {"T", "Z", "C"}.
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
                        reuse_disk_cache=reuse_disk_cache,
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
                            reuse_disk_cache=reuse_disk_cache,
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
# %% ALL
__all__ = [name for name in globals() if not name.startswith("__")]
# %% END
