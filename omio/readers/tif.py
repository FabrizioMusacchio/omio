""" 
OMIO TIFF/LSM READER

This module provides functions to read TIFF, OME-TIFF, multi-file 
OME-TIFF series, and Zeiss LSM files into OMIO's canonical representation.

author: Fabrizio Musacchio  
first version: December 2025
ported to modularized structure: August 2026
"""
# %% IMPORTS
from ..core import *
from ..cache import *
# %% TIFF FUNCTIONS
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

def read_tif(fname, physicalsize_xyz=None, pixelunit="micron", 
             zarr_store=None, zarr_store_path=None, return_list=False,
             reuse_disk_cache=False, verbose=True):
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
    zarr_store_path : str, os.PathLike, or None, optional
        Parent directory in which OMIO creates ``.omio_cache`` when
        ``zarr_store="disk"``. If None, the cache is created next to the source
        file as before. Passing the ``.omio_cache`` folder itself is also
        accepted. Default is None.
    return_list : bool, optional
        If True, force backward-compatible list return for non-paginated inputs by
        returning ``[image]`` and ``[metadata]``. Default is False.
    reuse_disk_cache : bool, optional
        If True and ``zarr_store="disk"``, OMIO first checks whether a compatible
        on-disk cache already exists and reuses it instead of rebuilding the Zarr
        store from the original TIFF. The existing cache is reused only if its
        persisted manifest matches the current source file and read settings.
        Default is False.
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
      ``.omio_cache``. OMIO metadata and cache validation info are persisted in the
      Zarr attributes so the store can later be reopened without rereading the
      original file.
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
    cache_override = physicalsize_xyz_ext if set_input_pixelsize else None

    if zarr_store == "disk" and reuse_disk_cache:
        cached_image, cached_metadata = _try_reuse_disk_cache(
            fname=fname,
            reader_name="tif",
            pixelunit=pixelunit,
            physicalsize_xyz_override=cache_override,
            zarr_store_path=zarr_store_path,
            verbose=verbose,
        )
        if cached_image is not None:
            if "P" in cached_metadata.get("axes", ""):
                return _split_paginated_tiff_stack(
                    cached_image,
                    cached_metadata,
                    fname=fname,
                    zarr_store=zarr_store,
                    zarr_store_path=zarr_store_path,
                    verbose=verbose,
                )
            if verbose:
                print("Finished reading TIFF from reused disk cache.")
            if return_list:
                return [cached_image], [cached_metadata]
            return cached_image, cached_metadata

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
                zarr_cache_path = _get_disk_cache_path(fname, zarr_store_path=zarr_store_path)
                os.makedirs(os.path.dirname(zarr_cache_path), exist_ok=True)
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
                cache_folder = _get_disk_cache_folder(fname, zarr_store_path=zarr_store_path)
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

            if zarr_store == "disk" and isinstance(image, zarr.core.array.Array):
                cache_info = _build_disk_cache_info(
                    fname=fname,
                    reader_name="tif",
                    pixelunit=pixelunit,
                    physicalsize_xyz_override=cache_override,
                    cache_kind="primary",
                )
                metadata = _annotate_disk_cache_metadata(
                    metadata,
                    fname=fname,
                    zarr_path=_get_zarr_array_store_path(
                        image,
                        _get_disk_cache_path(fname, zarr_store_path=zarr_store_path)),
                    zarr_store_path=zarr_store_path)
                _write_disk_cache_payload(image, metadata, cache_info, verbose=verbose)
            return _split_paginated_tiff_stack(
                image,
                metadata,
                fname=fname,
                zarr_store=zarr_store,
                zarr_store_path=zarr_store_path,
                verbose=verbose,
            )

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

        if zarr_store == "disk" and isinstance(image, zarr.core.array.Array):
            cache_info = _build_disk_cache_info(
                fname=fname,
                reader_name="tif",
                pixelunit=pixelunit,
                physicalsize_xyz_override=cache_override,
                cache_kind="primary",
            )
            metadata = _annotate_disk_cache_metadata(
                metadata,
                fname=fname,
                zarr_path=_get_zarr_array_store_path(
                    image,
                    _get_disk_cache_path(fname, zarr_store_path=zarr_store_path)),
                zarr_store_path=zarr_store_path)
            _write_disk_cache_payload(image, metadata, cache_info, verbose=verbose)
        
        if verbose:
            print("Finished reading TIFF.")
            
        if return_list:
            return [image], [metadata]
        else:
            return image, metadata
# %% ALL
__all__ = [name for name in globals() if not name.startswith("__")]
# %% END
