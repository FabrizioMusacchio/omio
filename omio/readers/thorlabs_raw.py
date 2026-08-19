""" 
OMIO THORLABS RAW READER

This module provides functions to read Thorlabs RAW files into OMIO's 
canonical representation.

author: Fabrizio Musacchio  
first version: December 2025
ported to modularized structure: August 2026
"""
# %% IMPORTS
from ..core import *
from ..cache import *
# %% THORLABS RAW FUNCTIONS
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

def read_thorlabs_raw(fname, physicalsize_xyz=None, pixelunit="micron",
                      zarr_store=None, zarr_store_path=None, return_list=False,
                      reuse_disk_cache=False, on_error="raise", verbose=True):
    """
    Read Thorlabs RAW files into OMIO's canonical representation.

    This function reads a Thorlabs RAW file and constructs an image array together
    with an OMIO metadata dictionary that follows the canonical OME axis convention
    TZCYX. Dimensions and acquisition metadata are obtained from an accompanying XML
    file in the same folder. If no XML is present, or if the XML is present but
    incomplete or inconsistent, the function falls back to a single YAML metadata
    file located in the same folder.

    The RAW payload is interpreted as a contiguous raster of pixel values that must
    be reshaped into a 5D stack ``(T, Z, C, Y, X)``. If requested, the data are
    materialized into a Zarr array either in memory or on disk. For Zarr output,
    copying is performed slice-wise over the last two spatial dimensions to limit
    peak RAM usage.
    
    **YAML fallback in case of missing or unusable XML:**
    In case no XML metadata file is found, or if the XML metadata file is incomplete
    or inconsistent, the function looks for a YAML file in the same folder. If found,
    it extracts the necessary dimensions and pixel size information from the YAML
    keys ``T``, ``Z``, ``C``, ``Y``, ``X``, ``bits``, ``PhysicalSizeX``,
    ``PhysicalSizeY``, ``PhysicalSizeZ``, and ``pixelunit``.
    
    The YAML file is not generated automatically by OMIO; it must be created
    manually if no usable XML metadata are available.
    
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
        Default is False.
    on_error : {"raise", "return_none"}, optional
        Error policy for unrecoverable Thorlabs metadata problems. ``"raise"``
        preserves the default behavior and raises a ValueError. ``"return_none"``
        emits a warning and returns ``(None, None)`` or ``([None], [None])``
        instead, which is useful for batch pipelines that want to skip unreadable
        RAW files explicitly. Default is ``"raise"``.
    verbose : bool, optional
        If True, print diagnostic progress messages. Default is True.

    Returns
    -------
    tuple
        Returns ``(image, metadata)`` with image data in canonical OME axis order
        TZCYX and the aligned metadata dictionary. If `return_list=True`, returns
        ``([image], [metadata])`` for backward compatibility. If dimensions cannot
        be inferred from XML or YAML, returns ``(None, None)`` or ``([None], [None])``.

    Raises
    ------
    ValueError
        If `zarr_store` is not one of {None, "memory", "disk"}, if `on_error` is not
        one of {"raise", "return_none"}, or if an XML file is present but incomplete
        or inconsistent, no YAML fallback is available, and `on_error="raise"`.
    FileNotFoundError
        If `fname` does not exist.
    ImportError
        If `zarr_store` is "memory" or "disk" but Zarr support is unavailable.

    Notes
    -----
    * RAW reading requires the dimensions T, Z, C, Y, X and a bit depth to infer the
      dtype and reshape the pixel stream. XML metadata is preferred. YAML is used
      if XML is absent or if XML parsing/validation fails.
    * YAML fallback expects at minimum the keys ``T``, ``Z``, ``C``, ``Y``, ``X``,
      and ``bits``. Additional keys such as ``pixelunit``, ``PhysicalSizeX/Y/Z``,
      and ``TimeIncrement`` are optional.
    * For `zarr_store` not None, the function uses ``numpy.memmap`` and slice-wise
      copying to avoid loading the full RAW into RAM before writing.
    * Axis normalization to TZCYX is applied at the end and may insert singleton
      dimensions or reorder axes. The updated axis string and shape are stored in
      the returned metadata.
    * When `zarr_store="disk"`, the function may create and overwrite paths under
      ``.omio_cache``. OMIO metadata and cache validation info are persisted in the
      Zarr attributes so the store can later be reopened without rereading the
      original file.
    """

    if zarr_store not in (None, "memory", "disk"):
        raise ValueError("read_thorlabs_raw: zarr_store must be one of None, 'memory', or 'disk'. "
                         f"Got: {zarr_store!r}")
    if on_error not in ("raise", "return_none"):
        raise ValueError("read_thorlabs_raw: on_error must be one of 'raise' or 'return_none'. "
                         f"Got: {on_error!r}")

    if verbose:
        print(f"Reading Thorlabs RAW file: {fname}")

    if not os.path.exists(fname):
        raise FileNotFoundError(f"The Thorlabs RAW file {fname} does not exist.")

    if zarr_store in ("memory", "disk") and zarr is None:
        raise ImportError("zarr is required for zarr_store='memory' or 'disk'.")

    cache_override = (
        tuple(float(v) for v in physicalsize_xyz)
        if physicalsize_xyz is not None else None
    )

    if zarr_store == "disk" and reuse_disk_cache:
        cached_image, cached_metadata = _try_reuse_disk_cache(
            fname=fname,
            reader_name="raw",
            pixelunit=pixelunit,
            physicalsize_xyz_override=cache_override,
            zarr_store_path=zarr_store_path,
            verbose=verbose,
        )
        if cached_image is not None:
            if verbose:
                print("Finished reading Thorlabs RAW file from reused disk cache.")
            if return_list:
                return [cached_image], [cached_metadata]
            return cached_image, cached_metadata

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

    
    # preferred: XML metadata in same folder. Hidden dot files such as
    # ._Experiment.xml or .Experiment.xml are ignored because they may be macOS
    # sidecar/resource-fork files or accidental editor artifacts.
    xml_files = sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(".xml") and not os.path.basename(f).startswith(".")
    )
    xml_path = None
    xml_error = None
    if xml_files:
        xml_path = os.path.join(folder, xml_files[0])
        if verbose:
            print(f"  Found XML file: {xml_files[0]}. Will use it for metadata extraction...")

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception as e:
            xml_error = e
            root = None

        try:
            if root is None:
                raise xml_error

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
                raise ValueError(
                    f"RAW file size {file_size} is not an integer multiple of "
                    f"X*Y*C*T*bytes_per_pixel={denom}."
                )
            Z_from_file_size = file_size // denom
            if Z_from_file_size <= 0:
                raise ValueError(
                    f"RAW file size {file_size} and XML dimensions imply invalid "
                    f"Z_from_file_size={Z_from_file_size}."
                )
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
            xml_error = e
            if verbose:
                print(f"  WARNING: XML file {xml_path} is incomplete or inconsistent: {e}")
                print("           Will try YAML fallback if available.")

    
    # fallback: YAML metadata in same folder if XML is missing or unusable:
    yaml_path = None
    if dims is None:
        yaml_path = _find_single_yaml(folder)
        if yaml_path is not None:
            if xml_error is not None:
                warnings.warn(
                    f"XML metadata file {xml_path} is incomplete or inconsistent: {xml_error}. "
                    f"Falling back to YAML metadata file {yaml_path}.")
            elif verbose:
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
            unit_from_meta = ymd.get("pixelunit", ymd.get("PixelUnit", None))
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
    
    if dims is None and xml_error is not None and yaml_path is None:
        msg = f"The XML file {xml_path} is incomplete or inconsistent: {xml_error}"
        if on_error == "return_none":
            warnings.warn(msg + " Returning (None, None) because on_error='return_none'.")
            if return_list:
                return [None], [None]
            return None, None
        raise ValueError(msg)

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
    metadata_source = metadata.get("original_metadata_type", "")
    if metadata_source == "thorlabs_yaml_metadata":
        size_mismatch_msg = (
            f"RAW data size mismatch after YAML metadata fallback: expected "
            f"{expected_elements} elements from YAML metadata, got {{actual_elements}}. "
            "Check the YAML dimensions and bit depth.")
    else:
        size_mismatch_msg = (
            f"RAW data size mismatch: expected {expected_elements} elements, "
            "got {actual_elements}. Check XML/YAML metadata.")

    if zarr_store is None:
        if verbose:
            print("  Reading entire Thorlabs RAW file into RAM...")
        with open(fname, "rb") as f:
            raw_data = np.frombuffer(f.read(), dtype=dtype)

        if raw_data.size != expected_elements:
            warnings.warn(size_mismatch_msg.format(actual_elements=raw_data.size))
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
            warnings.warn(size_mismatch_msg.format(actual_elements=raw_data.size))
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
            zarr_cache_path = _get_disk_cache_path(fname, zarr_store_path=zarr_store_path)
            os.makedirs(os.path.dirname(zarr_cache_path), exist_ok=True)
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

    if zarr_store == "disk" and isinstance(image, zarr.core.array.Array):
        cache_info = _build_disk_cache_info(
            fname=fname,
            reader_name="raw",
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
        print("Finished reading Thorlabs RAW file.")

    if return_list:
        return [image], [metadata]
    return image, metadata
# %% ALL
__all__ = [name for name in globals() if not name.startswith("__")]
# %% END
