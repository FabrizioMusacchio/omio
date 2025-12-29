""" 
This script contains usage examples for the interactive use of OMIO. You can adapt these
examples for your own scripts or Jupyter notebooks. Using VS Code, you can run this script
cell-by-cell in an interactive Python window.


How to install OMIO
---------------------

```bash
conda create -n omio python=3.12 -y
conda activate omio
pip install omio-microscopy
```

Example dataset
-------------------
To run the examples in this script, you may use your own image data or download our provided
example dataset from Zenodo: https://doi.org/10.5281/zenodo.18078231

The paths defined in this script are adapted to the following folder structure:

.. code-block:: text
    example_dataset/
    ├── example_folder_1        # <- example data folder (PLEASE ADAPT)
    │   ├── example_image.tif   # <- example image file (PLACEHOLDER)
    ├── example_folder_2
    └── ...
    scripts/
    ├── usage_example_interactive.py  # <- this script

Thus, to point to the example dataset from this script, we define the path relative to the 
script location (e.g., "../example_dataset/example_image.tif").

Script information
----------------------
author: Fabrizio Musacchio
date: December 2025
"""
# %% IMPORTS
import omio as om
import pprint
# %% HELLO WORLD
""" 
OMIO has a simple hello world function to verify that the installation was successful. The
following command should print "Hello from omio.py! OMIO version: 0.1.0" (note that the version
number may vary depending on the installed version):
"""
om.hello_world()
# %% SINGLE FILE READING AND METADATA INSPECTION
""" 
To open a single file such as a TIFF file, use the `imread` function. This function returns the image data
as a NumPy array (as default) along with the associated metadata as a dictionary:
"""
fname = "../example_data/tif_cell_single_tif/13374.tif"
image, metadata = om.imread(fname)
print(f"Image shape: {image.shape}")

""" 
`imread` automatically interprets the OME metadata stored in the TIFF file and
re-arranges the image axes to follow the OME axis order convention:
`(T)ime, (C)hannel, (Z)depth, (Y)height, (X)width`. If any of these axes are singleton
(i.e., size 1), they are retained in the returned image array to preserve the full
5D structure to ensure OME compliance and, thus, compatibility with downstream OME-based
pipelines.

`imread` always returns the read image data (as a NumPy array by default) and the 
associated metadata as a dictionary. The metadata dictionary contains OME-relevant
entries such as PhysicalSizeX, PhysicalSizeY, PhysicalSizeZ, TimeIncrement, Channels. 
OMIO will always assign these entries and tries to infer missing metadata from the
available information in the file or by assigning pre-defined defaults (which can be
customized by the user upon function call).

Let's inspect some of the read metadata:
"""

print(f"Metadata keys: {list(metadata.keys())}")
pprint.pprint(metadata)

""" 
You may notice that `imread` has, apart from the correct and OME-compliant axis order,
PhysicalSizeX, PhysicalSizeY, and PhysicalSizeZ entries in microns (and other OME defaults), 
also added an entry called `"Annotations"` that contains additional metadata parsed from the
TIFF file. OMIO tries to extract as much metadata as possible from the file and store it
in a structured manner in the metadata dictionary. Any non-OME metadata is stored under the
`"Annotations"` key to avoid conflicts with standard OME entries, but also to preserve
valuable metadata that may be useful for downstream processing.

Of course, you can always add or change metadata entries as needed. 

Let's add an "Experimenter" entry to the metadata dictionary:
"""

metadata["Experimenter"] = "Your Name"
pprint.pprint(metadata)

""" 
If would save `image` and its associated `metadata` back to an OME-TIFF file, our additional
"Experimenter" entry would not be written, as it is not part of the OME standard. However,
OMIO offers a check-up function called `OME_metadata_checkup()` that normalizes the metadata
dictionary to be fully OME-compliant by moving any non-OME entries under the `"Annotations"` key:
"""
metadata = om.OME_metadata_checkup(metadata)
pprint.pprint(metadata)

# %% OPEN IN NAPARI AND METADATA MODIFICATION
""" 
OMIO comes with built-in support to open images directly in Napari for interactive. Let's
open the previously read image in Napari:
"""
om.open_in_napari(image, metadata, fname)

""" 
For the sake of demonstration, we change the PhysicalSizeZ metadata entry to a wrong value
and re-open the image in Napari to see that Napari correctly rescales the Z axis based on
the provided metadata:
"""
print(f"Original PhysicalSizeZ: {metadata['PhysicalSizeZ']} microns")
metadata["PhysicalSizeZ"] = 5  # wrong value in microns
print(f"Modified PhysicalSizeZ: {metadata['PhysicalSizeZ']} microns")
om.open_in_napari(image, metadata, fname)

""" 
If you don't want to see terminal outputs from OMIO, you can set `verbose=False` in any
OMIO function call. For example:
"""

om.open_in_napari(image, metadata, fname, verbose=False)

# %% ENSURED OME-COMPLIANCE UPON READING
"""
OMIO ensures OME-compliance of the read image and metadata upon reading. This accounts
regardless of whether the input file is already OME-compliant or not, has incomplete OME
metadata, or even does not contain any OME metadata at all. It also accounts for non-OME
files such as Zeiss CZI files or Thorlabs RAW files. And it doesn't matter whether the 
input image is 2D (XY), 3D (Z-stack), 4D (time-lapse or multi-channel), or 5D
(time-lapse multi-channel Z-stack) data.

Let's try this out with a few more example files. the example data folder 

`tif_dummy_data/tif_single_files`

contains a few TIFF files with different dimensionalities. They are all generated with the
utility script `additional_scripts/generate_dummy_tif_files.py` using
tifffile. They do not contain any specific metadata declarations such like ImageJ Hyperstack
compatibility or OME-TIFF metadata:
"""

fname_5d = "../example_data/tif_dummy_data/tif_single_files/TZCYX_T5_Z10_C2.tif"
image_5d, metadata_5d = om.imread(fname_5d)
print(f"5D Image shape: {image_5d.shape} with axes {metadata_5d.get('axes', 'N/A')}")
pprint.pprint(metadata_5d)
om.open_in_napari(image_5d, metadata_5d, fname_5d)

fname_2d = "../example_data/tif_dummy_data/tif_single_files/YX.tif"
image_2d, metadata_2d = om.imread(fname_2d)
print(f"2D Image shape: {image_2d.shape} with axes {metadata_2d.get('axes', 'N/A')}")
pprint.pprint(metadata_2d)
om.open_in_napari(image_2d, metadata_2d, fname_2d)

"""
As you can see, OMIO correctly infers the OME-compliant axes and adds default OME metadata
entries as needed. Let's try out other tif files with different tifffile-specific metadata
declarations, such as ImageJ Hyperstack compatibility. Note, that these files contain 
additional to their actual axes (T, Z, C, Y, or X) also singleton axes (S), which is 
required for ImageJ Hyperstack compatibility:
"""

fname_4d = "../example_data/tif_dummy_data/tif_with_ImageJ/TYXS_T1.tif"
image_4d, metadata_4d = om.imread(fname_4d)
print(f"4D Image shape: {image_4d.shape} with axes {metadata_4d.get('axes', 'N/A')}")
pprint.pprint(metadata_4d)
om.open_in_napari(image_4d, metadata_4d, fname_4d)

fname_6d = "../example_data/tif_dummy_data/tif_with_ImageJ/TZCYXS_C1_Z10_T2.tif"
image_6d, metadata_6d = om.imread(fname_6d)
print(f"6D Image shape: {image_6d.shape} with axes {metadata_6d.get('axes', 'N/A')}")
pprint.pprint(metadata_6d)
om.open_in_napari(image_6d, metadata_6d, fname_6d)

""" 
Note, that due to the extra singleton axes, the artificially created tif files were saved
with photometric interpretation "rgb" instead of "minisblack" and `imread` interprets them
as three channel images. If the image additionally contains more than one channel axis,
this results in multiple channel axes in the read image. This is not a bug, but OMIO 
always tries to retain the full dimensionality of the read image to avoid any loss of
information:
"""

fname_6d = "../example_data/tif_dummy_data/tif_with_ImageJ/TZCYXS_T5_Z10_C2.tif"
image_6d, metadata_6d = om.imread(fname_6d)
print(f"6D Image shape: {image_6d.shape} with axes {metadata_6d.get('axes', 'N/A')}")
pprint.pprint(metadata_6d)
om.open_in_napari(image_6d, metadata_6d, fname_6d)

""" 
Let's also open an OME-TIFF file:
"""

fname_ometiff = "../example_data/tif_dummy_data/ome_tif/TZCYX_T5_Z10_C2.ome.tif"
image_ometiff, metadata_ometiff = om.imread(fname_ometiff)
print(f"OME-TIFF Image shape: {image_ometiff.shape} with axes {metadata_ometiff.get('axes', 'N/A')}")
pprint.pprint(metadata_ometiff)
om.open_in_napari(image_ometiff, metadata_ometiff, fname_ometiff)

# %% ENSURED OME-COMPLIANCE UPON WRITING
""" 
OMIO's writing function `imwrite` also ensures OME-compliance of the written image
and metadata:
"""

fname_2d = "../example_data/tif_dummy_data/tif_single_files/YX.tif"
image_2d, metadata_2d = om.imread(fname_2d)
print(f"2D Image shape: {image_2d.shape} with axes {metadata_2d.get('axes', 'N/A')}")

om.imwrite(fname_2d, image_2d, metadata_2d, relative_path="omio_converted")

""" 
`imwrite` requires as minimum the image data, the associated metadata dictionary, 
and the output file name. For teh latter, you can hand over the original file name. By
default, `overwrite` is set to `False`, so that any existing file with the same name
will not be overwritten (OMIO appends a number to the file name instead). Additionally, 
you can provide a `relative_path` argument to write the converted OME-TIFF file into a
sub-folder of the input file's folder. This further ensures that the original file is not
overwritten and it helps in keeping converted files organized. The newly created file
gets the file extension `.ome.tif` to indicate that it is an OME-TIFF file.

Let's inspect the written OME-TIFF file:
"""

fname_2d_written = "../example_data/tif_dummy_data/tif_single_files/omio_converted/YX.ome.tif"
image_2d_written, metadata_2d_written = om.imread(fname_2d_written)
print(f"Written 2D Image shape: {image_2d_written.shape} with axes {metadata_2d_written.get('axes', 'N/A')}")
pprint.pprint(metadata_2d_written)
om.open_in_napari(image_2d_written, metadata_2d_written, fname_2d_written)

""" 
Of course, you can open the written OME-TIFF file in any OME-compliant software such as
ImageJ/Fiji by dragging and dropping the file into the application window, or by using
FIJI's Bio-Formats Importer. In both cases, FIJI will correctly interpret the OME metadata
and display the image with the correct axes and scalings, except for one limitation: When
dragging and dropping the OME-TIFF file into FIJI, FIJI does not interpret the physical unit
`microns` correctly and displays "pixels" instead. This is a know limitation of FIJI's SCIFIO
library, which jumps in when opening files via drag and drop. Using Bio-Formats Importer
correctly interprets the physical unit as `microns`.
"""

# %% OMIO'S IMCONVERT FUNCTION
""" 
OMIO also provides a convenience function called `imconvert` that combines reading and writing:
"""

fname_5d = "../example_data/tif_dummy_data/tif_single_files/TZCYX_T5_Z10_C2.tif"
om.imconvert(fname_5d, relative_path="omio_converted")

""" 
`imconvert` accepts all arguments of both `imread` and `imwrite`, allowing you to
customize the reading and writing behavior as needed.

`imconvert` has an additional optional argument called `return_fnames` (default: `False`),
which if set to `True`, returns the output file names upon conversion for further downstream
processing:
"""

output_fnames = om.imconvert(fname_5d, relative_path="omio_converted", return_fnames=True)
print(f"Converted file names: {output_fnames}")


# %% READING LSM FILES
"""
`imread` and its associated reading function `read_tif` is based on the `tifffile` library,
which also supports reading of Zeiss LSM files. Thus, you can read LSM files directly with
`imread` as well:
"""

fname_lsm = "../example_data/lsm_test_file/032113-18.lsm"
image_lsm, metadata_lsm = om.imread(fname_lsm)
print(f"LSM image shape: {image_lsm.shape}")
pprint.pprint(metadata_lsm)
om.open_in_napari(image_lsm, metadata_lsm, fname_lsm)

# %% READING CZI FILES
"""
OMIO also supports reading of Zeiss CZI files via the `imread` function, which internally
calls the `read_czi` function based on the `czifile` library:
"""

fname_czi = "../example_data/czi_test_file/xt-scan-lsm980.czi"
image_czi, metadata_czi = om.imread(fname_czi)
print(f"CZI image shape: {image_czi.shape}")
pprint.pprint(metadata_czi)
om.open_in_napari(image_czi, metadata_czi, fname_czi)

# %% READING THORLABS RAW FILES
"""
OMIO supports reading of Thorlabs RAW files via the `imread` function, which internally
calls the `read_thorlabs_raw` function. `read_thorlabs_raw` is a custom OMIO function. 
For older Python versions (<=3.9), the PyPI package `utils2p` was a common solution to read
Thorlabs RAW files, but this package is no longer maintained and does not support Python
3.10 and above. Thus, OMIO provides its own implementation to read Thorlabs RAW files.
"""

fname_raw = "../example_data/thorlabs_dummy_data/case_C2_Z10_T5/example_C2_Z10_T5.raw"
""" 
This folder contains dummy Thorlabs RAW files generated with the script 
`additional_scripts/generate_thorlabs_dummy_raws.py`. It also contains
the associated example XML files required to read the RAW files correctly. Note: reading
Thorlabs RAW files always requires both the RAW file and its associated XML file to be present.
"""
image_raw, metadata_raw = om.imread(fname_raw)
print(f"Thorlabs RAW image shape: {image_raw.shape}")
pprint.pprint(metadata_raw)
om.open_in_napari(image_raw, metadata_raw, fname_raw)

""" 
If the according XML file is missing or cannot be found, `imread` will give a warning 
and returns None for both image and metadata:
"""
fname_raw = "../example_data/thorlabs_dummy_data/case_C2_Z10_T5_missing_xml/example_C2_Z10_T5.raw"
image_raw, metadata_raw = om.imread(fname_raw)

""" 
In such cases, you can provide a YAML file with the required metadata as a fallback. The YAML
file must be located in the same folder as the RAW file and has the following structure:
  T: 1  
  Z: 1  
  C: 1  
  Y: 512  
  X: 512  
  bits: 16  
  pixelunit: micron  
  PhysicalSizeX: 0.5  
  PhysicalSizeY: 0.5  
  PhysicalSizeZ: 1.0  
  TimeIncrement: 1.0  
  TimeIncrementUnit: seconds  
  
Avoid placing more than one YAML file in the same folder as the RAW file to prevent ambiguity.

You can also use OMIO's utility function `create_thorlabs_raw_yaml(fname)` to create an empty YAML
template that you can fill in manually. It will be created in the same folder as the RAW file.
The function uses default values for the metadata entries, which you can then modify as needed:
"""
fname_raw = "../example_data/thorlabs_dummy_data/case_C2_Z10_T5_yaml/example_C2_Z10_T5.raw"
om.create_thorlabs_raw_yaml(fname_raw, T=5, Z=10, C=2, Y=20, X=20, bits=16,
                            pixelunit="micron", physicalsize_xyz=(0.5, 0.5, 1.0),
                            time_increment=1.0, time_increment_unit="seconds")
image_raw, metadata_raw = om.imread(fname_raw)
om.open_in_napari(image_raw, metadata_raw, fname_raw)

# %% READING MULTI-SERIES TIFF STACKS
"""
OMIO's `imread` function also supports reading of multi-series TIFF and LSM stacks, however,
with some limitations.

TIFF and LSM containers may store multiple datasets ("series") in a single file.
While tifffile exposes these as tif-series, OMIO enforces a strict and predictable
policy to avoid ambiguous interpretations:

* If a file contains exactly one series (`len(tif.series) == 1`), OMIO guarantees
  correct reading and normalization to canonical OME axis order (TZCYX).
* If a file contains multiple series (`len(tif.series) > 1`), OMIO will process
  **only the first series (series 0)** and ignore all others.  A warning is emitted 
  in this case, and the policy decision is recorded in the returned metadata.
* OMIO does not attempt to infer relationships between multiple series, does not
  concatenate them, and does not inspect their shapes, axes, or photometric
  interpretation beyond series 0.

This policy is intentional and favors reproducibility and explicit behavior over
heuristic reconstruction of complex TIFF layouts.
"""

fname_multi_series = "../example_data/tif_dummy_data/multiseries_tif/multiseries_rgb_with_equal_shapes.tif"
image_multi_series, metadata_multi_series = om.imread(fname_multi_series)
pprint.pprint(metadata_multi_series)

""" 
Inspecting the "Annotations" in the retrieved metadata shows that OMIO has detected a multi-series
TIFF file (`'OMIO_MultiSeriesDetected': True`) which initially contained 2 series with axes
`'OMIO_MultiSeriesAxes': ['YXS', 'YXS']` and shapes `'OMIO_MultiSeriesShapes': [[16, 16, 3], [16, 16, 3]],`.
Thus, the two series seem to be compatible for concatenation along a new axis. However, OMIO 
does not infer - by intention - any such relationships and only reads the first series (series 0) with
shape `(16, 16, 3)` and axes `YXS`. The reason for this policy is to avoid ambiguous interpretations
of multi-series TIFF files, which may contain series with different dimensionalities, axes,
or photometric interpretations: 
"""

fname_multi_series = "../example_data/tif_dummy_data/multiseries_tif/multiseries_rgb_with_unequal_series.tif"
image_multi_series, metadata_multi_series = om.imread(fname_multi_series)
pprint.pprint(metadata_multi_series)

fname_multi_series = "../example_data/tif_dummy_data/multiseries_tif/multiseries_rgb_minisblack_mixture.tif"
image_multi_series, metadata_multi_series = om.imread(fname_multi_series)
pprint.pprint(metadata_multi_series)

fname_multi_series = "../example_data/tif_dummy_data/multiseries_tif/multiseries_minisblack.tif"
image_multi_series, metadata_multi_series = om.imread(fname_multi_series)
pprint.pprint(metadata_multi_series)


fname_multi_series = "../example_data/tif_dummy_data/multiseries_tif/multiseries_TCYXS.ome.tif"
image_multi_series, metadata_multi_series = om.imread(fname_multi_series)
pprint.pprint(metadata_multi_series)

""" 
If you want to process all series in a multi-series TIFF file, you have to manually separate them
with tools such as ImageJ/Fiji and store each series in its own single-series TIFF file.
"""

# %% READING PAGINATED TIFF STACKS
"""
OMIO's `imread` function also supports reading of paginated LSM stacks, that contain multiple pages
or tiles stored sequentially. OMIO's policy here is: Each page/tile is treated as a separate image 
stack and the returned image becomes a list of images and a list of metadata dictionaries, one for 
each page. This allows for flexible handling of paginated stacks, where each page may have different 
dimensionalities, axes, or metadata:
"""

fname_paginated = "../example_data/tif_dummy_data/paginated_tif/paginated_tif.tif"
images, metadata_paginated = om.imread(fname_paginated)

print(f"Number of pages read: {len(images)}")

for i, (img, meta) in enumerate(zip(images, metadata_paginated)):
    print(f"Page {i}: shape={img.shape}, axes={meta.get('axes', 'N/A')}")

pprint.pprint(metadata_paginated[0])
pprint.pprint(metadata_paginated[1])
pprint.pprint(metadata_paginated[2])

""" 
Note: `imread` has an optional argument `return_list` which is set to `False` by default. If
set to `True`, `imread` will always return a list of images and a list of metadata dictionaries,
even if the input file contains only a single page. This can be useful for consistent handling
of paginated stacks in batch processing scenarios.
"""


# %% READING MULTI-FILE OME-TIFF STACKS
"""
A multi-file OME-TIFF series consists of multiple TIFF files, each representing a single
time point, channel, or Z-slice of a larger multidimensional dataset. OMIO supports reading such
multi-file OME-TIFF series via the `imread` function by providing the file name of any
of the individual TIFF files in the series. OMIO will automatically detect and read all
files in the series, sort them correctly based on their OME metadata, and assemble them
into a single multidimensional NumPy array along with the associated OME-compliant metadata:
"""

fname_multifile_ometiff = "../example_data/tif_dummy_data/tif_ome_multi_file_series/TZCYX_T5_Z10_C2_Z00_C0_T0.ome.tif"
image_multifile_ometiff, metadata_multifile_ometiff = om.imread(fname_multifile_ometiff)
print(f"Multi-file OME-TIFF image shape: {image_multifile_ometiff.shape}")
pprint.pprint(metadata_multifile_ometiff)
om.open_in_napari(image_multifile_ometiff, metadata_multifile_ometiff, fname_multifile_ometiff)

""" 
Note: This only works for multi-file OME-TIFF series where each individual TIFF file contains
the necessary OME metadata to correctly sort and assemble the files into a multidimensional
dataset. You can not simply provide a list of arbitrary TIFF files and expect OMIO to
assemble them correctly without the required OME metadata, even though the single TIFF files'
names may contain hints about their position in the series (e.g., Z-slice or time point).
"""

# %% ---------------------------------------------------------
# %% READ LARGE FILES LAZILY WITH ZARR BACKEND
"""
OMIO supports reading large image files that do not fit into memory by Zarr-backed lazy loading
and optional memory mapping on disk. To read a large TIFF file lazily, use the `imread` function
with the `zarr_store="memory"` or `zarr_store="disk"` argument:
"""

fname = "../example_data/tif_large_Ca_imaging_large/1MP_SIMPLE_Stephan__001_001.tif"
image_lazy, metadata_lazy = om.imread(fname, zarr_store="memory")
print(f"Lazy image shape: {image_lazy.shape}")
print(f"Lazy image type: {type(image_lazy)}")
image_lazy

# You can now manipulate `image_lazy` as a Zarr array without loading the entire dataset into memory.
# For example, you can read a small chunk of the data:
sub_stack = image_lazy[0, 0:10, 0:100, 0:100]
print(f"Sub-stack shape: {sub_stack.shape}")    

""" 
With `zarr_store="disk"`, OMIO creates a temporary Zarr store on disk and memory-maps 
the data for efficient access. The default location of the temporary Zarr store fname's
parent directory, where a folder called `.omio_cache` is created to hold the temporary 
data:
"""

image_lazy_memmap, metadata_lazy_memmap = om.imread(fname, zarr_store="disk")
print(f"Lazy memmap image shape: {image_lazy_memmap.shape}")
print(f"Lazy memmap image type: {type(image_lazy_memmap)}")
image_lazy_memmap

om.open_in_napari(image_lazy_memmap, metadata_lazy_memmap, fname)

""" 
Note: If you have opened an image with Napari in the same interactive session before, OMIO
will reuse the existing Napari viewer instance to avoid opening multiple windows. In practice,
any new image opened with `om.open_in_napari` will be added as a new layer to the existing
Napari viewer.
"""

# There is by intention no automatic cleanup of the temporary Zarr stores (the user
# may want to reuse them for any downstream processing). To manually cleanup the
# temporary Zarr stores created by OMIO, use the following function:
om.cleanup_omio_cache(fname, full_cleanup=False)  # set full_cleanup=True to remove the entire .omio_cache folder
om.cleanup_omio_cache(fname, full_cleanup=True)

# %% EFFICIENTLY VIEW LARGE IMAGES IN NAPARI WITH OMIO'S DASK SUPPORT
"""
To efficiently view large images in Napari without loading the entire dataset into memory,
you can use OMIO's built-in support for lazy loading and combine it with OMIO's Napari
integration, which supports a) handling of in-memory and on-disk memory-mapped Zarr arrays,
b) automatic axis reordering based on OME semantics, and c) DASK support for out-of-core
parallel processing.
"""

# this file is 1.1 GB 3D stacks with multiple channels:
fname = "../example_data/tif_files_from_3P_paper/Supplementary_Video_4.tif"
""" 
File name: Supplementary Video 4
Description: In-vivo 3P Cortex to Hippocampus z-scan of 265 x-y frames from surface 
to 1325 um below taken at a depth increment of 5 um with 1300nm excitation in a GFP.M.:Cx3cr1-
CreERuRosa25_tdTomato transgenic mouse.
"""

# let's first memory-map the image on disk:
image_large, metadata_large = om.imread(fname, zarr_store="disk")

""" 
This stack has stored an erroneous PhysicalSizeZ in its ImageJ metadata, which is set to
0.0000185 microns instead of the correct value of 5 microns according to the SI description
of the paper. Thus, let's correct the corresponding metadata entry so that Napari can
correctly scale the Z axis upon viewing:
"""
metadata_large["PhysicalSizeZ"] = 5  # in microns

# Now open the large image in Napari. First, we do it w/o DASK (`zarr_mode="zarr_nodask"`):
om.open_in_napari(image_large, metadata_large, fname, zarr_mode="zarr_nodask")

"""
Internally, OMIO's Napari viewing function will correctly handle the true image scalings
and axes, but need to re-arrange the axes to the Napari expected order. Without DASK,
this may take some time for very large images, as a temporary Zarr store is created
with the re-ordered axes. This temporary store is created in the same `.omio_cache` folder
as before. To speed up this process, OMIO has an option `zarr_mode="zarr_dask"` to use DASK 
for parallelized re-ordering and writing of the temporary Zarr store.
"""

# and now with DASK support for out-of-core processing:
om.open_in_napari(image_large, metadata_large, fname, zarr_mode="zarr_dask")

""" 
With `returns=True`, the Napari viewer instance, the created Napari layers, the used
Zarr array, and the used axes order are also returned for further programmatic use:
"""
napari_viewer, napari_layers, napari_datas, napari_axes = om.open_in_napari(image_large, 
                                                                            metadata_large, 
                                                                            fname, 
                                                                            zarr_mode="zarr_dask", 
                                                                            returns=True)

om.cleanup_omio_cache(fname, full_cleanup=True)



# %% ---------------------------------------------------------
# %% IMREAD'S FOLDER READING ADN MERGING CAPABILITY
""" 
`imread`'s `fname` argument is not restricted to single file names. You can also provide
a folder name containing multiple image files of the same type (e.g., multiple TIFF files)
or different types (e.g., TIFF, CZI, LSM, RAW). In this case, `imread` will scan the provided
folder for all supported image files, read them one by one, and returns, by default, a list of 
images and a list of metadata dictionaries, one for each read file.
"""

fname_folder = "../example_data/tif_dummy_data/tif_folder_with_multiple_files/"
images_folder, metadata_folder = om.imread(fname_folder)
print(f"Number of images read from folder: {len(images_folder)}\n")
for i, (img, meta) in enumerate(zip(images_folder, metadata_folder)):
    print(f"Image {i}: shape={img.shape}, axes={meta.get('axes', 'N/A')}")

""" 
Depending on your use case, you may want to merge the read images into a single
multidimensional array along a new axis (e.g., time, channel). To do so, you can set
the `merge_multiple_files_in_folder` argument to `True` and specify the desired
axis along which to merge the images via the `merge_along_axis` argument:
"""

images_merged, metadata_merged = om.imread(fname_folder, 
                                           merge_multiple_files_in_folder=True, 
                                           merge_along_axis="T")
print(f"Merged image shape: {images_merged.shape} with axes {metadata_merged.get('axes', 'N/A')}")

""" 
In case of unequal image shapes, merging will still work if the optional argument
`zeropadding` is set to `True` (which is the default). In this case, smaller images
will be zero-padded to match the largest image shape along each axis before merging:
"""

fname_folder = "../example_data/tif_dummy_data/tif_folder_with_multiple_files_unequal_shapes/"
images_merged, metadata_merged = om.imread(fname_folder, 
                                           merge_multiple_files_in_folder=True, 
                                           merge_along_axis="T")
print(f"Merged image shape: {images_merged.shape} with axes {metadata_merged.get('axes', 'N/A')}")


""" 
In case `zeropadding` is set to `False`, imread will not merge the images and returns 
None for both image and metadata:
"""
images_merged, metadata_merged = om.imread(fname_folder, 
                                           merge_multiple_files_in_folder=True, 
                                           merge_along_axis="T",
                                           zeropadding=False)
print(f"type of merged images: {type(images_merged)},\ntype of merged metadata: {type(metadata_merged)}")

# %% FOLDER-STACKS READING AND MERGING
"""
OMIO also supports reading of "tagged" folders or folder stacks, where sub-folders are 
named according to specific preceding tags. For example,

.. code-block:: text
    example_dataset/
    ├── T0_FOV1...
    ├── T0_FOV2...
    ├── T0_FOV3...
    ├── T1_FOV1...
    ├── T1_FOV2...
    ├── HC_FOV1...
    ├── HC_FOV2...
    └── ...

In this example, 

* `T0`, `T1` are tags indicating different time points
* `HC` is a tag indicating, e.g., hippocampus region, and
* `FOV1`, `FOV2` are tags indicating different fields of view

OMIO can read such tagged folders and merge the read images along multiple new axes. To do 
so, provide one of the desired tag-folders (`T0_FOV1...`, ...) as `fname` argument to `imread`
and set the optinal argument `folder_stacks` to `True`. `imread` will then split `fname` based 
on the underscore (`_`) character and set the first part as the folder stack tag. It will then scan
for all folders in the parent directory of `fname` that start with any of the detected tags
(e.g., `T0_FOV2`, `T0_FOV3`), read the image files in each of these folders:
"""

fname_folder_stacks = "../example_data/tif_dummy_data/tif_folder_stacks/FOV1_time001"
images_folder_stacks, metadata_folder_stacks = om.imread(fname_folder_stacks, 
                                                           folder_stacks=True)
print(f"Number of images read from folder stacks: {len(images_folder_stacks)}\n")
for i, (img, meta) in enumerate(zip(images_folder_stacks, metadata_folder_stacks)):
    print(f"Image {i}: shape={img.shape}, axes={meta.get('axes', 'N/A')}")
    
""" 
In the example above, `imread` interpreted `FOV1` as the folder stack tag and read all folders
starting with `FOV1` in the parent directory of `fname_folder_stacks`. Note, that folders 
starting with, e..g, `FOV2` were ignored. `imread` also ignores any additional parts in the folder 
names after the first underscore (`_`), so that folders such as `FOV1_time002` are also
correctly recognized. This gives the suer more flexibility in naming the tagged/stacked folders
by, e.g., adding imaging depth or any other additional note (which is common practice in
real-world imaging campaigns).

What are stacked folders good for? They allow to merge the read images along a specified axes.
To do so, set the optional argument `merge_folder_stacks` to `True` and specify the desired
axis along which to merge the images via the `merge_along_axis` argument:
"""

images_folder_stacks_merged, metadata_folder_stacks_merged = om.imread(fname_folder_stacks, 
                                                                   folder_stacks=True,
                                                                   merge_folder_stacks=True,
                                                                   merge_along_axis="T")
print(f"Merged image shape from folder stacks: {images_folder_stacks_merged.shape} " 
       f"with axes {metadata_folder_stacks_merged.get('axes', 'N/A')}")

""" 
`imconvert` also supports reading and merging of tagged folder stacks by providing. It
is recommended, to set `relative_path` to write the converted OME-TIFF file into a
sub-folder of the input folder's parent directory to avoid overwriting the original files.
Furthermore, we recommend to set the relative path one level up (`"../..."`) as
otherwise, the created sub-folder would be placed into the folder defined in `fname`:
"""

output_fnames_folder_stacks = om.imconvert(fname_folder_stacks, 
                                           folder_stacks=True,
                                           merge_folder_stacks=True,
                                           merge_along_axis="T",
                                           relative_path="../omio_converted_FOV1",
                                           return_fnames=True)
for ofname in output_fnames_folder_stacks:
    print(f"Converted file name from folder stacks: {ofname}")
    

# %% ---------------------------------------------------------
# %% OMIO'S BATCH CONVERSION FUNCTION
""" 
OMIO provides a convenience function called `bids_batch_convert` to convert entire
folders of image files into OME-TIFF format in a single function call. It is required
that the folder structures follow the BIDS-like naming conventions, where sub-folders
are named according to specific tags such as `sub-<subject_id>`, `ses-<session_id>`,
`acq-<acquisition_id>`, `run-<run_id>`, etc. Here is a general example of a BIDS-like
folder structure:

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

To perform a batch conversion of all image files in a BIDS-like folder structure,
provide the root folder path as `fname` argument, the subject folder tag as `sub` argument,
and the experiment folder tag as `exp` argument to `bids_batch_convert` as minimum:
"""
fname = "../example_data/tif_dummy_data/BIDS_project_example/"
id_tag = "ID"
exp_tag = "TP001" # contains tif files
om.bids_batch_convert(fname, sub=id_tag, exp=exp_tag, relative_path="omio_bids_converted")

""" 
Of course, `bids_batch_convert` has the same functionalities as `imconvert`, `imread`, and
`imwrite`, so that it is able to, e.g., handle Thorlabs RAW files,
"""

exp_tag = "TP003" # contains thorlabs raw files
om.bids_batch_convert(fname, sub=id_tag, exp=exp_tag, relative_path="omio_bids_converted")

""" 
Also tagged folder stacks can be processed, while the arguments to be provided differ slightly
from those of `imread`/ `imconvert`. Here, you have to provide the `tagfolder` argument
to indicate the tag prefix of the tagfolders to be processed:
"""
exp_tag = "TP005" # contains tagged folder stacks
stackfolder_tag = "FOV1"
om.bids_batch_convert(fname, sub=id_tag, exp=exp_tag, 
                       tagfolder=stackfolder_tag,
                       merge_tagfolders=True,
                       merge_along_axis="T",
                       relative_path="omio_bids_converted_FOV1")

""" 
Note: Since `bids_batch_convert` processes multiple files and folders in a batch and
the additionally provide the `tagfolder`, it is not necessary to set the `relative_path`
one level up as done before with `imconvert`.
"""

""" 
`bids_batch_convert` can also handle multi-file OME-TIFF series correctly:
"""
exp_tag = "TP006" # contains multi-file ome-tiff series
om.bids_batch_convert(fname, sub=id_tag, exp=exp_tag, relative_path="omio_bids_converted")
fname_converted = "../example_data/tif_dummy_data/BIDS_project_example/ID0001/TP006_tif_multi_file_stack/omio_bids_converted/TZCYX_T5_Z10_C2_Z00_C0_T0.ome.tif"
image, metadata = om.imread(fname_converted)
print(f"Multi-file OME-TIFF image shape: {image.shape} with axes {metadata.get('axes', 'N/A')}")
om.open_in_napari(image, metadata, fname_converted)

# %% CREATING EMPTY, OME-COMPLIANT IMAGE ARRAYS AND METADATA
"""
OMIO provides a utility functions called `create_empty_image`, `create_empty_metadata`, and
`update_metadata_from_image` to create empty, OME-compliant image arrays and metadata dictionaries
based on user-defined specifications.
"""
import numpy as np
import os
my_image, my_metadata = om.create_empty_image(return_metadata=True)
print(f"Created empty image with shape: {my_image.shape}, dtype {my_image.dtype} and axes {my_metadata.get('axes', 'N/A')}.")

""" 
Without providing any arguments, `create_empty_image` creates a default empty image
with shape `(1, 1, 1, 512, 512)` and dtype `uint16`. The axes are OME-compliant `TZCYX`.
With the optional argument `return_metadata=True` (default is `False`), the associated 
metadata dictionary is also returned.

You can customize the created empty image by providing the desired `shape` and `dtype`
as arguments:
"""

my_image, my_metadata = om.create_empty_image(shape=(5, 10, 512, 512),  dtype=np.uint16, return_metadata=True)
print(f"Failed to create empty image. Type of my_image is {type(my_image)} and of my_metadata {type(my_metadata)}.")

""" 
The attempt above fails because the provided shape has only 4 dimensions instead of the required
5 dimensions for OME-compliant images. In such cases, OMIO will raise a warning and
return None for both image and metadata. 
"""

my_image, my_metadata = om.create_empty_image(shape=(5, 20, 2, 512, 512),  dtype=np.uint16, return_metadata=True)
print(f"Created empty image with shape: {my_image.shape}, dtype {my_image.dtype} and axes {my_metadata.get('axes', 'N/A')}.")

""" 
You can now manipulate the created empty image as needed:
"""

# for each timepoint, z-slice and channel, we fill the slice with an increasing integer value:
for t in range(my_image.shape[0]):
    for z in range(my_image.shape[1]):
        for c in range(my_image.shape[2]):
            my_image[t, z, c, :, :] = t * 100 + z * 10 + c

pathname_save = "../example_data/custom_created_images/"
os.makedirs(pathname_save, exist_ok=True)
om.imwrite(os.path.join(pathname_save, "my_empty_image_filled.ome.tif"), my_image, my_metadata)
read_my_image, read_my_metadata = om.imread(os.path.join(pathname_save, "my_empty_image_filled.ome.tif"))
om.open_in_napari(read_my_image, read_my_metadata, os.path.join(pathname_save, "my_empty_image_filled.ome.tif"))

""" 
When changing the image shape, e.g., by cropping or padding, 
"""
my_cropped_image = my_image[:, 2:8, :, 100:400, 100:400]  # crop Z and spatial dimensions
print(f"Cropped image shape: {my_cropped_image.shape}")

""" 
you need to update the associated metadata dictionary accordingly. You can do so
by manually updating the relevant metadata entries, or by using OMIO's utility function
`update_metadata_from_image`:
"""
my_cropped_metadata = om.update_metadata_from_image(my_metadata, my_cropped_image)
print(f"Updated cropped image metadata axes: {my_cropped_metadata.get('axes', 'N/A')} with shape: {my_cropped_image.shape}.")
om.imwrite(os.path.join(pathname_save, "my_cropped_image.ome.tif"), my_cropped_image, my_cropped_metadata)
read_my_cropped_image, read_my_cropped_metadata = om.imread(os.path.join(pathname_save, "my_cropped_image.ome.tif"))
om.open_in_napari(read_my_cropped_image, read_my_cropped_metadata, os.path.join(pathname_save, "my_cropped_image.ome.tif"))

# %% END


