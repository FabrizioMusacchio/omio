Core Workflow: Read, Inspect, View, Write
==========================================================

This section introduces the core OMIO workflow using single image files. It covers
reading image data, inspecting and modifying metadata, visualizing images in Napari,
and writing OME-TIFF output files.

Hello World
-----------

OMIO has a simple ``hello_world()`` function to verify that the installation 
was successful:

.. code-block:: python

   import omio as om
   import pprint
   om.hello_world()

The command above should print something like:

``Hello from omio.py! OMIO version: 0.1.0``

If you see this message, OMIO is correctly installed and ready to use. 
Note that the version number may vary depending on the installed version.


Single File Reading and Metadata Inspection
--------------------------------------------

To open a single file such as a TIFF file, use the ``imread`` function. This function
returns the image data as a NumPy array (by default) along with the associated metadata
as a dictionary.

.. code-block:: python

   fname = "example_data/tif_cell_single_tif/13374.tif"
   image, metadata = om.imread(fname)
   print(f"Image shape: {image.shape}")

``imread`` automatically interprets the OME metadata stored in the TIFF file and
re-arranges the image axes to follow the OME axis order convention:
(Time, Channel, Z depth, Y height, X width).

If any of these axes are singleton (i.e. size 1), they are retained in the returned
image array to preserve the full 5D structure. This ensures OME compliance and thus
compatibility with downstream OME-based pipelines.

``imread`` always returns the read image data (as a NumPy array by default) and the
associated metadata as a dictionary. The metadata dictionary contains OME-relevant
entries such as ``PhysicalSizeX``, ``PhysicalSizeY``, ``PhysicalSizeZ``,
``TimeIncrement``, and ``Channels``. OMIO always assigns these entries and tries to infer
missing metadata from the available information in the file or by assigning predefined
defaults, which can be customized by the user upon function call.

Let's inspect some of the read metadata:

.. code-block:: python

   print(f"Metadata keys: {list(metadata.keys())}")
   pprint.pprint(metadata)

You may notice that ``imread`` has, apart from the correct and OME-compliant axis order
and physical size entries in microns, also added an entry called ``"Annotations"`` that
contains additional metadata parsed from the TIFF file.

OMIO tries to extract as much metadata as possible from the file and store it in a
structured manner in the metadata dictionary. Any non-OME metadata is stored under the
``"Annotations"`` key to avoid conflicts with standard OME entries, while preserving
potentially valuable information for downstream processing.

Of course, you can always add or change metadata entries as needed. For example, let's
add an ``"Experimenter"`` entry to the metadata dictionary:

.. code-block:: python

   metadata["Experimenter"] = "Your Name"
   pprint.pprint(metadata)

If we would save ``image`` and its associated ``metadata`` back to an OME-TIFF file, this
additional ``"Experimenter"`` entry would not be written, as it is not part of the OME
standard.

However, OMIO offers a check-up function called ``OME_metadata_checkup()`` that normalizes
the metadata dictionary to be fully OME-compliant by moving any non-OME entries under the
``"Annotations"`` key:

.. code-block:: python

   metadata = om.OME_metadata_checkup(metadata)
   pprint.pprint(metadata)


Opening Images in Napari and Metadata Modification
----------------------------------------------------

OMIO comes with built-in support to open images directly in Napari for interactive
visualization. Let's open the previously read image in Napari:

.. code-block:: python

   om.open_in_napari(image, metadata, fname)

For demonstration purposes, we change the ``PhysicalSizeZ`` metadata entry to an
incorrect value and re-open the image in Napari to see that Napari correctly rescales
the Z axis based on the provided metadata:

.. code-block:: python

   print(f"Original PhysicalSizeZ: {metadata['PhysicalSizeZ']} microns")
   metadata["PhysicalSizeZ"] = 5  # wrong value in microns
   print(f"Modified PhysicalSizeZ: {metadata['PhysicalSizeZ']} microns")
   om.open_in_napari(image, metadata, fname)

If you do not want to see terminal output from OMIO, you can set ``verbose=False`` in any
OMIO function call. For example:

.. code-block:: python

   om.open_in_napari(image, metadata, fname, verbose=False)


Ensured OME Compliance upon Reading
-------------------------------------

OMIO ensures OME compliance of the read image and metadata upon reading. This applies
regardless of whether the input file is already OME-compliant, has incomplete OME
metadata, or does not contain any OME metadata at all.

This also applies to non-OME formats such as Zeiss CZI files or Thorlabs RAW files, and it
does not matter whether the input image is 2D (XY), 3D (Z stack), 4D (time lapse or
multichannel), or 5D (time lapse multichannel Z stack).

The example data folder ``tif_dummy_data/tif_single_files`` contains several TIFF files
with different dimensionalities. These files were generated using
``additional_scripts/generate_dummy_tif_files.py`` and do not contain ImageJ Hyperstack
or OME-TIFF metadata.

.. code-block:: python

   fname_5d = "example_data/tif_dummy_data/tif_single_files/TZCYX_T5_Z10_C2.tif"
   image_5d, metadata_5d = om.imread(fname_5d)
   print(f"5D Image shape: {image_5d.shape} with axes {metadata_5d.get('axes', 'N/A')}")
   pprint.pprint(metadata_5d)
   om.open_in_napari(image_5d, metadata_5d, fname_5d)

.. code-block:: python

   fname_2d = "example_data/tif_dummy_data/tif_single_files/YX.tif"
   image_2d, metadata_2d = om.imread(fname_2d)
   print(f"2D Image shape: {image_2d.shape} with axes {metadata_2d.get('axes', 'N/A')}")
   pprint.pprint(metadata_2d)
   om.open_in_napari(image_2d, metadata_2d, fname_2d)

As shown above, OMIO correctly infers OME-compliant axes and adds default OME metadata
entries as needed.

Let's also try TIFF files with ImageJ Hyperstack metadata. These files contain additional
singleton axes (S) required for ImageJ compatibility:

.. code-block:: python

   fname_4d = "example_data/tif_dummy_data/tif_with_ImageJ/TYXS_T1.tif"
   image_4d, metadata_4d = om.imread(fname_4d)
   print(f"4D Image shape: {image_4d.shape} with axes {metadata_4d.get('axes', 'N/A')}")
   pprint.pprint(metadata_4d)
   om.open_in_napari(image_4d, metadata_4d, fname_4d)

.. code-block:: python

   fname_6d = "example_data/tif_dummy_data/tif_with_ImageJ/TZCYXS_C1_Z10_T2.tif"
   image_6d, metadata_6d = om.imread(fname_6d)
   print(f"6D Image shape: {image_6d.shape} with axes {metadata_6d.get('axes', 'N/A')}")
   pprint.pprint(metadata_6d)
   om.open_in_napari(image_6d, metadata_6d, fname_6d)

Due to the extra singleton axes, these files were saved with photometric interpretation
``rgb`` instead of ``minisblack``. ``imread`` therefore interprets them as three-channel
images. If the image additionally contains more than one channel axis, this results in
multiple channel axes in the read image.

This behavior is intentional. OMIO always tries to retain the full dimensionality of the
image to avoid any loss of information.

.. code-block:: python

   fname_6d = "example_data/tif_dummy_data/tif_with_ImageJ/TZCYXS_T5_Z10_C2.tif"
   image_6d, metadata_6d = om.imread(fname_6d)
   print(f"6D Image shape: {image_6d.shape} with axes {metadata_6d.get('axes', 'N/A')}")
   pprint.pprint(metadata_6d)
   om.open_in_napari(image_6d, metadata_6d, fname_6d)

Let's also open an OME-TIFF file:

.. code-block:: python

   fname_ometiff = "example_data/tif_dummy_data/ome_tif/TZCYX_T5_Z10_C2.ome.tif"
   image_ometiff, metadata_ometiff = om.imread(fname_ometiff)
   print(f"OME-TIFF Image shape: {image_ometiff.shape} with axes {metadata_ometiff.get('axes', 'N/A')}")
   pprint.pprint(metadata_ometiff)
   om.open_in_napari(image_ometiff, metadata_ometiff, fname_ometiff)


Ensured OME Compliance upon Writing
------------------------------------

OMIO’s writing function ``imwrite`` also ensures OME compliance of the written image and
metadata.

.. code-block:: python

   fname_2d = "example_data/tif_dummy_data/tif_single_files/YX.tif"
   image_2d, metadata_2d = om.imread(fname_2d)
   print(f"2D Image shape: {image_2d.shape} with axes {metadata_2d.get('axes', 'N/A')}")

   om.imwrite(fname_2d, image_2d, metadata_2d, relative_path="omio_converted")

``imwrite`` requires, at minimum, the image data, the associated metadata dictionary, and
the output file name. By default, ``overwrite`` is set to ``False``, so existing files are
not overwritten. Instead, OMIO appends a numeric suffix to the file name.

A ``relative_path`` argument can be provided to write the converted OME-TIFF file into a
subfolder of the input file’s directory. The written file receives the extension
``.ome.tif``.

Let's inspect the written OME-TIFF file:

.. code-block:: python

   fname_2d_written = "example_data/tif_dummy_data/tif_single_files/omio_converted/YX.ome.tif"
   image_2d_written, metadata_2d_written = om.imread(fname_2d_written)
   print(f"Written 2D Image shape: {image_2d_written.shape} with axes {metadata_2d_written.get('axes', 'N/A')}")
   om.open_in_napari(image_2d_written, metadata_2d_written, fname_2d_written)

The written OME-TIFF file can be opened in any OME-compliant software such as ImageJ or
Fiji. When using drag and drop, Fiji does not correctly interpret the physical unit
``microns`` and displays ``pixels`` instead. This is a known limitation of Fiji’s SCIFIO
library. Using the Bio-Formats Importer correctly interprets the physical unit.


The imconvert Convenience Function
------------------------------------

OMIO also provides a convenience function called ``imconvert`` that combines reading and
writing in a single step.

.. code-block:: python

   fname_5d = "example_data/tif_dummy_data/tif_single_files/TZCYX_T5_Z10_C2.tif"
   om.imconvert(fname_5d, relative_path="omio_converted")

``imconvert`` accepts all arguments of both ``imread`` and ``imwrite``, allowing full
control over reading and writing behavior.

An additional optional argument called ``return_fnames`` (default ``False``) returns the
output file names upon conversion for further downstream processing:

.. code-block:: python

   output_fnames = om.imconvert(fname_5d, relative_path="omio_converted", return_fnames=True)
   print(f"Converted file names: {output_fnames}")