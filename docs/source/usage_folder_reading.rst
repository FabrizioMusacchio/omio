Folder Reading and Semantic Merging
=====================================

The examples below assume the following imports:

.. code-block:: python

   import omio as om
   import pprint

Imread’s Folder Reading and Merging Capability
-------------------------------------------------

``imread``’s ``fname`` argument is not restricted to single file names. You can also provide
a folder name containing multiple image files of the same type (e.g., multiple TIFF files)
or different types (e.g., TIFF, CZI, LSM, RAW). In this case, ``imread`` will scan the provided
folder for all supported image files, read them one by one, and returns, by default, a list
of images and a list of metadata dictionaries, one for each read file.

.. code-block:: python

   fname_folder = "example_data/tif_dummy_data/tif_folder_with_multiple_files/"
   images_folder, metadata_folder = om.imread(fname_folder)

   print(f"Number of images read from folder: {len(images_folder)}\n")
   for i, (img, meta) in enumerate(zip(images_folder, metadata_folder)):
       print(f"Image {i}: shape={img.shape}, axes={meta.get('axes', 'N/A')}")

.. code-block:: text

    >>> 
    Number of images read from folder: 3

    Image 0: shape=(1, 10, 2, 20, 100), axes=TZCYX
    Image 1: shape=(1, 10, 2, 20, 100), axes=TZCYX
    Image 2: shape=(1, 10, 2, 20, 100), axes=TZCYX

Depending on your use case, you may want to merge the read images into a single
multidimensional array along a new axis (e.g., time, channel). To do so, you can set
the ``merge_multiple_files_in_folder`` argument to ``True`` and specify the desired axis
along which to merge the images via the ``merge_along_axis`` argument:

.. code-block:: python

   images_merged, metadata_merged = om.imread(
       fname_folder,
       merge_multiple_files_in_folder=True,
       merge_along_axis="T")

   print(f"Merged image shape: {images_merged.shape} "
         f"with axes {metadata_merged.get('axes', 'N/A')}")

.. code-block:: text

    >>> 
    Merged image shape: (3, 10, 2, 20, 100) with axes TZCYX

In case of unequal image shapes, merging will still work if the optional argument
``zeropadding`` is set to ``True`` (which is the default). In this case, smaller images
will be zero-padded to match the largest image shape along each axis before merging:

.. code-block:: python

   fname_folder = "example_data/tif_dummy_data/tif_folder_with_multiple_files_unequal_shapes/"
   images_merged, metadata_merged = om.imread(
       fname_folder,
       merge_multiple_files_in_folder=True,
       merge_along_axis="T")

   print(f"Merged image shape: {images_merged.shape} "
         f"with axes {metadata_merged.get('axes', 'N/A')}")

.. code-block:: text

    >>> 
    Merged image shape: (5, 10, 2, 20, 100) with axes TZCYX

In case ``zeropadding`` is set to ``False``, ``imread`` will not merge the images and returns
``None`` for both image and metadata:

.. code-block:: python

   images_merged, metadata_merged = om.imread(
       fname_folder,
       merge_multiple_files_in_folder=True,
       merge_along_axis="T",
       zeropadding=False)

   print(f"type of merged images: {type(images_merged)},\n"
         f"type of merged metadata: {type(metadata_merged)}")

.. code-block:: text

    >>> 
    type of merged images: <class 'NoneType'>,
    type of merged metadata: <class 'NoneType'>


Folder Stacks Reading and Merging
------------------------------------

OMIO also supports reading of tagged folders or folder stacks, where sub-folders are
named according to specific preceding tags. For example:

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

In this example:

* ``T0``, ``T1`` are tags indicating different time points
* ``HC`` is a tag indicating, e.g., hippocampus region
* ``FOV1``, ``FOV2`` are tags indicating different fields of view

OMIO can read such tagged folders and merge the read images along multiple new axes.
To do so, provide one of the desired tag-folders (``T0_FOV1...``, ...) as ``fname`` argument
to ``imread`` and set the optional argument ``folder_stacks`` to ``True``.

``imread`` will then split ``fname`` based on the underscore (`_`) character and set the
first part as the folder stack tag. It will then scan for all folders in the parent
directory of ``fname`` that start with any of the detected tags and read the image files
in each of these folders.

.. code-block:: python

   fname_folder_stacks = "example_data/tif_dummy_data/tif_folder_stacks/FOV1_time001"
   images_folder_stacks, metadata_folder_stacks = om.imread(
       fname_folder_stacks,
       folder_stacks=True)

Terminal output during execution:

.. code-block:: text

    >>>
    folder_stacks=True, merge_folder_stacks=False ⟶ will read from tagged folder stacks.
    Detected folder stack tag: 'FOV1_'.
    Reading TIFF fully into RAM...
    Correcting for OME axes order...
        Got NumPy array as input. Will return reordered NumPy array.
    Finished reading TIFF.
    Reading TIFF fully into RAM...
    Correcting for OME axes order...
        Got NumPy array as input. Will return reordered NumPy array.
    Finished reading TIFF.

.. code-block:: python

   print(f"Number of images read from folder stacks: {len(images_folder_stacks)}\n")
   for i, (img, meta) in enumerate(zip(images_folder_stacks, metadata_folder_stacks)):
       print(f"Image {i}: shape={img.shape}, axes={meta.get('axes', 'N/A')}")

.. code-block:: text

    >>> 
    Number of images read from folder stacks: 2

    Image 0: shape=(5, 10, 2, 20, 100), axes=TZCYX
    Image 1: shape=(5, 10, 2, 20, 100), axes=TZCYX

In the example above, ``imread`` interpreted ``FOV1`` as the folder stack tag and read all
folders starting with, e.g., ``FOV1`` in the parent directory of ``fname_folder_stacks``. Note 
that folders starting with, e.g., ``FOV2`` were ignored. ``imread`` also ignores any
additional parts in the folder names after the first underscore (`_`), so that folders
such as ``FOV1_time002`` are also correctly recognized. This gives the user more
flexibility in naming the tagged or stacked folders by, for example, adding imaging
depth or other additional notes.

What are stacked folders good for? They allow merging of the read images along a
specified axis. To do so, set the optional argument ``merge_folder_stacks`` to ``True``
and specify the desired axis via ``merge_along_axis``:

.. code-block:: python

   images_folder_stacks_merged, metadata_folder_stacks_merged = om.imread(
       fname_folder_stacks,
       folder_stacks=True,
       merge_folder_stacks=True,
       merge_along_axis="T")

   print(f"Merged image shape from folder stacks: "
         f"{images_folder_stacks_merged.shape} "
         f"with axes {metadata_folder_stacks_merged.get('axes', 'N/A')}")

.. code-block:: text

    >>> 
    Merged image shape from folder stacks: (10, 10, 2, 20, 100) with axes TZCYX

``imconvert`` also supports reading and merging of tagged folder stacks. It is recommended 
to set in this case ``relative_path`` to write the converted OME-TIFF file into a sub-folder 
of the  input folder’s parent directory to avoid overwriting the original
files. Furthermore, we recommend setting the relative path one level up (`"../..."`),
as otherwise the created sub-folder would be placed into the folder defined in ``fname``:

.. code-block:: python

   output_fnames_folder_stacks = om.imconvert(
       fname_folder_stacks,
       folder_stacks=True,
       merge_folder_stacks=True,
       merge_along_axis="T",
       relative_path="../omio_converted_FOV1",
       return_fnames=True)

   for ofname in output_fnames_folder_stacks:
       print(f"Converted file name from folder stacks: {ofname}")

.. code-block:: text

    >>> 
    Converted file name from folder stacks: ../example_data/tif_dummy_data/tif_folder_stacks/FOV1_time001/../omio_converted_FOV1/TZCYX_T5_Z10_C2_FOV1_time001_merged.ome.tif
