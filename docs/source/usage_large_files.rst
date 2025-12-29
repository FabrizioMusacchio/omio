Large File Handling and Out of Core Workflows
==========================================================


OMIO provides explicit support for working with large image data that do not fit into
main memory. This includes Zarr-backed lazy loading, optional memory mapping on disk,
and efficient visualization of large datasets in napari.

The examples below assume the following imports:

.. code-block:: python

   import omio as om
   import pprint

Read Large Files Lazily With Zarr Backend
-------------------------------------------

OMIO supports reading large image files that do not fit into memory by Zarr-backed lazy
loading and optional memory mapping on disk.

To read a large TIFF file lazily, use the ``imread`` function with the
``zarr_store="memory"`` or ``zarr_store="disk"`` argument:

.. code-block:: python

   fname = "example_data/tif_large_Ca_imaging_large/1MP_SIMPLE_Stephan__001_001.tif"
   image_lazy, metadata_lazy = om.imread(fname, zarr_store="memory")

   print(f"Lazy image shape: {image_lazy.shape}")
   print(f"Lazy image type: {type(image_lazy)}")
   
   image_lazy


.. code-block:: text

   >>>
   Lazy image shape: (1, 2000, 1, 355, 350)
   Lazy image type: <class 'zarr.core.array.Array'>

   <Array <FsspecStore(AsyncFileSystemWrapper, /14069212288)> shape=(1, 2000, 1, 355, 350) dtype=uint16>

You can now manipulate ``image_lazy`` as a Zarr array without loading the entire dataset
into memory. For example, you can read a small chunk of the data:

.. code-block:: python

   sub_stack = image_lazy[0, 0:10, 0:100, 0:100]
   print(f"Sub-stack shape: {sub_stack.shape}")

With ``zarr_store="disk"``, ``imread`` creates a temporary Zarr store on disk and 
memory-maps the data for efficient access. The default location of that temporary Zarr 
store is the parent directory of ``fname``, where a folder called ``.omio_cache`` 
is created to hold the temporary data:

.. code-block:: python

   image_lazy_memmap, metadata_lazy_memmap = om.imread(fname, zarr_store="disk")
   print(f"Lazy memmap image shape: {image_lazy_memmap.shape}")
   print(f"Lazy memmap image type: {type(image_lazy_memmap)}")

   image_lazy_memmap


.. code-block:: text

   >>>
   Lazy memmap image shape: (1, 2000, 1, 355, 350)
   Lazy memmap image type: <class 'zarr.core.array.Array'>

   <Array file://example_data/tif_large_Ca_imaging_large/.omio_cache/1MP_SIMPLE_Stephan__001_001.zarr shape=(1, 2000, 1, 355, 350) dtype=uint16>

.. code-block:: python

   om.open_in_napari(image_lazy_memmap, metadata_lazy_memmap, fname)

.. image:: _static/figures/open_1MP_SIMPLE_Stephan__001_001.tif_in_napari.jpg
   :target: _static/figures/open_1MP_SIMPLE_Stephan__001_001.tif_in_napari.jpg
   :alt: Screenshot of large TIFF opened in napari


Note that if you have opened an image with napari in the same interactive session before,
OMIO will reuse the existing napari viewer instance to avoid opening multiple windows.
In practice, any new image opened with ``om.open_in_napari`` will be added as a new layer
to the existing napari viewer.

There is intentionally no automatic cleanup of the temporary Zarr stores, as users may
want to reuse them for downstream processing. To manually clean up the temporary Zarr 
stores created by OMIO, use:

.. code-block:: python

   om.cleanup_omio_cache(fname, full_cleanup=False)

This command cleans up only the temporary Zarr store associated with the given
``fname``. To clean up all temporary Zarr stores created by OMIO, use:

.. code-block:: python

   om.cleanup_omio_cache(fname, full_cleanup=True)


Efficiently View Large Images in Napari With OMIO’s DASK Support
------------------------------------------------------------------

To efficiently view large images in napari without loading the entire dataset into
memory, you can use OMIO’s built-in support for lazy loading and combine it with OMIO’s
napari integration. This integration supports:

* handling of in-memory and on-disk memory-mapped Zarr arrays  
* automatic axis reordering based on OME semantics  
* DASK support for out-of-core parallel processing  

The following example uses a 1.1 GB 3D image stack with multiple channels:

.. code-block:: python

   fname = "example_data/tif_files_from_3P_paper/Supplementary_Video_4.tif"

First, memory-map the image on disk:

.. code-block:: python

   image_large, metadata_large = om.imread(fname, zarr_store="disk")

This stack has stored an erroneous ``PhysicalSizeZ`` in its ImageJ metadata, which is set
to ``0.0000185`` microns instead of the correct value of ``5`` microns according to the
`supplementary information <https://static-content.springer.com/esm/art%3A10.1038%2Fs42003-025-08079-8/MediaObjects/42003_2025_8079_MOESM1_ESM.pdf>`_ of the paper. Thus, let’s correct the corresponding metadata 
entry so that napari can correctly scale the Z axis upon viewing:

.. code-block:: python

   metadata_large["PhysicalSizeZ"] = 5  # in microns

Now open the large image in napari without DASK support:

.. code-block:: python

   om.open_in_napari(image_large, metadata_large, fname, zarr_mode="zarr_nodask")

.. image:: _static/figures/open_Supplementary_Video_4_in_napari.jpg
   :target: _static/figures/open_Supplementary_Video_4_in_napari.jpg
   :alt: Structural 3D sample file opened in napari


Internally, OMIO’s napari viewing function correctly handles the true image scalings and
axes, but needs to re-arrange the axes to the napari-expected order. Without DASK, this 
may take some time for very large images, as a temporary Zarr store is created with the 
re-ordered axes. This temporary store is created in the same ``.omio_cache`` folder as before.

To speed up this process, OMIO provides ``zarr_mode="zarr_dask"`` to use DASK for
parallelized re-ordering and writing of the temporary Zarr store:

.. code-block:: python

   om.open_in_napari(image_large, metadata_large, fname, zarr_mode="zarr_dask")

With ``returns=True``, the napari viewer instance, the created napari layers, the used
Zarr array, and the used axes order are also returned for further programmatic use:

.. code-block:: python

   napari_viewer, napari_layers, napari_datas, napari_axes = om.open_in_napari(
       image_large,
       metadata_large,
       fname,
       zarr_mode="zarr_dask",
       returns=True)

After finishing the inspection, the temporary Zarr stores can be removed manually:

.. code-block:: python

   om.cleanup_omio_cache(fname, full_cleanup=True)

