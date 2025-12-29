TIFF Container Policies and Special Layouts
=============================================

This section describes how OMIO handles more complex TIFF and LSM container layouts,
including multi-series files, paginated stacks, and multi-file OME-TIFF series.
OMIO follows strict and explicit policies to avoid ambiguous interpretations.


Reading Multi-Series TIFF Stacks
-----------------------------------

OMIO’s ``imread`` function also supports reading of multi-series TIFF and LSM stacks,
however, with some limitations.

TIFF and LSM containers may store multiple datasets (“series”) in a single file.
While ``tifffile`` exposes these as TIFF series, OMIO enforces a strict and predictable
policy to avoid ambiguous interpretations:

* If a file contains exactly one series (``len(tif.series) == 1``), OMIO guarantees
  correct reading and normalization to canonical OME axis order (TZCYX).
* If a file contains multiple series (``len(tif.series) > 1``), OMIO will process
  **only the first series (series 0)** and ignore all others. A warning is emitted
  in this case, and the policy decision is recorded in the returned metadata.
* OMIO does not attempt to infer relationships between multiple series, does not
  concatenate them, and does not inspect their shapes, axes, or photometric
  interpretation beyond series 0.

This policy is intentional and favors reproducibility and explicit behavior over
heuristic reconstruction of complex TIFF layouts.

.. code-block:: python

   fname_multi_series = "example_data/tif_dummy_data/multiseries_tif/multiseries_rgb_with_equal_shapes.tif"
   image_multi_series, metadata_multi_series = om.imread(fname_multi_series)
   pprint.pprint(metadata_multi_series)

.. code-block:: text

   >>>

   {'Annotations': {'Namespace': 'omio:metadata',
                  'OMIO_MultiSeriesAxes': ['YXS', 'YXS'],
                  'OMIO_MultiSeriesDetected': True,
                  'OMIO_MultiSeriesPhotometric': ['RGB', 'RGB'],
                  'OMIO_MultiSeriesPolicy': 'only_series_0',
                  'OMIO_MultiSeriesShapes': [[16, 16, 3], [16, 16, 3]],
                  'OMIO_ProcessedSeries': 0,
                  'OMIO_TotalSeries': 2,
                  'OMIO_VERSION': '0.1.4',
                  'original_creation_or_change_date': '2025-12-27T16:57:29',
                  'original_filename': 'multiseries_rgb_with_equal_shapes.tif',
                  'original_filetype': 'tif',
                  'original_metadata_type': 'multipage RGB TIFF',
                  'original_parentfolder': 'example_data/tif_dummy_data/multiseries_tif',
                  'spacing': 1.0,
                  'unit': 'micron'},
   'PhysicalSizeX': 1.0,
   'PhysicalSizeXUnit': 'micron',
   'PhysicalSizeY': 1.0,
   'PhysicalSizeYUnit': 'micron',
   'PhysicalSizeZ': 1.0,
   'PhysicalSizeZUnit': 'micron',
   'SizeC': 3,
   'SizeT': 1,
   'SizeX': 16,
   'SizeY': 16,
   'SizeZ': 1,
   'axes': 'TZCYX',
   'shape': (1, 1, 3, 16, 16)}


Inspecting the ``"Annotations"`` in the retrieved metadata shows that OMIO has detected a
multi-series TIFF file (``'OMIO_MultiSeriesDetected': True``) which initially contained
two series with axes ``['YXS', 'YXS']`` and shapes ``[[16, 16, 3], [16, 16, 3]]``. Thus, 
the two series seem to be compatible for concatenation along a new axis. However,
OMIO does not infer, by intention, any such relationships and only reads the first series
(series 0) with shape ``(16, 16, 3)`` and axes ``YXS``.

The reason for this policy is to avoid ambiguous interpretations of multi-series TIFF
files, which may contain series with different dimensionalities, axes, or photometric
interpretations:

.. code-block:: python

   fname_multi_series = "example_data/tif_dummy_data/multiseries_tif/multiseries_rgb_with_unequal_series.tif"
   image_multi_series, metadata_multi_series = om.imread(fname_multi_series)
   pprint.pprint(metadata_multi_series)

.. code-block:: text

   >>>

   {'Annotations': {'Namespace': 'omio:metadata',
                  'OMIO_MultiSeriesAxes': ['YXS', 'YXS'],
                  'OMIO_MultiSeriesDetected': True,
                  'OMIO_MultiSeriesPhotometric': ['RGB', 'RGB'],
                  'OMIO_MultiSeriesPolicy': 'only_series_0',
                  'OMIO_MultiSeriesShapes': [[16, 16, 3], [17, 17, 3]],
                  'OMIO_ProcessedSeries': 0,
                  'OMIO_TotalSeries': 2,
                  'OMIO_VERSION': '0.1.4',
                  'original_creation_or_change_date': '2025-12-27T16:57:29',
                  'original_filename': 'multiseries_rgb_with_unequal_series.tif',
                  'original_filetype': 'tif',
                  'original_metadata_type': 'multipage RGB TIFF',
                  'original_parentfolder': 'example_data/tif_dummy_data/multiseries_tif',
                  'spacing': 1.0,
                  'unit': 'micron'},
   'PhysicalSizeX': 1.0,
   'PhysicalSizeXUnit': 'micron',
   'PhysicalSizeY': 1.0,
   'PhysicalSizeYUnit': 'micron',
   'PhysicalSizeZ': 1.0,
   'PhysicalSizeZUnit': 'micron',
   'SizeC': 3,
   'SizeT': 1,
   'SizeX': 16,
   'SizeY': 16,
   'SizeZ': 1,
   'axes': 'TZCYX',
   'shape': (1, 1, 3, 16, 16)}


.. code-block:: python

   fname_multi_series = "example_data/tif_dummy_data/multiseries_tif/multiseries_rgb_minisblack_mixture.tif"
   image_multi_series, metadata_multi_series = om.imread(fname_multi_series)
   pprint.pprint(metadata_multi_series)

.. code-block:: text

   >>>

   {'Annotations': {'Namespace': 'omio:metadata',
                  'OMIO_MultiSeriesAxes': ['YXS', 'QYX'],
                  'OMIO_MultiSeriesDetected': True,
                  'OMIO_MultiSeriesPhotometric': ['RGB', 'MINISBLACK'],
                  'OMIO_MultiSeriesPolicy': 'only_series_0',
                  'OMIO_MultiSeriesShapes': [[16, 16, 3], [2, 32, 32]],
                  'OMIO_ProcessedSeries': 0,
                  'OMIO_TotalSeries': 2,
                  'OMIO_VERSION': '0.1.4',
                  'original_creation_or_change_date': '2025-12-27T16:57:29',
                  'original_filename': 'multiseries_rgb_minisblack_mixture.tif',
                  'original_filetype': 'tif',
                  'original_metadata_type': 'multipage RGB TIFF',
                  'original_parentfolder': 'example_data/tif_dummy_data/multiseries_tif',
                  'spacing': 1.0,
                  'unit': 'micron'},
   'PhysicalSizeX': 1.0,
   'PhysicalSizeXUnit': 'micron',
   'PhysicalSizeY': 1.0,
   'PhysicalSizeYUnit': 'micron',
   'PhysicalSizeZ': 1.0,
   'PhysicalSizeZUnit': 'micron',
   'SizeC': 3,
   'SizeT': 1,
   'SizeX': 16,
   'SizeY': 16,
   'SizeZ': 1,
   'axes': 'TZCYX',
   'shape': (1, 1, 3, 16, 16)}



.. code-block:: python

   fname_multi_series = "example_data/tif_dummy_data/multiseries_tif/multiseries_minisblack.tif"
   image_multi_series, metadata_multi_series = om.imread(fname_multi_series)
   pprint.pprint(metadata_multi_series)

.. code-block:: text

   >>>

   {'Annotations': {'Namespace': 'omio:metadata',
                  'OMIO_MultiSeriesAxes': ['QYX', 'QYX'],
                  'OMIO_MultiSeriesDetected': True,
                  'OMIO_MultiSeriesPhotometric': ['MINISBLACK', 'MINISBLACK'],
                  'OMIO_MultiSeriesPolicy': 'only_series_0',
                  'OMIO_MultiSeriesShapes': [[2, 32, 32], [2, 32, 32]],
                  'OMIO_ProcessedSeries': 0,
                  'OMIO_TotalSeries': 2,
                  'OMIO_VERSION': '0.1.4',
                  'original_creation_or_change_date': '2025-12-27T16:57:29',
                  'original_filename': 'multiseries_minisblack.tif',
                  'original_filetype': 'tif',
                  'original_metadata_type': 'N/A',
                  'original_parentfolder': 'example_data/tif_dummy_data/multiseries_tif',
                  'spacing': 1.0,
                  'unit': 'micron'},
   'PhysicalSizeX': 1.0,
   'PhysicalSizeXUnit': 'micron',
   'PhysicalSizeY': 1.0,
   'PhysicalSizeYUnit': 'micron',
   'PhysicalSizeZ': 1.0,
   'PhysicalSizeZUnit': 'micron',
   'SizeC': 2,
   'SizeT': 1,
   'SizeX': 32,
   'SizeY': 32,
   'SizeZ': 1,
   'axes': 'TZCYX',
   'shape': (1, 1, 2, 32, 32)}


.. code-block:: python

   fname_multi_series = "example_data/tif_dummy_data/multiseries_tif/multiseries_TCYXS.ome.tif"
   image_multi_series, metadata_multi_series = om.imread(fname_multi_series)
   pprint.pprint(metadata_multi_series)

.. code-block:: text

   >>>

   {'Annotations': {'1': '256 256',
                  '2': '128 128',
                  'Namespace': 'omio:metadata',
                  'OMIO_MultiSeriesAxes': ['TCYXS', 'YXS'],
                  'OMIO_MultiSeriesDetected': True,
                  'OMIO_MultiSeriesPhotometric': ['RGB', 'RGB'],
                  'OMIO_MultiSeriesPolicy': 'only_series_0',
                  'OMIO_MultiSeriesShapes': [[8, 2, 20, 100, 3], [3, 13, 3]],
                  'OMIO_ProcessedSeries': 0,
                  'OMIO_TotalSeries': 2,
                  'OMIO_VERSION': '0.1.4',
                  'original_creation_or_change_date': '2025-12-28T10:24:12',
                  'original_filename': 'multiseries_TCYXS.ome.tif',
                  'original_filetype': 'tif',
                  'original_metadata_type': 'OME_XML',
                  'original_parentfolder': 'example_data/tif_dummy_data/multiseries_tif',
                  'spacing': 1.0,
                  'unit': 'micron'},
   'Channel_Count': 2,
   'PhysicalSizeX': 0.29,
   'PhysicalSizeXUnit': 'micron',
   'PhysicalSizeY': 0.29,
   'PhysicalSizeYUnit': 'micron',
   'PhysicalSizeZ': 1.0,
   'PhysicalSizeZUnit': 'micron',
   'SizeC': 6,
   'SizeT': 8,
   'SizeX': 100,
   'SizeY': 20,
   'SizeZ': 1,
   'TimeIncrement': 0.1,
   'TimeIncrementUnit': 's',
   'axes': 'TZCYX',
   'shape': (8, 1, 6, 20, 100)}


As a conequence of this policy, users who wish to work with multiple series in a
multi-series TIFF file must explicitly handle the separation and reading of each
series themselves (e.g., by using as ImageJ/Fiji and store each series in its own
single-series TIFF file).


Reading Paginated TIFF Stacks
--------------------------------

OMIO’s ``imread`` function also supports reading of paginated LSM stacks that contain
multiple pages or tiles stored sequentially.

OMIO’s policy here is that each page or tile is treated as a separate image stack,
and the returned image becomes a list of images and a list of metadata dictionaries,
one for each page. This allows for flexible handling of paginated stacks, where each
page may have different dimensionalities, axes, or metadata.

.. code-block:: python

   fname_paginated = "example_data/tif_dummy_data/paginated_tif/paginated_tif.tif"
   images, metadata_paginated = om.imread(fname_paginated)

   print(f"Number of pages read: {len(images)}")
   for i, (img, meta) in enumerate(zip(images, metadata_paginated)):
       print(f"Page {i}: shape={img.shape}, axes={meta.get('axes', 'N/A')}")

   pprint.pprint(metadata_paginated[0])
   pprint.pprint(metadata_paginated[1])
   pprint.pprint(metadata_paginated[2])

Output (truncated):

.. code-block:: text

   >>>

   Number of pages read: 3

   Page 0: shape=(1, 1, 1, 16, 16), axes=TZCYX
   Page 1: shape=(1, 1, 1, 16, 16), axes=TZCYX
   Page 2: shape=(1, 1, 1, 16, 16), axes=TZCYX

Note that ``imread`` has an optional argument ``return_list`` which is set to ``False``
by default. If set to ``True``, ``imread`` will always return a list of images and a
list of metadata dictionaries, even if the input file contains only a single page.
This can be useful for consistent handling of paginated stacks in batch processing
scenarios.


Reading Multi-File OME-TIFF Stacks
------------------------------------

A multi-file OME-TIFF series consists of multiple TIFF files, each representing a
single time point, channel, or Z-slice of a larger multidimensional dataset.

OMIO supports reading such multi-file OME-TIFF series via the ``imread`` function by
providing the file name of any one of the individual TIFF files in the series.
OMIO will automatically detect and read all files in the series, sort them correctly
based on their OME metadata, and assemble them into a single multidimensional NumPy
array along with the associated OME-compliant metadata:

.. code-block:: python

   fname_multifile_ometiff = "example_data/tif_dummy_data/tif_ome_multi_file_series/TZCYX_T5_Z10_C2_Z00_C0_T0.ome.tif"
   image_multifile_ometiff, metadata_multifile_ometiff = om.imread(fname_multifile_ometiff)

   print(f"Multi-file OME-TIFF image shape: {image_multifile_ometiff.shape}")
   pprint.pprint(metadata_multifile_ometiff)
   om.open_in_napari(image_multifile_ometiff, metadata_multifile_ometiff, fname_multifile_ometiff)

.. code-block:: text

   >>>

   Multi-file OME-TIFF image shape: (5, 10, 2, 20, 100)

   {'Annotations': {'Namespace': 'omio:metadata',
                  'OMIO_MultiSeriesDetected': False,
                  'OMIO_VERSION': '0.1.4',
                  'original_creation_or_change_date': '2025-12-27T16:57:29',
                  'original_filename': 'TZCYX_T5_Z10_C2_Z00_C0_T0.ome.tif',
                  'original_filetype': 'tif',
                  'original_metadata_type': 'OME_XML',
                  'original_parentfolder': 'example_data/tif_dummy_data/tif_ome_multi_file_series',
                  'spacing': 2.0,
                  'unit': 'micron'},
   'Channel_Count': 2,
   'PhysicalSizeX': 0.19,
   'PhysicalSizeXUnit': 'micron',
   'PhysicalSizeY': 0.19,
   'PhysicalSizeYUnit': 'micron',
   'PhysicalSizeZ': 2.0,
   'PhysicalSizeZUnit': 'micron',
   'SizeC': 2,
   'SizeT': 5,
   'SizeX': 100,
   'SizeY': 20,
   'SizeZ': 10,
   'TimeIncrement': 3.0,
   'TimeIncrementUnit': 's',
   'axes': 'TZCYX',
   'shape': (5, 10, 2, 20, 100)}

.. image:: _static/figures/open_TZCYX_T5_Z10_C2_Z00_C0_T0_in_napari.jpg
   :target: _static/figures/open_TZCYX_T5_Z10_C2_Z00_C0_T0_in_napari.jpg
   :alt: The read multi-file OME-TIFF series opened in napari.

Note that this only works for multi-file OME-TIFF series where each individual TIFF file
contains the necessary OME metadata to correctly sort and assemble the files into a
multidimensional dataset. You cannot simply provide a list of arbitrary TIFF files and 
expect OMIO to assemble them correctly without the required OME metadata, even though 
the single TIFF files’ names may contain hints about their position in the series 
(for example Z-slice or time point).