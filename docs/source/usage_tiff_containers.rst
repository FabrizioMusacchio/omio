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

Inspecting the ``"Annotations"`` in the retrieved metadata shows that OMIO has detected a
multi-series TIFF file (``'OMIO_MultiSeriesDetected': True``) which initially contained
two series with axes ``['YXS', 'YXS']`` and shapes ``[[16, 16, 3], [16, 16, 3]]``.

Thus, the two series seem to be compatible for concatenation along a new axis. However,
OMIO does not infer, by intention, any such relationships and only reads the first series
(series 0) with shape ``(16, 16, 3)`` and axes ``YXS``.

The reason for this policy is to avoid ambiguous interpretations of multi-series TIFF
files, which may contain series with different dimensionalities, axes, or photometric
interpretations.

.. code-block:: python

   fname_multi_series = "example_data/tif_dummy_data/multiseries_tif/multiseries_rgb_with_unequal_series.tif"
   image_multi_series, metadata_multi_series = om.imread(fname_multi_series)
   pprint.pprint(metadata_multi_series)

   fname_multi_series = "example_data/tif_dummy_data/multiseries_tif/multiseries_rgb_minisblack_mixture.tif"
   image_multi_series, metadata_multi_series = om.imread(fname_multi_series)
   pprint.pprint(metadata_multi_series)

   fname_multi_series = "example_data/tif_dummy_data/multiseries_tif/multiseries_minisblack.tif"
   image_multi_series, metadata_multi_series = om.imread(fname_multi_series)
   pprint.pprint(metadata_multi_series)

   fname_multi_series = "example_data/tif_dummy_data/paginated_tif/paginated_TCYXS.ome.tif"
   image_multi_series, metadata_multi_series = om.imread(fname_multi_series)
   pprint.pprint(metadata_multi_series)

If you want to process all series in a multi-series TIFF file, you have to manually
separate them with tools such as ImageJ/Fiji and store each series in its own
single-series TIFF file.


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
array along with the associated OME-compliant metadata.

.. code-block:: python

   fname_multifile_ometiff = "example_data/tif_dummy_data/tif_ome_multi_file_series/TZCYX_T5_Z10_C2_Z00_C0_T0.ome.tif"
   image_multifile_ometiff, metadata_multifile_ometiff = om.imread(fname_multifile_ometiff)

   print(f"Multi-file OME-TIFF image shape: {image_multifile_ometiff.shape}")
   pprint.pprint(metadata_multifile_ometiff)
   om.open_in_napari(image_multifile_ometiff, metadata_multifile_ometiff, fname_multifile_ometiff)

Note that this only works for multi-file OME-TIFF series where each individual TIFF file
contains the necessary OME metadata to correctly sort and assemble the files into a
multidimensional dataset.

You cannot simply provide a list of arbitrary TIFF files and expect OMIO to assemble
them correctly without the required OME metadata, even though the single TIFF files’
names may contain hints about their position in the series (for example Z-slice or
time point).