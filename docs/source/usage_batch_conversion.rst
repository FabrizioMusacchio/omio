Batch processing over a BIDS-like tree
========================================

The examples below assume the following imports:

.. code-block:: python

   import omio as om
   import pprint


OMIO's flexible batch processing function
-----------------------------------------

OMIO provides ``bids_batch_process`` to process entire BIDS-like microscopy
projects in one robust batch run. The function discovers image files, optionally
follows arbitrary folder-token levels, processes each image independently, skips
already processed outputs when requested, and writes persistent run/error reports.

A typical BIDS-like project can look like this::

   project_root
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

The structure is intentionally flexible. A simple project might use one level
below each subject, for example ``project_root / ID0001 / TP000 / image.tif``.
A deeper Thorlabs-style project might use multiple folder-token levels, for
example ``project_root / ID0001 / DC000_FOV1 / TL_000 / Image_001_001.raw``.


Discovery rules
---------------

``bids_batch_process`` follows explicit discovery rules:

* ``project_root`` is the root folder.
* If ``subject_ids`` is provided, only those subject folders are processed.
* If ``subject_ids=None``, all direct child folders whose names start with
  ``subject_prefix`` are processed.
* ``tag_folder_levels`` defines flexible folder-token levels below each subject.
* Each level may contain multiple tokens. Tokens are matched by containment.
* Empty levels, ``None``, ``()``, and ``[]`` inside an explicitly provided list
  are skipped.
* If ``tag_folder_levels=None``, OMIO uses one default experiment level matching
  ``"TP"``.

For example:

.. code-block:: python

   tag_folder_levels = [
       ("DC000_FOV", "DA000_FOV"),
       ("TL_000",)]

This means:

* Below each subject, find folders whose names contain either ``DC000_FOV`` or
  ``DA000_FOV``.
* Below each of those folders, find folders whose names contain ``TL_000``.
* Search the final matched folders for image files.


File-type and exclude filtering
-------------------------------

If ``image_patterns=None``, OMIO uses sensible microscopy defaults:

.. code-block:: python

   ("*.ome.tif", "*.ome.tiff", "*.tif", "*.tiff", "*.czi", "*.lsm", "*.raw")

If the user provides ``image_patterns`` explicitly, only those glob patterns are
processed:

.. code-block:: python

   image_patterns = ("*.ome.tif", "*.czi")

``exclude_name_contains`` filters both files and folders by name. If any token is
contained in a file or folder name, that item is skipped during discovery:

.. code-block:: python

   exclude_name_contains = ("Preview", "ROIMask")

OME multi-file TIFF series are collapsed during discovery by default. In other
words, if one logical OME-TIFF dataset is stored across multiple TIFF files,
``bids_batch_process`` keeps one representative file as the batch input and
``om.imread`` then loads the full multi-file series. To intentionally process
every TIFF member separately, set ``collapse_ome_multifile_series=False``.


Basic batch processing
----------------------

By default, ``bids_batch_process`` loads each discovered image with ``om.imread``,
performs an identity processing step, and writes an OME-TIFF with ``om.imwrite``.
The default processing behavior thus is image conversion to OME-TIFF with correct axis
order and metadata normalization.
Outputs are written into an ``omio_converted`` folder next to the input image.
Use ``output_folder_name`` to control the output location. Relative paths are
resolved below each discovered image folder, while absolute paths are used
directly. ``save_options`` is reserved for writer options passed to ``om.imwrite``,
for example ``overwrite`` or ``compression_level``.

.. code-block:: python

   result = om.bids_batch_process(
      project_root          = "example_data/tif_dummy_data/BIDS_project_example",
      subject_ids           = None,
      subject_prefix        = "ID",
      tag_folder_levels     = [("TP",)],
      image_patterns        = None,
      exclude_name_contains = ("Preview",),
      collapse_ome_multifile_series = True,
      output_folder_name    = "omio_converted",
      skip_processed        = True,
      load_options          = {"zarr_store": "disk",
                               "reuse_disk_cache": True},
      save_options          = {"overwrite": False},
      verbose               = True)

   # Example for a single absolute output folder:
   # output_folder_name="/Users/me/omio_batch_outputs"

   print(f"Discovered: {len(result.discovered)}")
   print(f"Processed:  {len(result.processed)}")
   print(f"Skipped:    {len(result.skipped)}")
   print(f"Failed:     {len(result.failed)}")
   print(result.report_path)

OME-TIFF multi-file series continue to use OMIO's standard multi-file handling in
batch mode. With ``collapse_ome_multifile_series=True`` (the default), the discovery
step keeps one representative TIFF member per logical series, and the subsequent
``om.imread`` call reads and converts the complete multi-file OME-TIFF dataset as
one image.


Folder-stack tags
-----------------

Some acquisitions are intentionally split across multiple tagged subfolders, for
example ``time01``, ``time02``, and ``time03`` below the same experiment folder.
The batch processor can discover these folders and pass the first matching
folder group through OMIO's established folder-stack merge path so that the matching
co-folders are loaded and merged into one image before processing and saving.

Pass ``folder_stacks`` as a tag or list of tags. If ``folder_stacks=None`` (the
default), no folder-stack discovery is performed.

.. code-block:: python

   result = om.bids_batch_process(
      project_root      = "example_data/tif_dummy_data/BIDS_project_example",
      subject_prefix    = "ID",
      tag_folder_levels = [("TP005",)],
      folder_stacks     = ("FOV1",),
      merge_along_axis  = "T",
      zeropadding       = True,
      output_folder_name = "omio_converted_FOV1",
      verbose           = True)

In this example, OMIO searches each final ``TP005`` folder for child folders whose
names start with ``FOV1`` and merges them along the time axis. Normal image files
in the same final folder are still discovered and processed as regular file inputs.


Skip and error behavior
-----------------------

If ``skip_processed=True`` and the expected output already exists, OMIO skips the
file and records it as already converted. This is not treated as an error, and
the text run report marks the file as ``[CONVERTED]``.

Every real per-file error is caught when ``continue_on_error=True``. The batch
continues with the next file and records:

* timestamp
* input path
* project-relative path
* failure stage, for example ``"load"``, ``"process"``, or ``"save"``
* exception type and message
* output directory or output path, where available
* optional extra information

At the end, OMIO prints compact processed/skipped/failed counts and lists skipped
or failed image paths.


Persistent run and error reports
--------------------------------

OMIO writes two persistent report types into ``project_root``:

* ``omio_batch_run_report.yaml`` and ``omio_batch_run_report.txt`` are updated
  across runs and keep a per-file run history.
* ``omio_batch_error_report_<timestamp>.txt`` is written only when failures
  occurred. Shorter local error reports are also written next to affected files.

The human-readable run report looks like this::

   OMIO batch run report
   Project root: /path/to/project_root
   Last updated: 2026-08-20_09-10-44

   ├─ ID000001/
   │  ├─ TP000/
   │  │  └─ image_01.ome.tif [CONVERTED]
   │  │     output: ID000001/TP000/omio_converted/image_01_omio_converted.ome.tif
   │  │     runs:
   │  │       - 2026-08-20_09-10-22 | processed | omio_batch_process
   │  │       - 2026-08-20_09-10-44 | skipped/already converted
   │  └─ TP001/
   │     └─ image_01.raw [FAILED]
   │        runs:
   │          - 2026-08-20_09-11-01 | failed | load | ValueError: ...


Creating Thorlabs RAW YAML sidecars from reports
------------------------------------------------

For Thorlabs RAW files with missing, incomplete, or inconsistent XML metadata,
the root error report contains an editable Python dictionary named
``OMIO_BATCH_FAILED_RAW_FILES``. The intended workflow is:

1. Run ``bids_batch_process`` once and let OMIO collect RAW failures.
2. Open the root ``omio_batch_error_report_<timestamp>.txt``.
3. Edit the ``template_metadata`` values for the failed RAW files.
4. Create YAML sidecars in batch:

.. code-block:: python

   result = om.batch_create_thorlabs_raw_yaml_templates(
       project_root="my_project",
       report_name="omio_batch_error_report_2026-08-20_12-00-00.txt",
       overwrite_existing=False,
       verbose=True)

   print(f"Created YAML templates: {len(result.created)}")
   print(f"Skipped RAW files:      {len(result.skipped)}")


Flexible batch processing with custom callables
------------------------------------------------

Advanced callers can replace the three central stages with custom callables:

.. code-block:: python

   result = om.bids_batch_process(
      project_root       = "my_project",
      load_func          = my_load_function,
      process_func       = my_processing_function,
      save_func          = my_save_function,
      load_options       = {...},
      processing_options = {...},
      save_options       = {...})

The intention behind this design is to enable flexible batch pipelines
that can load, process, and save images in one seamless pipeline.
Processing can be any arbitrary function, for example a denoising step,
projection, or a custom analysis. The load and save functions can also be
replaced with custom implementations, for example to read from or write to
non-standard file formats, databases, or cloud storage.

If no custom callable is provided, OMIO uses its standard behavior:

* ``load_func=None`` uses ``om.imread``
* ``process_func=None`` keeps image and metadata unchanged
* ``save_func=None`` uses ``om.imwrite``

This means that you can also replace only one stage. For example, the following
batch keeps OMIO's default loading and saving behavior, but inserts a custom
Z-projection step between them. The processing callable receives a canonical OMIO
image in ``TZCYX`` order, the matching metadata dictionary, the current batch
record, and the supplied ``processing_options``.

.. code-block:: python

   import numpy as np
   import omio as om

   def z_projection_process(*, image, metadata, record, processing_options):
       projection = processing_options.get("projection", "max")
       axes = metadata.get("axes", "TZCYX")
       if "Z" not in axes:
           return image, metadata, {"details": "no Z axis found; image unchanged"}

       z_axis = axes.index("Z")
       z_slices = image.shape[z_axis]
       if z_slices <= 1:
           return image, metadata, {"details": f"Z={z_slices}; image unchanged"}

       if projection == "max":
           image_projected = np.max(image, axis=z_axis, keepdims=True)
       elif projection == "mean":
           image_projected = np.mean(image, axis=z_axis, keepdims=True)
       elif projection == "median":
           image_projected = np.median(image, axis=z_axis, keepdims=True)
       elif projection == "std":
           image_projected = np.std(image, axis=z_axis, keepdims=True)
       elif projection == "var":
           image_projected = np.var(image, axis=z_axis, keepdims=True)
       else:
           raise ValueError(
               "projection must be one of {'max', 'mean', 'median', 'std', 'var'}")

       metadata_projected = om.update_metadata_from_image(
           metadata,
           image_projected,
           verbose=False)
       return image_projected, metadata_projected, {
           "details": f"{projection} projection along Z; {z_slices} slices -> 1 slice"}

   result = om.bids_batch_process(
       project_root="my_project",
       subject_prefix="ID",
       tag_folder_levels=[("TP",)],
       image_patterns=("*.ome.tif", "*.tif"),
       output_folder_name="omio_z_projection",
       output_suffix="_z_projected",
       process_func=z_projection_process,
       processing_options={
           "projection": "max"},
       save_options={
           "overwrite": False})

The custom processing callable should return either ``(image, metadata)`` or
``(image, metadata, info)``. The optional ``info`` dictionary can add compact
details to successful run-report entries. If ``process_func`` is provided and
``method_name`` is not set explicitly, OMIO records the processing callable's
function name in the run report. The report also includes the supplied
``processing_options`` so that successful entries remain interpretable later,
for example::

   2026-08-20_12-00-00 | processed | z_projection_process | max projection along Z; 5 slices -> 1 slice; processing_options: projection='max'
