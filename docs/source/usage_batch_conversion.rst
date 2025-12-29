Batch Conversion over a BIDS-like Tree
========================================

The examples below assume the following imports:

.. code-block:: python

   import omio as om
   import pprint

OMIO’s Batch Conversion Function
-----------------------------------

OMIO provides a convenience function called ``bids_batch_convert`` to convert entire
folders of image files into OME-TIFF format in a single function call. It is required
that the folder structures follow the BIDS-like naming conventions, where sub-folders
are named according to specific tags such as ``sub-<subject_id>``, ``ses-<session_id>``,
``acq-<acquisition_id>``, ``run-<run_id>``, etc.
Here is a general example of a BIDS-like folder structure::

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
provide the root folder path as ``fname`` argument, the subject folder tag as ``sub``
argument, and the experiment folder tag as ``exp`` argument to ``bids_batch_convert``
as minimum:

.. code-block:: python

   fname = "example_data/tif_dummy_data/BIDS_project_example/"
   id_tag = "ID"
   exp_tag = "TP001"  # contains tif files

   om.bids_batch_convert(
       fname,
       sub=id_tag,
       exp=exp_tag,
       relative_path="omio_bids_converted")

Of course, ``bids_batch_convert`` has the same functionalities as ``imconvert``, ``imread``,
and ``imwrite``, so that it is able to, for example, handle Thorlabs RAW files:

.. code-block:: python

   exp_tag = "TP003"  # contains thorlabs raw files

   om.bids_batch_convert(
       fname,
       sub=id_tag,
       exp=exp_tag,
       relative_path="omio_bids_converted")

Also tagged folder stacks can be processed, while the arguments to be provided differ
slightly from those of ``imread`` and ``imconvert``. Here, you have to provide the
``tagfolder`` argument to indicate the tag prefix of the tagfolders to be processed:

.. code-block:: python

   exp_tag = "TP005"  # contains tagged folder stacks
   stackfolder_tag = "FOV1"

   om.bids_batch_convert(
       fname,
       sub=id_tag,
       exp=exp_tag,
       tagfolder=stackfolder_tag,
       merge_tagfolders=True,
       merge_along_axis="T",
       relative_path="omio_bids_converted_FOV1")

Note: Since ``bids_batch_convert`` processes multiple files and folders in a batch and
additionally provides the ``tagfolder``, it is not necessary to set the ``relative_path``
one level up as done before with ``imconvert``.

``bids_batch_convert`` can also handle multi-file OME-TIFF series correctly:

.. code-block:: python

   exp_tag = "TP006"  # contains multi-file ome-tiff series

   om.bids_batch_convert(
       fname,
       sub=id_tag,
       exp=exp_tag,
       relative_path="omio_bids_converted")

   fname_converted = (
       "example_data/tif_dummy_data/BIDS_project_example/"
       "ID0001/TP006_tif_multi_file_stack/"
       "omio_bids_converted/"
       "TZCYX_T5_Z10_C2_Z00_C0_T0.ome.tif")

   image, metadata = om.imread(fname_converted)
   print(f"Multi-file OME-TIFF image shape: {image.shape} "
         f"with axes {metadata.get('axes', 'N/A')}")

   om.open_in_napari(image, metadata, fname_converted)

.. code-block:: text

   >>>
   Multi-file OME-TIFF image shape: (5, 10, 2, 20, 100) with axes TZCYX