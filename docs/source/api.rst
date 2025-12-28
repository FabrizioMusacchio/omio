API Reference
=============

OMIO's public API functions


.. currentmodule:: omio.omio

High level I/O
---------------------------
OMIO's main high-level API functions for reading, writing, and converting
microscopy image data:

.. autofunction:: imread
.. autofunction:: imwrite
.. autofunction:: imconvert
.. autofunction:: bids_batch_convert

Checks and utilities
---------------------------
API functions for miscellaneous checks and utilities, including the napari
viewer opener:

.. autofunction:: hello_world
.. autofunction:: OME_metadata_checkup
.. autofunction:: cleanup_omio_cache
.. autofunction:: open_in_napari

Readers
---------------------------
API functions to currently implemented image files readers:

.. autofunction:: read_tif
.. autofunction:: read_czi
.. autofunction:: read_thorlabs_raw

Metadata and image helpers
---------------------------
API functions to create empty metadata/image structures and to update
metadata from image data:

.. autofunction:: create_empty_metadata
.. autofunction:: create_empty_image
.. autofunction:: update_metadata_from_image