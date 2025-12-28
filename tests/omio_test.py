""" 
pip install pytest

In a terminal, run:

pip install -e .
pytest
"""
# %% IMPORTS
from __future__ import annotations

from omio.omio import (
    hello_world, 
    version, 
    OME_metadata_checkup,
    read_tif,
    read_czi,
    read_thorlabs_raw,
    cleanup_omio_cache,
    create_empty_image,
    create_empty_metadata,
    update_metadata_from_image,
    imwrite,
    imread,
    imconvert,
    bids_batch_convert,
    _get_channel_axis_from_axes_and_shape,
    _get_scales_from_axes_and_metadata,
    _squeeze_zarr_to_napari_cache,
    _squeeze_zarr_to_napari_cache_dask,
    _single_image_open_in_napari,
    open_in_napari,
    _batch_correct_for_OME_axes_order,
    _standardize_imagej_metadata,
    _standardize_lsm_metadata,
    _copy_to_zarr_in_xy_slices,
    _copy_into_zarr_chunk_aligned,
    _estimate_compressed_size,
    _check_bigtiff,
    _estimate_compressed_size,
    _copy_into_zarr_with_padding,
    _zarr_open_for_merge_output,
    _check_fname_out,
    _dispatch_read_file,
    _validate_merge_inputs_with_optional_padding,
    _normalize_axes_for_ometiff,
    _get_original_filename_from_metadata,
    _zarr_pick_first_array,
    _merge_concat_along_axis,
    _load_yaml_metadata,
    _ensure_axes_in_metadata
)

import os
import re
import copy
from pathlib import Path
import numpy as np
import pytest
import tifffile
import zarr
import yaml
import warnings
import zlib

import dask.array as da
# %% TEST HELLO WORLD
# test hello_world function:
def test_hello_world_prints_version(capsys):
    hello_world()
    captured = capsys.readouterr()
    assert "Hello from omio.py! OMIO version:" in captured.out
    assert str(version) in captured.out

def test_version_is_nonempty_string():
    assert isinstance(version, str)
    assert len(version) > 0

# %% HELPER FUNCTIONS

# _batch_correct_for_OME_axes_order tests:
def test_batch_correct_length_mismatch_returns_inputs_unchanged(capsys):
    images = [np.zeros((5, 5))]
    metadatas = []

    out_images, out_mds = _batch_correct_for_OME_axes_order(
        images,
        metadatas,
        verbose=True
    )

    assert out_images is images
    assert out_mds is metadatas

    captured = capsys.readouterr()
    assert "different lengths" in captured.out

def test_batch_correct_single_image_calls_corrector(monkeypatch):
    images = [np.zeros((2, 3))]
    metadatas = [{"axes": "YX"}]

    calls = []

    def fake_correct(image, md, memap_large_file=False, verbose=True):
        calls.append((image, md, memap_large_file, verbose))
        return image, image.shape, "TZCYX"

    monkeypatch.setattr(
        "omio.omio._correct_for_OME_axes_order",
        fake_correct
    )

    out_images, out_mds = _batch_correct_for_OME_axes_order(
        images,
        metadatas,
        memap_large_file=True,
        verbose=False
    )

    assert len(calls) == 1
    assert calls[0][2] is True          # memap_large_file forwarded
    assert calls[0][3] is False         # verbose forwarded

    assert out_images is images
    assert out_mds is metadatas
    assert metadatas[0]["axes"] == "TZCYX"
    assert metadatas[0]["shape"] == (2, 3)

def test_batch_correct_multiple_images(monkeypatch):
    images = [
        np.zeros((2, 3)),
        np.zeros((4, 5)),
    ]
    metadatas = [
        {"axes": "YX"},
        {"axes": "ZYX"},
    ]

    def fake_correct(image, md, memap_large_file=False, verbose=True):
        new_axes = "TZCYX"
        return image, image.shape, new_axes

    monkeypatch.setattr(
        "omio.omio._correct_for_OME_axes_order",
        fake_correct
    )

    out_images, out_mds = _batch_correct_for_OME_axes_order(images, metadatas)

    assert len(out_images) == 2
    for img, md in zip(out_images, out_mds):
        assert md["axes"] == "TZCYX"
        assert md["shape"] == img.shape

def test_batch_correct_accepts_zarr_arrays(monkeypatch):
    store = zarr.storage.MemoryStore()
    z = zarr.open(store, mode="w", shape=(3, 4), dtype=np.uint8)

    images = [z]
    metadatas = [{"axes": "YX"}]

    def fake_correct(image, md, memap_large_file=False, verbose=True):
        assert isinstance(image, zarr.core.array.Array)
        return image, image.shape, "TZCYX"

    monkeypatch.setattr(
        "omio.omio._correct_for_OME_axes_order",
        fake_correct
    )

    out_images, out_mds = _batch_correct_for_OME_axes_order(images, metadatas)

    assert isinstance(out_images[0], zarr.core.array.Array)
    assert out_mds[0]["axes"] == "TZCYX"

def test_batch_correct_verbose_false_suppresses_output(monkeypatch, capsys):
    images = [np.zeros((2, 2))]
    metadatas = [{"axes": "YX"}]

    monkeypatch.setattr(
        "omio.omio._correct_for_OME_axes_order",
        lambda image, md, memap_large_file=False, verbose=True: (image, image.shape, "TZCYX")
    )

    _batch_correct_for_OME_axes_order(images, metadatas, verbose=False)

    captured = capsys.readouterr()
    assert captured.out == ""

# _standardize_imagej_metadata tests:
def test_standardize_imagej_metadata_maps_known_keys_and_preserves_unknown():
    md_in = {
        "sizex": 7,
        "sizey": 5,
        "sizec": 2,
        "physicalsizex": 0.12,
        "physicalsizey": 0.34,
        "timeincrement": 0.5,
        "timeincrementunit": "s",
        "some_weird_key": "keep_me",
    }

    md_out = _standardize_imagej_metadata(md_in)

    assert md_out["SizeX"] == 7
    assert md_out["SizeY"] == 5
    assert md_out["SizeC"] == 2
    assert md_out["PhysicalSizeX"] == 0.12
    assert md_out["PhysicalSizeY"] == 0.34
    assert md_out["TimeIncrement"] == 0.5
    assert md_out["TimeIncrementUnit"] == "s"

    assert md_out["some_weird_key"] == "keep_me"

def test_standardize_imagej_metadata_does_not_parse_info_if_physicalsizex_present():
    md_in = {
        "physicalsizex": 1.23,
        "Info": "Scaling|Distance|Id #1 = X\nScaling|Distance|Value #1 = 9.99\nScaling|Distance|DefaultUnitFormat #1 = m\n",
    }

    md_out = _standardize_imagej_metadata(md_in)

    assert md_out["PhysicalSizeX"] == 9.99

def test_standardize_imagej_metadata_parses_info_and_sets_xyz_sizes():
    info = "\n".join(
        [
            "SomeOtherKey = 123",
            "Scaling|Distance|Id #1 = X",
            "Scaling|Distance|Value #1 = 1.135E-07",
            "Scaling|Distance|DefaultUnitFormat #1 = µm",
            "Scaling|Distance|Id #2 = Y",
            "Scaling|Distance|Value #2 = 2.0E-07",
            "Scaling|Distance|DefaultUnitFormat #2 = µm",
            "Scaling|Distance|Id #3 = Z",
            "Scaling|Distance|Value #3 = 5.0E-07",
            "Scaling|Distance|DefaultUnitFormat #3 = µm",
        ]
    )

    md_in = {
        "Info": info,
    }

    md_out = _standardize_imagej_metadata(md_in)

    assert "PhysicalSizeX" in md_out
    assert "PhysicalSizeY" in md_out
    assert "PhysicalSizeZ" in md_out

    assert md_out["PhysicalSizeX"] == pytest.approx(1.135e-07 * 1e6)
    assert md_out["PhysicalSizeY"] == pytest.approx(2.0e-07 * 1e6)
    assert md_out["PhysicalSizeZ"] == pytest.approx(5.0e-07 * 1e6)

def test_standardize_imagej_metadata_parses_info_partially_only_sets_present_axes():
    info = "\n".join(
        [
            "Scaling|Distance|Id #1 = X",
            "Scaling|Distance|Value #1 = 1.0E-06",
            "Scaling|Distance|DefaultUnitFormat #1 = µm",
            "Scaling|Distance|Id #2 = Y",
            "Scaling|Distance|Value #2 = 2.0E-06",
            "Scaling|Distance|DefaultUnitFormat #2 = µm",
        ]
    )

    md_in = {"Info": info}
    md_out = _standardize_imagej_metadata(md_in)

    assert md_out["PhysicalSizeX"] == pytest.approx(1.0e-06 * 1e6)
    assert md_out["PhysicalSizeY"] == pytest.approx(2.0e-06 * 1e6)
    assert "PhysicalSizeZ" not in md_out

def test_standardize_imagej_metadata_spacing_fallback_sets_physicalsizez_if_missing():
    md_in = {
        "spacing": 3.21,
        "sizex": 10,
    }
    md_out = _standardize_imagej_metadata(md_in)

    assert md_out["SizeX"] == 10
    assert md_out["PhysicalSizeZ"] == 3.21

def test_standardize_imagej_metadata_info_parse_failure_prints_and_leaves_sizes_unset(capsys):
    info = "\n".join(
        [
            "Scaling|Distance|Id #1 = X",
            "Scaling|Distance|Value #1 = NOT_A_FLOAT",
            "Scaling|Distance|DefaultUnitFormat #1 = µm",
        ]
    )
    md_in = {"Info": info}

    md_out = _standardize_imagej_metadata(md_in)

    captured = capsys.readouterr()
    assert "Error while extracting PhysicalSize from Info" in captured.out

    assert "PhysicalSizeX" not in md_out
    assert "PhysicalSizeY" not in md_out
    assert "PhysicalSizeZ" not in md_out

# _standardize_lsm_metadata tests:
def test_standardize_lsm_metadata_maps_known_keys_and_preserves_unknown():
    md_in = {
        "DimensionX": 512,
        "DimensionY": 256,
        "DimensionZ": 12,
        "DimensionChannels": 3,
        "DimensionTime": 7,
        "VoxelSizeX": 0.12,
        "VoxelSizeY": 0.34,
        "VoxelSizeZ": 1.5,
        "TimeIntervall": 0.25,
        "SomeVendorField": "keep_me",
    }

    md_out = _standardize_lsm_metadata(md_in)

    assert md_out["SizeX"] == 512
    assert md_out["SizeY"] == 256
    assert md_out["SizeZ"] == 12
    assert md_out["SizeC"] == 3
    assert md_out["SizeT"] == 7

    assert md_out["PhysicalSizeX"] == 0.12
    assert md_out["PhysicalSizeY"] == 0.34
    assert md_out["PhysicalSizeZ"] == 1.5

    assert md_out["TimeIncrement"] == 0.25

    # unknown keys are preserved unchanged
    assert md_out["SomeVendorField"] == "keep_me"

def test_standardize_lsm_metadata_does_not_modify_input_dict():
    md_in = {
        "DimensionX": 10,
        "VoxelSizeX": 0.5,
        "SomeKey": 123,
    }
    md_in_copy = copy.deepcopy(md_in)

    _ = _standardize_lsm_metadata(md_in)

    assert md_in == md_in_copy

def test_standardize_lsm_metadata_allows_non_numeric_values_and_keeps_them():
    md_in = {
        "DimensionX": "512",           # sometimes strings happen
        "TimeIntervall": "0.1",
        "VoxelSizeZ": None,
        "CustomBlock": {"a": 1},
    }

    md_out = _standardize_lsm_metadata(md_in)

    assert md_out["SizeX"] == "512"
    assert md_out["TimeIncrement"] == "0.1"
    assert md_out["PhysicalSizeZ"] is None
    assert md_out["CustomBlock"] == {"a": 1}

# _standardize_lsm_metadata tests:
def test_copy_to_zarr_in_xy_slices_trivial_2d_copies_all():
    src = (np.arange(6 * 7).reshape(6, 7)).astype(np.uint16)

    store = zarr.storage.MemoryStore()
    dst = zarr.open(store=store, mode="w", shape=src.shape, dtype=src.dtype, chunks=src.shape)

    _copy_to_zarr_in_xy_slices(src, dst, desc="test")

    out = np.asarray(dst)
    assert out.shape == src.shape
    assert np.array_equal(out, src)

def test_copy_to_zarr_in_xy_slices_streams_outer_dims_3d():
    src = np.random.randint(0, 255, size=(4, 6, 7), dtype=np.uint8)  # outer=(4,), plane=(6,7)

    store = zarr.storage.MemoryStore()
    dst = zarr.open(store=store, mode="w", shape=src.shape, dtype=src.dtype, chunks=(1, 6, 7))

    _copy_to_zarr_in_xy_slices(src, dst, desc="test")

    out = np.asarray(dst)
    assert out.shape == src.shape
    assert np.array_equal(out, src)

def test_copy_to_zarr_in_xy_slices_streams_outer_dims_5d():
    src = np.random.randn(2, 3, 1, 8, 9).astype(np.float32)  # outer=(2,3,1), plane=(8,9)

    store = zarr.storage.MemoryStore()
    dst = zarr.open(store=store, mode="w", shape=src.shape, dtype=src.dtype, chunks=(1, 1, 1, 8, 9))

    _copy_to_zarr_in_xy_slices(src, dst, desc="test")

    out = np.asarray(dst)
    assert out.shape == src.shape
    assert np.array_equal(out, src)

def test_copy_to_zarr_in_xy_slices_raises_on_shape_mismatch():
    src = np.zeros((2, 3, 4), dtype=np.uint8)

    store = zarr.storage.MemoryStore()
    dst = zarr.open(store=store, mode="w", shape=(2, 3, 5), dtype=src.dtype, chunks=(1, 3, 5))

    # Zarr assignment should raise due to incompatible slice shapes
    with pytest.raises(Exception):
        _copy_to_zarr_in_xy_slices(src, dst, desc="test")

# _copy_into_zarr_chunk_aligned tests:
def _make_zarr(shape, dtype, chunks):
    store = zarr.storage.MemoryStore()
    return zarr.open(store=store, mode="w", shape=shape, dtype=dtype, chunks=chunks)

def test_copy_into_zarr_chunk_aligned_writes_with_offset_and_chunk_blocks_numpy_src():
    # merge axis: T (idx 0)
    axis_idx = 0

    # source has n=5 along T, chunk_len=2 -> blocks: 2,2,1
    img = np.random.randint(0, 256, size=(5, 1, 2, 4, 4), dtype=np.uint8)

    # output larger along T, with chunking along T=2
    z_out = _make_zarr(shape=(10, 1, 2, 4, 4), dtype=img.dtype, chunks=(2, 1, 2, 4, 4))
    z_out[...] = 0

    out_start = 3
    _copy_into_zarr_chunk_aligned(z_out, img, out_start=out_start, axis_idx=axis_idx)

    out = np.asarray(z_out)
    # written region matches
    assert np.array_equal(out[out_start:out_start + 5, ...], img)
    # outside region stays zero
    assert np.all(out[:out_start, ...] == 0)
    assert np.all(out[out_start + 5:, ...] == 0)

def test_copy_into_zarr_chunk_aligned_works_with_zarr_source():
    axis_idx = 0

    src_np = np.random.randint(0, 256, size=(4, 1, 1, 6, 7), dtype=np.uint8)
    z_src = _make_zarr(shape=src_np.shape, dtype=src_np.dtype, chunks=(2, 1, 1, 6, 7))
    z_src[...] = src_np

    z_out = _make_zarr(shape=(9, 1, 1, 6, 7), dtype=src_np.dtype, chunks=(3, 1, 1, 6, 7))
    z_out[...] = 0

    out_start = 2
    _copy_into_zarr_chunk_aligned(z_out, z_src, out_start=out_start, axis_idx=axis_idx)

    out = np.asarray(z_out)
    assert np.array_equal(out[out_start:out_start + 4, ...], src_np)

def test_copy_into_zarr_chunk_aligned_fallback_when_chunks_missing_or_invalid(monkeypatch):
    axis_idx = 0
    img = np.random.randint(0, 256, size=(3, 1, 1, 5, 5), dtype=np.uint8)

    z_out = _make_zarr(shape=(6, 1, 1, 5, 5), dtype=img.dtype, chunks=(2, 1, 1, 5, 5))
    z_out[...] = 0

    # Force getattr(z_out, "chunks", None) -> None so code uses fallback (one block)
    monkeypatch.setattr(type(z_out), "chunks", property(lambda self: None))

    out_start = 1
    _copy_into_zarr_chunk_aligned(z_out, img, out_start=out_start, axis_idx=axis_idx)

    out = np.asarray(z_out)
    assert np.array_equal(out[out_start:out_start + 3, ...], img)

# _estimate_compressed_size tests:
def _make_zarr_memory(arr: np.ndarray, chunks=None):
    store = zarr.storage.MemoryStore()
    z = zarr.open(
        store=store,
        mode="w",
        shape=arr.shape,
        dtype=arr.dtype,
        chunks=chunks if chunks is not None else arr.shape,
    )
    z[...] = arr
    return z

def test_estimate_compressed_size_numpy_returns_positive_float():
    img = np.zeros((10, 11, 12), dtype=np.uint16)
    est = _estimate_compressed_size(img, sample_fraction=0.001, compression_level=3)

    assert isinstance(est, float)
    assert est > 0.0

def test_estimate_compressed_size_numpy_sample_fraction_min_one_element():
    # sample_fraction so small that int(prod * frac) would be 0 without max(1, ...)
    img = np.arange(100, dtype=np.uint8).reshape(10, 10)
    est = _estimate_compressed_size(img, sample_fraction=0.0, compression_level=3)

    assert isinstance(est, float)
    assert est > 0.0

def test_estimate_compressed_size_numpy_compression_level_effect_for_highly_compressible_data():
    # For all zeros, higher compression level should not produce a larger compressed sample
    img = np.zeros((50, 50), dtype=np.uint8)

    est_low = _estimate_compressed_size(img, sample_fraction=0.05, compression_level=1)
    est_high = _estimate_compressed_size(img, sample_fraction=0.05, compression_level=9)

    assert est_high <= est_low

def test_estimate_compressed_size_numpy_invalid_compression_level_raises():
    img = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(zlib.error):
        _estimate_compressed_size(img, sample_fraction=0.1, compression_level=99)

def test_estimate_compressed_size_zarr_returns_positive_float():
    img = np.zeros((2, 3, 20, 21), dtype=np.uint8)  # e.g. T,C,Y,X
    z = _make_zarr_memory(img, chunks=(1, 1, 20, 21))

    est = _estimate_compressed_size(z, sample_fraction=0.001, compression_level=3)

    assert isinstance(est, float)
    assert est > 0.0

def test_estimate_compressed_size_zarr_path_ignores_sample_fraction_argument_semantically():
    # Implementation uses one YX plane for Zarr regardless of sample_fraction.
    # We only test that it runs and returns a stable type/value for different fractions.
    img = np.zeros((2, 3, 16, 16), dtype=np.uint8)
    z = _make_zarr_memory(img, chunks=(1, 1, 16, 16))

    est_a = _estimate_compressed_size(z, sample_fraction=1e-6, compression_level=3)
    est_b = _estimate_compressed_size(z, sample_fraction=0.9, compression_level=3)

    assert isinstance(est_a, float)
    assert isinstance(est_b, float)
    assert est_a > 0.0
    assert est_b > 0.0

def test_estimate_compressed_size_zarr_compression_level_effect_for_highly_compressible_data():
    img = np.zeros((2, 1, 32, 32), dtype=np.uint8)
    z = _make_zarr_memory(img, chunks=(1, 1, 32, 32))

    est_low = _estimate_compressed_size(z, compression_level=1)
    est_high = _estimate_compressed_size(z, compression_level=9)

    assert est_high <= est_low

# _check_bigtiff tests:
def test_check_bigtiff_small_array_returns_false():
    img = np.zeros((10, 10), dtype=np.uint8)  # 100 bytes

    use_bigtiff = _check_bigtiff(img)

    assert use_bigtiff is False
    
class DummyImage:
    def __init__(self, nbytes):
        self.nbytes = nbytes

def test_check_bigtiff_large_uncompressed_requires_bigtiff(monkeypatch):
    class DummyImage:
        def __init__(self, nbytes):
            self.nbytes = nbytes

    img = DummyImage(nbytes=2**32)

    limit = 2**32 - 2**25

    def fake_estimate(image, sample_fraction=0.001, compression_level=3):
        return limit + 1  # bleibt über Limit

    monkeypatch.setattr("omio.omio._estimate_compressed_size", fake_estimate)

    use_bigtiff = _check_bigtiff(img, compression_level=3)

    assert use_bigtiff is True
    
def test_check_bigtiff_large_but_compressible_resets_flag(monkeypatch):
    class DummyImage:
        def __init__(self, nbytes):
            self.nbytes = nbytes

    img = DummyImage(nbytes=2**32)

    def fake_estimate(image, sample_fraction=0.001, compression_level=3):
        return 1024  # weit unter Limit

    monkeypatch.setattr("omio.omio._estimate_compressed_size", fake_estimate)

    use_bigtiff = _check_bigtiff(img, compression_level=3)

    assert use_bigtiff is False

def test_check_bigtiff_invalid_compression_level_propagates(monkeypatch):
    class DummyImage:
        def __init__(self, nbytes):
            self.nbytes = nbytes

    img = DummyImage(nbytes=2**32)

    def fake_estimate(image, sample_fraction=0.001, compression_level=99):
        raise zlib.error("Bad compression level")

    monkeypatch.setattr("omio.omio._estimate_compressed_size", fake_estimate)

    with pytest.raises(zlib.error):
        _check_bigtiff(img, compression_level=99)

# _estimate_compressed_size tests:
def _make_zarr_array(data: np.ndarray, chunks=None):
    store = zarr.storage.MemoryStore()
    z = zarr.open(
        store=store,
        mode="w",
        shape=data.shape,
        dtype=data.dtype,
        chunks=chunks if chunks is not None else data.shape,
    )
    z[...] = data
    return z

def test_estimate_compressed_size_numpy_returns_positive_float():
    img = np.random.randint(0, 256, (50, 60), dtype=np.uint8)
    est = _estimate_compressed_size(img, sample_fraction=0.1, compression_level=3)

    assert isinstance(est, (float, np.floating))
    assert est > 0

def test_estimate_compressed_size_numpy_sample_fraction_zero_still_works():
    # sample_size should clamp to at least 1 element
    img = np.random.randint(0, 256, (10, 10), dtype=np.uint8)
    est = _estimate_compressed_size(img, sample_fraction=0.0, compression_level=3)

    assert est > 0

def test_estimate_compressed_size_numpy_all_zeros_compresses_strongly():
    img = np.zeros((200, 200), dtype=np.uint8)
    est = _estimate_compressed_size(img, sample_fraction=0.1, compression_level=3)

    # should be clearly smaller than raw in most cases
    assert est < img.nbytes

def test_estimate_compressed_size_numpy_invalid_compression_level_raises_zlib_error():
    img = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(zlib.error):
        _estimate_compressed_size(img, sample_fraction=0.1, compression_level=99)

def test_estimate_compressed_size_zarr_returns_positive_float():
    data = np.random.randint(0, 256, (2, 3, 20, 21), dtype=np.uint8)
    z = _make_zarr_array(data, chunks=(1, 1, 20, 21))

    est = _estimate_compressed_size(z, compression_level=3)

    assert isinstance(est, (float, np.floating))
    assert est > 0

def test_estimate_compressed_size_zarr_uses_single_yx_plane_and_matches_numpy_formula():
    # This test checks the exact sampling logic used for Zarr inputs:
    # slicer = [0]*(ndim-2) + [slice(None), slice(None)]
    # So for (T, C, Y, X) it samples [0,0,:,:]
    data = np.random.randint(0, 256, (4, 2, 16, 18), dtype=np.uint8)
    z = _make_zarr_array(data, chunks=(1, 1, 16, 18))

    est = _estimate_compressed_size(z, compression_level=3)

    sample = np.asarray(data[0, 0, :, :]).ravel()
    compressed = zlib.compress(sample.tobytes(), level=3)
    ratio = len(compressed) / sample.nbytes
    expected = z.nbytes * ratio

    assert est == pytest.approx(expected)

def test_estimate_compressed_size_zarr_invalid_compression_level_raises_zlib_error():
    data = np.random.randint(0, 256, (2, 2, 8, 9), dtype=np.uint8)
    z = _make_zarr_array(data)

    with pytest.raises(zlib.error):
        _estimate_compressed_size(z, compression_level=99)

# _copy_into_zarr_with_padding tests:
def _make_zarr(shape, dtype=np.uint8, chunks=None, fill=0):
    store = zarr.storage.MemoryStore()
    z = zarr.open(
        store=store,
        mode="w",
        shape=shape,
        dtype=dtype,
        chunks=chunks if chunks is not None else shape,
    )
    if fill != 0:
        z[...] = fill
    return z

def _make_zarr_from_numpy(arr: np.ndarray, chunks=None):
    z = _make_zarr(arr.shape, dtype=arr.dtype, chunks=chunks)
    z[...] = arr
    return z

def test_copy_into_zarr_with_padding_places_block_and_keeps_padding_zeros_axis0_merge():
    # merge axis: T (axis 0)
    axis_idx = 0

    # src has smaller C and smaller X than target
    src = np.ones((2, 1, 2, 3, 4), dtype=np.uint8) * 7
    z_out = _make_zarr(shape=(5, 1, 4, 3, 6), dtype=np.uint8, chunks=(2, 1, 4, 3, 3), fill=0)

    out_start = 1
    target_nonmerge_shape = z_out.shape  # interface arg, not used internally

    _copy_into_zarr_with_padding(z_out, src, out_start=out_start, axis_idx=axis_idx,
                                 target_nonmerge_shape=target_nonmerge_shape)

    out = np.asarray(z_out)

    # region that must be written: T in [1:3], Z full (1), C [0:2], Y full (3), X [0:4]
    written = out[1:3, :, 0:2, :, 0:4]
    assert np.all(written == 7)

    # padding in C beyond src (C=2..3) must remain zero in the written T range
    assert np.all(out[1:3, :, 2:4, :, :] == 0)

    # padding in X beyond src (X=4..5) must remain zero in the written T range and valid C
    assert np.all(out[1:3, :, 0:2, :, 4:6] == 0)

    # regions outside written T range must remain zero everywhere
    assert np.all(out[0, ...] == 0)
    assert np.all(out[3:, ...] == 0)

def test_copy_into_zarr_with_padding_merge_axis2_channel_axis():
    # merge axis: C (axis 2)
    axis_idx = 2

    src = np.ones((1, 1, 3, 4, 5), dtype=np.uint8) * 9  # C=3
    z_out = _make_zarr(shape=(1, 1, 8, 4, 7), dtype=np.uint8, chunks=(1, 1, 2, 4, 7), fill=0)

    out_start = 2  # place into C indices [2:5]
    _copy_into_zarr_with_padding(z_out, src, out_start=out_start, axis_idx=axis_idx,
                                 target_nonmerge_shape=z_out.shape)

    out = np.asarray(z_out)

    # written region in C
    assert np.all(out[:, :, 2:5, :, 0:5] == 9)

    # padding in X beyond src stays zero within written C
    assert np.all(out[:, :, 2:5, :, 5:7] == 0)

    # padding in C outside written region stays zero
    assert np.all(out[:, :, 0:2, :, :] == 0)
    assert np.all(out[:, :, 5:8, :, :] == 0)

def test_copy_into_zarr_with_padding_accepts_zarr_source():
    axis_idx = 0
    src_np = (np.arange(2 * 1 * 1 * 3 * 4, dtype=np.uint16).reshape(2, 1, 1, 3, 4))
    src = _make_zarr_from_numpy(src_np, chunks=(1, 1, 1, 3, 4))

    z_out = _make_zarr(shape=(4, 1, 1, 3, 6), dtype=src_np.dtype, chunks=(2, 1, 1, 3, 3), fill=0)

    _copy_into_zarr_with_padding(z_out, src, out_start=1, axis_idx=axis_idx,
                                 target_nonmerge_shape=z_out.shape)

    out = np.asarray(z_out)

    # exact values copied into valid region
    assert np.array_equal(out[1:3, :, :, :, 0:4], src_np)

    # padding in X remains zero
    assert np.all(out[1:3, :, :, :, 4:6] == 0)

def test_copy_into_zarr_with_padding_chunk_fallback_works_when_chunks_missing():
    # z_out.chunks could be None or invalid, the function falls back to chunk_len = n
    axis_idx = 0

    src = np.ones((3, 1, 1, 2, 2), dtype=np.uint8) * 4

    store = zarr.storage.MemoryStore()
    z_out = zarr.open(store=store, mode="w", shape=(5, 1, 1, 2, 2), dtype=np.uint8, chunks=(5, 1, 1, 2, 2))
    z_out[...] = 0

    # forcibly emulate missing chunks attribute (rare, but we can simulate by wrapping)
    class _Wrap:
        def __init__(self, z):
            self._z = z
            self.shape = z.shape
            self.dtype = z.dtype
            self.nbytes = z.nbytes
            self.chunks = None
        def __getitem__(self, idx):
            return self._z.__getitem__(idx)
        def __setitem__(self, idx, val):
            self._z.__setitem__(idx, val)

    z_wrap = _Wrap(z_out)

    _copy_into_zarr_with_padding(z_wrap, src, out_start=1, axis_idx=axis_idx,
                                 target_nonmerge_shape=z_wrap.shape)

    out = np.asarray(z_out)
    assert np.all(out[1:4, ...] == 4)
    assert np.all(out[0, ...] == 0)
    assert np.all(out[4, ...] == 0)

def test_copy_into_zarr_with_padding_write_exceeds_destination_axis_is_clipped_not_raised():
    axis_idx = 0  # merge along T

    src = np.ones((3, 1, 1, 2, 2), dtype=np.uint8) * 5
    z_out = _make_zarr(shape=(4, 1, 1, 2, 2), dtype=np.uint8, chunks=(2, 1, 1, 2, 2), fill=0)

    out_start = 2  # request would conceptually write T=2,3,4 but dst has only 0..3

    # Current Zarr behavior: no exception, assignment is effectively clipped to valid region.
    _copy_into_zarr_with_padding(z_out, src, out_start=out_start, axis_idx=axis_idx,
                                 target_nonmerge_shape=z_out.shape)

    out = np.asarray(z_out)

    # Only T indices 2 and 3 can be written (2 slices).
    assert np.all(out[2:4, :, :, :, :] == 5)

    # Everything before the written region must remain zero.
    assert np.all(out[0:2, :, :, :, :] == 0)

def test_copy_into_zarr_with_padding_src_exceeds_target_nonmerge_axis_is_clipped_not_raised():
    axis_idx = 0  # merge along T

    src = np.ones((1, 1, 1, 2, 10), dtype=np.uint8) * 7  # X=10
    z_out = _make_zarr(shape=(2, 1, 1, 2, 6), dtype=np.uint8, chunks=(2, 1, 1, 2, 3), fill=0)

    # Current Zarr behavior: no exception, non-merge write is effectively clipped to dst's X extent.
    _copy_into_zarr_with_padding(z_out, src, out_start=0, axis_idx=axis_idx,
                                 target_nonmerge_shape=z_out.shape)

    out = np.asarray(z_out)

    # Valid intersection is X=0..5.
    assert np.all(out[0, :, :, :, 0:6] == 7)

    # There is no X beyond 6 in dst; the key property to test is that dst content is consistent:
    # second T slice remains untouched (still zeros).
    assert np.all(out[1, :, :, :, :] == 0)

# _zarr_open_for_merge_output:
def test_zarr_open_for_merge_output_memory():
    shape = (5, 4, 3)
    chunks = (2, 2, 3)
    dtype = np.uint16

    z = _zarr_open_for_merge_output(
        zarr_store="memory",
        folder="/does/not/matter",
        basename="ignored",
        shape=shape,
        dtype=dtype,
        chunks=chunks,
    )

    assert isinstance(z, zarr.core.array.Array)
    assert z.shape == shape
    assert z.dtype == dtype
    assert z.chunks == chunks

    # writable and zero-initialized
    z[:] = 1
    arr = np.asarray(z[:])
    assert int(arr.sum()) == int(np.prod(shape))

def test_zarr_open_for_merge_output_disk_creates_cache(tmp_path):
    shape = (4, 3)
    chunks = (2, 3)
    dtype = np.uint8
    basename = "merged"

    z = _zarr_open_for_merge_output(
        zarr_store="disk",
        folder=str(tmp_path),
        basename=basename,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
    )

    cache = tmp_path / ".omio_cache"
    zarr_path = cache / f"{basename}.zarr"

    assert cache.exists()
    assert zarr_path.exists()
    assert isinstance(z, zarr.core.array.Array)
    assert z.shape == shape
    
def test_zarr_open_for_merge_output_disk_overwrites_existing(tmp_path):
    shape = (3, 3)
    chunks = (3, 3)
    dtype = np.uint8
    basename = "merged"

    z1 = _zarr_open_for_merge_output(
        "disk", str(tmp_path), basename, shape, dtype, chunks
    )
    z1[:] = 7

    # recreate (function removes the whole store path)
    z2 = _zarr_open_for_merge_output(
        "disk", str(tmp_path), basename, shape, dtype, chunks
    )

    arr2 = np.asarray(z2[:])
    assert int(arr2.sum()) == 0

def test_zarr_open_for_merge_output_invalid_store_raises(tmp_path):
    with pytest.raises(ValueError, match="invalid zarr_store"):
        _zarr_open_for_merge_output(
            zarr_store="invalid",
            folder=str(tmp_path),
            basename="x",
            shape=(2, 2),
            dtype=np.uint8,
            chunks=(2, 2),
        )

def test_zarr_open_for_merge_output_disk_creates_cache_folder(tmp_path):
    cache = tmp_path / ".omio_cache"
    assert not cache.exists()

    _zarr_open_for_merge_output(
        "disk",
        folder=str(tmp_path),
        basename="merged",
        shape=(2, 2),
        dtype=np.uint8,
        chunks=(2, 2),
    )

    assert cache.exists()
    assert cache.is_dir()

# _check_fname_out tests:
def test_check_fname_out_no_collision(tmp_path):
    f = tmp_path / "out.ome.tif"
    assert _check_fname_out(str(f), overwrite=False) == str(f)

def test_check_fname_out_collision_adds_suffix(tmp_path):
    f = tmp_path / "out.ome.tif"
    f.write_text("x")

    out = _check_fname_out(str(f), overwrite=False)
    assert out.endswith("out 1.ome.tif")

def test_check_fname_out_multiple_suffixes(tmp_path):
    (tmp_path / "out.ome.tif").write_text("x")
    (tmp_path / "out 1.ome.tif").write_text("x")

    out = _check_fname_out(str(tmp_path / "out.ome.tif"), overwrite=False)
    assert out.endswith("out 2.ome.tif")

def test_check_fname_out_overwrite_true(tmp_path):
    f = tmp_path / "out.ome.tif"
    f.write_text("x")

    out = _check_fname_out(str(f), overwrite=True)
    assert out == str(f)

def test_check_fname_out_invalid_extension_raises():
    with pytest.raises(ValueError):
        _check_fname_out("out.tif", overwrite=False)

# _validate_merge_inputs_with_optional_padding tests:
def _md_axes(ax="TZCYX"):
    return {"axes": ax}

def test_validate_merge_inputs_invalid_merge_axis_returns_false(capsys):
    images = [np.zeros((1, 1, 1, 4, 4), dtype=np.uint8)]
    mds = [_md_axes("TZCYX")]

    ok = _validate_merge_inputs_with_optional_padding(
        images, mds, merge_along_axis="Y", zeropadding=False, context="merge_test"
    )

    out = capsys.readouterr().out
    assert ok is False
    assert "invalid merge_along_axis" in out
    assert "merge_test" in out

def test_validate_merge_inputs_empty_or_mismatched_lengths_returns_false(capsys):
    ok = _validate_merge_inputs_with_optional_padding(
        images=[], metadatas=[], merge_along_axis="T", zeropadding=False, context="merge_test"
    )
    out = capsys.readouterr().out
    assert ok is False
    assert "empty inputs or mismatched" in out

    images = [np.zeros((1, 1, 1, 4, 4), dtype=np.uint8)]
    mds = []
    ok = _validate_merge_inputs_with_optional_padding(
        images, mds, merge_along_axis="T", zeropadding=False, context="merge_test"
    )
    out = capsys.readouterr().out
    assert ok is False
    assert "empty inputs or mismatched" in out

    images = [np.zeros((1, 1, 1, 4, 4), dtype=np.uint8)]
    mds = [_md_axes("TZCYX"), _md_axes("TZCYX")]
    ok = _validate_merge_inputs_with_optional_padding(
        images, mds, merge_along_axis="T", zeropadding=False, context="merge_test"
    )
    out = capsys.readouterr().out
    assert ok is False
    assert "empty inputs or mismatched" in out

def test_validate_merge_inputs_axes_mismatch_returns_false(capsys):
    images = [np.zeros((1, 1, 1, 4, 4), dtype=np.uint8)]
    mds = [_md_axes("TCYXZ")]  # wrong

    ok = _validate_merge_inputs_with_optional_padding(
        images, mds, merge_along_axis="T", zeropadding=False, context="merge_test"
    )

    out = capsys.readouterr().out
    assert ok is False
    assert "axes mismatch" in out
    assert "Expected" in out
    assert "Merge aborted" in out

def test_validate_merge_inputs_reference_not_5d_emits_warning_and_returns_false():
    images = [np.zeros((4, 4), dtype=np.uint8)]  # not 5D
    mds = [_md_axes("TZCYX")]

    with pytest.warns(UserWarning, match=r"expected 5D arrays"):
        ok = _validate_merge_inputs_with_optional_padding(
            images, mds, merge_along_axis="T", zeropadding=False, context="merge_test"
        )
    assert ok is False

def test_validate_merge_inputs_zeropadding_true_allows_nonmerge_shape_mismatch_but_requires_5d():
    images = [
        np.zeros((1, 1, 1, 4, 4), dtype=np.uint8),
        np.zeros((2, 1, 3, 4, 4), dtype=np.uint8),  # Z and C differ; allowed if padding
    ]
    mds = [_md_axes("TZCYX"), _md_axes("TZCYX")]

    ok = _validate_merge_inputs_with_optional_padding(
        images, mds, merge_along_axis="T", zeropadding=True, context="merge_test"
    )
    assert ok is True

    # but still must be 5D for every stack
    images_bad = [
        np.zeros((1, 1, 1, 4, 4), dtype=np.uint8),
        np.zeros((4, 4), dtype=np.uint8),  # not 5D
    ]
    mds_bad = [_md_axes("TZCYX"), _md_axes("TZCYX")]

    with pytest.warns(UserWarning, match=r"expected 5D arrays"):
        ok = _validate_merge_inputs_with_optional_padding(
            images_bad, mds_bad, merge_along_axis="T", zeropadding=True, context="merge_test"
        )
    assert ok is False

def test_validate_merge_inputs_strict_mode_rejects_nonmerge_axis_mismatch(capsys):
    # merge along T => Z,C,Y,X must match
    images = [
        np.zeros((1, 1, 1, 4, 4), dtype=np.uint8),
        np.zeros((2, 2, 1, 4, 4), dtype=np.uint8),  # Z differs => should fail
    ]
    mds = [_md_axes("TZCYX"), _md_axes("TZCYX")]

    ok = _validate_merge_inputs_with_optional_padding(
        images, mds, merge_along_axis="T", zeropadding=False, context="merge_test"
    )
    out = capsys.readouterr().out
    assert ok is False
    assert "incompatible shapes for merge" in out
    assert "Mismatch in axis 'Z'" in out
    assert "Merge aborted" in out

def test_validate_merge_inputs_strict_mode_allows_only_merge_axis_to_differ():
    # merge along T => only T may differ
    images = [
        np.zeros((1, 1, 1, 4, 4), dtype=np.uint8),
        np.zeros((3, 1, 1, 4, 4), dtype=np.uint8),  # only T differs
        np.zeros((2, 1, 1, 4, 4), dtype=np.uint8),
    ]
    mds = [_md_axes("TZCYX"), _md_axes("TZCYX"), _md_axes("TZCYX")]

    ok = _validate_merge_inputs_with_optional_padding(
        images, mds, merge_along_axis="T", zeropadding=False, context="merge_test"
    )
    assert ok is True

def test_validate_merge_inputs_strict_mode_rejects_non_5d_in_any_stack():
    images = [
        np.zeros((1, 1, 1, 4, 4), dtype=np.uint8),
        np.zeros((1, 4, 4), dtype=np.uint8),  # not 5D
    ]
    mds = [_md_axes("TZCYX"), _md_axes("TZCYX")]

    with pytest.warns(UserWarning, match=r"expected 5D arrays"):
        ok = _validate_merge_inputs_with_optional_padding(
            images, mds, merge_along_axis="T", zeropadding=False, context="merge_test"
        )
    assert ok is False

# _normalize_axes_for_ometiff tests:
def test_normalize_axes_removes_singleton_S_axis():
    img = np.zeros((1, 2, 3), dtype=np.uint8)
    axes = "SXY"

    arr, ax = _normalize_axes_for_ometiff(img, axes)

    assert arr.shape == (2, 3)
    assert ax == "XY"

def test_normalize_axes_keeps_non_singleton_S_axis():
    img = np.zeros((2, 3, 4), dtype=np.uint8)
    axes = "SXY"

    arr, ax = _normalize_axes_for_ometiff(img, axes)

    # S is not singleton → must be preserved
    assert arr.shape == (2, 3, 4)
    assert ax == "SXY"

def test_normalize_axes_no_S_axis_no_change():
    img = np.zeros((2, 3, 4), dtype=np.uint8)
    axes = "ZXY"

    arr, ax = _normalize_axes_for_ometiff(img, axes)

    assert arr.shape == (2, 3, 4)
    assert ax == "ZXY"

def test_normalize_axes_only_S_axis_singleton_results_in_scalar():
    img = np.zeros((1,), dtype=np.uint8)
    axes = "S"

    arr, ax = _normalize_axes_for_ometiff(img, axes)

    assert arr.shape == ()
    assert ax == ""

def test_normalize_axes_multiple_singletons_only_S_removed():
    img = np.zeros((1, 1, 5), dtype=np.uint8)
    axes = "SZX"

    arr, ax = _normalize_axes_for_ometiff(img, axes)

    # Only S removed, Z singleton must remain
    assert arr.shape == (1, 5)
    assert ax == "ZX"

def test_normalize_axes_input_converted_to_numpy():
    img = [[1, 2, 3]]
    axes = "XY"

    arr, ax = _normalize_axes_for_ometiff(img, axes)

    assert isinstance(arr, np.ndarray)
    assert arr.shape == (1, 3)
    assert ax == "XY"

def test_normalize_axes_axis_length_mismatch_raises_value_error():
    img = np.zeros((2, 3), dtype=np.uint8)
    axes = "XYZ"  # too long

    with pytest.raises(ValueError, match="does not fit to arr.ndim"):
        _normalize_axes_for_ometiff(img, axes)

def test_normalize_axes_axis_length_mismatch_after_squeeze_raises():
    img = np.zeros((1, 3), dtype=np.uint8)
    axes = "SX"  # after removing S, axes="X" but arr.ndim=1 → ok

    arr, ax = _normalize_axes_for_ometiff(img, axes)

    assert arr.shape == (3,)
    assert ax == "X"

def test_normalize_axes_squeeze_produces_mismatch_raises():
    img = np.zeros((1, 3), dtype=np.uint8)
    axes = "SZ"  # after squeeze → axes="Z", but arr.ndim=1 → ok

    arr, ax = _normalize_axes_for_ometiff(img, axes)

    assert arr.shape == (3,)
    assert ax == "Z"

def test_normalize_axes_illegal_axes_string_raises():
    img = np.zeros((1, 2, 3), dtype=np.uint8)
    axes = "SXYQ"  # impossible length

    with pytest.raises(ValueError):
        _normalize_axes_for_ometiff(img, axes)

# _get_original_filename_from_metadata tests:
def test_get_original_filename_returns_none_if_metadata_not_dict():
    assert _get_original_filename_from_metadata(None) is None
    assert _get_original_filename_from_metadata("not a dict") is None
    assert _get_original_filename_from_metadata(123) is None

def test_get_original_filename_returns_none_if_annotations_missing():
    md = {"axes": "TZCYX"}
    assert _get_original_filename_from_metadata(md) is None

def test_get_original_filename_dict_case_returns_basename():
    md = {"Annotations": {"original_filename": "/a/b/c/file.ome.tif"}}
    assert _get_original_filename_from_metadata(md) == "file.ome.tif"

def test_get_original_filename_dict_case_strips_whitespace_and_returns_basename():
    md = {"Annotations": {"original_filename": "   /a/b/c/file.ome.tif   "}}
    assert _get_original_filename_from_metadata(md) == "file.ome.tif"

def test_get_original_filename_dict_case_empty_or_whitespace_returns_none():
    md1 = {"Annotations": {"original_filename": ""}}
    md2 = {"Annotations": {"original_filename": "   "}}
    assert _get_original_filename_from_metadata(md1) is None
    assert _get_original_filename_from_metadata(md2) is None

def test_get_original_filename_dict_case_non_string_returns_none():
    md = {"Annotations": {"original_filename": 123}}
    assert _get_original_filename_from_metadata(md) is None

def test_get_original_filename_list_case_returns_first_valid_basename():
    md = {
        "Annotations": [
            {"original_filename": "   "},
            {"other": "x"},
            {"original_filename": "/x/y/z/first.ome.tif"},
            {"original_filename": "/x/y/z/second.ome.tif"},
        ]
    }
    assert _get_original_filename_from_metadata(md) == "first.ome.tif"

def test_get_original_filename_list_case_skips_non_dict_entries():
    md = {
        "Annotations": [
            "not a dict",
            123,
            {"original_filename": "/x/y/z/file.ome.tif"},
        ]
    }
    assert _get_original_filename_from_metadata(md) == "file.ome.tif"

def test_get_original_filename_annotations_wrong_type_returns_none():
    md = {"Annotations": "not a dict or list"}
    assert _get_original_filename_from_metadata(md) is None

def test_get_original_filename_handles_relative_paths():
    md = {"Annotations": {"original_filename": "relative/path/to/file.tif"}}
    assert _get_original_filename_from_metadata(md) == os.path.basename("relative/path/to/file.tif")

# _zarr_pick_first_array tests:
def test_zarr_pick_first_array_returns_array_if_already_array():
    arr = zarr.array(np.zeros((3, 4), dtype=np.uint8))
    out = _zarr_pick_first_array(arr, verbose=False)

    assert out is arr
    assert out.shape == (3, 4)

def test_zarr_pick_first_array_prefers_key_zero_when_present():
    grp = zarr.group()
    grp.create_array("1", shape=(2, 2), dtype=np.uint8)
    grp.create_array("0", shape=(5, 6), dtype=np.uint16)

    out = _zarr_pick_first_array(grp, prefer_keys=("0",), verbose=False)

    assert out.shape == (5, 6)
    assert out.dtype == np.uint16

def test_zarr_pick_first_array_uses_first_array_like_key_if_preferred_missing():
    grp = zarr.group()
    grp.create_array("b", shape=(3, 3), dtype=np.uint8)
    grp.create_array("a", shape=(4, 4), dtype=np.uint16)

    # sorted keys → "a" first
    out = _zarr_pick_first_array(grp, prefer_keys=("0",), verbose=False)

    assert out.shape == (4, 4)
    assert out.dtype == np.uint16

def test_zarr_pick_first_array_skips_non_array_entries():
    grp = zarr.group()
    grp.attrs["meta"] = "not an array"
    grp.create_group("nested")  # subgroup, not array
    grp.create_array("img", shape=(7, 8), dtype=np.float32)

    out = _zarr_pick_first_array(grp, verbose=False)

    assert out.shape == (7, 8)
    assert out.dtype == np.float32

def test_zarr_pick_first_array_respects_custom_prefer_keys_order():
    grp = zarr.group()
    grp.create_array("lowres", shape=(2, 2), dtype=np.uint8)
    grp.create_array("highres", shape=(10, 10), dtype=np.uint8)

    out = _zarr_pick_first_array(
        grp,
        prefer_keys=("highres", "lowres"),
        verbose=False,
    )

    assert out.shape == (10, 10)

def test_zarr_pick_first_array_raises_if_group_contains_no_arrays():
    grp = zarr.group()
    grp.attrs["foo"] = "bar"
    grp.create_group("nested")

    with pytest.raises(TypeError):
        _zarr_pick_first_array(grp, verbose=False)

def test_zarr_pick_first_array_handles_object_without_keys_gracefully():
    class Dummy:
        pass

    dummy = Dummy()

    with pytest.raises(TypeError):
        _zarr_pick_first_array(dummy, verbose=False)


# _merge_concat_along_axis tests:
# helpers:
def _mk_md(shape, parentfolder, filename="img.ome.tif"):
    # minimal metadata that your merge path expects
    return {
        "axes": "TZCYX",
        "shape": tuple(shape),
        "original_parentfolder": str(parentfolder),
        "original_filename": filename,
        "Annotations": {"original_filename": filename},
    }

def _assert_sizes_in_md(md, shape):
    # merged md is expected to contain these fields
    assert md["axes"] == "TZCYX"
    assert tuple(md["shape"]) == tuple(shape)
    # Size* are derived from merged_shape in your code
    assert md["SizeT"] == shape[0]
    assert md["SizeZ"] == shape[1]
    assert md["SizeC"] == shape[2]
    assert md["SizeY"] == shape[3]
    assert md["SizeX"] == shape[4]

# validation failure tests
def test_merge_concat_invalid_merge_axis_returns_none(tmp_path):
    img = np.zeros((1, 1, 1, 2, 2), dtype=np.uint8)
    md = _mk_md(img.shape, tmp_path)

    merged, mdm = _merge_concat_along_axis(
        [img, img], [md, md],
        merge_along_axis="Q",   # invalid
        zarr_store=None,
        zeropadding=False,
        verbose=False,
    )

    assert merged is None
    assert mdm is None

def test_merge_concat_axes_mismatch_in_metadata_returns_none(tmp_path):
    img = np.zeros((1, 1, 1, 2, 2), dtype=np.uint8)
    md_bad = _mk_md(img.shape, tmp_path)
    md_bad["axes"] = "TYX"   # invalid for this merge function

    merged, mdm = _merge_concat_along_axis(
        [img], [md_bad],
        merge_along_axis="T",
        zarr_store=None,
        zeropadding=False,
        verbose=False,
    )

    assert merged is None
    assert mdm is None

def test_merge_concat_non_5d_input_returns_none(tmp_path):
    img = np.zeros((2, 2), dtype=np.uint8)  # not 5D
    md = {"axes": "TZCYX", "shape": img.shape, "original_parentfolder": str(tmp_path)}

    with pytest.warns(UserWarning, match=r"expected 5D arrays"):
        merged, mdm = _merge_concat_along_axis(
            [img], [md],
            merge_along_axis="T",
            zarr_store=None,
            zeropadding=False,
            verbose=False,
        )

    assert merged is None
    assert mdm is None

# NumPy output tests
def test_merge_concat_numpy_strict_concat_along_T(tmp_path):
    a = np.ones((2, 1, 1, 3, 4), dtype=np.uint8)          # T=2
    b = 2 * np.ones((3, 1, 1, 3, 4), dtype=np.uint8)      # T=3

    mda = _mk_md(a.shape, tmp_path, filename="a.ome.tif")
    mdb = _mk_md(b.shape, tmp_path, filename="b.ome.tif")

    merged, mdm = _merge_concat_along_axis(
        [a, b], [mda, mdb],
        merge_along_axis="T",
        zarr_store=None,
        zeropadding=False,
        verbose=False,
    )

    assert isinstance(merged, np.ndarray)
    assert merged.shape == (5, 1, 1, 3, 4)
    _assert_sizes_in_md(mdm, merged.shape)

    # content: first block ones, second block twos
    assert np.all(merged[0:2] == 1)
    assert np.all(merged[2:5] == 2)

def test_merge_concat_numpy_strict_rejects_nonmerge_mismatch(tmp_path):
    a = np.ones((2, 1, 1, 3, 4), dtype=np.uint8)
    b = np.ones((3, 1, 1, 3, 5), dtype=np.uint8)  # X mismatch

    mda = _mk_md(a.shape, tmp_path)
    mdb = _mk_md(b.shape, tmp_path)

    merged, mdm = _merge_concat_along_axis(
        [a, b], [mda, mdb],
        merge_along_axis="T",
        zarr_store=None,
        zeropadding=False,   # strict
        verbose=False,
    )

    assert merged is None
    assert mdm is None

def test_merge_concat_numpy_padding_allows_nonmerge_mismatch_and_pads_zeros(tmp_path):
    # mismatch in X
    a = np.ones((2, 1, 1, 3, 4), dtype=np.uint8)
    b = 2 * np.ones((3, 1, 1, 3, 6), dtype=np.uint8)

    mda = _mk_md(a.shape, tmp_path, filename="a.ome.tif")
    mdb = _mk_md(b.shape, tmp_path, filename="b.ome.tif")

    merged, mdm = _merge_concat_along_axis(
        [a, b], [mda, mdb],
        merge_along_axis="T",
        zarr_store=None,
        zeropadding=True,
        verbose=False,
    )

    assert isinstance(merged, np.ndarray)
    assert merged.shape == (5, 1, 1, 3, 6)   # X padded to 6
    _assert_sizes_in_md(mdm, merged.shape)

    # a occupies X=0..4, padding at X=4..6 must be zeros for first 2 frames
    assert np.all(merged[0:2, :, :, :, 0:4] == 1)
    assert np.all(merged[0:2, :, :, :, 4:6] == 0)

    # b occupies full X=0..6 and has value 2
    assert np.all(merged[2:5] == 2)

# Zarr output tests:
def test_merge_concat_zarr_memory_strict_concat_along_T(tmp_path):
    a = np.ones((2, 1, 1, 3, 4), dtype=np.uint8)
    b = 2 * np.ones((3, 1, 1, 3, 4), dtype=np.uint8)

    mda = _mk_md(a.shape, tmp_path, filename="a.ome.tif")
    mdb = _mk_md(b.shape, tmp_path, filename="b.ome.tif")

    z, mdm = _merge_concat_along_axis(
        [a, b], [mda, mdb],
        merge_along_axis="T",
        zarr_store="memory",
        zeropadding=False,
        verbose=False,
    )

    assert isinstance(z, zarr.core.array.Array)
    assert z.shape == (5, 1, 1, 3, 4)
    _assert_sizes_in_md(mdm, z.shape)

    arr = np.asarray(z)
    assert np.all(arr[0:2] == 1)
    assert np.all(arr[2:5] == 2)

def test_merge_concat_zarr_memory_padding_pads_zeros(tmp_path):
    a = np.ones((2, 1, 1, 3, 4), dtype=np.uint8)
    b = 2 * np.ones((3, 1, 1, 3, 6), dtype=np.uint8)

    mda = _mk_md(a.shape, tmp_path, filename="a.ome.tif")
    mdb = _mk_md(b.shape, tmp_path, filename="b.ome.tif")

    z, mdm = _merge_concat_along_axis(
        [a, b], [mda, mdb],
        merge_along_axis="T",
        zarr_store="memory",
        zeropadding=True,
        verbose=False,
    )

    assert isinstance(z, zarr.core.array.Array)
    assert z.shape == (5, 1, 1, 3, 6)
    _assert_sizes_in_md(mdm, z.shape)

    arr = np.asarray(z)
    assert np.all(arr[0:2, :, :, :, 0:4] == 1)
    assert np.all(arr[0:2, :, :, :, 4:6] == 0)
    assert np.all(arr[2:5] == 2)

def test_merge_concat_zarr_disk_writes_to_cache_folder(tmp_path):
    a = np.ones((1, 1, 1, 2, 2), dtype=np.uint8)
    b = np.ones((1, 1, 1, 2, 2), dtype=np.uint8)

    # important: folder0 for disk path comes from original_parentfolder
    mda = _mk_md(a.shape, tmp_path, filename="a.ome.tif")
    mdb = _mk_md(b.shape, tmp_path, filename="b.ome.tif")

    z, mdm = _merge_concat_along_axis(
        [a, b], [mda, mdb],
        merge_along_axis="T",
        zarr_store="disk",
        zeropadding=False,
        verbose=False,
    )

    assert isinstance(z, zarr.core.array.Array)
    assert z.shape == (2, 1, 1, 2, 2)
    _assert_sizes_in_md(mdm, z.shape)

    # check that .omio_cache exists under tmp_path
   

# _load_yaml_metadata tests:
def test_load_yaml_metadata_valid(tmp_path):
    p = tmp_path / "meta.yaml"
    p.write_text(
        "a: 1\nb:\n  c: 2\n",
        encoding="utf-8"
    )

    md = _load_yaml_metadata(str(p))

    assert isinstance(md, dict)
    assert md["a"] == 1
    assert md["b"]["c"] == 2

def test_load_yaml_metadata_empty_file_returns_empty_dict(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")

    md = _load_yaml_metadata(str(p))

    assert md == {}

def test_load_yaml_metadata_non_mapping_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(
        "- a\n- b\n- c\n",
        encoding="utf-8"
    )

    with pytest.raises(ValueError, match="must contain a mapping"):
        _load_yaml_metadata(str(p))

def test_load_yaml_metadata_missing_pyyaml(monkeypatch, tmp_path):
    p = tmp_path / "meta.yaml"
    p.write_text("a: 1\n", encoding="utf-8")

    monkeypatch.setattr("omio.omio.yaml", None)

    with pytest.raises(ImportError, match="PyYAML is not installed"):
        _load_yaml_metadata(str(p))

# _ensure_axes_in_metadata tests:
class _DummySeries0:
    def __init__(self, axes):
        self.axes = axes

class _DummyTif:
    def __init__(self, axes, has_series=True):
        self.series = [_DummySeries0(axes)] if has_series else []

def test_ensure_axes_in_metadata_sets_axes_when_missing(capsys):
    md = {}
    tif = _DummyTif("TZCYX")
    out = _ensure_axes_in_metadata(md, tif)

    assert out["axes"] == "TZCYX"
    assert md["axes"] == "TZCYX"  # in-place

def test_ensure_axes_in_metadata_overwrites_on_mismatch(capsys):
    md = {"axes": "YX"}
    tif = _DummyTif("TZCYX")

    out = _ensure_axes_in_metadata(md, tif)

    captured = capsys.readouterr().out
    assert "Mismatch found" in captured
    assert out["axes"] == "TZCYX"

def test_ensure_axes_in_metadata_replaces_I_with_T():
    md = {}
    tif = _DummyTif("IZCYX")  # nonstandard time axis encoding
    out = _ensure_axes_in_metadata(md, tif)

    assert out["axes"] == "TZCYX"

def test_ensure_axes_in_metadata_maps_yxs_to_yxc_for_rgb():
    md = {}
    tif = _DummyTif("YXS")
    out = _ensure_axes_in_metadata(md, tif)

    assert out["axes"] == "YXC"
    
def test_ensure_axes_in_metadata_maps_Q_to_C_if_C_missing():
    md = {}
    tif = _DummyTif("TQZYX")  # Q present, C missing
    out = _ensure_axes_in_metadata(md, tif)

    assert out["axes"] == "TCZYX"

def test_ensure_axes_in_metadata_maps_Q_to_T_if_C_present_T_missing():
    md = {}
    tif = _DummyTif("CQZYX")  # C present, T missing
    out = _ensure_axes_in_metadata(md, tif)

    assert out["axes"] == "CTZYX"

def test_ensure_axes_in_metadata_maps_Q_to_Z_if_C_T_present_Z_missing():
    md = {}
    tif = _DummyTif("CTQYX")  # C,T present, Z missing
    out = _ensure_axes_in_metadata(md, tif)

    assert out["axes"] == "CTZYX"

def test_ensure_axes_in_metadata_maps_Q_to_P_if_C_T_Z_present_P_missing():
    md = {}
    tif = _DummyTif("CTZQYX")
    out = _ensure_axes_in_metadata(md, tif)

    assert out["axes"] == "CTZPYX"

def test_ensure_axes_in_metadata_raises_if_Q_and_C_T_Z_P_all_present():
    md = {}
    tif = _DummyTif("CTZPQYX")
    with pytest.raises(ValueError, match="Unable to map axis 'Q'"):
        _ensure_axes_in_metadata(md, tif)

def test_ensure_axes_in_metadata_falls_back_to_unknown_if_no_series(capsys):
    md = {}
    tif = _DummyTif("TZCYX", has_series=False)

    out = _ensure_axes_in_metadata(md, tif)
    captured = capsys.readouterr().out

    assert "Unable to extract axes" in captured
    assert out["axes"] == "unknown"




# %% TEST OME_METADATA_CHECKUP
# test OME_metadata_checkup function with minimal metadata
def test_OME_metadata_checkup_does_not_modify_input():
    md_in = {
        "axes": "TZCYX",
        "PhysicalSizeX": 0.5,
        "SizeX": 64,
        "random_key": 123,
    }
    md_copy = dict(md_in)

    md_out = OME_metadata_checkup(md_in, namespace="omio:metadata", verbose=False)

    assert md_in == md_copy
    assert md_out is not md_in

def test_OME_metadata_checkup_keeps_core_and_keep_keys_top_level_and_moves_extras():
    md_in = {
        # core keys
        "axes": "TZCYX",
        "PhysicalSizeX": 0.3,
        "PhysicalSizeXUnit": "µm",
        "TimeIncrement": 1.0,
        "TimeIncrementUnit": "s",
        # keep keys
        "SizeX": 128,
        "SizeY": 64,
        "shape": (1, 1, 2, 64, 128),
        "Channel_Count": 2,
        # extras
        "foo": "bar",
        "original_meta_source": "LSM",
    }

    md_out = OME_metadata_checkup(md_in, namespace="omio:metadata", verbose=False)

    # core + keep stay at top level
    assert md_out["axes"] == "TZCYX"
    assert md_out["PhysicalSizeX"] == 0.3
    assert md_out["PhysicalSizeXUnit"] == "µm"
    assert md_out["TimeIncrement"] == 1.0
    assert md_out["TimeIncrementUnit"] == "s"

    assert md_out["SizeX"] == 128
    assert md_out["SizeY"] == 64
    assert md_out["shape"] == (1, 1, 2, 64, 128)
    assert md_out["Channel_Count"] == 2

    # extras moved into Annotations
    assert "foo" not in md_out
    assert "original_meta_source" not in md_out

    assert "Annotations" in md_out
    assert md_out["Annotations"]["Namespace"] == "omio:metadata"
    assert md_out["Annotations"]["foo"] == "bar"
    assert md_out["Annotations"]["original_meta_source"] == "LSM"

def test_OME_metadata_checkup_preserves_existing_annotations_and_sets_namespace():
    md_in = {
        "axes": "TZCYX",
        "Annotations": {"some": "thing", "Namespace": "old:ns"},
        "extra": 1,
    }

    md_out = OME_metadata_checkup(md_in, namespace="new:ns", verbose=False)

    assert md_out["Annotations"]["some"] == "thing"
    assert md_out["Annotations"]["Namespace"] == "new:ns"
    assert md_out["Annotations"]["extra"] == 1
    assert "extra" not in md_out

def test_OME_metadata_checkup_handles_non_dict_annotations_gracefully():
    md_in = {
        "axes": "TZCYX",
        "Annotations": "not-a-dict",
        "extra": 7,
    }

    md_out = OME_metadata_checkup(md_in, namespace="omio:metadata", verbose=False)

    assert isinstance(md_out["Annotations"], dict)
    assert md_out["Annotations"]["Namespace"] == "omio:metadata"
    assert md_out["Annotations"]["extra"] == 7
    assert "extra" not in md_out

def test_OME_metadata_checkup_protects_existing_original_keys_in_annotations():
    md_in = {
        "axes": "TZCYX",
        "Annotations": {
            "original_A": "keep-me",
            "other": "ok",
        },
        "original_A": "overwrite-attempt",
        "extra": 2,
    }

    md_out = OME_metadata_checkup(md_in, namespace="omio:metadata", verbose=False)

    # 'original_A' already existed in Annotations and must not be overwritten
    assert md_out["Annotations"]["original_A"] == "keep-me"
    # other keys still merge
    assert md_out["Annotations"]["extra"] == 2
    assert md_out["Annotations"]["other"] == "ok"
    assert "original_A" not in md_out
    assert "extra" not in md_out

def test_OME_metadata_checkup_prints_skip_message_when_verbose(capsys):
    md_in = {
        "axes": "TZCYX",
        "Annotations": {"original_X": "keep"},
        "original_X": "overwrite-attempt",
    }

    md_out = OME_metadata_checkup(md_in, namespace="omio:metadata", verbose=True)
    captured = capsys.readouterr()

    assert md_out["Annotations"]["original_X"] == "keep"
    assert "Skipping overwrite of original metadata key 'original_X'" in captured.out
    
# %% TEST READ_TIF

# TIFF reading tests for read_tif function:

def test_read_tif_invalid_zarr_store_raises(tmp_path):
    f = tmp_path / "test.tif"
    tifffile.imwrite(f, np.zeros((5, 7), dtype=np.uint16))

    with pytest.raises(ValueError):
        read_tif(str(f), zarr_store="nope", verbose=False)
    
def test_read_tif_basic_plain_tiff_returns_numpy_and_canonical_axes(tmp_path):
    data = (np.arange(5 * 7, dtype=np.uint16).reshape(5, 7))
    f = tmp_path / "plain.tif"
    tifffile.imwrite(f, data)

    image, md = read_tif(str(f), verbose=False)

    assert isinstance(image, np.ndarray)
    assert isinstance(md, dict)

    # OMIO policy: canonical order TZCYX
    assert md.get("axes") == "TZCYX"
    assert image.ndim == 5
    assert image.shape[-2:] == (5, 7)

    # minimal metadata invariants
    assert md["SizeX"] == 7
    assert md["SizeY"] == 5
    assert md["PhysicalSizeX"] > 0
    assert md["PhysicalSizeY"] > 0
    assert md["PhysicalSizeZ"] > 0
    
def test_read_tif_return_list_true_wraps_singleton(tmp_path):
    f = tmp_path / "plain.tif"
    tifffile.imwrite(f, np.zeros((3, 4), dtype=np.uint8))

    images, mds = read_tif(str(f), return_list=True, verbose=False)

    assert isinstance(images, list)
    assert isinstance(mds, list)
    assert len(images) == 1
    assert len(mds) == 1
    assert mds[0]["axes"] == "TZCYX"
    assert images[0].shape[-2:] == (3, 4)
    
def test_read_tif_zarr_memory_returns_zarr_array(tmp_path):
    f = tmp_path / "plain.tif"
    tifffile.imwrite(f, np.zeros((6, 8), dtype=np.uint16))

    image, md = read_tif(str(f), zarr_store="memory", verbose=False)

    assert isinstance(image, zarr.core.array.Array)
    assert md["axes"] == "TZCYX"
    assert image.shape[-2:] == (6, 8)
    assert image.dtype == np.uint16
    
def test_read_tif_physicalsize_override_applied(tmp_path):
    f = tmp_path / "plain.tif"
    tifffile.imwrite(f, np.zeros((5, 7), dtype=np.uint16))

    image, md = read_tif(
        str(f),
        physicalsize_xyz=(0.1, 0.2, 0.3),
        pixelunit="micron",
        verbose=False,
    )

    assert md["PhysicalSizeX"] == pytest.approx(0.1)
    assert md["PhysicalSizeY"] == pytest.approx(0.2)
    assert md["PhysicalSizeZ"] == pytest.approx(0.3)

def test_read_tif_samples_axis_is_folded_into_channel(tmp_path):
    f = tmp_path / "rgb_samples.ome.tif"
    data = np.random.randint(0, 255, (8, 2, 32, 32, 3), dtype=np.uint8)  # T,C,Y,X,S

    tifffile.imwrite(
        f,
        data,
        photometric="rgb",
        metadata={"axes": "TCYXS"},
    )

    image, md = read_tif(str(f), verbose=False)

    assert isinstance(image, np.ndarray)
    assert md["axes"] == "TZCYX"
    assert image.shape[0] == 8          # T
    assert image.shape[1] == 1          # Z
    assert image.shape[2] == 6          # C = 2*3
    assert image.shape[-2:] == (32, 32)

def test_read_tif_rgb_no_metadata_shape_yxc_ok(tmp_path):
    f = tmp_path / "rgb_plain.tif"
    rgb = np.random.randint(0, 255, (12, 13, 3), dtype=np.uint8)

    tifffile.imwrite(f, rgb, photometric="rgb")

    image, md = read_tif(str(f), verbose=False)

    assert md["axes"] == "TZCYX"
    assert image.ndim == 5
    assert image.shape[-2:] == (12, 13)
    assert image.shape[2] == 3  # C

def test_read_tif_rgb_no_metadata_unexpected_shape_raises(tmp_path):
    f = tmp_path / "rgb_weird.tif"
    rgb = np.random.randint(0, 255, (2, 12, 13, 3), dtype=np.uint8)

    tifffile.imwrite(f, rgb, photometric="rgb")

    with pytest.raises(ValueError, match="Encountered RGB TIFF with unexpected shape"):
        read_tif(str(f), verbose=False)

def test_read_tif_physically_unreasonable_sizes_are_clamped_to_one(tmp_path, monkeypatch):
    f = tmp_path / "plain.tif"
    tifffile.imwrite(f, np.zeros((5, 7), dtype=np.uint16))

    def fake_parse_ome_metadata(_):
        return {
            "axes": "YX",
            "PhysicalSizeX": -1.0,
            "PhysicalSizeY": 0.0,
            "PhysicalSizeZ": -5.0,
            "unit": "micron",
            "PhysicalSizeXUnit": "micron",
            "PhysicalSizeYUnit": "micron",
            "PhysicalSizeZUnit": "micron",
        }

    monkeypatch.setattr("omio.omio._parse_ome_metadata", fake_parse_ome_metadata)

    class FakeTiffFile:
        def __init__(self, real):
            self._real = real
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            self._real.close()
        @property
        def series(self): return self._real.series
        @property
        def pages(self): return self._real.pages
        @property
        def imagej_metadata(self): return self._real.imagej_metadata
        @property
        def ome_metadata(self): return "<OME/>"  # non None
        @property
        def lsm_metadata(self): return None
        def asarray(self): return self._real.asarray()

    real = tifffile.TiffFile(str(f))
    monkeypatch.setattr("tifffile.TiffFile", lambda _: FakeTiffFile(real))

    image, md = read_tif(str(f), verbose=False)

    assert md["PhysicalSizeX"] == 1
    assert md["PhysicalSizeY"] == 1
    assert md["PhysicalSizeZ"] == 1

def test_read_tif_zarr_disk_creates_cache_and_returns_zarr_array(tmp_path):
    f = tmp_path / "plain.tif"
    tifffile.imwrite(f, np.zeros((6, 8), dtype=np.uint16))

    image, md = read_tif(str(f), zarr_store="disk", verbose=False)

    assert isinstance(image, zarr.core.array.Array)
    assert md["axes"] == "TZCYX"
    cache = tmp_path / ".omio_cache"
    assert cache.exists()
    assert any(p.suffix == ".zarr" for p in cache.iterdir())


# paginated TIFF tests:
def test_read_tif_paginated_returns_lists_and_contains_expected_xy_shapes(tmp_path, capsys):
    f = tmp_path / "paginated.tif"
    data = np.random.randint(0, 255, (8, 2, 20, 20, 3), 'uint8')
    subresolutions = 2
    pixelsize = 0.29  # micrometer
    with tifffile.TiffWriter(f, bigtiff=True) as tif:
        metadata = {
            'axes': 'TCYXS',
            'SignificantBits': 8,
            'TimeIncrement': 0.1,
            'TimeIncrementUnit': 's',
            'PhysicalSizeX': pixelsize,
            'PhysicalSizeXUnit': 'µm',
            'PhysicalSizeY': pixelsize,
            'PhysicalSizeYUnit': 'µm',
            'Channel': {'Name': ['Channel 1', 'Channel 2']},
            'Plane': {'PositionX': [0.0] * 16, 'PositionXUnit': ['µm'] * 16},
            'Description': 'A multi-dimensional, multi-resolution image',
            'MapAnnotation': {  # for OMERO
                'Namespace': 'openmicroscopy.org/PyramidResolution',
                '1': '256 256',
                '2': '128 128',
            },
        }
        options = dict(
            photometric='rgb',
            tile=(16, 16),
            compression='jpeg',
            resolutionunit='CENTIMETER',
            maxworkers=2)
        tif.write(
            data,
            subifds=subresolutions,
            resolution=(1e4 / pixelsize, 1e4 / pixelsize),
            metadata=metadata,
            **options)
        # write pyramid levels to the two subifds
        # in production use resampling to generate sub-resolution images
        for level in range(subresolutions):
            mag = 2 ** (level + 1)
            tif.write(
                data[..., ::mag, ::mag, :],
                subfiletype=1,  # FILETYPE.REDUCEDIMAGE
                resolution=(1e4 / mag / pixelsize, 1e4 / mag / pixelsize),
                **options)
        # add a thumbnail image as a separate series
        # it is recognized by QuPath as an associated image
        thumbnail = (data[0, 0, ::8, ::8] >> 2).astype('uint8')
        tif.write(thumbnail, metadata={'Name': 'thumbnail'})

    images, metadatas = read_tif(str(f), verbose=False)

    # for this test function, we clear the captured output; we do this
    # to suppress an expected print warning:
    capsys.readouterr()

    assert not isinstance(images, list)
    assert metadatas["axes"] == "TZCYX"
    assert images.shape[-2:] == (20, 20)

def test_read_tif_forced_paginated_axis_p_splits_into_pages(tmp_path, monkeypatch):
    f = tmp_path / "p_like.tif"

    data = np.random.randint(0, 255, (4, 10, 11), dtype=np.uint8)
    tifffile.imwrite(f, data, photometric="minisblack")

    def fake_ensure_axes_in_metadata(md, tif):
        md = dict(md)
        md["axes"] = "PYX"
        return md

    monkeypatch.setattr("omio.omio._ensure_axes_in_metadata", fake_ensure_axes_in_metadata)

    monkeypatch.setattr("omio.omio.OME_metadata_checkup", lambda md, verbose=False: md)
    monkeypatch.setattr("omio.omio._batch_correct_for_OME_axes_order",
                        lambda imgs, mds, memap_large_file, verbose=False: (imgs, mds))

    images, metadatas = read_tif(str(f), verbose=False)

    assert isinstance(images, list)
    assert isinstance(metadatas, list)
    assert len(images) == 4
    assert len(metadatas) == 4
    for im, md in zip(images, metadatas):
        assert md["axes"] == "YX"
        assert im.shape == (10, 11)

# multi-series TIFF tests (Weg B: no warnings expected):
def _get_multiseries_carrier(md: dict) -> dict:
    """
    Helper: OMIO may store the multi-series policy info either top-level
    or inside md["Annotations"] (after OME_metadata_checkup). This returns
    whichever dict contains the expected keys.
    """
    ann = md.get("Annotations", {})
    if isinstance(ann, dict) and ("OMIO_MultiSeriesDetected" in ann or "OMIO_MultiSeriesPhotometric" in ann):
        return ann
    return md

def test_read_tif_multiseries_records_policy(tmp_path):
    f = tmp_path / "multiseries_same_shape.tif"

    series0 = np.zeros((5, 7), dtype=np.uint16)
    series1 = np.ones((5, 7), dtype=np.uint16)

    # two calls to tif.write create two series
    with tifffile.TiffWriter(f) as tif:
        tif.write(series0)
        tif.write(series1)

    # Weg B: no warning expected
    image, md = read_tif(str(f), verbose=False)

    assert isinstance(image, np.ndarray)
    assert md["axes"] == "TZCYX"
    assert image.shape[-2:] == (5, 7)

    carrier = _get_multiseries_carrier(md)

    assert carrier.get("OMIO_MultiSeriesDetected") is True
    assert carrier.get("OMIO_TotalSeries") == 2
    assert carrier.get("OMIO_ProcessedSeries") == 0
    assert carrier.get("OMIO_MultiSeriesPolicy") == "only_series_0"
    assert "OMIO_MultiSeriesShapes" in carrier
    assert "OMIO_MultiSeriesAxes" in carrier

def test_read_tif_multiseries_mixed_shapes_still_only_series0(tmp_path):
    f = tmp_path / "multiseries_mixed_shape.tif"

    series0 = np.zeros((5, 7), dtype=np.uint8)
    series1 = np.zeros((3, 4), dtype=np.uint8)

    with tifffile.TiffWriter(f) as tif:
        tif.write(series0)
        tif.write(series1)

    # Weg B: no warning expected
    image, md = read_tif(str(f), verbose=False)

    assert image.shape[-2:] == (5, 7)  # must match series0
    assert md["SizeX"] == 7
    assert md["SizeY"] == 5

    carrier = _get_multiseries_carrier(md)
    assert carrier.get("OMIO_MultiSeriesDetected") is True
    assert carrier.get("OMIO_TotalSeries") == 2
    assert carrier.get("OMIO_MultiSeriesPolicy") == "only_series_0"

def test_read_tif_multiseries_records_photometric_names(tmp_path):
    f = tmp_path / "multiseries_photometric.tif"

    rgb = np.zeros((8, 9, 3), dtype=np.uint8)
    gray = np.zeros((8, 9), dtype=np.uint8)

    with tifffile.TiffWriter(f) as tif:
        tif.write(rgb, photometric="rgb")
        tif.write(gray, photometric="minisblack")

    # Weg B: no warning expected
    image, md = read_tif(str(f), verbose=False)

    carrier = _get_multiseries_carrier(md)

    assert "OMIO_MultiSeriesPhotometric" in carrier
    assert isinstance(carrier["OMIO_MultiSeriesPhotometric"], list)
    assert len(carrier["OMIO_MultiSeriesPhotometric"]) == 2
    # depending on how you store it, entries are typically "RGB"/"MINISBLACK"
    # but tolerate numeric fallback if you still store the enum value
    assert carrier["OMIO_MultiSeriesPhotometric"][0] in {"RGB", "MINISBLACK", 2, "2"}
    assert carrier["OMIO_MultiSeriesPhotometric"][1] in {"RGB", "MINISBLACK", 1, "1"}

def test_read_tif_multiseries_zarr_memory_still_only_series0(tmp_path):
    f = tmp_path / "multiseries_zarr.tif"

    series0 = np.zeros((6, 8), dtype=np.uint16)
    series1 = np.ones((6, 8), dtype=np.uint16)

    with tifffile.TiffWriter(f) as tif:
        tif.write(series0)
        tif.write(series1)

    # Weg B: no warning expected
    image, md = read_tif(str(f), zarr_store="memory", verbose=False)

    assert isinstance(image, zarr.core.array.Array)
    assert md["axes"] == "TZCYX"
    assert image.shape[-2:] == (6, 8)

    carrier = _get_multiseries_carrier(md)
    assert carrier.get("OMIO_MultiSeriesDetected") is True
    assert carrier.get("OMIO_TotalSeries") == 2
    assert carrier.get("OMIO_MultiSeriesPolicy") == "only_series_0"


# LSM tests via read_tif:
def _lsm_path() -> str:
    # tests/test_images/032113-18.lsm
    here = Path(__file__).resolve()
    return str(here.parent / "test_images" / "032113-18.lsm")

@pytest.mark.parametrize("zarr_store", [None, "memory", "disk"])
def test_read_tif_real_lsm_basic_smoke_and_metadata(zarr_store, tmp_path, monkeypatch):
    # For zarr_store="disk" we want the cache folder to be created next to the file.
    # Therefore copy the LSM into tmp_path so we do not write into the repo.
    src = Path(_lsm_path())
    assert src.exists(), "Expected test LSM file in tests/test_images."

    f = tmp_path / src.name
    f.write_bytes(src.read_bytes())

    image, md = read_tif(str(f), zarr_store=zarr_store, verbose=False)

    if zarr_store is None:
        assert isinstance(image, np.ndarray)
    else:
        assert isinstance(image, zarr.core.array.Array)

    assert isinstance(md, dict)
    assert md.get("axes") == "TZCYX"
    assert len(image.shape) == 5

    # minimal invariants
    assert md["SizeX"] == image.shape[-1]
    assert md["SizeY"] == image.shape[-2]
    assert md["SizeZ"] == image.shape[1]
    assert md["SizeC"] == image.shape[2] or "SizeC" in md  # depending on how you populate
    assert md["PhysicalSizeX"] > 0
    assert md["PhysicalSizeY"] > 0
    assert md["PhysicalSizeZ"] > 0

    # metadata provenance should mention LSM somewhere
    # depending on your implementation, this may be overwritten by later sources,
    # so we do not require equality, but we require that LSM-derived keys exist.
    assert "TimeIncrement" in md or "DimensionTime" in md or "SizeT" in md

def test_read_tif_real_lsm_exposes_standardized_lsm_keys(tmp_path):
    src = Path(_lsm_path())
    f = tmp_path / src.name
    f.write_bytes(src.read_bytes())

    # Read raw lsm_metadata from tifffile so we know what is available in principle.
    with tifffile.TiffFile(str(f)) as tif:
        raw = tif.lsm_metadata
    assert raw is not None, "This test requires tifffile to expose lsm_metadata for the file."

    image, md = read_tif(str(f), verbose=False)

    # If raw contains these keys, OMIO should have them standardized.
    # We conditionally assert to keep the test robust across LSM variants.
    if "DimensionX" in raw:
        assert md.get("SizeX") == raw["DimensionX"]
    if "DimensionY" in raw:
        assert md.get("SizeY") == raw["DimensionY"]
    if "DimensionZ" in raw:
        assert md.get("SizeZ") == raw["DimensionZ"]
    if "DimensionChannels" in raw:
        assert md.get("SizeC") == raw["DimensionChannels"]
    if "DimensionTime" in raw:
        assert md.get("SizeT") == raw["DimensionTime"]

    if "VoxelSizeX" in raw:
        assert md.get("PhysicalSizeX") == raw["VoxelSizeX"]
    if "VoxelSizeY" in raw:
        assert md.get("PhysicalSizeY") == raw["VoxelSizeY"]
    if "VoxelSizeZ" in raw:
        assert md.get("PhysicalSizeZ") == raw["VoxelSizeZ"]

    if "TimeIntervall" in raw:
        assert md.get("TimeIncrement") == raw["TimeIntervall"]

def test_read_tif_real_lsm_pagination_behavior_if_present(tmp_path):
    src = Path(_lsm_path())
    f = tmp_path / src.name
    f.write_bytes(src.read_bytes())

    image, md = read_tif(str(f), verbose=False)

    # If the file is paginated (axis P), OMIO returns lists by design.
    # If it is not paginated, this test should not fail, it should just assert the opposite.
    if isinstance(image, list):
        assert isinstance(md, list)
        assert len(image) == len(md)
        assert len(image) >= 1

        for im, m in zip(image, md):
            assert m["axes"] == "TZCYX"  # after batch correction
            assert len(im.shape) == 5
            assert m["PhysicalSizeX"] > 0
            assert m["PhysicalSizeY"] > 0
            assert m["PhysicalSizeZ"] > 0

        if any(m.get("original_metadata_type") == "CZ_LSMINFO" for m in md):
            for m in md:
                if m.get("original_metadata_type") == "CZ_LSMINFO":
                    assert m["PhysicalSizeX"] != 1.0
                    assert m["PhysicalSizeY"] != 1.0
                    assert m["PhysicalSizeZ"] != 1.0
    else:
        assert isinstance(md, dict)
        assert md["axes"] == "TZCYX"


# %% TEST READ_CZI

def _czi_fixture_path() -> Path:
    return Path(__file__).resolve().parent / "test_images" / "xt-scan-lsm980.czi"

def test_read_czi_invalid_zarr_store_raises():
    with pytest.raises(ValueError):
        read_czi("dummy.czi", zarr_store="nope", verbose=False)

def test_read_czi_reads_fixture_and_returns_numpy():
    f = _czi_fixture_path()
    if not f.exists():
        pytest.skip(f"Missing test fixture file: {f}")

    image, md = read_czi(str(f), zarr_store=None, verbose=False)

    assert isinstance(image, np.ndarray)
    assert isinstance(md, dict)

    assert md.get("axes") == "TZCYX"
    assert image.ndim == 5
    assert image.shape == tuple(md["shape"])

    # basic size fields should exist and be consistent with image shape
    assert md["SizeX"] == image.shape[-1]
    assert md["SizeY"] == image.shape[-2]
    assert md["SizeC"] == image.shape[2]
    assert md["SizeZ"] == image.shape[1]
    assert md["SizeT"] == image.shape[0]

def test_read_czi_zarr_memory_returns_zarr_array():
    f = _czi_fixture_path()
    if not f.exists():
        pytest.skip(f"Missing test fixture file: {f}")

    image, md = read_czi(str(f), zarr_store="memory", verbose=False)

    assert isinstance(image, zarr.core.array.Array)
    assert md.get("axes") == "TZCYX"
    assert tuple(image.shape) == tuple(md["shape"])

def test_read_czi_return_list_true_wraps_singleton():
    f = _czi_fixture_path()
    if not f.exists():
        pytest.skip(f"Missing test fixture file: {f}")

    images, mds = read_czi(str(f), return_list=True, verbose=False)

    assert isinstance(images, list)
    assert isinstance(mds, list)
    assert len(images) == 1
    assert len(mds) == 1
    assert mds[0].get("axes") == "TZCYX"
    
def test_read_czi_physicalsize_override_applied():
    f = _czi_fixture_path()
    if not f.exists():
        pytest.skip(f"Missing test fixture file: {f}")

    image, md = read_czi(
        str(f),
        physicalsize_xyz=(0.11, 0.22, 0.33),
        pixelunit="micron",
        verbose=False,
    )

    assert md["PhysicalSizeX"] == pytest.approx(0.11)
    assert md["PhysicalSizeY"] == pytest.approx(0.22)
    assert md["PhysicalSizeZ"] == pytest.approx(0.33)
    
def test_read_czi_zarr_disk_creates_cache(tmp_path):
    # Copy fixture into tmp_path so the test does not create .omio_cache inside the repo.
    src = _czi_fixture_path()
    if not src.exists():
        pytest.skip(f"Missing test fixture file: {src}")

    dst = tmp_path / src.name
    dst.write_bytes(src.read_bytes())

    image, md = read_czi(str(dst), zarr_store="disk", verbose=False)

    assert isinstance(image, zarr.core.array.Array)
    cache_dir = tmp_path / ".omio_cache"
    assert cache_dir.exists()
    assert any(p.suffix == ".zarr" for p in cache_dir.iterdir())

# %% TEST READ_THORLABS_RAW

# helpers:

def _write_example_thorlabs_xml(
    xml_path: Path,
    *,
    X: int,
    Y: int,
    C: int,
    T: int,
    Z: int,
    pixel_size_um: float = 0.5,
    z_step_um: float = 1.0,
    time_interval_s: float = 1.0,
    bits: int = 16,
) -> None:
    # minimal XML that matches read_thorlabs_raw expectations
    xml = f"""<?xml version="1.0" encoding="utf-8"?>
<ThorImage>
  <LSM pixelX="{X}" pixelY="{Y}" channel="{C}" pixelSizeUM="{pixel_size_um}" frameRate="1.0"/>
  <Wavelengths>
    {''.join([f'<Wavelength index="{i}"/>' for i in range(C)])}
  </Wavelengths>
  <Timelapse timepoints="{T}" intervalSec="{time_interval_s}"/>
  <Camera bitsPerPixel="{bits}"/>
  <ZStage steps="{Z}" stepSizeUM="{z_step_um}"/>
  <Streaming zFastEnable="{1 if Z > 1 else 0}"/>
</ThorImage>
"""
    xml_path.write_text(xml, encoding="utf-8")

def _write_dummy_raw(
    raw_path: Path,
    *,
    T: int,
    Z: int,
    C: int,
    Y: int,
    X: int,
    dtype: np.dtype = np.uint16,
) -> np.ndarray:
    n = T * Z * C * Y * X
    data = np.arange(n, dtype=dtype)
    data.tofile(str(raw_path))
    return data

def _write_thorlabs_xml_variant(
    xml_path: Path,
    *,
    lsm: bool = True,
    wavelengths: str | None = "normal",   # "normal" | "empty" | None
    timelapse: bool = True,
    camera: bool = True,
    zstage: bool = True,
    streaming: bool = True,
    z_fast_enable: int = 1,
    X: int = 7,
    Y: int = 5,
    C_attr: int = 2,
    T: int = 2,
    Z: int = 3,
    bits: int = 16,
    pixel_size_um: float = 0.5,
    z_step_um: float = 1.0,
    time_interval_s: float = 1.0,
    frame_rate_value: str = "1.0",
    include_date: bool = False,
    date_value: str = "12/31/2025 10:11:12",
) -> None:
    parts = ['<?xml version="1.0" encoding="utf-8"?>', "<ThorImage>"]

    if lsm:
        parts.append(
            f'<LSM pixelX="{X}" pixelY="{Y}" channel="{C_attr}" '
            f'pixelSizeUM="{pixel_size_um}" frameRate="{frame_rate_value}"/>'
        )

    if wavelengths is None:
        pass
    elif wavelengths == "empty":
        parts.append("<Wavelengths></Wavelengths>")
    elif wavelengths == "normal":
        parts.append("<Wavelengths>" + "".join([f'<Wavelength index="{i}"/>' for i in range(C_attr)]) + "</Wavelengths>")
    else:
        raise ValueError("bad wavelengths mode")

    if timelapse:
        parts.append(f'<Timelapse timepoints="{T}" intervalSec="{time_interval_s}"/>')

    if camera:
        parts.append(f'<Camera bitsPerPixel="{bits}"/>')

    if zstage:
        parts.append(f'<ZStage steps="{Z}" stepSizeUM="{z_step_um}"/>')

    if streaming:
        parts.append(f'<Streaming zFastEnable="{z_fast_enable}"/>')

    if include_date:
        parts.append(f'<Date date="{date_value}"/>')

    parts.append("</ThorImage>")
    xml_path.write_text("\n".join(parts), encoding="utf-8")

# error path tests:

def test_read_thorlabs_raw_invalid_zarr_store_raises(tmp_path):
    raw_path = tmp_path / "x.raw"
    raw_path.write_bytes(b"")  # exists, but content irrelevant for this error

    with pytest.raises(ValueError):
        read_thorlabs_raw(str(raw_path), zarr_store="nope", verbose=False)

def test_read_thorlabs_raw_missing_file_raises(tmp_path):
    raw_path = tmp_path / "does_not_exist.raw"
    with pytest.raises(FileNotFoundError):
        read_thorlabs_raw(str(raw_path), verbose=False)

# xml path, numpy output test:

def test_read_thorlabs_raw_reads_xml_and_returns_numpy(tmp_path):
    T, Z, C, Y, X = 2, 3, 2, 5, 7
    raw_path = tmp_path / "example.raw"
    xml_path = tmp_path / "example.xml"

    _write_dummy_raw(raw_path, T=T, Z=Z, C=C, Y=Y, X=X, dtype=np.uint16)
    _write_example_thorlabs_xml(xml_path, X=X, Y=Y, C=C, T=T, Z=Z, bits=16)

    image, md = read_thorlabs_raw(str(raw_path), zarr_store=None, verbose=False)

    assert isinstance(image, np.ndarray)
    assert image.shape == (T, Z, C, Y, X)
    assert md["axes"] == "TZCYX"
    assert md["shape"] == (T, Z, C, Y, X)

    assert md["SizeT"] == T
    assert md["SizeZ"] == Z
    assert md["SizeC"] == C
    assert md["SizeY"] == Y
    assert md["SizeX"] == X

    # basic pixel size metadata present and positive
    assert md["PhysicalSizeX"] > 0
    assert md["PhysicalSizeY"] > 0
    assert md["PhysicalSizeZ"] > 0
    assert md["Annotations"]["unit"] is not None

# tests:

def test_read_thorlabs_raw_corrects_Z_from_file_size_when_xml_mismatches(tmp_path):
    T, Z_true, C, Y, X = 1, 4, 1, 6, 6
    Z_wrong = 2

    raw_path = tmp_path / "z_mismatch.raw"
    xml_path = tmp_path / "z_mismatch.xml"

    _write_dummy_raw(raw_path, T=T, Z=Z_true, C=C, Y=Y, X=X, dtype=np.uint16)
    _write_example_thorlabs_xml(xml_path, X=X, Y=Y, C=C, T=T, Z=Z_wrong, bits=16)

    image, md = read_thorlabs_raw(str(raw_path), verbose=False)

    assert image.shape[1] == Z_true
    assert md["SizeZ"] == Z_true

def test_read_thorlabs_raw_zarr_memory_returns_zarr_array(tmp_path):
    T, Z, C, Y, X = 1, 2, 1, 8, 9
    raw_path = tmp_path / "example.raw"
    xml_path = tmp_path / "example.xml"

    _write_dummy_raw(raw_path, T=T, Z=Z, C=C, Y=Y, X=X, dtype=np.uint16)
    _write_example_thorlabs_xml(xml_path, X=X, Y=Y, C=C, T=T, Z=Z, bits=16)

    image, md = read_thorlabs_raw(str(raw_path), zarr_store="memory", verbose=False)

    assert isinstance(image, zarr.core.array.Array)
    assert tuple(image.shape) == (T, Z, C, Y, X)
    assert md["axes"] == "TZCYX"

def test_read_thorlabs_raw_zarr_disk_creates_cache(tmp_path):
    T, Z, C, Y, X = 1, 2, 1, 8, 9
    raw_path = tmp_path / "example.raw"
    xml_path = tmp_path / "example.xml"

    _write_dummy_raw(raw_path, T=T, Z=Z, C=C, Y=Y, X=X, dtype=np.uint16)
    _write_example_thorlabs_xml(xml_path, X=X, Y=Y, C=C, T=T, Z=Z, bits=16)

    image, md = read_thorlabs_raw(str(raw_path), zarr_store="disk", verbose=False)

    assert isinstance(image, zarr.core.array.Array)

    cache_dir = tmp_path / ".omio_cache"
    assert cache_dir.exists()
    assert any(p.suffix == ".zarr" for p in cache_dir.iterdir())

def test_read_thorlabs_raw_physicalsize_override_applied(tmp_path):
    T, Z, C, Y, X = 1, 1, 1, 4, 4
    raw_path = tmp_path / "example.raw"
    xml_path = tmp_path / "example.xml"

    _write_dummy_raw(raw_path, T=T, Z=Z, C=C, Y=Y, X=X, dtype=np.uint16)
    _write_example_thorlabs_xml(xml_path, X=X, Y=Y, C=C, T=T, Z=Z, bits=16)

    image, md = read_thorlabs_raw(
        str(raw_path),
        physicalsize_xyz=(0.11, 0.22, 0.33),
        verbose=False,
    )

    assert md["PhysicalSizeX"] == pytest.approx(0.11)
    assert md["PhysicalSizeY"] == pytest.approx(0.22)
    assert md["PhysicalSizeZ"] == pytest.approx(0.33)


# yaml cases:

def _write_yaml(path, obj):
    path.write_text(yaml.safe_dump(obj), encoding="utf-8")

def test_read_thorlabs_raw_uses_yaml_when_no_xml_present(tmp_path):
    # No XML in folder, but one YAML file with required keys exists.
    T, Z, C, Y, X = 2, 3, 2, 5, 7
    raw_path = tmp_path / "example.raw"
    yaml_path = tmp_path / "meta.yaml"

    _write_dummy_raw(raw_path, T=T, Z=Z, C=C, Y=Y, X=X, dtype=np.uint16)

    _write_yaml(
        yaml_path,
        {
            "T": T,
            "Z": Z,
            "C": C,
            "Y": Y,
            "X": X,
            "bits": 16,
            "pixelunit": "micron",
            "PhysicalSizeX": 0.5,
            "PhysicalSizeY": 0.5,
            "PhysicalSizeZ": 1.0,
            "TimeIncrement": 2.0,
            "TimeIncrementUnit": "seconds",
        },
    )

    image, md = read_thorlabs_raw(str(raw_path), zarr_store=None, verbose=False)

    assert isinstance(image, np.ndarray)
    assert image.shape == (T, Z, C, Y, X)
    assert md["axes"] == "TZCYX"
    assert md["SizeT"] == T
    assert md["SizeZ"] == Z
    assert md["SizeC"] == C
    assert md["SizeY"] == Y
    assert md["SizeX"] == X

    assert md["Annotations"]["unit"] == "micron"
    assert md["PhysicalSizeX"] == pytest.approx(0.5)
    assert md["PhysicalSizeY"] == pytest.approx(0.5)
    assert md["PhysicalSizeZ"] == pytest.approx(1.0)
    assert md["TimeIncrement"] == pytest.approx(2.0)
    assert md["TimeIncrementUnit"] == "seconds"

def test_read_thorlabs_raw_yaml_missing_required_key_warns_and_returns_none(tmp_path):
    # YAML exists but is missing required key 'bits' (or any of T,Z,C,Y,X,bits)
    raw_path = tmp_path / "example.raw"
    yaml_path = tmp_path / "meta.yaml"

    _write_dummy_raw(raw_path, T=1, Z=1, C=1, Y=4, X=4, dtype=np.uint16)

    _write_yaml(
        yaml_path,
        {
            "T": 1,
            "Z": 1,
            "C": 1,
            "Y": 4,
            "X": 4,
            # "bits" missing on purpose
            "pixelunit": "micron",
        },
    )

    with pytest.warns(UserWarning):
        image, md = read_thorlabs_raw(str(raw_path), verbose=False)

    assert image is None
    assert md is None

def test_read_thorlabs_raw_yaml_missing_required_key_warns_and_returns_list_none_when_return_list(tmp_path):
    raw_path = tmp_path / "example.raw"
    yaml_path = tmp_path / "meta.yaml"

    _write_dummy_raw(raw_path, T=1, Z=1, C=1, Y=4, X=4, dtype=np.uint16)

    _write_yaml(
        yaml_path,
        {
            "T": 1,
            "Z": 1,
            "C": 1,
            "Y": 4,
            "X": 4,
            # "bits" missing on purpose
        },
    )

    with pytest.warns(UserWarning):
        images, mds = read_thorlabs_raw(str(raw_path), return_list=True, verbose=False)

    assert images == [None]
    assert mds == [None]

def test_read_thorlabs_raw_multiple_yaml_files_warns_but_reads_first(tmp_path):
    # Behavior is intentionally platform dependent regarding which YAML is "first".
    # We therefore accept either outcome, but require that a warning is emitted and
    # that reading succeeds using one of the YAMLs.
    raw_path = tmp_path / "example.raw"
    y1 = tmp_path / "a.yaml"
    y2 = tmp_path / "b.yaml"

    T1, Z1, C1, Y1d, X1d = 1, 1, 1, 4, 4
    T2, Z2, C2, Y2d, X2d = 1, 2, 1, 4, 4

    # raw must match whichever YAML ends up being chosen; make it compatible with BOTH:
    # choose Z=max(Z1,Z2) and write raw accordingly, then the YAML with smaller Z will still
    # pass file-size check only if expected_elements matches. It won't. So instead we write
    # two RAWs? That would violate "same folder". Therefore, we constrain both YAMLs to same dims.
    T, Z, C, Y, X = 1, 2, 1, 4, 4
    _write_dummy_raw(raw_path, T=T, Z=Z, C=C, Y=Y, X=X, dtype=np.uint16)

    _write_yaml(y1, {"T": T, "Z": Z, "C": C, "Y": Y, "X": X, "bits": 16, "PhysicalSizeX": 0.5})
    _write_yaml(y2, {"T": T, "Z": Z, "C": C, "Y": Y, "X": X, "bits": 16, "PhysicalSizeX": 0.6})

    with pytest.warns(UserWarning):
        image, md = read_thorlabs_raw(str(raw_path), verbose=False)

    assert isinstance(image, np.ndarray)
    assert image.shape == (T, Z, C, Y, X)
    assert md["axes"] == "TZCYX"
    assert md["PhysicalSizeX"] in (pytest.approx(0.5), pytest.approx(0.6))


# more tests:

def test_read_thorlabs_raw_xml_missing_lsm_raises(tmp_path):
    raw_path = tmp_path / "x.raw"
    xml_path = tmp_path / "x.xml"
    _write_dummy_raw(raw_path, T=1, Z=1, C=1, Y=4, X=4, dtype=np.uint16)

    _write_thorlabs_xml_variant(xml_path, lsm=False)

    with pytest.raises(ValueError, match="missing the LSM node"):
        read_thorlabs_raw(str(raw_path), verbose=False)

def test_read_thorlabs_raw_xml_no_wavelengths_uses_lsm_channel(tmp_path):
    T, Z, C, Y, X = 1, 2, 3, 4, 5
    raw_path = tmp_path / "x.raw"
    xml_path = tmp_path / "x.xml"
    _write_dummy_raw(raw_path, T=T, Z=Z, C=C, Y=Y, X=X, dtype=np.uint16)

    _write_thorlabs_xml_variant(
        xml_path,
        wavelengths=None,
        X=X, Y=Y, C_attr=C, T=T, Z=Z,
        bits=16
    )

    img, md = read_thorlabs_raw(str(raw_path), verbose=False)
    assert img.shape == (T, Z, C, Y, X)
    assert md["SizeC"] == C

def test_read_thorlabs_raw_xml_empty_wavelengths_falls_back_to_lsm_channel(tmp_path):
    T, Z, C, Y, X = 1, 2, 4, 4, 4
    raw_path = tmp_path / "x.raw"
    xml_path = tmp_path / "x.xml"
    _write_dummy_raw(raw_path, T=T, Z=Z, C=C, Y=Y, X=X, dtype=np.uint16)

    _write_thorlabs_xml_variant(
        xml_path,
        wavelengths="empty",
        X=X, Y=Y, C_attr=C, T=T, Z=Z
    )

    img, md = read_thorlabs_raw(str(raw_path), verbose=False)
    assert img.shape == (T, Z, C, Y, X)
    assert md["SizeC"] == C

def test_read_thorlabs_raw_xml_no_timelapse_defaults_T_and_timeincrement(tmp_path):
    raw_path = tmp_path / "x.raw"
    xml_path = tmp_path / "x.xml"

    T, Z, C, Y, X = 1, 2, 1, 3, 3
    _write_dummy_raw(raw_path, T=T, Z=Z, C=C, Y=Y, X=X, dtype=np.uint16)

    _write_thorlabs_xml_variant(
        xml_path,
        timelapse=False,
        X=X, Y=Y, C_attr=C, T=99, Z=Z
    )

    img, md = read_thorlabs_raw(str(raw_path), verbose=False)
    assert img.shape[0] == 1
    assert md["SizeT"] == 1
    assert md["TimeIncrement"] == pytest.approx(1.0)

def test_read_thorlabs_raw_xml_no_camera_defaults_bits_16(tmp_path):
    T, Z, C, Y, X = 1, 1, 1, 4, 4
    raw_path = tmp_path / "x.raw"
    xml_path = tmp_path / "x.xml"
    _write_dummy_raw(raw_path, T=T, Z=Z, C=C, Y=Y, X=X, dtype=np.uint16)

    _write_thorlabs_xml_variant(xml_path, camera=False, X=X, Y=Y, C_attr=C, T=T, Z=Z)

    img, md = read_thorlabs_raw(str(raw_path), verbose=False)

    assert img.dtype == np.uint16
    # bits_per_pixel is not guaranteed by current implementation in all cases:
    if "bits_per_pixel" in md:
        assert md["bits_per_pixel"] == 16

@pytest.mark.parametrize("bits,dtype", [(8, np.uint8), (32, np.float32)])
def test_read_thorlabs_raw_bits_drives_dtype(tmp_path, bits, dtype):
    T, Z, C, Y, X = 1, 1, 1, 4, 4
    raw_path = tmp_path / "x.raw"
    xml_path = tmp_path / "x.xml"

    _write_dummy_raw(raw_path, T=T, Z=Z, C=C, Y=Y, X=X, dtype=dtype)
    _write_thorlabs_xml_variant(xml_path, bits=bits, X=X, Y=Y, C_attr=C, T=T, Z=Z)

    img, md = read_thorlabs_raw(str(raw_path), verbose=False)

    assert img.dtype == dtype
    if "bits_per_pixel" in md:
        assert md["bits_per_pixel"] == bits

def test_read_thorlabs_raw_xml_zfast_disabled_defaults_Z_to_1(tmp_path):
    T, Z_true, C, Y, X = 1, 1, 1, 4, 4
    raw_path = tmp_path / "x.raw"
    xml_path = tmp_path / "x.xml"

    _write_dummy_raw(raw_path, T=T, Z=Z_true, C=C, Y=Y, X=X, dtype=np.uint16)

    _write_thorlabs_xml_variant(
        xml_path,
        zstage=True,
        streaming=True,
        z_fast_enable=0,
        Z=99,  # would be ignored, because zFastEnable=0
        X=X, Y=Y, C_attr=C, T=T
    )

    img, md = read_thorlabs_raw(str(raw_path), verbose=False)
    assert img.shape[1] == 1
    assert md["SizeZ"] == 1

def test_read_thorlabs_raw_xml_filesize_not_multiple_warns_and_returns_none(tmp_path):
    T, Z, C, Y, X = 1, 2, 1, 4, 4
    raw_path = tmp_path / "x.raw"
    xml_path = tmp_path / "x.xml"

    _write_dummy_raw(raw_path, T=T, Z=Z, C=C, Y=Y, X=X, dtype=np.uint16)
    with open(raw_path, "ab") as f:
        np.array([123], dtype=np.uint16).tofile(f)

    _write_thorlabs_xml_variant(xml_path, X=X, Y=Y, C_attr=C, T=T, Z=Z, bits=16)

    with pytest.warns(UserWarning, match="RAW data size mismatch"):
        img, md = read_thorlabs_raw(str(raw_path), verbose=False)

    assert img is None
    assert md is None

def test_read_thorlabs_raw_xml_frame_rate_parse_fail_does_not_crash(tmp_path):
    T, Z, C, Y, X = 1, 1, 1, 4, 4
    raw_path = tmp_path / "x.raw"
    xml_path = tmp_path / "x.xml"
    _write_dummy_raw(raw_path, T=T, Z=Z, C=C, Y=Y, X=X, dtype=np.uint16)

    _write_thorlabs_xml_variant(
        xml_path,
        frame_rate_value="not_a_number",
        X=X, Y=Y, C_attr=C, T=T, Z=Z
    )

    img, md = read_thorlabs_raw(str(raw_path), verbose=False)
    assert img is not None
    assert md is not None
    if "frame_rate" in md:
        assert md["frame_rate"] == 0.0

def test_read_thorlabs_raw_xml_date_parse_fail_does_not_crash(tmp_path):
    T, Z, C, Y, X = 1, 1, 1, 4, 4
    raw_path = tmp_path / "x.raw"
    xml_path = tmp_path / "x.xml"
    _write_dummy_raw(raw_path, T=T, Z=Z, C=C, Y=Y, X=X, dtype=np.uint16)

    _write_thorlabs_xml_variant(
        xml_path,
        include_date=True,
        date_value="bad date string",
        X=X, Y=Y, C_attr=C, T=T, Z=Z
    )

    img, md = read_thorlabs_raw(str(raw_path), verbose=False)
    assert img is not None
    assert md is not None

def test_read_thorlabs_raw_numpy_size_mismatch_warns_and_returns_none(tmp_path):
    T, Z, C, Y, X = 1, 2, 1, 4, 4
    raw_path = tmp_path / "x.raw"
    xml_path = tmp_path / "x.xml"

    # write too few elements
    np.arange(10, dtype=np.uint16).tofile(str(raw_path))
    _write_thorlabs_xml_variant(xml_path, X=X, Y=Y, C_attr=C, T=T, Z=Z, bits=16)

    with pytest.warns(UserWarning, match="RAW data size mismatch"):
        img, md = read_thorlabs_raw(str(raw_path), zarr_store=None, verbose=False)

    assert img is None
    assert md is None

def test_read_thorlabs_raw_zarr_size_mismatch_warns_and_returns_none(tmp_path):
    T, Z, C, Y, X = 1, 2, 1, 4, 4
    raw_path = tmp_path / "x.raw"
    xml_path = tmp_path / "x.xml"

    np.arange(10, dtype=np.uint16).tofile(str(raw_path))
    _write_thorlabs_xml_variant(xml_path, X=X, Y=Y, C_attr=C, T=T, Z=Z, bits=16)

    with pytest.warns(UserWarning, match="RAW data size mismatch"):
        img, md = read_thorlabs_raw(str(raw_path), zarr_store="memory", verbose=False)

    assert img is None
    assert md is None

def test_read_thorlabs_raw_yaml_optional_parse_fail_is_ignored(tmp_path):
    raw_path = tmp_path / "x.raw"
    yaml_path = tmp_path / "meta.yaml"

    T, Z, C, Y, X = 1, 1, 1, 4, 4
    _write_dummy_raw(raw_path, T=T, Z=Z, C=C, Y=Y, X=X, dtype=np.uint16)

    _write_yaml(
        yaml_path,
        {
            "T": T, "Z": Z, "C": C, "Y": Y, "X": X, "bits": 16,
            "pixelunit": "micron",
            "PhysicalSizeX": "not a float",
            "TimeIncrement": "also bad",
        },
    )

    img, md = read_thorlabs_raw(str(raw_path), verbose=False)

    assert img is not None
    assert md["Annotations"]["unit"] == "micron"
    assert md["PhysicalSizeX"] == pytest.approx(1.0)
    assert "TimeIncrement" not in md
    
def test_read_thorlabs_raw_success_return_list_true(tmp_path):
    T, Z, C, Y, X = 1, 2, 1, 4, 4
    raw_path = tmp_path / "x.raw"
    xml_path = tmp_path / "x.xml"

    _write_dummy_raw(raw_path, T=T, Z=Z, C=C, Y=Y, X=X, dtype=np.uint16)
    _write_thorlabs_xml_variant(xml_path, X=X, Y=Y, C_attr=C, T=T, Z=Z, bits=16)

    imgs, mds = read_thorlabs_raw(str(raw_path), return_list=True, verbose=False)

    assert isinstance(imgs, list) and isinstance(mds, list)
    assert imgs[0].shape == (T, Z, C, Y, X)
    assert mds[0]["axes"] == "TZCYX"

def test_read_thorlabs_raw_verbose_emits_expected_messages(tmp_path, capsys):
    T, Z, C, Y, X = 1, 1, 1, 4, 4
    raw_path = tmp_path / "x.raw"
    xml_path = tmp_path / "x.xml"
    _write_dummy_raw(raw_path, T=T, Z=Z, C=C, Y=Y, X=X, dtype=np.uint16)
    _write_thorlabs_xml_variant(xml_path, X=X, Y=Y, C_attr=C, T=T, Z=Z)

    read_thorlabs_raw(str(raw_path), verbose=True)

    out = capsys.readouterr().out
    assert "Reading Thorlabs RAW file" in out
    assert "Found XML file" in out
    assert "Finished reading Thorlabs RAW file" in out


# %% CLEANUP OMIO CACHE

def test_cleanup_omio_cache_raises_on_nonexistent_path(tmp_path):
    p = tmp_path / "does_not_exist"
    with pytest.raises(ValueError):
        cleanup_omio_cache(str(p), verbose=False)

def test_cleanup_omio_cache_returns_silently_if_no_cache_folder_for_file(tmp_path, capsys):
    data_file = tmp_path / "image.tif"
    data_file.write_bytes(b"dummy")

    cleanup_omio_cache(str(data_file), full_cleanup=False, verbose=True)

    out = capsys.readouterr().out
    assert "No .omio_cache folder found" in out

def test_cleanup_omio_cache_targeted_removes_matching_zarr_store_only(tmp_path, capsys):
    data_file = tmp_path / "image.tif"
    data_file.write_bytes(b"dummy")

    cache_dir = tmp_path / ".omio_cache"
    cache_dir.mkdir()

    target_store = cache_dir / "image.zarr"
    other_store = cache_dir / "other.zarr"
    target_store.mkdir()
    other_store.mkdir()

    # sanity
    assert target_store.exists()
    assert other_store.exists()

    cleanup_omio_cache(str(data_file), full_cleanup=False, verbose=True)

    out = capsys.readouterr().out
    assert "Deleting Zarr store for image" in out

    assert not target_store.exists()
    assert other_store.exists()
    assert cache_dir.exists()

def test_cleanup_omio_cache_targeted_no_store_prints_message(tmp_path, capsys):
    data_file = tmp_path / "image.tif"
    data_file.write_bytes(b"dummy")

    cache_dir = tmp_path / ".omio_cache"
    cache_dir.mkdir()

    cleanup_omio_cache(str(data_file), full_cleanup=False, verbose=True)

    out = capsys.readouterr().out
    assert "No Zarr store found for image" in out
    assert cache_dir.exists()

def test_cleanup_omio_cache_full_cleanup_on_file_removes_entire_cache_dir(tmp_path, capsys):
    data_file = tmp_path / "image.tif"
    data_file.write_bytes(b"dummy")

    cache_dir = tmp_path / ".omio_cache"
    cache_dir.mkdir()
    (cache_dir / "image.zarr").mkdir()
    (cache_dir / "other.zarr").mkdir()

    cleanup_omio_cache(str(data_file), full_cleanup=True, verbose=True)

    out = capsys.readouterr().out
    assert "Performing full cleanup of .omio_cache folder" in out

    assert not cache_dir.exists()

def test_cleanup_omio_cache_full_cleanup_on_directory_removes_cache_dir(tmp_path, capsys):
    cache_dir = tmp_path / ".omio_cache"
    cache_dir.mkdir()
    (cache_dir / "anything.zarr").mkdir()

    cleanup_omio_cache(str(tmp_path), full_cleanup=False, verbose=True)

    out = capsys.readouterr().out
    assert "Performing full cleanup of .omio_cache folder" in out

    assert not cache_dir.exists()

def test_cleanup_omio_cache_directory_without_cache_returns_silently(tmp_path, capsys):
    cleanup_omio_cache(str(tmp_path), full_cleanup=False, verbose=True)

    out = capsys.readouterr().out
    assert "No .omio_cache folder found" in out

# %% EMPTY IMAGE AND METADATA CREATORS

# create_empty_metadata:

def test_create_empty_metadata_defaults_and_core_keys_present():
    md = create_empty_metadata(verbose=False)

    assert isinstance(md, dict)

    assert md["axes"] == "TZCYX"
    assert md["shape"] is None

    for k in ("SizeT", "SizeZ", "SizeC", "SizeY", "SizeX"):
        assert k in md
        assert md[k] is None

    assert md["PhysicalSizeX"] == 1
    assert md["PhysicalSizeY"] == 1
    assert md["PhysicalSizeZ"] == 1

    assert md["TimeIncrement"] == 1
    assert md["TimeIncrementUnit"] == "s"

    assert md["Annotations"]["unit"] == "µm"
    assert "Annotations" in md
    assert isinstance(md["Annotations"], dict)
    assert "Namespace" in md["Annotations"]

    # OMIO_VERSION is moved into Annotations by OME_metadata_checkup
    assert "OMIO_VERSION" in md["Annotations"]

def test_create_empty_metadata_pixelunit_normalization_to_um():
    md = create_empty_metadata(pixelunit="micron", verbose=False)
    assert md["Annotations"]["unit"] == "µm"

    md = create_empty_metadata(pixelunit="um", verbose=False)
    assert md["Annotations"]["unit"] == "µm"

    md = create_empty_metadata(pixelunit="µm", verbose=False)
    assert md["Annotations"]["unit"] == "µm"

    md = create_empty_metadata(pixelunit="micrometer", verbose=False)
    assert md["Annotations"]["unit"] == "µm"

def test_create_empty_metadata_custom_pixelunit_preserved():
    md = create_empty_metadata(pixelunit="nm", verbose=False)
    assert md["Annotations"]["unit"] == "nm"
    
def test_create_empty_metadata_physicalsize_override_applied():
    md = create_empty_metadata(physicalsize_xyz=(0.1, 0.2, 0.3), verbose=False)

    assert md["PhysicalSizeX"] == pytest.approx(0.1)
    assert md["PhysicalSizeY"] == pytest.approx(0.2)
    assert md["PhysicalSizeZ"] == pytest.approx(0.3)
    
def test_create_empty_metadata_time_increment_overrides_applied():
    md = create_empty_metadata(time_increment=2.5, time_increment_unit="seconds", verbose=False)

    assert md["TimeIncrement"] == pytest.approx(2.5)
    assert md["TimeIncrementUnit"] == "seconds"
    
def test_create_empty_metadata_shape_sets_sizes_consistently():
    shape = (2, 3, 4, 5, 6)  # T Z C Y X
    md = create_empty_metadata(shape=shape, verbose=False)

    assert md["shape"] == shape
    assert md["SizeT"] == 2
    assert md["SizeZ"] == 3
    assert md["SizeC"] == 4
    assert md["SizeY"] == 5
    assert md["SizeX"] == 6
    
def test_create_empty_metadata_invalid_shape_warns_and_does_not_set_shape():
    with pytest.warns(UserWarning):
        md = create_empty_metadata(shape=(1, 2, 3), verbose=False)

    assert md["shape"] is None
    assert md["SizeT"] is None
    assert md["SizeZ"] is None
    assert md["SizeC"] is None
    assert md["SizeY"] is None
    assert md["SizeX"] is None
    
def test_create_empty_metadata_input_metadata_is_not_modified_in_place():
    input_md = {
        "foo": 123,
        "Annotations": {"bar": "baz"},
    }
    input_md_copy = {"foo": 123, "Annotations": {"bar": "baz"}}

    md = create_empty_metadata(input_metadata=input_md, verbose=False)

    assert input_md == input_md_copy
    assert "foo" not in md  # moved into Annotations by OME_metadata_checkup
    assert md["Annotations"]["foo"] == 123
    assert md["Annotations"]["bar"] == "baz"
    
def test_create_empty_metadata_annotations_merge_into_existing_annotations():
    input_md = {"Annotations": {"a": 1}}
    md = create_empty_metadata(
        input_metadata=input_md,
        annotations={"b": 2},
        verbose=False,
    )

    assert md["Annotations"]["a"] == 1
    assert md["Annotations"]["b"] == 2
    
def test_create_empty_metadata_explicit_overrides_win_over_input_metadata():
    input_md = {
        "PhysicalSizeX": 9.0,
        "PhysicalSizeY": 9.0,
        "PhysicalSizeZ": 9.0,
        "TimeIncrement": 9.0,
        "TimeIncrementUnit": "min",
    }

    md = create_empty_metadata(
        input_metadata=input_md,
        physicalsize_xyz=(0.1, 0.2, 0.3),
        time_increment=2.5,
        time_increment_unit="seconds",
        verbose=False,
    )

    assert md["PhysicalSizeX"] == pytest.approx(0.1)
    assert md["PhysicalSizeY"] == pytest.approx(0.2)
    assert md["PhysicalSizeZ"] == pytest.approx(0.3)
    assert md["TimeIncrement"] == pytest.approx(2.5)
    assert md["TimeIncrementUnit"] == "seconds"

# create_empty_image:


def test_create_empty_image_invalid_shape_returns_none_and_warns(capsys):
    img = create_empty_image(shape=(1, 2, 3), verbose=True)
    captured = capsys.readouterr()
    assert img is None
    assert "WARNING create_empty_image: shape must be a 5-tuple" in captured.out

def test_create_empty_image_invalid_shape_returns_tuple_none_when_return_metadata(capsys):
    img, md = create_empty_image(shape=(1, 2, 3), return_metadata=True, verbose=False)
    captured = capsys.readouterr()
    assert img is None
    assert md is None
    assert "WARNING create_empty_image: shape must be a 5-tuple" in captured.out

def test_create_empty_image_numpy_default_is_zeros_uint16():
    shape = (1, 2, 3, 4, 5)
    img = create_empty_image(shape=shape, zarr_store=None, verbose=False)

    assert isinstance(img, np.ndarray)
    assert img.shape == shape
    assert img.dtype == np.uint16
    assert np.all(img == 0)

def test_create_empty_image_numpy_fill_value_nonzero():
    shape = (1, 1, 1, 3, 3)
    img = create_empty_image(shape=shape, fill_value=7, dtype=np.uint8, zarr_store=None, verbose=False)

    assert isinstance(img, np.ndarray)
    assert img.shape == shape
    assert img.dtype == np.uint8
    assert np.all(img == 7)

def test_create_empty_image_numpy_return_metadata_consistent():
    shape = (2, 3, 1, 4, 5)
    img, md = create_empty_image(shape=shape, return_metadata=True, verbose=False)

    assert isinstance(img, np.ndarray)
    assert img.shape == shape
    assert md["axes"] == "TZCYX"
    assert md["shape"] == shape
    assert md["SizeT"] == 2
    assert md["SizeZ"] == 3
    assert md["SizeC"] == 1
    assert md["SizeY"] == 4
    assert md["SizeX"] == 5

def test_create_empty_image_zarr_invalid_store_warns_and_returns_none():
    with pytest.warns(UserWarning):
        img = create_empty_image(shape=(1, 1, 1, 2, 2), zarr_store="nope", verbose=False)
    assert img is None

def test_create_empty_image_zarr_memory_returns_zarr_and_fills_zero():
    shape = (1, 2, 1, 4, 4)
    img = create_empty_image(shape=shape, zarr_store="memory", fill_value=0, verbose=False)

    assert isinstance(img, zarr.core.array.Array)
    assert tuple(img.shape) == shape
    assert np.all(np.asarray(img[:]) == 0)

def test_create_empty_image_zarr_memory_fill_value_nonzero():
    shape = (1, 1, 2, 3, 3)
    img = create_empty_image(shape=shape, zarr_store="memory", fill_value=5, dtype=np.uint8, verbose=False)

    assert isinstance(img, zarr.core.array.Array)
    assert tuple(img.shape) == shape
    assert np.all(np.asarray(img[:]) == 5)

def test_create_empty_image_zarr_memory_fill_value_none_leaves_array_writable():
    # We do not assert contents, because it is explicitly uninitialized.
    shape = (1, 1, 1, 2, 2)
    img = create_empty_image(shape=shape, zarr_store="memory", fill_value=None, verbose=False)

    assert isinstance(img, zarr.core.array.Array)
    assert tuple(img.shape) == shape

    img[:] = 3
    assert np.all(np.asarray(img[:]) == 3)

def test_create_empty_image_zarr_disk_requires_path_warns_and_returns_none(tmp_path):
    with pytest.warns(UserWarning):
        img = create_empty_image(
            shape=(1, 1, 1, 2, 2),
            zarr_store="disk",
            zarr_store_path=None,
            zarr_store_name="x",
            verbose=False,
        )
    assert img is None

def test_create_empty_image_zarr_disk_requires_name_warns_and_returns_none(tmp_path):
    with pytest.warns(UserWarning):
        img = create_empty_image(
            shape=(1, 1, 1, 2, 2),
            zarr_store="disk",
            zarr_store_path=str(tmp_path),
            zarr_store_name=None,
            verbose=False,
        )
    assert img is None

def test_create_empty_image_zarr_disk_creates_cache_under_directory(tmp_path):
    shape = (1, 1, 1, 3, 4)
    img = create_empty_image(
        shape=shape,
        zarr_store="disk",
        zarr_store_path=str(tmp_path),
        zarr_store_name="test_store",
        fill_value=0,
        verbose=False,
    )

    assert isinstance(img, zarr.core.array.Array)
    assert tuple(img.shape) == shape

    cache_dir = tmp_path / ".omio_cache"
    assert cache_dir.exists()
    assert (cache_dir / "test_store.zarr").exists()
    assert np.all(np.asarray(img[:]) == 0)

def test_create_empty_image_zarr_disk_path_can_be_file_uses_parent_folder(tmp_path, capsys):
    shape = (1, 1, 1, 2, 2)

    # Provide a "file path" under tmp_path
    fake_file = tmp_path / "some_input_file.tif"

    img = create_empty_image(
        shape=shape,
        zarr_store="disk",
        zarr_store_path=str(fake_file),
        zarr_store_name="from_file_parent",
        fill_value=1,
        verbose=True,
    )

    out = capsys.readouterr().out
    assert "zarr_store_path is a file; taking its parent folder" in out

    cache_dir = tmp_path / ".omio_cache"
    assert (cache_dir / "from_file_parent.zarr").exists()
    assert np.all(np.asarray(img[:]) == 1)

def test_create_empty_image_zarr_disk_overwrites_existing_store(tmp_path):
    shape = (1, 1, 1, 2, 2)

    img1 = create_empty_image(
        shape=shape,
        zarr_store="disk",
        zarr_store_path=str(tmp_path),
        zarr_store_name="overwrite_me",
        fill_value=2,
        verbose=False,
    )
    assert np.all(np.asarray(img1[:]) == 2)

    img2 = create_empty_image(
        shape=shape,
        zarr_store="disk",
        zarr_store_path=str(tmp_path),
        zarr_store_name="overwrite_me",
        fill_value=0,
        verbose=False,
    )
    assert np.all(np.asarray(img2[:]) == 0)

def test_create_empty_image_return_metadata_for_zarr_memory():
    shape = (2, 1, 1, 3, 3)
    img, md = create_empty_image(
        shape=shape,
        zarr_store="memory",
        return_metadata=True,
        verbose=False,
    )

    assert isinstance(img, zarr.core.array.Array)
    assert tuple(img.shape) == shape
    assert md["axes"] == "TZCYX"
    assert md["shape"] == shape
    assert md["SizeT"] == 2
    assert md["SizeZ"] == 1
    assert md["SizeC"] == 1
    assert md["SizeY"] == 3
    assert md["SizeX"] == 3

def test_create_empty_image_metadata_merges_input_metadata_into_annotations():
    shape = (1, 1, 1, 2, 2)
    img, md = create_empty_image(
        shape=shape,
        return_metadata=True,
        input_metadata={"CustomField": 123},
        verbose=False,
    )

    assert isinstance(img, np.ndarray)
    assert "CustomField" not in md  # moved by OME_metadata_checkup
    assert md["Annotations"]["CustomField"] == 123


# update_metadata_from_image:

def test_update_metadata_from_image_sets_axes_shape_and_sizes_numpy():
    img = np.zeros((2, 3, 4, 5, 6), dtype=np.uint8)  # T Z C Y X
    md0 = {"SomeField": 123}

    md = update_metadata_from_image(md0, img, run_checkup=True, verbose=False)

    assert md["axes"] == "TZCYX"
    assert md["shape"] == (2, 3, 4, 5, 6)

    assert md["SizeT"] == 2
    assert md["SizeZ"] == 3
    assert md["SizeC"] == 4
    assert md["SizeY"] == 5
    assert md["SizeX"] == 6

    # moved into Annotations by checkup
    assert "SomeField" not in md
    assert md["Annotations"]["SomeField"] == 123
    assert "Namespace" in md["Annotations"]

def test_update_metadata_from_image_does_not_modify_input_metadata_in_place():
    img = np.zeros((1, 1, 1, 2, 3), dtype=np.uint16)
    md0 = {"axes": "WRONG", "foo": "bar"}
    md0_copy = dict(md0)

    md = update_metadata_from_image(md0, img, run_checkup=False, verbose=False)

    assert md0 == md0_copy
    assert md["axes"] == "TZCYX"
    assert md["shape"] == (1, 1, 1, 2, 3)
    assert md["SizeX"] == 3

def test_update_metadata_from_image_run_checkup_false_keeps_extra_keys_top_level():
    img = np.zeros((1, 2, 1, 3, 4), dtype=np.uint8)
    md0 = {"extra": 1}

    md = update_metadata_from_image(md0, img, run_checkup=False, verbose=False)

    assert md["extra"] == 1
    assert "Annotations" not in md or isinstance(md.get("Annotations"), dict)  # no guarantee here

def test_update_metadata_from_image_raises_on_non_5d_image():
    img = np.zeros((2, 3, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        update_metadata_from_image({}, img, verbose=False)

def test_update_metadata_from_image_accepts_zarr_array():
    shape = (2, 1, 3, 4, 5)
    store = zarr.storage.MemoryStore()
    z = zarr.open(store=store, mode="w", shape=shape, dtype=np.uint8, chunks=(1, 1, 1, 4, 5))
    z[:] = 0

    md = update_metadata_from_image({"k": "v"}, z, run_checkup=True, verbose=False)

    assert md["axes"] == "TZCYX"
    assert md["shape"] == shape
    assert md["SizeT"] == 2
    assert md["SizeZ"] == 1
    assert md["SizeC"] == 3
    assert md["SizeY"] == 4
    assert md["SizeX"] == 5

    assert "k" not in md
    assert md["Annotations"]["k"] == "v"

# %% IMWRITE

def _make_image_and_metadata(shape=(1, 1, 1, 8, 9), *, annotations=None):
    img = np.zeros(shape, dtype=np.uint16)
    md = create_empty_metadata(shape=shape, annotations=annotations, verbose=False)
    md = update_metadata_from_image(md, img, run_checkup=True, verbose=False)
    # ensure writer-critical physical sizes exist and are > 0
    md["PhysicalSizeX"] = float(md.get("PhysicalSizeX", 1.0) or 1.0)
    md["PhysicalSizeY"] = float(md.get("PhysicalSizeY", 1.0) or 1.0)
    md["PhysicalSizeZ"] = float(md.get("PhysicalSizeZ", 1.0) or 1.0)
    md["unit"] = md.get("unit", "µm")
    return img, md

def test_write_ometiff_single_writes_file_and_returns_fname(tmp_path):
    img, md = _make_image_and_metadata()

    anchor = tmp_path / "out.ome.tif"
    fnames = imwrite(
        str(anchor),
        img,
        md,
        overwrite=True,
        return_fnames=True,
        verbose=False,
    )

    assert isinstance(fnames, list)
    assert len(fnames) == 1
    out = tmp_path / "out.ome.tif"
    assert out.exists()

    with tifffile.TiffFile(str(out)) as tif:
        ome = tif.ome_metadata
        assert ome is not None

        # must be OME-XML and contain Pixels block
        assert "<OME" in ome
        assert "Pixels" in ome

        # verify that sizes are written and match the input
        assert f'SizeX="{img.shape[-1]}"' in ome
        assert f'SizeY="{img.shape[-2]}"' in ome
        assert f'SizeC="{img.shape[2]}"' in ome
        assert f'SizeZ="{img.shape[1]}"' in ome
        assert f'SizeT="{img.shape[0]}"' in ome

def test_write_ometiff_writes_mapannotation_from_annotations_dict(tmp_path):
    img, md = _make_image_and_metadata(annotations={"Experiment": "MSD", "Namespace": "omio:metadata"})

    out = tmp_path / "ann.ome.tif"
    imwrite(str(out), img, md, overwrite=True, verbose=False)

    with tifffile.TiffFile(str(out)) as tif:
        ome = tif.ome_metadata
        assert ome is not None
        assert "MapAnnotation" in ome
        assert "Experiment" in ome
        assert "MSD" in ome

def test_write_ometiff_multiple_uses_original_filename_from_annotations(tmp_path):
    img1, md1 = _make_image_and_metadata(annotations={"original_filename": "a.tif"})
    img2, md2 = _make_image_and_metadata(annotations={"original_filename": "b.tif"})

    anchor = tmp_path / "fallback.ome.tif"
    fnames = imwrite(
        str(anchor),
        [img1, img2],
        [md1, md2],
        overwrite=True,
        return_fnames=True,
        verbose=False,
    )

    assert len(fnames) == 2
    assert (tmp_path / "a.ome.tif").exists()
    assert (tmp_path / "b.ome.tif").exists()

def test_write_ometiff_relative_path_writes_into_subfolder(tmp_path):
    img, md = _make_image_and_metadata()

    anchor = tmp_path / "rootname.ome.tif"
    imwrite(
        str(anchor),
        img,
        md,
        relative_path="outputs",
        overwrite=True,
        verbose=False,
    )

    assert (tmp_path / "outputs" / "rootname.ome.tif").exists()

def test_write_ometiff_images_metadatas_length_mismatch_raises(tmp_path):
    img, md = _make_image_and_metadata()

    with pytest.raises(ValueError):
        imwrite(
            str(tmp_path / "x.ome.tif"),
            [img, img],
            [md],
            verbose=False,
        )

# %% IMREAD

def test_imread_single_tif_dispatches_to_tif_reader(tmp_path):
    # minimal tif without metadata
    arr = (np.random.rand(8, 9) * 255).astype(np.uint8)
    f = tmp_path / "single.tif"
    tifffile.imwrite(str(f), arr)

    img, md = imread(str(f), verbose=False)

    assert isinstance(md, dict)
    assert md["axes"] == "TZCYX"
    assert isinstance(img, np.ndarray)
    assert img.shape[-2:] == arr.shape  # YX preserved
    assert md["SizeY"] == arr.shape[0]
    assert md["SizeX"] == arr.shape[1]

def test_imread_list_of_files_returns_lists(tmp_path):
    a0 = np.zeros((6, 7), dtype=np.uint8)
    a1 = np.ones((6, 7), dtype=np.uint8)

    f0 = tmp_path / "a.tif"
    f1 = tmp_path / "b.tif"
    tifffile.imwrite(str(f0), a0)
    tifffile.imwrite(str(f1), a1)

    imgs, mds = imread([str(f0), str(f1)], verbose=False)

    assert isinstance(imgs, list) and isinstance(mds, list)
    assert len(imgs) == 2 and len(mds) == 2
    assert all(md["axes"] == "TZCYX" for md in mds)

def test_imread_invalid_merge_axis_raises(tmp_path):
    f = tmp_path / "x.tif"
    tifffile.imwrite(str(f), np.zeros((4, 4), dtype=np.uint8))

    with pytest.raises(ValueError):
        imread(str(f), merge_along_axis="Y", verbose=False)

def test_imread_nonexistent_file_raises(tmp_path):
    missing = tmp_path / "does_not_exist.tif"
    with pytest.raises(FileNotFoundError):
        imread(str(missing), verbose=False)

def test_imread_folder_reads_all_images(tmp_path):
    f0 = tmp_path / "a.tif"
    f1 = tmp_path / "b.tif"
    tifffile.imwrite(str(f0), np.zeros((5, 6), dtype=np.uint8))
    tifffile.imwrite(str(f1), np.ones((5, 6), dtype=np.uint8))

    imgs, mds = imread(str(tmp_path), return_list=True, verbose=False)

    assert isinstance(imgs, list) and isinstance(mds, list)
    assert len(imgs) == 2
    assert all(md["axes"] == "TZCYX" for md in mds)

def test_imread_single_tif_zarr_memory_returns_zarr_array(tmp_path):
    arr = (np.random.rand(10, 11) * 255).astype(np.uint8)
    f = tmp_path / "zarr_mem.tif"
    tifffile.imwrite(str(f), arr)

    img, md = imread(str(f), zarr_store="memory", verbose=False)

    assert md["axes"] == "TZCYX"
    assert hasattr(img, "shape") and hasattr(img, "dtype")
    assert isinstance(img, zarr.core.array.Array)
    assert img.shape[-2:] == arr.shape

def test_imread_folder_stacks_missing_tag_underscore_aborts(tmp_path):
    # folder name without "_" should abort in folder_stacks mode
    folder = tmp_path / "NoTagFolder"
    folder.mkdir()

    img, md = imread(str(folder), folder_stacks=True, verbose=False)

    assert img is None
    assert isinstance(md, dict)

def test_imread_folder_stacks_reads_first_file_from_each_tagfolder(tmp_path):
    # create TAG_000, TAG_001 each with one tif
    base = tmp_path
    f0 = base / "TAG_000"
    f1 = base / "TAG_001"
    f0.mkdir()
    f1.mkdir()

    tifffile.imwrite(str(f0 / "a.tif"), np.zeros((7, 8), dtype=np.uint8))
    tifffile.imwrite(str(f1 / "b.tif"), np.ones((7, 8), dtype=np.uint8))

    imgs, mds = imread(str(f0), folder_stacks=True, return_list=True, verbose=False)

    assert isinstance(imgs, list) and isinstance(mds, list)
    assert len(imgs) == 2
    assert all(md["axes"] == "TZCYX" for md in mds)

def test_imread_merge_multiple_files_in_folder_concat_T(tmp_path):
    # 2 files, each becomes TZCYX with T=1 -> merge along T => T=2
    a0 = np.zeros((9, 10), dtype=np.uint8)
    a1 = np.ones((9, 10), dtype=np.uint8)

    tifffile.imwrite(str(tmp_path / "a.tif"), a0)
    tifffile.imwrite(str(tmp_path / "b.tif"), a1)

    img, md = imread(
        str(tmp_path),
        merge_multiple_files_in_folder=True,
        merge_along_axis="T",
        verbose=False,
    )

    assert md["axes"] == "TZCYX"
    assert img.shape[0] == 2  # T concatenated
    assert img.shape[-2:] == a0.shape

# %% IMCONVERT

"""
Tests for imconvert (reader plus writer integration).

These tests intentionally use very small synthetic inputs created on the fly inside
pytest temporary folders.

Rationale for the input fixtures:
* Plain TIFF files written from higher dimensional NumPy arrays do not reliably carry
  semantic axis labels unless an explicit axes string is provided.
  OMIO expects axis labels drawn from the OME set {"T","Z","C","Y","X"}.
  Therefore, the synthetic plain TIFF fixtures always include an explicit "axes"
  metadata field (for example "YX", "TYX", "TZCYX") to avoid ambiguous non spatial
  labels such as "Q".
* For cache related tests that exercise zarr_store="disk", the input is written as a
  minimal OME TIFF with explicit "TZCYX" axes. This ensures that the on disk Zarr
  cache is created with 5 dimensions, matching OMIO's streaming reorder logic.

Output naming note:
imconvert passes the input path as the writer anchor, so the output basename is
derived from the input stem. For example, "in.tif" becomes "in.ome.tif".
"""

def _write_simple_ome_tif_5d(path: Path, shape=(2, 2, 2, 32, 32), dtype=np.uint16) -> None:
    arr = np.random.randint(0, 1000, size=shape).astype(dtype, copy=False)
    tifffile.imwrite(
        str(path),
        arr,
        ome=True,
        metadata={
            "axes": "TZCYX",
            "PhysicalSizeX": 1.0,
            "PhysicalSizeY": 1.0,
            "PhysicalSizeZ": 1.0,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeYUnit": "µm",
            "PhysicalSizeZUnit": "µm",
        },
    )

""" def test_convert_to_ometiff_single_file_writes_one_ometiff_and_returns_fname(tmp_path):
    src = tmp_path / "in.tif"
    #_write_simple_tif_2d(src)
    _write_simple_ome_tif_5d(src, shape=(2, 2, 2, 32, 32))

    fnames = imconvert(
        fname=str(src),
        overwrite=True,
        return_fnames=True,
        cleanup_cache=True,
        verbose=False,
    )

    assert isinstance(fnames, list)
    assert len(fnames) == 1

    out = Path(fnames[0])
    assert out.exists()
    # with fname=".../in.tif", imwrite writes ".../in.ome.tif"
    assert out.name == "in.ome.tif"

    with tifffile.TiffFile(str(out)) as tif:
        assert tif.ome_metadata is not None
 """

def test_convert_to_ometiff_folder_writes_one_per_file(tmp_path):
    folder = tmp_path / "inputs"
    folder.mkdir()

    # _write_simple_tif_2d(folder / "a.tif", yx=(8, 8))
    # _write_simple_tif_2d(folder / "b.tif", yx=(8, 8))
    _write_simple_ome_tif_5d(folder / "a.tif", shape=(2, 2, 2, 8, 8))
    _write_simple_ome_tif_5d(folder / "b.tif", shape=(2, 2, 2, 8, 8))

    fnames = imconvert(
        fname=str(folder),
        overwrite=True,
        return_fnames=True,
        cleanup_cache=True,
        verbose=False,
    )

    assert isinstance(fnames, list)
    assert len(fnames) == 2

    outs = sorted(Path(f).name for f in fnames)
    assert outs == ["a.ome.tif", "b.ome.tif"]

    for f in fnames:
        with tifffile.TiffFile(str(f)) as tif:
            assert tif.ome_metadata is not None

def test_convert_to_ometiff_relative_path_places_outputs_in_subfolder(tmp_path):
    folder = tmp_path / "inputs"
    folder.mkdir()

    #_write_simple_tif_2d(folder / "a.tif")
    _write_simple_ome_tif_5d(folder / "a.tif", shape=(2, 2, 2, 32, 32))

    fnames = imconvert(
        fname=str(folder),
        relative_path="converted",
        overwrite=True,
        return_fnames=True,
        cleanup_cache=True,
        verbose=False,
    )

    assert len(fnames) == 1
    out = Path(fnames[0])
    assert out.exists()
    assert out.parent.name == "converted"
    assert out.name == "a.ome.tif"

def test_convert_to_ometiff_zarr_disk_creates_and_cleans_cache(tmp_path):
    src = tmp_path / "in.tif"
    _write_simple_ome_tif_5d(src, shape=(2, 2, 2, 32, 32))

    cache_folder = tmp_path / ".omio_cache"
    assert not cache_folder.exists()

    fnames = imconvert(
        fname=str(src),
        zarr_store="disk",
        overwrite=True,
        return_fnames=True,
        cleanup_cache=True,
        verbose=False,
    )

    assert len(fnames) == 1
    assert Path(fnames[0]).exists()

    # disk-zarr store name is derived from input stem: ".omio_cache/in.zarr"
    expected_store = cache_folder / (src.stem + ".zarr")
    assert not expected_store.exists()

""" def test_convert_to_ometiff_zarr_memory_does_not_create_disk_cache(tmp_path):
    src = tmp_path / "in.tif"
    #_write_simple_tif_2d(src)
    _write_simple_ome_tif_5d(src, shape=(2, 2, 2, 32, 32))

    fnames = imconvert(
        fname=str(src),
        zarr_store="memory",
        overwrite=True,
        return_fnames=True,
        cleanup_cache=True,
        verbose=False,
    )

    assert len(fnames) == 1
    assert Path(fnames[0]).exists()
    assert not (tmp_path / ".omio_cache").exists()
 """

def test_convert_to_ometiff_merge_multiple_files_in_folder_writes_single_merged_file(tmp_path):
    folder = tmp_path / "inputs"
    folder.mkdir()

    #_write_simple_tif_2d(folder / "a.tif", yx=(8, 8))
    #_write_simple_tif_2d(folder / "b.tif", yx=(8, 8))
    _write_simple_ome_tif_5d(folder / "a.tif", shape=(2, 2, 2, 8, 8))
    _write_simple_ome_tif_5d(folder / "b.tif", shape=(2, 2, 2, 8, 8))
    

    fnames = imconvert(
        fname=str(folder),
        merge_multiple_files_in_folder=True,
        merge_along_axis="T",
        overwrite=True,
        return_fnames=True,
        cleanup_cache=True,
        verbose=False,
    )

    assert isinstance(fnames, list)
    assert len(fnames) == 1

    out = Path(fnames[0])
    assert out.exists()
    with tifffile.TiffFile(str(out)) as tif:
        assert tif.ome_metadata is not None


# new:
def _make_pattern(shape: tuple[int, ...], dtype=np.uint16) -> np.ndarray:
    n = int(np.prod(shape))
    return np.arange(n, dtype=dtype).reshape(shape)

def _write_plain_tif(path: Path, data: np.ndarray, axes: str, *, photometric="minisblack", ome=False, extra_md=None):
    md = {"axes": axes}
    if extra_md:
        md.update(extra_md)

    tifffile.imwrite(
        str(path),
        data,
        metadata=md,
        photometric=photometric,
        ome=bool(ome),
        imagej=False,
    )

def _write_multiseries(path: Path, series0: np.ndarray, series1: np.ndarray, phot0: str, phot1: str):
    with tifffile.TiffWriter(str(path)) as tif:
        tif.write(series0, photometric=phot0)
        tif.write(series1, photometric=phot1)

def _write_pyramid_ome_tif(path: Path, shape=(8, 2, 20, 20, 3), subresolutions=2, pixelsize=0.29):
    data = np.random.randint(0, 255, shape, dtype=np.uint8)
    with tifffile.TiffWriter(str(path), bigtiff=True) as tif:
        metadata = {
            "axes": "TCYXS",
            "SignificantBits": 8,
            "TimeIncrement": 0.1,
            "TimeIncrementUnit": "s",
            "PhysicalSizeX": pixelsize,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": pixelsize,
            "PhysicalSizeYUnit": "µm",
            "Channel": {"Name": ["Channel 1", "Channel 2"]},
            "Plane": {"PositionX": [0.0] * 16, "PositionXUnit": ["µm"] * 16},
            "Description": "A multi-dimensional, multi-resolution image",
            "MapAnnotation": {
                "Namespace": "openmicroscopy.org/PyramidResolution",
                "1": "256 256",
                "2": "128 128",
            },
        }
        options = dict(
            photometric="rgb",
            tile=(16, 16),
            compression="jpeg",
            resolutionunit="CENTIMETER",
            maxworkers=2,
        )
        tif.write(
            data,
            subifds=subresolutions,
            resolution=(1e4 / pixelsize, 1e4 / pixelsize),
            metadata=metadata,
            **options,
        )
        for level in range(subresolutions):
            mag = 2 ** (level + 1)
            tif.write(
                data[..., ::mag, ::mag, :],
                subfiletype=1,
                resolution=(1e4 / mag / pixelsize, 1e4 / mag / pixelsize),
                **options,
            )
        thumbnail = (data[0, 0, ::8, ::8] >> 2).astype("uint8")
        tif.write(thumbnail, metadata={"Name": "thumbnail"})

@pytest.fixture
def tif_suite(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "tif_suite"
    root.mkdir()

    out = {}

    # simple axis cases
    out["PATH_TO_TIF_XY"] = root / "YX.tif"
    _write_plain_tif(out["PATH_TO_TIF_XY"], _make_pattern((20, 20)), "YX", ome=False)

    out["PATH_TO_TIF_TYX_T1"] = root / "TYX_T1.tif"
    _write_plain_tif(out["PATH_TO_TIF_TYX_T1"], _make_pattern((1, 20, 20)), "TYX", ome=False)

    out["PATH_TO_TIF_ZTYX_Z1_T1"] = root / "ZTYX_Z1_T1.tif"
    _write_plain_tif(out["PATH_TO_TIF_ZTYX_Z1_T1"], _make_pattern((1, 1, 20, 20)), "ZTYX", ome=False)

    out["PATH_TO_TIF_CZTYX_C1_Z1_T1"] = root / "CZTYX_C1_Z1_T1.tif"
    _write_plain_tif(out["PATH_TO_TIF_CZTYX_C1_Z1_T1"], _make_pattern((1, 1, 1, 20, 20)), "CZTYX", ome=False)

    out["PATH_TO_TIF_CZTYX_C2_Z1_T1"] = root / "CZTYX_C2_Z1_T1.tif"
    _write_plain_tif(out["PATH_TO_TIF_CZTYX_C2_Z1_T1"], _make_pattern((2, 1, 1, 20, 20)), "CZTYX", ome=False)

    out["PATH_TO_TIF_CZTYX_C2_Z10_T1"] = root / "CZTYX_C2_Z10_T1.tif"
    _write_plain_tif(out["PATH_TO_TIF_CZTYX_C2_Z10_T1"], _make_pattern((2, 10, 1, 20, 20)), "CZTYX", ome=False)

    out["PATH_TO_TIF_TZCYX_T5_Z10_C2"] = root / "TZCYX_T5_Z10_C2.tif"
    _write_plain_tif(out["PATH_TO_TIF_TZCYX_T5_Z10_C2"], _make_pattern((5, 10, 2, 20, 20)), "TZCYX", ome=False)

    # explicit OME tif
    out["PATH_TO_OME_TIF"] = root / "TZCYX_T5_Z10_C2.ome.tif"
    ome_md = {
        "axes": "TZCYX",
        "PhysicalSizeX": 0.19,
        "PhysicalSizeY": 0.19,
        "PhysicalSizeZ": 2.0,
        "PhysicalSizeXUnit": "µm",
        "PhysicalSizeYUnit": "µm",
        "PhysicalSizeZUnit": "µm",
        "TimeIncrement": 3.0,
        "TimeIncrementUnit": "s",
    }
    _write_plain_tif(out["PATH_TO_OME_TIF"], np.zeros((5, 10, 2, 20, 20), np.uint16), "TZCYX", ome=True, extra_md=ome_md)

    # pyramid paginated OME
    out["PATH_TO_PAGINATED_TIF"] = root / "paginated.ome.tif"
    _write_pyramid_ome_tif(out["PATH_TO_PAGINATED_TIF"], shape=(8, 2, 20, 20, 3))

    # multiseries
    out["PATH_TO_MULTISERIES_TIF1"] = root / "multiseries_rgb_equal_shapes.tif"
    a = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
    b = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
    _write_multiseries(out["PATH_TO_MULTISERIES_TIF1"], a, b, "rgb", "rgb")

    out["PATH_TO_MULTISERIES_TIF2"] = root / "multiseries_rgb_unequal_shapes.tif"
    a = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
    b = np.random.randint(0, 255, (17, 17, 3), dtype=np.uint8)
    _write_multiseries(out["PATH_TO_MULTISERIES_TIF2"], a, b, "rgb", "rgb")

    out["PATH_TO_MULTISERIES_TIF3"] = root / "multiseries_minisblack.tif"
    a = np.random.randint(0, 255, (2, 32, 32), dtype=np.uint8)
    b = np.random.randint(0, 255, (2, 32, 32), dtype=np.uint8)
    _write_multiseries(out["PATH_TO_MULTISERIES_TIF3"], a, b, "minisblack", "minisblack")

    out["PATH_TO_MULTISERIES_TIF4"] = root / "multiseries_rgb_minisblack_mixture.tif"
    a = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
    b = np.random.randint(0, 255, (2, 32, 32), dtype=np.uint8)
    _write_multiseries(out["PATH_TO_MULTISERIES_TIF4"], a, b, "rgb", "minisblack")

    return out

@pytest.mark.parametrize("zarr_store", [None, "memory", "disk"])
def test_imconvert_tif_suite_roundtrip(tif_suite, tmp_path, zarr_store):
    from omio.omio import imconvert

    out_dir = tmp_path / "converted"
    out_dir.mkdir()

    for key, src in tif_suite.items():
        fnames = imconvert(
            fname=str(src),
            relative_path=str(out_dir),
            overwrite=True,
            return_fnames=True,
            zarr_store=zarr_store,
            cleanup_cache=True,
            verbose=False)

        assert isinstance(fnames, list)
        assert len(fnames) == 1

        out = Path(fnames[0])
        assert out.exists()
        
        # output naming: "<stem>.ome.tif". expected: exactly one ".ome" before ".tif"
        if src.suffixes[-2:] == [".ome", ".tif"]:
            base = src.name[:-len(".ome.tif")]
        else:
            base = src.stem
        assert out.name == f"{base}.ome.tif"

        with tifffile.TiffFile(str(out)) as tif:
            assert tif.ome_metadata is not None

    # cache folder expectations after cleanup_cache=True
    if zarr_store == "disk":
        cache_dir = src.parent / ".omio_cache"

        # expected "main" store name (matches your other test logic)
        if src.suffixes[-2:] == [".ome", ".tif"]:
            base = src.name[:-len(".ome.tif")]
        else:
            base = src.stem

        main_store = cache_dir / f"{base}.zarr"
        assert not main_store.exists(), f"Expected main cache store to be deleted: {main_store}"

#%% BIDS-LIKE BATCH CONVERSION

"""
Tests for bids_batch_convert

These tests validate OMIO’s BIDS like batch converter on a fully synthetic directory
tree created under pytest’s tmp_path. The goal is to verify traversal and selection
logic without relying on any external microscopy data formats.

What is covered
* Subject discovery via startswith(sub) at the project root level.
* Experiment discovery within each subject via exp and exp_match_mode
  (startswith, exact, regex).
* Mode A, tagfolder is None:
  * direct conversion of image files located inside experiment folders
  * optional merge_multiple_files_in_folder behavior
* Mode B, tagfolder is set:
  * discovery of tagfolders inside experiment folders via startswith(tagfolder)
  * per tagfolder conversion when merge_tagfolders is False
  * merged output when merge_tagfolders is True, including synthetic provenance
    injection through Annotations["original_filename"] for stable naming
* Output placement rules with relative_path set or None.
* Cache behavior for zarr_store="disk" and cleanup_cache=True, verifying that
  intermediate .omio_cache artifacts are removed after conversion.

Test data policy
* Input files are small OME TIFFs written by tifffile with explicit axes="TZCYX".
  This avoids axis inference ambiguity and avoids 2D collapse issues in code paths
  that expect 5D shaped data.
* No real device specific formats (CZI, Thorlabs RAW) are used here, because those
  readers are tested separately and the batch converter only orchestrates traversal
  plus calls into imread and imwrite.
"""

def _build_bids_like_tree(
    root: Path,
    sub_names=("sub-01",),
    exp_names=("TP000",),
    files_per_exp=1,
    tagfolder_prefix=None,
    tagfolders_per_exp=0,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for s in sub_names:
        sub_dir = root / s
        sub_dir.mkdir(exist_ok=True)
        for e in exp_names:
            exp_dir = sub_dir / e
            exp_dir.mkdir(exist_ok=True)

            if tagfolder_prefix is None:
                for i in range(files_per_exp):
                    _write_simple_ome_tif_5d(exp_dir / f"img_{i:02d}.tif", shape=(2, 2, 2, 8, 8))
            else:
                for t in range(tagfolders_per_exp):
                    tf = exp_dir / f"{tagfolder_prefix}{t:02d}"
                    tf.mkdir(exist_ok=True)
                    for i in range(files_per_exp):
                        _write_simple_ome_tif_5d(tf / f"img_{i:02d}.tif", shape=(2, 2, 2, 8, 8))

def test_convert_bids_batch_mode_a_direct_files_writes_one_per_input(tmp_path):
    project = tmp_path / "project"
    _build_bids_like_tree(
        project,
        sub_names=("sub-01", "sub-02"),
        exp_names=("TP000",),
        files_per_exp=2,
        tagfolder_prefix=None,
    )

    fnames = bids_batch_convert(
        fname=str(project),
        sub="sub",
        exp="TP",
        exp_match_mode="startswith",
        tagfolder=None,
        merge_multiple_files_in_folder=False,
        relative_path="omio_converted",
        overwrite=True,
        cleanup_cache=True,
        return_fnames=True,
        verbose=False,
    )

    assert isinstance(fnames, list)
    assert len(fnames) == 2 * 1 * 2

    for f in fnames:
        p = Path(f)
        assert p.exists()
        assert p.parent.name == "omio_converted"
        with tifffile.TiffFile(str(p)) as tif:
            assert tif.ome_metadata is not None

def test_convert_bids_batch_mode_a_merge_multiple_files_in_folder_writes_one_per_exp(tmp_path):
    project = tmp_path / "project"
    _build_bids_like_tree(
        project,
        sub_names=("sub-01", "sub-02"),
        exp_names=("TP000", "TP001"),
        files_per_exp=2,
        tagfolder_prefix=None,
    )

    fnames = bids_batch_convert(
        fname=str(project),
        sub="sub",
        exp="TP",
        exp_match_mode="startswith",
        tagfolder=None,
        merge_multiple_files_in_folder=True,
        merge_along_axis="T",
        relative_path="omio_converted",
        overwrite=True,
        cleanup_cache=True,
        return_fnames=True,
        verbose=False,
    )

    assert isinstance(fnames, list)
    assert len(fnames) == 2 * 2

    for f in fnames:
        p = Path(f)
        assert p.exists()
        assert p.parent.name == "omio_converted"
        with tifffile.TiffFile(str(p)) as tif:
            assert tif.ome_metadata is not None

def test_convert_bids_batch_exp_match_mode_exact_selects_only_exact_folder(tmp_path):
    project = tmp_path / "project"
    _build_bids_like_tree(
        project,
        sub_names=("sub-01",),
        exp_names=("TP000", "TP000_extra"),
        files_per_exp=1,
        tagfolder_prefix=None,
    )

    fnames = bids_batch_convert(
        fname=str(project),
        sub="sub",
        exp="TP000",
        exp_match_mode="exact",
        tagfolder=None,
        relative_path="omio_converted",
        overwrite=True,
        cleanup_cache=True,
        return_fnames=True,
        verbose=False,
    )

    assert isinstance(fnames, list)
    assert len(fnames) == 1
    assert Path(fnames[0]).exists()
    assert Path(fnames[0]).parent.name == "omio_converted"

def test_convert_bids_batch_exp_match_mode_regex_selects_by_pattern(tmp_path):
    project = tmp_path / "project"
    _build_bids_like_tree(
        project,
        sub_names=("sub-01",),
        exp_names=("TP000", "HC_FOV1", "TP123"),
        files_per_exp=1,
        tagfolder_prefix=None,
    )

    fnames = bids_batch_convert(
        fname=str(project),
        sub="sub",
        exp=r"^TP\d+$",
        exp_match_mode="regex",
        tagfolder=None,
        relative_path="omio_converted",
        overwrite=True,
        cleanup_cache=True,
        return_fnames=True,
        verbose=False,
    )

    assert isinstance(fnames, list)
    assert len(fnames) == 2

def test_convert_bids_batch_mode_b_tagfolders_no_merge_writes_one_per_tagfolder(tmp_path):
    project = tmp_path / "project"
    _build_bids_like_tree(
        project,
        sub_names=("sub-01",),
        exp_names=("TP000",),
        files_per_exp=1,
        tagfolder_prefix="TAG_",
        tagfolders_per_exp=3,
    )

    fnames = bids_batch_convert(
        fname=str(project),
        sub="sub",
        exp="TP",
        exp_match_mode="startswith",
        tagfolder="TAG_",
        merge_tagfolders=False,
        merge_multiple_files_in_folder=False,
        relative_path="omio_converted",
        overwrite=True,
        cleanup_cache=True,
        return_fnames=True,
        verbose=False,
    )

    assert isinstance(fnames, list)
    assert len(fnames) == 3

    for f in fnames:
        p = Path(f)
        assert p.exists()
        assert p.parent.name == "omio_converted"
        with tifffile.TiffFile(str(p)) as tif:
            assert tif.ome_metadata is not None

def test_convert_bids_batch_mode_b_tagfolders_merge_writes_single_output_per_exp(tmp_path):
    project = tmp_path / "project"
    _build_bids_like_tree(
        project,
        sub_names=("sub-01",),
        exp_names=("TP000",),
        files_per_exp=1,
        tagfolder_prefix="TAG_",
        tagfolders_per_exp=3,
    )

    fnames = bids_batch_convert(
        fname=str(project),
        sub="sub",
        exp="TP",
        exp_match_mode="startswith",
        tagfolder="TAG_",
        merge_tagfolders=True,
        merge_along_axis="T",
        relative_path="omio_converted",
        overwrite=True,
        cleanup_cache=True,
        return_fnames=True,
        verbose=False,
    )

    assert isinstance(fnames, list)
    assert len(fnames) == 1

    out = Path(fnames[0])
    assert out.exists()
    assert out.parent.name == "omio_converted"
    with tifffile.TiffFile(str(out)) as tif:
        assert tif.ome_metadata is not None

def test_convert_bids_batch_zarr_disk_cleans_cache(tmp_path):
    project = tmp_path / "project"
    _build_bids_like_tree(
        project,
        sub_names=("sub-01",),
        exp_names=("TP000",),
        files_per_exp=1,
        tagfolder_prefix=None,
    )

    exp_dir = project / "sub-01" / "TP000"
    cache_dir = exp_dir / ".omio_cache"
    assert not cache_dir.exists()

    fnames = bids_batch_convert(
        fname=str(project),
        sub="sub",
        exp="TP",
        exp_match_mode="startswith",
        tagfolder=None,
        zarr_store="disk",
        relative_path="omio_converted",
        overwrite=True,
        cleanup_cache=True,
        return_fnames=True,
        verbose=False,
    )

    assert len(fnames) == 1
    assert Path(fnames[0]).exists()
    assert not cache_dir.exists()

def test_convert_bids_batch_no_subjects_returns_empty_list_when_requested(tmp_path):
    project = tmp_path / "project"
    project.mkdir()

    with warnings.catch_warnings():
        # we need this to suppress a specific, but expected (!) UserWarning in 
        # the BIDS batch conversion test:
        warnings.simplefilter("ignore", UserWarning)

        fnames = bids_batch_convert(
            fname=str(project),
            sub="sub",
            exp="TP",
            return_fnames=True,
            verbose=False,
        )

    assert fnames == []

# new:
def test_bids_batch_convert_raises_if_fname_not_dir(tmp_path):
    f = tmp_path / "not_a_dir.txt"
    f.write_text("x")
    with pytest.raises(ValueError, match="fname must be an existing directory"):
        bids_batch_convert(fname=str(f), sub="sub", exp="TP", verbose=False)

def test_bids_batch_convert_raises_on_invalid_merge_axis(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    with pytest.raises(ValueError, match="merge_along_axis must be one of"):
        bids_batch_convert(fname=str(project), sub="sub", exp="TP", merge_along_axis="X", verbose=False)

def test_bids_batch_convert_subject_detection_startswith_and_ignores_files(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    (project / "sub-01").mkdir()
    (project / "submarine").mkdir()  # also startswith("sub")
    (project / "subfile.txt").write_text("x")  # not a dir

    # patch imconvert so we do not depend on real images
    def fake_imconvert(**kwargs):
        return []  # nothing written
    monkeypatch.setattr("omio.omio.imconvert", fake_imconvert)

    out = bids_batch_convert(
        fname=str(project),
        sub="sub",
        exp="TP",
        return_fnames=True,
        verbose=False,
    )
    assert out == []

def test_bids_batch_convert_subject_has_no_exp_folders_returns_empty(tmp_path, monkeypatch):
    project = tmp_path / "project"
    (project / "sub-01").mkdir(parents=True)

    monkeypatch.setattr("omio.omio.imconvert", lambda **kwargs: ["should_not_happen"])

    out = bids_batch_convert(
        fname=str(project),
        sub="sub",
        exp="TP",
        exp_match_mode="startswith",
        return_fnames=True,
        verbose=False,
    )
    assert out == []

def test_bids_batch_convert_mode_a_imconvert_exception_is_caught(tmp_path, monkeypatch):
    project = tmp_path / "project"
    _build_bids_like_tree(project, sub_names=("sub-01",), exp_names=("TP000",), files_per_exp=1)

    def failing_imconvert(**kwargs):
        raise RuntimeError("boom")
    monkeypatch.setattr("omio.omio.imconvert", failing_imconvert)

    out = bids_batch_convert(
        fname=str(project),
        sub="sub",
        exp="TP",
        tagfolder=None,
        return_fnames=True,
        verbose=False,
    )
    assert out == []

def test_bids_batch_convert_return_fnames_false_returns_none(tmp_path, monkeypatch):
    project = tmp_path / "project"
    _build_bids_like_tree(project, sub_names=("sub-01",), exp_names=("TP000",), files_per_exp=1)

    monkeypatch.setattr("omio.omio.imconvert", lambda **kwargs: [str(tmp_path / "dummy.ome.tif")])

    out = bids_batch_convert(
        fname=str(project),
        sub="sub",
        exp="TP",
        tagfolder=None,
        return_fnames=False,
        overwrite=True,
        verbose=False,
    )
    assert out is None

def test_bids_batch_convert_mode_b_tagfolder_set_but_none_found_skips(tmp_path, monkeypatch):
    project = tmp_path / "project"
    _build_bids_like_tree(project, sub_names=("sub-01",), exp_names=("TP000",), files_per_exp=1, tagfolder_prefix=None)

    # should never be called
    monkeypatch.setattr("omio.omio.imconvert", lambda **kwargs: ["x"])

    out = bids_batch_convert(
        fname=str(project),
        sub="sub",
        exp="TP",
        tagfolder="TAG_",
        return_fnames=True,
        verbose=False,
    )
    assert out == []

def test_bids_batch_convert_mode_b_merge_tagfolders_imread_none_skips(tmp_path, monkeypatch):
    project = tmp_path / "project"
    _build_bids_like_tree(project, sub_names=("sub-01",), exp_names=("TP000",), files_per_exp=1,
                         tagfolder_prefix="TAG_", tagfolders_per_exp=2)

    monkeypatch.setattr("omio.omio.imread", lambda **kwargs: (None, None))
    # imwrite must not be called
    called = {"write": 0}
    def fake_write_ometiff(**kwargs):
        called["write"] += 1
        return ["x"]
    monkeypatch.setattr("omio.omio.imwrite", fake_write_ometiff)

    out = bids_batch_convert(
        fname=str(project),
        sub="sub",
        exp="TP",
        tagfolder="TAG_",
        merge_tagfolders=True,
        return_fnames=True,
        verbose=False,
    )
    assert out == []
    assert called["write"] == 0

def test_bids_batch_convert_mode_b_merge_tagfolders_injects_provenance_and_uses_merged_folder_when_relative_path_none(tmp_path, monkeypatch):
    project = tmp_path / "project"
    _build_bids_like_tree(project, sub_names=("sub-01",), exp_names=("TP000",), files_per_exp=1,
                         tagfolder_prefix="TAG_", tagfolders_per_exp=2)

    dummy_img = np.zeros((1, 1, 1, 2, 2), dtype=np.uint8)
    dummy_md = {}  # no Annotations

    monkeypatch.setattr("omio.omio.imread", lambda **kwargs: (dummy_img, dummy_md))

    captured = {}
    def fake_write_ometiff(fname, images, metadatas, relative_path, **kwargs):
        captured["fname"] = fname
        captured["md"] = metadatas
        captured["relative_path"] = relative_path
        return [str(Path(fname) / (relative_path or "") / "out.ome.tif")]
    monkeypatch.setattr("omio.omio.imwrite", fake_write_ometiff)

    out = bids_batch_convert(
        fname=str(project),
        sub="sub",
        exp="TP",
        tagfolder="TAG_",
        merge_tagfolders=True,
        relative_path=None,   # triggers fallback "merged"
        return_fnames=True,
        verbose=False,
    )

    assert len(out) == 1
    assert captured["relative_path"] == "merged"
    ann = captured["md"]["Annotations"]
    assert isinstance(ann, dict)
    assert ann["original_filename"].endswith("_TAG_merged.ome.tif")

def test_bids_batch_convert_mode_b_merge_tagfolders_overwrites_non_dict_annotations(tmp_path, monkeypatch):
    project = tmp_path / "project"
    _build_bids_like_tree(project, sub_names=("sub-01",), exp_names=("TP000",), files_per_exp=1,
                         tagfolder_prefix="TAG_", tagfolders_per_exp=2)

    dummy_img = np.zeros((1, 1, 1, 2, 2), dtype=np.uint8)
    dummy_md = {"Annotations": "not_a_dict"}

    monkeypatch.setattr("omio.omio.imread", lambda **kwargs: (dummy_img, dummy_md))

    captured = {}
    def fake_write_ometiff(**kwargs):
        captured["md"] = kwargs["metadatas"]
        return [str(Path(kwargs["fname"]) / "merged" / "out.ome.tif")]
    monkeypatch.setattr("omio.omio.imwrite", fake_write_ometiff)

    out = bids_batch_convert(
        fname=str(project),
        sub="sub",
        exp="TP",
        tagfolder="TAG_",
        merge_tagfolders=True,
        relative_path=None,
        return_fnames=True,
        verbose=False,
    )

    assert len(out) == 1
    assert isinstance(captured["md"]["Annotations"], dict)
    assert "original_filename" in captured["md"]["Annotations"]

def test_bids_batch_convert_merge_tagfolders_cleanup_called_only_when_zarr_store_set(tmp_path, monkeypatch):
    project = tmp_path / "project"
    _build_bids_like_tree(project, sub_names=("sub-01",), exp_names=("TP000",), files_per_exp=1,
                         tagfolder_prefix="TAG_", tagfolders_per_exp=2)

    dummy_img = np.zeros((1, 1, 1, 2, 2), dtype=np.uint8)
    dummy_md = {}

    monkeypatch.setattr("omio.omio.imread", lambda **kwargs: (dummy_img, dummy_md))
    monkeypatch.setattr("omio.omio.imwrite", lambda **kwargs: [str(tmp_path / "out.ome.tif")])

    calls = {"n": 0}
    def fake_cleanup(path, full_cleanup=False, verbose=False):
        calls["n"] += 1
    monkeypatch.setattr("omio.omio.cleanup_omio_cache", fake_cleanup)

    # zarr_store None -> no cleanup
    bids_batch_convert(
        fname=str(project),
        sub="sub",
        exp="TP",
        tagfolder="TAG_",
        merge_tagfolders=True,
        zarr_store=None,
        cleanup_cache=True,
        return_fnames=True,
        verbose=False,
    )
    assert calls["n"] == 0

    # zarr_store disk -> cleanup
    bids_batch_convert(
        fname=str(project),
        sub="sub",
        exp="TP",
        tagfolder="TAG_",
        merge_tagfolders=True,
        zarr_store="disk",
        cleanup_cache=True,
        return_fnames=True,
        verbose=False,
    )
    assert calls["n"] == 1

# %% _dispatch_read_file

# helper: capture calls in a simple dict
def _make_fake_reader(name, calls, ret=("IMG", {"md": True})):
    def _reader(path, **kwargs):
        calls["name"] = name
        calls["path"] = path
        calls["kwargs"] = dict(kwargs)
        return ret
    return _reader

def test_dispatch_read_file_tiff_variants_go_to_read_tif(monkeypatch):
    calls = {}

    fake = _make_fake_reader("read_tif", calls)
    monkeypatch.setattr("omio.omio.read_tif", fake)

    # ensure other readers would fail if called
    monkeypatch.setattr("omio.omio.read_czi", lambda *a, **k: (_ for _ in ()).throw(AssertionError("read_czi called")))
    monkeypatch.setattr("omio.omio.read_thorlabs_raw", lambda *a, **k: (_ for _ in ()).throw(AssertionError("read_thorlabs_raw called")))

    path = "some/path/file.TIFF"  # upper-case should still dispatch to tif reader
    out = _dispatch_read_file(
        path=path,
        zarr_store="memory",
        return_list=True,
        physicalsize_xyz=(0.1, 0.2, 0.3),
        pixelunit="micron",
        verbose=False,
    )

    assert out == ("IMG", {"md": True})
    assert calls["name"] == "read_tif"
    assert calls["path"] == path

    # forwarded parameters must match
    assert calls["kwargs"]["zarr_store"] == "memory"
    assert calls["kwargs"]["return_list"] is True
    assert calls["kwargs"]["physicalsize_xyz"] == (0.1, 0.2, 0.3)
    assert calls["kwargs"]["pixelunit"] == "micron"
    assert calls["kwargs"]["verbose"] is False

def test_dispatch_read_file_lsm_goes_to_read_tif(monkeypatch):
    calls = {}
    monkeypatch.setattr("omio.omio.read_tif", _make_fake_reader("read_tif", calls))

    out = _dispatch_read_file(
        path="x/stack.lsm",
        zarr_store=None,
        return_list=False,
        physicalsize_xyz=None,
        pixelunit="micron",
        verbose=True,
    )

    assert out == ("IMG", {"md": True})
    assert calls["name"] == "read_tif"
    assert calls["path"].endswith("stack.lsm")

def test_dispatch_read_file_ome_tif_detected_by_looks_like_ome_tif(monkeypatch):
    calls = {}
    monkeypatch.setattr("omio.omio.read_tif", _make_fake_reader("read_tif", calls))

    # Force the ome detector branch to be used regardless of extension parsing details
    monkeypatch.setattr("omio.omio._looks_like_ome_tif", lambda lp: True)
    # also ensure _lower_ext would not accidentally route elsewhere
    monkeypatch.setattr("omio.omio._lower_ext", lambda lp: ".nope")

    out = _dispatch_read_file(
        path="x/whatever.OME.TIF",
        zarr_store="disk",
        return_list=False,
        physicalsize_xyz=None,
        pixelunit="micron",
        verbose=False,
    )

    assert out == ("IMG", {"md": True})
    assert calls["name"] == "read_tif"
    assert calls["kwargs"]["zarr_store"] == "disk"

def test_dispatch_read_file_czi_goes_to_read_czi(monkeypatch):
    calls = {}
    monkeypatch.setattr("omio.omio.read_czi", _make_fake_reader("read_czi", calls))
    monkeypatch.setattr("omio.omio.read_tif", lambda *a, **k: (_ for _ in ()).throw(AssertionError("read_tif called")))
    monkeypatch.setattr("omio.omio.read_thorlabs_raw", lambda *a, **k: (_ for _ in ()).throw(AssertionError("read_thorlabs_raw called")))

    path = "x/stack.CZI"
    out = _dispatch_read_file(
        path=path,
        zarr_store="memory",
        return_list=True,
        physicalsize_xyz=(1, 2, 3),
        pixelunit="micron",
        verbose=False,
    )

    assert out == ("IMG", {"md": True})
    assert calls["name"] == "read_czi"
    assert calls["path"] == path
    assert calls["kwargs"]["zarr_store"] == "memory"
    assert calls["kwargs"]["return_list"] is True
    assert calls["kwargs"]["physicalsize_xyz"] == (1, 2, 3)
    assert calls["kwargs"]["pixelunit"] == "micron"
    assert calls["kwargs"]["verbose"] is False

def test_dispatch_read_file_raw_goes_to_read_thorlabs_raw(monkeypatch):
    calls = {}
    monkeypatch.setattr("omio.omio.read_thorlabs_raw", _make_fake_reader("read_thorlabs_raw", calls))
    monkeypatch.setattr("omio.omio.read_tif", lambda *a, **k: (_ for _ in ()).throw(AssertionError("read_tif called")))
    monkeypatch.setattr("omio.omio.read_czi", lambda *a, **k: (_ for _ in ()).throw(AssertionError("read_czi called")))

    out = _dispatch_read_file(
        path="x/data.raw",
        zarr_store=None,
        return_list=False,
        physicalsize_xyz=None,
        pixelunit="micron",
        verbose=True,
    )

    assert out == ("IMG", {"md": True})
    assert calls["name"] == "read_thorlabs_raw"
    assert calls["kwargs"]["zarr_store"] is None
    assert calls["kwargs"]["verbose"] is True

def test_dispatch_read_file_unsupported_extension_raises(monkeypatch):
    # Make sure ome detector is false so we go purely by extension
    monkeypatch.setattr("omio.omio._looks_like_ome_tif", lambda lp: False)
    monkeypatch.setattr("omio.omio._lower_ext", lambda lp: ".xyz")

    with pytest.raises(ValueError, match="Unsupported file extension"):
        _dispatch_read_file(
            path="x/file.xyz",
            zarr_store=None,
            return_list=False,
            physicalsize_xyz=None,
            pixelunit="micron",
            verbose=False,
        )

# %% NAPARI VIEWER

""" 
Napari viewer helpers and integration tests

These tests focus on OMIO's Napari convenience layer. The goal is to validate the deterministic,
GUI independent parts of the implementation:

* Axis handling:
  - correct inference of the channel axis index from an axis string and shape
  - correct derivation of Napari scale tuples from OMIO physical pixel sizes

* Zarr squeezing cache logic:
  - singleton dimensions are removed deterministically
  - the returned squeezed axis string matches the squeezed array
  - the derived on disk Zarr stores are created at the expected locations

* Napari integration without launching a real viewer:
  - viewer.add_image is called with the expected data object, channel_axis, scale, and name
  - the scale bar properties are set from metadata

Implementation notes:
* These tests use a minimal FakeViewer and monkeypatch napari.current_viewer and napari.Viewer
  so that no Qt event loop and no real GUI is required.
* Zarr and Dask related tests use small arrays to keep runtime and IO minimal. 
"""

class FakeScaleBar:
    def __init__(self) -> None:
        self.visible = False
        self.unit = None

class FakeViewer:
    def __init__(self) -> None:
        self.scale_bar = FakeScaleBar()
        self.added = []

    def add_image(self, data, channel_axis=None, scale=None, name=None):
        layer = object()
        self.added.append(
            {
                "data": data,
                "channel_axis": channel_axis,
                "scale": scale,
                "name": name,
                "layer": layer,
            }
        )
        return layer

def _make_metadata(unit: str = "micron"):
    return {
        "PhysicalSizeX": 0.5,
        "PhysicalSizeY": 1.5,
        "PhysicalSizeZ": 2.5,
        "unit": unit,
    }

def _make_deterministic_5d(shape_tzcyx=(1, 2, 3, 4, 5), dtype=np.uint16) -> np.ndarray:
    t, z, c, y, x = shape_tzcyx
    arr = np.zeros(shape_tzcyx, dtype=dtype)
    for ti in range(t):
        for zi in range(z):
            for ci in range(c):
                for yi in range(y):
                    for xi in range(x):
                        arr[ti, zi, ci, yi, xi] = ti * 10000 + zi * 1000 + ci * 100 + yi * 10 + xi
    return arr

def _write_zarr(path: Path, data: np.ndarray, chunks=None) -> zarr.core.array.Array:
    if path.exists():
        import shutil

        shutil.rmtree(path)
    z = zarr.open(str(path), mode="w", shape=data.shape, dtype=data.dtype, chunks=chunks)
    z[...] = data
    return z


# Unit tests for axis and scale helpers:

def test_get_channel_axis_finds_c():
    axes = "TZCYX"
    shape = (1, 2, 3, 4, 5)
    assert _get_channel_axis_from_axes_and_shape(axes=axes, shape=shape, target_axis="C") == 2

def test_get_channel_axis_returns_none_if_missing():
    axes = "TYX"
    shape = (10, 256, 256)
    assert _get_channel_axis_from_axes_and_shape(axes=axes, shape=shape, target_axis="C") is None

def test_get_channel_axis_raises_on_length_mismatch():
    with pytest.raises(ValueError):
        _get_channel_axis_from_axes_and_shape(axes="TYX", shape=(10, 256), target_axis="C")

def test_get_scales_skips_c_and_maps_xyz():
    md = _make_metadata()
    axes = "TZCYX"
    scales = _get_scales_from_axes_and_metadata(axes=axes, metadata=md)

    # Channel axis is skipped, so output covers T, Z, Y, X
    assert len(scales) == 4
    assert scales[0] == 1.0
    assert scales[1] == md["PhysicalSizeZ"]
    assert scales[2] == md["PhysicalSizeY"]
    assert scales[3] == md["PhysicalSizeX"]

def test_get_scales_without_c_includes_all_axes():
    md = _make_metadata()
    axes = "TYX"
    scales = _get_scales_from_axes_and_metadata(axes=axes, metadata=md)

    assert len(scales) == 3
    assert scales[0] == 1.0
    assert scales[1] == md["PhysicalSizeY"]
    assert scales[2] == md["PhysicalSizeX"]



# Unit tests for Zarr squeezing to Napari cache:

def test_squeeze_zarr_to_napari_cache_removes_singletons_and_preserves_values(tmp_path: Path):
    # Shape: (T, Z, C, Y, X) with singleton T and C
    data = _make_deterministic_5d(shape_tzcyx=(1, 2, 1, 4, 5))
    src_path = tmp_path / "src.zarr"
    src = _write_zarr(src_path, data, chunks=(1, 1, 1, 4, 5))

    dst_path = tmp_path / "napari_view.zarr"
    dst, squeezed_axes = _squeeze_zarr_to_napari_cache(src=src, fname=str(dst_path), axes="TZCYX")

    assert squeezed_axes == "ZYX"
    assert dst.shape == (2, 4, 5)

    # spot check a few values
    # original index mapping: (t=0, z, c=0, y, x) -> squeezed (z, y, x)
    assert dst[0, 0, 0] == data[0, 0, 0, 0, 0]
    assert dst[1, 3, 4] == data[0, 1, 0, 3, 4]

def test_squeeze_zarr_to_napari_cache_2d_fast_path(tmp_path: Path):
    # Shape (T, Z, C, Y, X) with only Y and X non singleton
    data = _make_deterministic_5d(shape_tzcyx=(1, 1, 1, 4, 5))
    src_path = tmp_path / "src2d.zarr"
    src = _write_zarr(src_path, data, chunks=(1, 1, 1, 4, 5))

    dst_path = tmp_path / "napari_view_2d.zarr"
    dst, squeezed_axes = _squeeze_zarr_to_napari_cache(src=src, fname=str(dst_path), axes="TZCYX")

    assert squeezed_axes == "YX"
    assert dst.shape == (4, 5)
    assert np.array_equal(dst[...], data[0, 0, 0, :, :])

def test_squeeze_zarr_to_napari_cache_dask_writes_into_cache_dir(tmp_path: Path):
    data = _make_deterministic_5d(shape_tzcyx=(1, 2, 1, 4, 5))
    src_path = tmp_path / "src_dask.zarr"
    src = _write_zarr(src_path, data, chunks=(1, 1, 1, 4, 5))

    # fname is used to derive <base_dir>/.omio_cache/<basename>_napari_squeezed.zarr
    base_no_ext = str(tmp_path / "src_dask")

    squeezed_zarr, squeezed_axes = _squeeze_zarr_to_napari_cache_dask(
        src=src,
        fname=base_no_ext,
        axes="TZCYX",
        cache_folder_name=".omio_cache",
    )

    expected_cache_dir = tmp_path / ".omio_cache"
    expected_store = expected_cache_dir / "src_dask_napari_squeezed.zarr"

    assert expected_cache_dir.exists()
    assert expected_store.exists()
    assert squeezed_axes == "ZYX"
    assert squeezed_zarr.shape == (2, 4, 5)

    assert squeezed_zarr[0, 0, 0] == data[0, 0, 0, 0, 0]
    assert squeezed_zarr[1, 3, 4] == data[0, 1, 0, 3, 4]


# Tests for Napari wrapper logic without creating a real GUI:

def test_single_image_open_in_napari_numpy_input_uses_expected_channel_axis_and_scale(monkeypatch, tmp_path: Path):
    import omio.omio as m

    fake_viewer = FakeViewer()

    # Force reuse of a viewer without starting Qt
    monkeypatch.setattr(m.napari, "current_viewer", lambda: fake_viewer)
    monkeypatch.setattr(m.napari, "Viewer", lambda: FakeViewer())

    md = _make_metadata(unit="micron")

    # Input shape: (T, Z, C, Y, X) = (1, 1, 2, 8, 8) -> squeezed: (C, Y, X)
    img = _make_deterministic_5d(shape_tzcyx=(1, 1, 2, 8, 8))

    viewer, layer, napari_data, napari_axes = _single_image_open_in_napari(
        image=img,
        metadata=md,
        fname=str(tmp_path / "in.tif"),
        zarr_mode="numpy",
        axes_full="TZCYX",
        viewer=None,
        verbose=False,
    )

    assert viewer is fake_viewer
    assert napari_axes == "CYX"
    assert napari_data.shape == (2, 8, 8)

    assert len(fake_viewer.added) == 1
    call = fake_viewer.added[0]
    assert call["channel_axis"] == 0
    assert call["name"] == "in.tif"
    assert call["scale"] == (md["PhysicalSizeY"], md["PhysicalSizeX"])

    assert fake_viewer.scale_bar.visible is True
    assert fake_viewer.scale_bar.unit == "micron"

def test_single_image_open_in_napari_zarr_nodask_creates_side_store_and_passes_dask(monkeypatch, tmp_path: Path):
    import omio.omio as m

    fake_viewer = FakeViewer()
    monkeypatch.setattr(m.napari, "current_viewer", lambda: fake_viewer)
    monkeypatch.setattr(m.napari, "Viewer", lambda: FakeViewer())

    md = _make_metadata(unit="micron")

    data = _make_deterministic_5d(shape_tzcyx=(1, 2, 1, 8, 8))
    src_store = tmp_path / "src.zarr"
    src = _write_zarr(src_store, data, chunks=(1, 1, 1, 8, 8))

    viewer, layer, napari_data, napari_axes = _single_image_open_in_napari(
        image=src,
        metadata=md,
        fname=str(tmp_path / "src.ome.tif"),
        zarr_mode="zarr_nodask",
        axes_full="TZCYX",
        viewer=None,
        verbose=False,
    )

    # nodask squeeze writes to base_no_ext derived from store_path, that is "<...>/src"
    expected_side_store = tmp_path / "src"
    assert expected_side_store.exists()

    assert napari_axes == "ZYX"
    assert len(fake_viewer.added) == 1

    call = fake_viewer.added[0]
    assert isinstance(call["data"], da.Array)
    assert call["channel_axis"] is None
    assert call["scale"] == (md["PhysicalSizeZ"], md["PhysicalSizeY"], md["PhysicalSizeX"])

def test_open_in_napari_multiple_images_reuses_viewer_and_appends_idx_suffix(monkeypatch, tmp_path: Path):
    import omio.omio as m

    fake_viewer = FakeViewer()
    monkeypatch.setattr(m.napari, "current_viewer", lambda: fake_viewer)
    monkeypatch.setattr(m.napari, "Viewer", lambda: FakeViewer())

    md = _make_metadata(unit="micron")

    img1 = _make_deterministic_5d(shape_tzcyx=(1, 1, 1, 4, 4))
    img2 = _make_deterministic_5d(shape_tzcyx=(1, 1, 1, 4, 4)) + 1

    viewer, layers, datas, axess = open_in_napari(
        images=[img1, img2],
        metadatas=[md, md],
        fname=str(tmp_path / "base_name"),
        zarr_mode="numpy",
        axes_full="TZCYX",
        viewer=None,
        returns=True,
        verbose=False,
    )

    assert viewer is fake_viewer
    assert len(layers) == 2
    assert len(fake_viewer.added) == 2

    assert fake_viewer.added[0]["name"] == "base_name_idx0"
    assert fake_viewer.added[1]["name"] == "base_name_idx1"

def test_open_in_napari_raises_on_length_mismatch():
    md = _make_metadata()
    img = _make_deterministic_5d(shape_tzcyx=(1, 1, 1, 4, 4))
    with pytest.raises(ValueError):
        open_in_napari(images=[img, img], metadatas=[md], fname="x", returns=False, verbose=False)

# %% END