#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate small dummy TIFF test files for OMIO.

Creates:
* plain TIFFs with explicit axes metadata (tifffile metadata['axes'])
* one OME-TIFF (C=2, Z=10, T=5, Y=20, X=20) with physical sizes and time increment
* two paginated multi-series TIFF examples (minisblack and rgb)

All numeric stacks are uint16. Spatial calibration:
* PhysicalSizeX = PhysicalSizeY = 0.19 µm
* PhysicalSizeZ = 2.0 µm
* TIFF resolution is set as (1/PhysicalSizeY, 1/PhysicalSizeX)

Adjust `OUT_DIR` as needed.
"""
# %% IMPORTS
import os
import numpy as np
import tifffile
# %% FUNCTIONS
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def write_tif(
    path: str,
    data: np.ndarray,
    axes: str,
    *,
    compression_level: int = 3,
    physical_xy: float = 0.19,
    bigtiff: bool = False,
    ome: bool = False,
    extra_metadata: dict | None = None,
    photometric: str = "minisblack",
) -> None:
    """
    Write a TIFF (or OME-TIFF if ome=True) with axes metadata and XY resolution.
    """
    md = {"axes": axes}
    if extra_metadata:
        md.update(extra_metadata)

    tifffile.imwrite(
        path,
        data,
        compression="zlib",
        compressionargs={"level": int(compression_level)},
        resolution=(1.0 / float(physical_xy), 1.0 / float(physical_xy)),
        metadata=md,
        photometric=photometric,
        imagej=False,
        bigtiff=bool(bigtiff),
        ome=bool(ome),
    )


def make_pattern(shape: tuple[int, ...], dtype=np.uint16) -> np.ndarray:
    """
    Deterministic pattern (~~ramp~~ random) to make debugging easier than pure zeros.
    """
    #n = int(np.prod(shape))
    #arr = np.arange(n, dtype=dtype).reshape(shape)
    arr = np.random.randint(0, 255, shape, dtype=dtype)
    return arr


def main() -> None:
    # Change this to your desired output directory
    OUT_DIR = "tif_dummy_data"
    # prepend path to folder of this script:
    OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUT_DIR)
    ensure_dir(OUT_DIR)

    physical_xy = 0.19  # µm
    physical_z = 2.0    # µm
    time_increment = 3.0
    time_unit = "s"

    # ---------------------------------------------------------------------
    # Requested stacks (Y=20, X=20 base)
    # ---------------------------------------------------------------------
    cases = [
        ("YX",               (20, 20),              "YX"),
        ("TYX_T1",           (1, 20, 20),           "TYX"),
        ("ZTYX_Z1_T1",       (1, 1, 20, 20),        "ZTYX"),
        ("CZTYX_C1_Z1_T1",   (1, 1, 1, 20, 20),     "CZTYX"),
        ("CZTYX_C2_Z1_T1",   (2, 1, 1, 20, 20),     "CZTYX"),
        ("CZTYX_C2_Z10_T1",  (2, 10, 1, 20, 20),    "CZTYX"),
        ("TZCYX_T5_Z10_C2",  (5, 10, 2, 20, 20),    "TZCYX"),
    ]

    for name, shape, axes in cases:
        data = make_pattern(shape, dtype=np.uint16)
        out_path = os.path.join(OUT_DIR, f"tif/{name}.tif")
        ensure_dir(os.path.dirname(out_path))
        write_tif(
            out_path,
            data,
            axes,
            physical_xy=physical_xy,
            compression_level=3,
            photometric="minisblack",
            ome=False)
        print(f"Wrote TIFF: {out_path}  shape={shape}  axes={axes}")

    # ---------------------------------------------------------------------
    # Also write an OME-TIFF with metadata:
    # ---------------------------------------------------------------------
    ome_shape = (5, 10, 2, 20, 20)
    dd = np.random.randint(0, 255, ome_shape).astype(np.uint8)

    ome_out = os.path.join(OUT_DIR, "ome_tif/TZCYX_T5_Z10_C2.ome.tif")
    ensure_dir(os.path.dirname(ome_out))
    
    ome_md = {
        "axes": "TZCYX",
        "PhysicalSizeX": float(physical_xy),
        "PhysicalSizeY": float(physical_xy),
        "PhysicalSizeZ": float(physical_z),
        "PhysicalSizeXUnit": "µm",
        "PhysicalSizeYUnit": "µm",
        "PhysicalSizeZUnit": "µm",
        "TimeIncrement": float(time_increment),
        "TimeIncrementUnit": str(time_unit),
    }
    write_tif(
        ome_out,
        dd,
        "TZCYX",
        physical_xy=physical_xy,
        compression_level=3,
        photometric="minisblack",
        ome=True,
        extra_metadata=ome_md)
    print(f"Wrote OME-TIFF: {ome_out}  shape={ome_shape}  axes=TZCYX")

    # ---------------------------------------------------------------------
    # Paginated / multi-series TIFFs
    # ---------------------------------------------------------------------
    series0 = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
    series1 = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
    paged_rgb_path = os.path.join(OUT_DIR, "multiseries_tif/multiseries_rgb_with_equal_shapes.tif")
    ensure_dir(os.path.dirname(paged_rgb_path))
    with tifffile.TiffWriter(paged_rgb_path) as tif:
        tif.write(series0, photometric="rgb")
        tif.write(series1, photometric="rgb")
    print(f"Wrote paginated rgb TIFF: {paged_rgb_path}")
    """ 
    if the image-slice shapes are identical, FIJI's Bio-Formats reader
    seems to interpret both pages as one multi-page RGB image, not as two
    separate series. Hence, we create some more examples with differing shapes.
    """
        
    series0 = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
    series1 = np.random.randint(0, 255, (17, 17, 3), dtype=np.uint8)
    paged_rgb_path = os.path.join(OUT_DIR, "multiseries_tif/multiseries_rgb_with_unequal_series.tif")
    ensure_dir(os.path.dirname(paged_rgb_path))
    with tifffile.TiffWriter(paged_rgb_path) as tif:
        tif.write(series0, photometric="rgb")
        tif.write(series1, photometric="rgb")
    print(f"Wrote paginated rgb TIFF: {paged_rgb_path}")
    
    series0 = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
    series1 = np.random.randint(0, 255, (2, 32, 32), dtype=np.uint8)
    paged_rgb_path = os.path.join(OUT_DIR, "multiseries_tif/multiseries_rgb_minisblack_mixture.tif")
    ensure_dir(os.path.dirname(paged_rgb_path))
    with tifffile.TiffWriter(paged_rgb_path) as tif:
        tif.write(series0, photometric="rgb")
        tif.write(series1, photometric="minisblack")
    print(f"Wrote paginated rgb TIFF: {paged_rgb_path}")
    
    series0 = np.random.randint(0, 255, (2, 32, 32), dtype=np.uint8)
    series1 = np.random.randint(0, 255, (2, 32, 32), dtype=np.uint8)
    paged_rgb_path = os.path.join(OUT_DIR, "multiseries_tif/multiseries_minisblack.tif")
    ensure_dir(os.path.dirname(paged_rgb_path))
    with tifffile.TiffWriter(paged_rgb_path) as tif:
        tif.write(series0, photometric="minisblack")
        tif.write(series1, photometric="minisblack")
    print(f"Wrote paginated rgb TIFF: {paged_rgb_path}")
    
    data = np.random.randint(0, 255, (8, 2, 20, 20, 3), 'uint16')
    subresolutions = 2
    pixelsize = 0.29  # micrometer
    paged_rgb_path = os.path.join(OUT_DIR, "paginated_tif/paginated.ome.tif")
    ensure_dir(os.path.dirname(paged_rgb_path))
    with tifffile.TiffWriter(paged_rgb_path, bigtiff=True) as tif:
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
            compression='zlib',
            resolutionunit='CENTIMETER',
            maxworkers=2,
        )
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

    print("\nDone.")

# %% MAIN
if __name__ == "__main__":
    main()