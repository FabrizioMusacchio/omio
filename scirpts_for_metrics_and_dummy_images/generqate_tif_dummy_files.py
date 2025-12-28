"""
This script generates small dummy TIFF files for testing OMIO.

Adjust `OUT_DIR` as needed.

author: Fabrizio Musacchio
date: December 2025
"""
# %% IMPORTS
import os
import numpy as np
import tifffile
# %% FUNCTIONS
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def write_tif(path: str,
              data: np.ndarray,
              axes: str,
              *,
              imagej: bool = False,
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
        imagej=bool(imagej),
        bigtiff=bool(bigtiff),
        ome=bool(ome),
    )

# old function left for reference:
def make_pattern(shape: tuple[int, ...], dtype=np.uint16) -> np.ndarray:
    """
    Deterministic pattern (~~ramp~~ random) to make debugging easier than pure zeros.
    """
    #n = int(np.prod(shape))
    #arr = np.arange(n, dtype=dtype).reshape(shape)
    arr = np.random.randint(0, 255, shape, dtype=dtype)
    return arr

# We create a minimal 5x7 bitmap font for needed chars: digits, '#', space, 
# and T Z C Y X. Each glyph: 7 rows, 5 columns, encoded as strings of '0'/'1':
_FONT_5x7 = {
    " ": [
        "00000","00000","00000","00000","00000","00000","00000",
    ],
    "#": [
        "01010","11111","01010","01010","11111","01010","01010",
    ],
    "0": [
        "01110","10001","10011","10101","11001","10001","01110",
    ],
    "1": [
        "00100","01100","00100","00100","00100","00100","01110",
    ],
    "2": [
        "01110","10001","00001","00010","00100","01000","11111",
    ],
    "3": [
        "11110","00001","00001","01110","00001","00001","11110",
    ],
    "4": [
        "00010","00110","01010","10010","11111","00010","00010",
    ],
    "5": [
        "11111","10000","10000","11110","00001","00001","11110",
    ],
    "6": [
        "00110","01000","10000","11110","10001","10001","01110",
    ],
    "7": [
        "11111","00001","00010","00100","01000","01000","01000",
    ],
    "8": [
        "01110","10001","10001","01110","10001","10001","01110",
    ],
    "9": [
        "01110","10001","10001","01111","00001","00010","01100",
    ],
    "T": [
        "11111","00100","00100","00100","00100","00100","00100",
    ],
    "Z": [
        "11111","00001","00010","00100","01000","10000","11111",
    ],
    "C": [
        "01110","10001","10000","10000","10000","10001","01110",
    ],
    "Y": [
        "10001","10001","01010","00100","00100","00100","00100",
    ],
    "X": [
        "10001","01010","00100","00100","00100","01010","10001",
    ],
}

def _dtype_max_value(dtype: np.dtype) -> int:
    dt = np.dtype(dtype)
    if np.issubdtype(dt, np.integer):
        return int(np.iinfo(dt).max)
    # float: use 1.0 as "bright" by convention
    return 1

def _draw_text_5x7(
    img2d_or_rgb: np.ndarray,
    text: str,
    *,
    top: int = 1,
    left: int = 1,
    scale: int = 1,
    value: int | None = None,
) -> None:
    """
    Draw 5x7 bitmap text into:
      - 2D image: (Y,X)
      - RGB image: (Y,X,3) -> writes into all 3 channels
    In-place.
    """
    if value is None:
        value = _dtype_max_value(img2d_or_rgb.dtype)

    is_rgb = (img2d_or_rgb.ndim == 3 and img2d_or_rgb.shape[-1] == 3)

    y = int(top)
    x = int(left)

    for ch in text:
        glyph = _FONT_5x7.get(ch, _FONT_5x7[" "])
        # glyph is 7 rows of 5 bits
        for gy, row in enumerate(glyph):
            for gx, bit in enumerate(row):
                if bit == "1":
                    yy0 = y + gy * scale
                    xx0 = x + gx * scale
                    yy1 = yy0 + scale
                    xx1 = xx0 + scale
                    if is_rgb:
                        img2d_or_rgb[yy0:yy1, xx0:xx1, :] = value
                    else:
                        img2d_or_rgb[yy0:yy1, xx0:xx1] = value
        # advance cursor: 5 columns + 1 column spacing
        x += (5 + 1) * scale

def _draw_two_line_label(
    img2d_or_rgb: np.ndarray,
    *,
    t: int,
    z: int,
    c: int,
    y: int,
    x: int,
    top: int = 1,
    left: int = 1,
    scale: int = 1,
    line_spacing: int = 1,
) -> None:
    """
    Draws:
      line 1: T#t Z#z C#c
      line 2: Y#y X#x
    """
    line1 = f"T#{t} Z#{z} C#{c}"
    line2 = f"Y#{y} X#{x}"

    _draw_text_5x7(
        img2d_or_rgb,
        line1,
        top=top,
        left=left,
        scale=scale,
    )

    line_height = 7 * scale + line_spacing

    _draw_text_5x7(
        img2d_or_rgb,
        line2,
        top=top + line_height,
        left=left,
        scale=scale,
    )

def _make_zeros_with_slice_labels(shape: tuple[int, ...], axes: str, dtype=np.uint16) -> np.ndarray:
    """
    Create zeros array with given shape/dtype and annotate every YX-plane with:
    T#<t> Z#<z> C#<c> Y#<Y> X#<X>
    Missing axes (T/Z/C) default to 0.
    """
    arr = np.zeros(shape, dtype=dtype)

    axes = str(axes)
    axpos = {a: i for i, a in enumerate(axes)}
    if "Y" not in axpos or "X" not in axpos:
        raise ValueError(f"axes must contain Y and X, got {axes!r}")

    Y = shape[axpos["Y"]]
    X = shape[axpos["X"]]

    # Determine iteration axes: all axes except Y, X, and optional sample axis S.
    iter_axes = [a for a in axes if a not in ("Y", "X", "S")]

    # Build ranges for each iter axis
    ranges = []
    for a in iter_axes:
        ranges.append(range(shape[axpos[a]]))

    # Cartesian product over iter axes (manual to avoid itertools if you prefer)
    # We'll use itertools for clarity:
    import itertools
    for idxs in itertools.product(*ranges) if ranges else [()]:
        # Build slicer
        sl = [slice(None)] * len(shape)
        # set iter axes indices
        for a, v in zip(iter_axes, idxs):
            sl[axpos[a]] = int(v)

        # Extract the plane containing Y,X (+ possibly RGB samples S)
        plane = arr[tuple(sl)]

        # Determine T/Z/C indices if present, else 0
        t = int(sl[axpos["T"]]) if "T" in axpos and isinstance(sl[axpos["T"]], (int, np.integer)) else 0
        z = int(sl[axpos["Z"]]) if "Z" in axpos and isinstance(sl[axpos["Z"]], (int, np.integer)) else 0
        c = int(sl[axpos["C"]]) if "C" in axpos and isinstance(sl[axpos["C"]], (int, np.integer)) else 0

        #text = f"T#{t} Z#{z} C#{c} Y#{Y} X#{X}"
        #_draw_text_5x7(plane, text, top=1, left=1, scale=1, value=_dtype_max_value(arr.dtype))
        
        _draw_two_line_label(plane,
                            t=t,
                            z=z,
                            c=c,
                            y=Y,
                            x=X,
                            top=1,
                            left=1,
                            scale=1,
                        )

    return arr

def main() -> None:
    # change this to desired output directory:
    OUT_DIR = "tif_dummy_data"
    # prepend path to folder of this script:
    OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUT_DIR)
    ensure_dir(OUT_DIR)

    physical_xy = 0.19  # µm
    physical_z = 2.0    # µm
    time_increment = 3.0
    time_unit = "s"

    # --------------------------------------------------------------------------------
    # sample stacks to create:
    base_resolution_X = 100
    base_resolution_Y = 20
    
    cases = [
        ("YX",               (base_resolution_Y, base_resolution_X),              "YX"),
        ("TYX_T1",           (1, base_resolution_Y, base_resolution_X),           "TYX"),
        ("ZTYX_Z1_T1",       (1, 1, base_resolution_Y, base_resolution_X),        "ZTYX"),
        ("CZTYX_C1_Z1_T1",   (1, 1, 1, base_resolution_Y, base_resolution_X),     "CZTYX"),
        ("CZTYX_C2_Z1_T1",   (2, 1, 1, base_resolution_Y, base_resolution_X),     "CZTYX"),
        ("CZTYX_C2_Z10_T1",  (2, 10, 1, base_resolution_Y, base_resolution_X),    "CZTYX"),
        ("TZCYX_T5_Z10_C2",  (5, 10, 2, base_resolution_Y, base_resolution_X),    "TZCYX")]

    for name, shape, axes in cases:
        #data = make_pattern(shape, dtype=np.uint16)
        data = _make_zeros_with_slice_labels(shape, axes, dtype=np.uint16)
        out_path = os.path.join(OUT_DIR, f"tif_single_files/{name}.tif")
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
    
    # write a README.md into the output folder describing the content:
    readme_path = os.path.join(OUT_DIR, "tif_single_files/README.md")
    with open(readme_path, "w") as f:
        f.write("# Dummy TIFF files with slice labels\n\n")
        f.write("This folder contains dummy TIFF files generated for testing OMIO.\n\n")
        f.write("Each YX plane in the multi-dimensional images is annotated with text indicating the indices of the T, Z, C, Y, and X axes.\n\n")
        f.write("The following files were generated:\n\n")
        for name, shape, axes in cases:
            f.write(f"- `tif/{name}.tif`: shape={shape}, axes={axes}\n")
    
    # --------------------------------------------------------------------------------
    # write an ImageJ TIFF version of the above:
        
    cases_ImageJ = [
        ("YXS",               (base_resolution_Y, base_resolution_X, 3),              "YXS"),
        ("TYXS_T1",           (1, base_resolution_Y, base_resolution_X, 3),           "TYXS"),
        ("TZYXS_Z1_T1",       (1, 1, base_resolution_Y, base_resolution_X, 3),        "TZYXS"),
        ("TZCYXS_C1_Z1_T1",   (1, 1, 1, base_resolution_Y, base_resolution_X, 3),     "TZCYXS"),
        ("TZCYXS_C2_Z1_T1",   (2, 1, 1, base_resolution_Y, base_resolution_X, 3),     "TZCYXS"),
        ("TZCYXS_C2_Z10_T1",  (2, 10, 1, base_resolution_Y, base_resolution_X, 3),    "TZCYXS"),
        ("TZCYXS_T5_Z10_C2",  (5, 10, 2, base_resolution_Y, base_resolution_X, 3),    "TZCYXS")]
    for name, shape, axes in cases_ImageJ:
        #data = make_pattern(shape, dtype=np.uint16)
        data = _make_zeros_with_slice_labels(shape, axes, dtype=np.uint16)
        out_path = os.path.join(OUT_DIR, f"tif_with_ImageJ/{name}.tif")
        ensure_dir(os.path.dirname(out_path))
        #axes = "TZCYXS"
        #shape = (5, 10, 2, base_resolution_Y, base_resolution_X, 3)
        #data = _make_zeros_with_slice_labels(shape, axes, dtype=np.uint16)
        write_tif(
            out_path,
            data,
            axes,
            imagej=True,
            physical_xy=physical_xy,
            compression_level=3,
            photometric="minisblack",
            ome=False)
        print(f"Wrote TIFF: {out_path}  shape={shape}  axes={axes}")
    """ 
    This files needs to opened with ImageJ/FIJI' Bio-Formats reader in order
    to correctly interpret the ImageJ metadata and the S (samples) axis.
    """
    
    # write a README.md into the output folder describing the content:
    readme_path = os.path.join(OUT_DIR, "tif_with_ImageJ/README.md")
    with open(readme_path, "w") as f:
        f.write("# Dummy ImageJ TIFF files with slice labels\n\n")
        f.write("This folder contains dummy ImageJ TIFF files generated with tifffile and option `imagej=True` for testing OMIO.\n\n")
        f.write("Each YX plane in the multi-dimensional images is annotated with text indicating the indices of the T, Z, C, Y, and X axes.\n\n")
        f.write("The following files were generated:\n\n")
        for name, shape, axes in cases_ImageJ:
            f.write(f"- `tif_with_ImageJ/{name}.tif`: shape={shape}, axes={axes}\n")
        f.write("\n")
        f.write("These files need to be opened with ImageJ/FIJI's Bio-Formats reader in order to correctly interpret the ImageJ metadata and the S (samples) axis.\n")

    # --------------------------------------------------------------------------------
    # also write an OME-TIFF with metadata:
    ome_shape = (5, 10, 2, base_resolution_Y, base_resolution_X)
    #dd = np.random.randint(0, 255, ome_shape).astype(np.uint8)
    dd = _make_zeros_with_slice_labels(ome_shape, "TZCYX", dtype=np.uint8)

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
    
    # write a README.md into the output folder describing the content:
    readme_path = os.path.join(OUT_DIR, "ome_tif/README.md")
    with open(readme_path, "w") as f:
        f.write("# Dummy OME-TIFF files with slice labels\n\n")
        f.write("This folder contains dummy OME-TIFF files generated for testing OMIO.\n\n")
        f.write("Each YX plane in the multi-dimensional images is annotated with text indicating the indices of the T, Z, C, Y, and X axes.\n\n")
        f.write("The following files were generated:\n\n")
        f.write(f"- `ome_tif/TZCYX_T5_Z10_C2.ome.tif`: shape={ome_shape}, axes=TZCYX\n")
        f.write("\n")

    # --------------------------------------------------------------------------------
    # paginated / multi-series TIFFs:
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
    
    #data = np.random.randint(0, 255, (8, 2, base_resolution_Y, base_resolution_X, 3), 'uint16')
    data = _make_zeros_with_slice_labels((8, 2, base_resolution_Y, base_resolution_X, 3), "TCYXS", dtype=np.uint16)
    subresolutions = 2
    pixelsize = 0.29  # micrometer
    paged_rgb_path = os.path.join(OUT_DIR, "multiseries_tif/multiseries_TCYXS.ome.tif")
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
        
    # write a README.md into the output folder describing the content:
    readme_path = os.path.join(OUT_DIR, "multiseries_tif/README.md")
    with open(readme_path, "w") as f:
        f.write("# Dummy paginated / multi-series TIFF files\n\n")
        f.write("This folder contains dummy paginated / multi-series TIFF files generated for testing OMIO.\n\n")
        f.write("The following files were generated:\n\n")
        f.write(f"- `multiseries_tif/multiseries_rgb_with_equal_shapes.tif`: two RGB series with identical shapes (16,16,3)\n")
        f.write(f"- `multiseries_tif/multiseries_rgb_with_unequal_series.tif`: two RGB series with differing shapes (16,16,3) and (17,17,3)\n")
        f.write(f"- `multiseries_tif/multiseries_rgb_minisblack_mixture.tif`: one RGB series (16,16,3) and one minisblack series (2,32,32)\n")
        f.write(f"- `multiseries_tif/multiseries_minisblack.tif`: two minisblack series (2,32,32) each\n")
        f.write(f"- `multiseries_tif/multiseries_TCYXS.ome.tif`: shape=(8, 2, {base_resolution_Y}, {base_resolution_X}, 3), axes=TCYXS, with 2 sub-resolution levels and a thumbnail image.\n")
        f.write("\n")
    
    # write a paginated TIFF with RGB series of differing shapes:
    series0 = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
    series1 = np.random.randint(0, 255, (17, 17, 3), dtype=np.uint8)
    paged_rgb_path = os.path.join(OUT_DIR, "paginated_tif/paginated_tif.tif")
    ensure_dir(os.path.dirname(paged_rgb_path))
    with tifffile.TiffWriter(paged_rgb_path) as tif:
        tif.write(series0, photometric="rgb", metadata={'axes': 'YXP'})
        tif.write(series1, photometric="rgb", metadata={'axes': 'YXP'})
    print(f"Wrote paginated OME-TIFF: {paged_rgb_path}")
    readme_path = os.path.join(OUT_DIR, "paginated_tif/README.md")
    with open(readme_path, "w") as f:
        f.write("# Dummy paginated OME-TIFF file\n\n")
        f.write("This folder contains a dummy paginated OME-TIFF file generated for testing OMIO.\n\n")
        f.write("The following file was generated:\n\n")
        f.write(f"- `paginated_tif/paginated_tif.ome.tif`: two RGB series with differing shapes (16,16,3) and (17,17,3)\n")
        f.write("\n")
    
    
    # --------------------------------------------------------------------------------
    # test files for testing OMIO's batch capabilities:
    
    """ 
    In tif_folder_with_multiple_files/, we now generate multiple ome.tif files with TZCYX
    in such a way:
    - 3 files with T=1,Z=10,C=2
    - 2 files with T=1,Z=5,C=2
    """
    base_path = os.path.join(OUT_DIR, "tif_folder_with_multiple_files")
    base_path2= os.path.join(OUT_DIR, "tif_folder_with_multiple_files_unequal_shapes")
    for i in range(3):
        shape = (1, 10, 2, base_resolution_Y, base_resolution_X)
        axes = "TZCYX"
        data = _make_zeros_with_slice_labels(shape, axes, dtype=np.uint16)
        out_path = os.path.join(base_path, f"TZCYX_T1_Z10_C2_file{i+1}.ome.tif")
        out_path2= os.path.join(base_path2, f"TZCYX_T1_Z10_C2_file{i+1}.ome.tif")
        ensure_dir(os.path.dirname(out_path))
        ensure_dir(os.path.dirname(out_path2))
        write_tif(
            out_path,
            data,
            axes,
            physical_xy=physical_xy,
            compression_level=3,
            photometric="minisblack",
            ome=True)
        write_tif(
            out_path2,
            data,
            axes,
            physical_xy=physical_xy,
            compression_level=3,
            photometric="minisblack",
            ome=True)
        print(f"Wrote TIFF: {out_path}  shape={shape}  axes={axes}")
    for i in range(2):
        shape = (1, 5, 2, base_resolution_Y, base_resolution_X)
        axes = "TZCYX"
        data = _make_zeros_with_slice_labels(shape, axes, dtype=np.uint16)
        out_path2 = os.path.join(base_path2, f"TZCYX_T1_Z5_C2_file{i+1}.ome.tif")
        ensure_dir(os.path.dirname(out_path2))
        write_tif(
            out_path2,
            data,
            axes,
            physical_xy=physical_xy,
            compression_level=3,
            photometric="minisblack",
            ome=True)
        print(f"Wrote TIFF: {out_path}  shape={shape}  axes={axes}")
    readme_path = os.path.join(base_path, "README.md")
    with open(readme_path, "w") as f:
        f.write("# Dummy OME-TIFF files for batch testing\n\n")
        f.write("This folder contains multiple dummy OME-TIFF files generated for testing OMIO's batch capabilities.\n\n")
        f.write("The following files were generated:\n\n")
        f.write("- 3 files with shape=(1, 10, 2, {base_resolution_Y}, {base_resolution_X}), axes=TZCYX\n")
        f.write("\n")
    readme_path = os.path.join(base_path2, "README.md")
    with open(readme_path, "w") as f:
        f.write("# Dummy OME-TIFF files for batch testing\n\n")
        f.write("This folder contains multiple dummy OME-TIFF files generated for testing OMIO's batch capabilities.\n\n")
        f.write("The following files were generated:\n\n")
        f.write("- 3 files with shape=(1, 10, 2, {base_resolution_Y}, {base_resolution_X}), axes=TZCYX\n")
        f.write("- 2 files with shape=(1, 5, 2, {base_resolution_Y}, {base_resolution_X}), axes=TZCYX\n")
        f.write("\n")
    
    # --------------------------------------------------------------------------------
    """ 
    Now we create in tif_folder_stacks several folders:
    * dummy_folder_1/
        * ⟶ remains empty
    * dummy_folder_2/
        * ⟶ remains empty
    * FOV1_time001/
        * TZCYX_T5_Z10_C2.ome.tif
    * FOV1_time002/
        * TZCYX_T5_Z10_C2.ome.tif
    * FOV1_time003_z_cropped/
        * TZCYX_T5_Z5_C2.ome.tif
    * FOV2_time001/
        * TZCYX_T5_Z10_C2.ome.tif
    * HC_day1_with_additional_file/
        * TZCYXS_T5_Z10_C2.tif
        * YXS.tif
    * HC_day2/
        * TZCYXS_T5_Z10_C2.tif
    """
    base_path = os.path.join(OUT_DIR, "tif_folder_stacks")
    # dummy_folder_1 and dummy_folder_2 remain empty
    empty_folder_1 = os.path.join(base_path, "dummy_folder_1")
    empty_folder_2 = os.path.join(base_path, "dummy_folder_2")
    fov1_time001_path = os.path.join(base_path, "FOV1_time001")
    fov1_time002_path = os.path.join(base_path, "FOV1_time002")
    fov1_time003_path = os.path.join(base_path, "FOV2_time003_z_cropped")
    fov2_time001_path = os.path.join(base_path, "FOV2_time001")
    hc_day1_path = os.path.join(base_path, "HC_day1_with_additional_file")
    hc_day2_path = os.path.join(base_path, "HC_day2")
    for p in [fov1_time001_path, fov1_time002_path, fov1_time003_path, fov2_time001_path, 
              hc_day1_path, hc_day2_path, empty_folder_1, empty_folder_2]:
        ensure_dir(p)
    # FOV1_time001
    shape = (5, 10, 2, base_resolution_Y, base_resolution_X)
    axes = "TZCYX"
    data = _make_zeros_with_slice_labels(shape, axes, dtype=np.uint16)
    out_path = os.path.join(fov1_time001_path, "TZCYX_T5_Z10_C2_FOV1_time001.ome.tif")
    write_tif(
        out_path,
        data,
        axes,
        physical_xy=physical_xy,
        compression_level=3,
        photometric="minisblack",
        ome=True)
    print(f"Wrote TIFF: {out_path}  shape={shape}  axes={axes}")
    # FOV1_time002
    out_path = os.path.join(fov1_time002_path, "TZCYX_T5_Z10_C2_FOV1_time002.ome.tif")
    write_tif(
        out_path,
        data,
        axes,
        physical_xy=physical_xy,
        compression_level=3,
        photometric="minisblack",
        ome=True)
    print(f"Wrote TIFF: {out_path}  shape={shape}  axes={axes}")
    # FOV1_time003_z_cropped
    shape = (5, 5, 2, base_resolution_Y, base_resolution_X)
    data = _make_zeros_with_slice_labels(shape, axes, dtype=np.uint16)
    out_path = os.path.join(fov1_time003_path, "TZCYX_T5_Z5_C2_FOV1_time003_z_cropped.ome.tif")
    write_tif(
        out_path,
        data,
        axes,
        physical_xy=physical_xy,
        compression_level=3,
        photometric="minisblack",
        ome=True)
    print(f"Wrote TIFF: {out_path}  shape={shape}  axes={axes}")
    # FOV2_time001
    shape = (5, 10, 2, base_resolution_Y, base_resolution_X)
    data = _make_zeros_with_slice_labels(shape, axes, dtype=np.uint16)
    out_path = os.path.join(fov2_time001_path, "TZCYX_T5_Z10_C2_FOV2_time001.ome.tif")
    write_tif(
        out_path,
        data,
        axes,
        physical_xy=physical_xy,
        compression_level=3,
        photometric="minisblack",
        ome=True)
    print(f"Wrote TIFF: {out_path}  shape={shape}  axes={axes}")
    # HC_day1_with_additional_file
    shape = (5, 10, 2, base_resolution_Y, base_resolution_X, 3)
    axes = "TZCYXS"
    data = _make_zeros_with_slice_labels(shape, axes, dtype=np.uint16)
    out_path = os.path.join(hc_day1_path, "TZCYXS_T5_Z10_C2_HC_day1_with_additional_file.tif")
    write_tif(
        out_path,
        data,
        axes,
        physical_xy=physical_xy,
        compression_level=3,
        photometric="minisblack",
        ome=False)
    print(f"Wrote TIFF: {out_path}  shape={shape}  axes={axes}")
    shape = (base_resolution_Y, base_resolution_X, 3)
    axes = "YXS"
    data = _make_zeros_with_slice_labels(shape, axes, dtype=np.uint16)
    out_path = os.path.join(hc_day1_path, "YXS_HC_day1_with_additional_file.tif")
    write_tif(
        out_path,
        data,
        axes,
        physical_xy=physical_xy,
        compression_level=3,
        photometric="minisblack",
        ome=False)
    print(f"Wrote TIFF: {out_path}  shape={shape}  axes={axes}")
    # HC_day2
    shape = (5, 10, 2, base_resolution_Y, base_resolution_X, 3)
    axes = "TZCYXS"
    data = _make_zeros_with_slice_labels(shape, axes, dtype=np.uint16)
    out_path = os.path.join(hc_day2_path, "TZCYXS_T5_Z10_C2_HC_day2.tif")
    write_tif(
        out_path,
        data,
        axes,
        physical_xy=physical_xy,
        compression_level=3,
        photometric="minisblack",
        ome=False)
    print(f"Wrote TIFF: {out_path}  shape={shape}  axes={axes}")
    readme_path = os.path.join(base_path, "README.md")
    with open(readme_path, "w") as f:
        f.write("# Dummy OME-TIFF files for batch stack testing\n\n")
        f.write("This folder contains multiple dummy OME-TIFF files organized in subfolders for testing OMIO's batch stack capabilities.\n\n")
        f.write("The following folder structure was created:\n\n")
        f.write("- `dummy_folder_1/`: (empty)\n")
        f.write("- `dummy_folder_2/`: (empty)\n")
        f.write("- `FOV1_time001/`\n")
        f.write("    - `TZCYX_T5_Z10_C2.ome.tif`: shape=(5, 10, 2, {base_resolution_Y}, {base_resolution_X}), axes=TZCYX\n")
        f.write("- `FOV1_time002/`\n")
        f.write("    - `TZCYX_T5_Z10_C2.ome.tif`: shape=(5, 10, 2, {base_resolution_Y}, {base_resolution_X}), axes=TZCYX\n")
        f.write("- `FOV1_time003_z_cropped/`\n")
        f.write("    - `TZCYX_T5_Z5_C2.ome.tif`: shape=(5, 5, 2, {base_resolution_Y}, {base_resolution_X}), axes=TZCYX\n")
        f.write("- `FOV2_time001/`\n")
        f.write("    - `TZCYX_T5_Z10_C2.ome.tif`: shape=(5, 10, 2, {base_resolution_Y}, {base_resolution_X}), axes=TZCYX\n")
        f.write("- `HC_day1_with_additional_file/`\n")
        f.write("    - `TZCYXS_T5_Z10_C2.tif`: shape=(5, 10, 2, {base_resolution_Y}, {base_resolution_X}, 3), axes=TZCYXS\n")
        f.write("    - `YXS.tif`: shape=({base_resolution_Y}, {base_resolution_X}, 3), axes=YXS\n")
        f.write("- `HC_day2/`\n")
        f.write("    - `TZCYXS_T5_Z10_C2.tif`: shape=(5, 10, 2, {base_resolution_Y}, {base_resolution_X}, 3), axes=TZCYXS\n")
        f.write("\n")
    
    # --------------------------------------------------------------------------------
    """ now we create a dummy BIDS project example in BIDS_project_example/ with the following structure:
    * ID0001/
        * OV/ (remains empty)
        * TP001_FOV1_tif/
            * TZCYX_T3_Z5_C2_ID0001_TP001_FOV1_.ome.tif
        * TP001_FOV2_tif/
            * TZCYX_T3_Z5_C2__ID0001_TP001_FOV2_.ome.tif
       * TP002_single_tif/
            * TZCYX_T1_Z5_C2_TP002.ome.tif
        * TP003_thorlabs_raw
            * leave empty; I will place manually some Thorlabs RAW files here later
        * TP004_all_files_in_this_folder/
            * TZCYX_T3_Z5_C2_ID0001_TP004_1.ome.tif
            * TZCYX_T3_Z5_C2_ID0001_TP004_2.ome.tif
            * TZCYX_T3_Z5_C2_ID0001_TP004_3.ome.tif
        * TP005_Tagged_Folders/
            * FOV1_time001/
                * TZCYX_T3_Z5_C2_ID0001_TP005_FOV1_time001.ome.tif
            * FOV1_time002/
                * TZCYX_T3_Z5_C2_ID0001_TP005_FOV1_time002.ome.tif
        * TP006_tif_multi_file_stack/
            * leave empty; I will place manually multi-file OME-TIFF series here later
    * ID0002/
        * TP001_FOV1 tif/
            * TZCYX_T3_Z5_C2_ID0002_TP001_FOV
        * TP003_thorlabs_raw/
            * leave empty; I will place manually some Thorlabs RAW files here later
        * TP005_Tagged_Folders
            * FOV1_time001/
                * TZCYX_T3_Z5_C2_ID0002_TP005_FOV1_time001.ome.tif
            * FOV1_time002/
                * TZCYX_T3_Z5_C2_ID0002_TP005_FOV1_time002.ome.tif
    """
    base_path = os.path.join(OUT_DIR, "BIDS_project_example")
    for subject_id in ["ID0001", "ID0002"]:
        subject_path = os.path.join(base_path, subject_id)
        ensure_dir(subject_path)
        if subject_id == "ID0001":
            # OV/
            ensure_dir(os.path.join(subject_path, "OV"))
            # TP001_FOV1_tif/
            tp001_fov1_path = os.path.join(subject_path, "TP001_FOV1_tif")
            ensure_dir(tp001_fov1_path)
            shape = (3, 5, 2, base_resolution_Y, base_resolution_X)
            axes = "TZCYX"
            data = _make_zeros_with_slice_labels(shape, axes, dtype=np.uint16)
            out_path = os.path.join(tp001_fov1_path, "TZCYX_T3_Z5_C2_ID0001_TP001_FOV1_.ome.tif")
            write_tif(
                out_path,
                data,
                axes,
                physical_xy=physical_xy,
                compression_level=3,
                photometric="minisblack",
                ome=True)
            print(f"Wrote TIFF: {out_path}  shape={shape}  axes={axes}")
            # TP001_FOV2_tif/
            tp001_fov2_path = os.path.join(subject_path, "TP001_FOV2_tif")
            ensure_dir(tp001_fov2_path)
            out_path = os.path.join(tp001_fov2_path, "TZCYX_T3_Z5_C2_ID0001_TP001_FOV2_.ome.tif")
            write_tif(
                out_path,
                data,
                axes,
                physical_xy=physical_xy,
                compression_level=3,
                photometric="minisblack",
                ome=True)
            print(f"Wrote TIFF: {out_path}  shape={shape}  axes={axes}")
            # TP002_single_tif/
            tp002_path = os.path.join(subject_path, "TP002_single_tif")
            ensure_dir(tp002_path)
            shape = (1, 5, 2, base_resolution_Y, base_resolution_X)
            data = _make_zeros_with_slice_labels(shape, axes, dtype=np.uint16)
            out_path = os.path.join(tp002_path, "TZCYX_T1_Z5_C2_TP002.ome.tif")
            write_tif(
                out_path,
                data,
                axes,
                physical_xy=physical_xy,
                compression_level=3,
                photometric="minisblack",
                ome=True)
            print(f"Wrote TIFF: {out_path}  shape={shape}  axes={axes}")
            # TP003_thorlabs_raw/ (leave empty)
            ensure_dir(os.path.join(subject_path, "TP003_thorlabs_raw"))
            # TP004_all_files_in_this_folder/
            tp004_path = os.path.join(subject_path, "TP004_all_files_in_this_folder")
            ensure_dir(tp004_path)
            for i in range(3):
                out_path = os.path.join(tp004_path, f"TZCYX_T3_Z5_C2_ID0001_TP004_{i+1}.ome.tif")
                write_tif(
                    out_path,
                    data,
                    axes,
                    physical_xy=physical_xy,
                    compression_level=3,
                    photometric="minisblack",
                    ome=True)
                print(f"Wrote TIFF: {out_path}  shape={shape}  axes={axes}")
            # TP005_Tagged_Folders/
            tp005_path = os.path.join(subject_path, "TP005_Tagged_Folders")
            ensure_dir(tp005_path)
            # FOV1_time001/
            fov1_time001_path = os.path.join(tp005_path, "FOV1_time001")
            ensure_dir(fov1_time001_path)
            out_path = os.path.join(fov1_time001_path, "TZCYX_T3_Z5_C2_ID0001_TP005_FOV1_time001.ome.tif")
            write_tif(
                out_path,
                data,
                axes,
                physical_xy=physical_xy,
                compression_level=3,
                photometric="minisblack",
                ome=True)
            print(f"Wrote TIFF: {out_path}  shape={shape}  axes={axes}")
            # FOV1_time002/
            fov1_time002_path = os.path.join(tp005_path, "FOV1_time002")
            ensure_dir(fov1_time002_path)
            out_path = os.path.join(fov1_time002_path, "TZCYX_T3_Z5_C2_ID0001_TP005_FOV1_time002.ome.tif")
            write_tif(
                out_path,
                data,
                axes,
                physical_xy=physical_xy,
                compression_level=3,
                photometric="minisblack",
                ome=True)
            print(f"Wrote TIFF: {out_path}  shape={shape}  axes={axes}")
            # TP006_tif_multi_file_stack/ (leave empty)
            ensure_dir(os.path.join(subject_path, "TP006_tif_multi_file_stack"))
        elif subject_id == "ID0002":
            # TP001_FOV1_tif/
            tp001_fov1_path = os.path.join(subject_path, "TP001_FOV1_tif")
            ensure_dir(tp001_fov1_path)
            shape = (3, 5, 2, base_resolution_Y, base_resolution_X)
            axes = "TZCYX"
            data = _make_zeros_with_slice_labels(shape, axes, dtype=np.uint16)
            out_path = os.path.join(tp001_fov1_path, "TZCYX_T3_Z5_C2_ID0002_TP001_FOV1_.ome.tif")
            write_tif(
                out_path,
                data,
                axes,
                physical_xy=physical_xy,
                compression_level=3,
                photometric="minisblack",
                ome=True)
            print(f"Wrote TIFF: {out_path}  shape={shape}  axes={axes}")
            # TP003_thorlabs_raw/ (leave empty)
            ensure_dir(os.path.join(subject_path, "TP003_thorlabs_raw"))
            # TP005_Tagged_Folders
            tp005_path = os.path.join(subject_path, "TP005_Tagged_Folders")
            ensure_dir(tp005_path)
            # FOV1_time001/
            fov1_time001_path = os.path.join(tp005_path, "FOV1_time001")
            ensure_dir(fov1_time001_path)
            out_path = os.path.join(fov1_time001_path, "TZCYX_T3_Z5_C2_ID0002_TP005_FOV1_time001.ome.tif")
            write_tif(
                out_path,
                data,
                axes,
                physical_xy=physical_xy,
                compression_level=3,
                photometric="minisblack",
                ome=True)
            print(f"Wrote TIFF: {out_path}  shape={shape}  axes={axes}")
            # FOV1_time002/
            fov1_time002_path = os.path.join(tp005_path, "FOV1_time002")
            ensure_dir(fov1_time002_path)
            out_path = os.path.join(fov1_time002_path, "TZCYX_T3_Z5_C2_ID0002_TP005_FOV1_time002.ome.tif")
            write_tif(
                out_path,
                data,
                axes,
                physical_xy=physical_xy,
                compression_level=3,
                photometric="minisblack",
                ome=True)
            print(f"Wrote TIFF: {out_path}  shape={shape}  axes={axes}")
    readme_path = os.path.join(base_path, "README.md")
    with open(readme_path, "w") as f:
        f.write("# Dummy BIDS project example\n\n")
        f.write("This folder contains a dummy BIDS project structure with multiple subjects and timepoints for testing OMIO's BIDS capabilities.\n\n")
        f.write("The following folder structure was created:\n\n")
        f.write("- `ID0001/`\n")
        f.write("    - `OV/`: (empty)\n")
        f.write("    - `TP001_FOV1_tif/`\n")
        f.write("        - `TZCYX_T3_Z5_C2_ID0001_TP001_FOV1_.ome.tif`: shape=(3, 5, 2, {base_resolution_Y}, {base_resolution_X}), axes=TZCYX\n")
        f.write("    - `TP001_FOV2_tif/`\n")
        f.write("        - `TZCYX_T3_Z5_C2__ID0001_TP001_FOV2_.ome.tif`: shape=(3, 5, 2, {base_resolution_Y}, {base_resolution_X}), axes=TZCYX\n")
        f.write("    - `TP002_single_tif/`\n")
        f.write("        - `TZCYX_T1_Z5_C2_TP002.ome.tif`: shape=(1, 5, 2, {base_resolution_Y}, {base_resolution_X}), axes=TZCYX\n")
        f.write("    - `TP003_thorlabs_raw/`: (empty)\n")
        f.write("    - `TP004_all_files_in_this_folder/`\n")
        f.write("        - 3 OME-TIFF files with shape=(3, 5, 2, {base_resolution_Y}, {base_resolution_X}), axes=TZCYX\n")
        f.write("    - `TP005_Tagged_Folders/`\n")
        f.write("        - `FOV1_time001/`\n")
        f.write("            - `TZCYX_T3_Z5_C2_ID0001_TP005_FOV1_time001.ome.tif`: shape=(3, 5, 2, {base_resolution_Y}, {base_resolution_X}), axes=TZCYX\n")
        f.write("        - `FOV1_time002/`\n")
        f.write("            - `TZCYX_T3_Z5_C2_ID0001_TP005_FOV1_time002.ome.tif`: shape=(3, 5, 2, {base_resolution_Y}, {base_resolution_X}), axes=TZCYX\n")
        f.write("    - `TP006_tif_multi_file_stack/`: (empty)\n")
        f.write("- `ID0002/`\n")
        f.write("    - `TP001_FOV1_tif/`\n")
        f.write("        - `TZCYX_T3_Z5_C2_ID0002_TP001_FOV1_.ome.tif`: shape=(3, 5, 2, {base_resolution_Y}, {base_resolution_X}), axes=TZCYX\n")
        f.write("    - `TP003_thorlabs_raw/`: (empty)\n")
        f.write("    - `TP005_Tagged_Folders/`\n")
        f.write("        - `FOV1_time001/`\n")
        f.write("            - `TZCYX_T3_Z5_C2_ID0002_TP005_FOV1_time001.ome.tif`: shape=(3, 5, 2, {base_resolution_Y}, {base_resolution_X}), axes=TZCYX\n")
        f.write("        - `FOV1_time002/`\n")
        f.write("            - `TZCYX_T3_Z5_C2_ID0002_TP005_FOV1_time002.ome.tif`: shape=(3, 5, 2, {base_resolution_Y}, {base_resolution_X}), axes=TZCYX\n")
        f.write("\n")
    
    # write a general README.md into the output folder and mention that all files in here are
    # artificially toy data for testing purposes only:
    readme_path = os.path.join(OUT_DIR, "README.md")
    with open(readme_path, "w") as f:
        f.write("# Dummy TIFF files for testing OMIO\n\n")
        f.write("This folder contains multiple dummy TIFF and OME-TIFF files generated for testing the OMIO library.\n\n")
        f.write("**Note:** All image data in these files are artificially generated toy data for testing purposes only. They do not represent real microscopy images.\n")
        f.write("Do not use these files for any analysis other than testing the functionality of OMIO.\n")
        f.write("\n")
    
    print("\nDone.")

# %% MAIN
if __name__ == "__main__":
    main()