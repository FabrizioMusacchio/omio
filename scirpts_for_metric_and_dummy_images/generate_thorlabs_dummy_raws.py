""" 
This script generates dummy Thorlabs RAW files along with matching XML metadata 
files for testing OMIO.

author: Fabrizio Musacchio
date: December 2025
"""
# %% IMPORTS
import os
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom

# %% FUNCTIONS
def write_example_xml(
    xml_path,
    *,
    X,
    Y,
    C,
    T,
    Z,
    pixel_size_um=0.5,
    z_step_um=1.0,
    time_interval_s=1.0,
    bits=16,
):
    """
    Create a minimal Thorlabs-like XML file that matches the expectations
    of read_thorlabs_raw().
    """

    root = ET.Element("ThorImage")

    lsm = ET.SubElement(root, "LSM")
    lsm.set("pixelX", str(X))
    lsm.set("pixelY", str(Y))
    lsm.set("channel", str(C))
    lsm.set("pixelSizeUM", str(pixel_size_um))
    lsm.set("frameRate", "1.0")

    wavelengths = ET.SubElement(root, "Wavelengths")
    for i in range(C):
        ET.SubElement(wavelengths, "Wavelength").set("index", str(i))

    timelapse = ET.SubElement(root, "Timelapse")
    timelapse.set("timepoints", str(T))
    timelapse.set("intervalSec", str(time_interval_s))

    camera = ET.SubElement(root, "Camera")
    camera.set("bitsPerPixel", str(bits))

    zstage = ET.SubElement(root, "ZStage")
    zstage.set("steps", str(Z))
    zstage.set("stepSizeUM", str(z_step_um))

    streaming = ET.SubElement(root, "Streaming")
    streaming.set("zFastEnable", "1" if Z > 1 else "0")

    # pretty-print
    rough_string = ET.tostring(root, "utf-8")
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")

    with open(xml_path, "w") as f:
        f.write(pretty_xml)


def write_dummy_raw(
    raw_path,
    *,
    T,
    Z,
    C,
    Y,
    X,
    dtype=np.uint16,
):
    """
    Write a RAW file with a simple ramp pattern.
    Data layout is contiguous and matches reshape((T,Z,C,Y,X)).
    """

    total_elements = T * Z * C * Y * X
    data = np.arange(total_elements, dtype=dtype)
    data.tofile(raw_path)


def generate_case(base_dir, name, *, T, Z, C, Y, X):
    os.makedirs(base_dir, exist_ok=True)

    raw_path = os.path.join(base_dir, f"{name}.raw")
    xml_path = os.path.join(base_dir, f"{name}.xml")

    print(f"Creating test case: {name}")
    print(f"  Shape: T={T}, Z={Z}, C={C}, Y={Y}, X={X}")

    write_dummy_raw(
        raw_path,
        T=T, Z=Z, C=C, Y=Y, X=X,
        dtype=np.uint16,
    )

    write_example_xml(
        xml_path,
        X=X,
        Y=Y,
        C=C,
        T=T,
        Z=Z,
        pixel_size_um=0.5,
        z_step_um=1.0,
        time_interval_s=1.0,
        bits=16,
    )

# %% MAIN
if __name__ == "__main__":
    out_root = "thorlabs_dummy_data"
    # prepend path to folder of this script:
    out_root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        out_root)

    # Case 1: C=1, Z=1, T=1
    generate_case(
        os.path.join(out_root, "case_C1_Z1_T1"),
        "example_C1_Z1_T1",
        T=1,
        Z=1,
        C=1,
        Y=20,
        X=20,
    )

    # Case 2: C=2, Z=1, T=1
    generate_case(
        os.path.join(out_root, "case_C2_Z1_T1"),
        "example_C2_Z1_T1",
        T=1,
        Z=1,
        C=2,
        Y=20,
        X=20,
    )

    # Case 3: C=2, Z=10, T=5
    generate_case(
        os.path.join(out_root, "case_C2_Z10_T5"),
        "example_C2_Z10_T5",
        T=5,
        Z=10,
        C=2,
        Y=20,
        X=20,
    )

    print("\nAll dummy Thorlabs RAW test cases created.")
