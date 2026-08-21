---
name: Feature request
about: Suggest an idea for this project
title: ''
labels: ''
assignees: ''

---

**Is your feature request related to a microscopy I/O problem? Please describe.**  
Please describe the image I/O issue you are trying to solve. For example: unsupported file format, incorrect axis order, missing or wrong metadata, physical pixel size problems, large-file handling, batch conversion, napari display, or OME-TIFF writing behavior.

**Describe the feature or behavior you would like**  
Please describe what you would like OMIO to do. If this concerns a new file format or metadata convention, please specify the expected input and the desired OMIO output, including axes, shape, physical pixel sizes, units, and output format if relevant.

**Provide an example dataset if possible**  
If the request depends on a specific file format or acquisition setup, please provide a small representative example file or stack. You can upload it to a public repository, Zenodo, OSF, institutional storage, or another accessible location. If the data cannot be shared publicly, you may provide a private access link or contact the maintainer to arrange transfer. Please remove sensitive or unpublished biological information when possible.

**Describe alternatives or workarounds you have tried**  
Please mention whether you tried other tools or OMIO options, such as `imread`, `imconvert`, `zarr_store="disk"`, `reuse_disk_cache=True`, `folder_stacks`, `bids_batch_process`, or manual metadata overrides.

**Environment and version information**  
Please include your OMIO version, Python version, operating system, and relevant package versions if known. If possible, paste the output of:

```python
import omio as om
om.hello_world()
```

**Additional context**  
Add any other context, screenshots, error messages, acquisition software details, microscope/vendor information, or expected downstream workflow.
