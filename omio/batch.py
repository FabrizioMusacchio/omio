""" 
OMIO BATCH PROCESSOR

This module provides functions to perform batch processing of image files
using OMIO's readers and writers.

author: Fabrizio Musacchio  
first version: December 2025
ported to modularized structure: August 2026
"""
# %% IMPORTS
from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime as _DateTime
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .core import *
from .cache import cleanup_omio_cache
from .read import (
    imread,
    _collapse_ome_multifile_series,
    _dispatch_read_file,
    _first_image_file_in_folder,
    _merge_folderstacks_with_padding)
from .writers.ome_tiff import imwrite
from .convert import imconvert
from .readers.thorlabs_raw import create_thorlabs_raw_yaml
# %% FLEXIBLE BATCH CONSTANTS
DEFAULT_BATCH_IMAGE_PATTERNS = (
    "*.ome.tif",
    "*.ome.tiff",
    "*.tif",
    "*.tiff",
    "*.czi",
    "*.lsm",
    "*.raw")

DEFAULT_BATCH_RAW_TEMPLATE_METADATA = {
    "T": 1,
    "Z": 1,
    "C": 1,
    "Y": 1,
    "X": 1,
    "bits": 16,
    "pixelunit": "micron",
    "physicalsize_xyz": (0.5, 0.5, 1.0),
    "time_increment": 1.0,
    "time_increment_unit": "seconds"}

_BATCH_FOLDER_STACK_ALLOWED_EXTENSIONS = {
    ".tif",
    ".tiff",
    ".lsm",
    ".czi",
    ".raw",
    ".ome.tif",
    ".ome.tiff"}
# %% FLEXIBLE BATCH DATA CLASSES
@dataclass(frozen=True)
class BatchImageRecord:
    """One image discovered in a flexible BIDS-like OMIO batch project."""

    subject_id: str
    tag_folders: tuple[str, ...]
    image_path: Path
    scan_dir: Path
    output_scope_dir: Path
    input_kind: str = "file"
    folder_stack_tag: str | None = None
    folder_stack_paths: tuple[Path, ...] = ()

@dataclass(frozen=True)
class BatchProcessedRecord:
    """One successfully processed image in an OMIO batch run."""

    input_path: Path
    output_path: Path
    subject_id: str
    tag_folders: tuple[str, ...]

@dataclass(frozen=True)
class BatchSkippedRecord:
    """One skipped image in an OMIO batch run."""

    input_path: Path
    reason: str
    subject_id: str
    tag_folders: tuple[str, ...] = ()
    stage: str = "unknown"
    output_path: Path | None = None

@dataclass(frozen=True)
class BatchErrorRecord:
    """Structured per-file error collected during an OMIO batch run."""

    timestamp: str
    input_path: Path
    relative_path: str
    stage: str
    exception_type: str
    message: str
    subject_id: str
    tag_folders: tuple[str, ...] = ()
    output_dir: Path | None = None
    output_path: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class BatchProcessResult:
    """Summary returned by :func:`bids_batch_process`."""

    processed: tuple[BatchProcessedRecord, ...] = ()
    skipped: tuple[BatchSkippedRecord, ...] = ()
    failed: tuple[BatchErrorRecord, ...] = ()
    discovered: tuple[BatchImageRecord, ...] = ()
    report_path: Path | None = None
    run_report_yaml_path: Path | None = None
    error_report_path: Path | None = None
    local_error_report_paths: tuple[Path, ...] = field(default_factory=tuple)

@dataclass(frozen=True)
class BatchRawYamlTemplateRecord:
    """One RAW file considered for OMIO YAML template creation."""

    raw_path: Path
    yaml_path: Path | None
    template_metadata: dict[str, Any]
    status: str
    reason: str = ""

@dataclass(frozen=True)
class BatchRawYamlTemplateResult:
    """Summary returned by :func:`batch_create_thorlabs_raw_yaml_templates`."""

    report_path: Path | None
    records: tuple[BatchRawYamlTemplateRecord, ...] = ()

    @property
    def created(self) -> tuple[BatchRawYamlTemplateRecord, ...]:
        """RAW files for which YAML template creation was attempted."""

        return tuple(record for record in self.records if record.status == "created")

    @property
    def skipped(self) -> tuple[BatchRawYamlTemplateRecord, ...]:
        """RAW files skipped during YAML template creation."""

        return tuple(record for record in self.records if record.status != "created")
# %% BATCH FUNCTIONS
def _match_name(name: str, pattern: str, mode: str) -> bool:
    """
    Match a string against a pattern using a selectable matching mode.

    This helper provides a small, explicit abstraction over common name matching
    strategies used throughout OMIO, for example when selecting files, folders,
    or tagged stack components.

    Supported matching modes
    ------------------------
    * "startswith":
        Return True if `name` starts with `pattern`, equivalent to
        `name.startswith(pattern)`.

    * "exact":
        Return True if `name` and `pattern` are identical strings.

    * "regex":
        Interpret `pattern` as a regular expression and return True if
        `re.match(pattern, name)` succeeds. The match is anchored at the beginning
        of `name`, following Python's `re.match` semantics.

    Parameters
    ----------
    name : str
        The string to be tested, typically a filename or folder name.
    pattern : str
        The pattern to match against `name`. Interpreted according to `mode`.
    mode : {"startswith", "exact", "regex"}
        Matching strategy to use.

    Returns
    -------
    bool
        True if the match succeeds under the selected mode, False otherwise.

    Raises
    ------
    ValueError
        If `mode` is not one of the supported values {"startswith", "exact", "regex"}.

    Notes
    -----
    This function does not perform any normalization (such as lowercasing) of
    either `name` or `pattern`. Callers are responsible for ensuring consistent
    string preprocessing when required.
    """
    if mode == "startswith":
        return name.startswith(pattern)
    if mode == "exact":
        return name == pattern
    if mode == "regex":
        return re.match(pattern, name) is not None
    raise ValueError(f"_match_name: invalid mode={mode!r}. Allowed: 'startswith','exact','regex'.")

def _batch_timestamp() -> str:
    """Return a filesystem-safe timestamp for batch reports."""

    return _DateTime.now().strftime("%Y-%m-%d_%H-%M-%S")

def _normalize_subject_ids(subject_ids: Iterable[str | Path] | None) -> tuple[str, ...] | None:
    """Normalize optional requested subject folder names."""

    if subject_ids is None:
        return None
    normalized = tuple(str(Path(subject_id).name) for subject_id in subject_ids if str(subject_id))
    return normalized

def _normalize_batch_patterns(image_patterns: str | Sequence[str] | None) -> tuple[str, ...]:
    """Return explicit image glob patterns, falling back to OMIO defaults."""

    if image_patterns is None:
        return DEFAULT_BATCH_IMAGE_PATTERNS
    if isinstance(image_patterns, str):
        return (image_patterns,)
    return tuple(str(pattern) for pattern in image_patterns if str(pattern))

def _normalize_exclude_tokens(exclude_name_contains: str | Sequence[str] | None) -> tuple[str, ...]:
    """Return name tokens used to exclude files or folders from discovery."""

    if exclude_name_contains is None:
        return ()
    if isinstance(exclude_name_contains, str):
        return (exclude_name_contains,)
    return tuple(str(token) for token in exclude_name_contains if str(token))

def _normalize_folder_stack_tags(folder_stacks: str | Path | Sequence[str | Path] | None) -> tuple[str, ...]:
    """Return optional folder-stack tags used by flexible batch discovery."""

    if folder_stacks is None:
        return ()
    if isinstance(folder_stacks, bool):
        raise ValueError("bids_batch_process(folder_stacks=...) expects a tag string or sequence of tag strings, not a boolean.")
    if isinstance(folder_stacks, (str, Path)):
        tag = str(Path(folder_stacks).name if isinstance(folder_stacks, Path) else folder_stacks)
        return (tag,) if tag else ()
    return tuple(
        str(Path(tag).name if isinstance(tag, Path) else tag)
        for tag in folder_stacks
        if str(tag))

def _name_is_excluded(name: str, exclude_tokens: Sequence[str]) -> bool:
    """Return True if any exclude token occurs in ``name``."""

    return any(token in name for token in exclude_tokens)

def _normalize_tag_folder_levels(tag_folder_levels: Sequence[Iterable[str | Path] | None] | None) -> tuple[tuple[str, ...], ...]:
    """
    Normalize flexible folder-tag levels below each subject.

    ``None`` as the complete value keeps OMIO's BIDS-like default of one
    experiment level matching ``"TP"``. Empty levels inside an explicitly
    provided sequence are skipped.
    """

    if tag_folder_levels is None:
        return (("TP",),)
    normalized: list[tuple[str, ...]] = []
    for level in tag_folder_levels:
        if level is None:
            continue
        if isinstance(level, (str, Path)):
            tokens = (str(Path(level).name if isinstance(level, Path) else level),)
        else:
            tokens = tuple(str(Path(token).name if isinstance(token, Path) else token) for token in level if str(token))
        if tokens:
            normalized.append(tokens)
    return tuple(normalized)

def _select_matching_child_dirs(parent: Path, tokens: Sequence[str], exclude_tokens: Sequence[str]) -> list[Path]:
    """Return child folders whose names contain one of ``tokens``."""

    if not parent.is_dir():
        return []
    return sorted(
        path
        for path in parent.iterdir()
        if path.is_dir()
        and not _name_is_excluded(path.name, exclude_tokens)
        and any(token in path.name for token in tokens))

def _iter_tag_folder_chains(root_dir: Path,
                            tag_folder_levels: Sequence[Sequence[str]],
                            exclude_tokens: Sequence[str]) -> list[list[Path]]:
    """Return matched folder chains for an arbitrary number of folder levels."""

    if not tag_folder_levels:
        return [[]]
    current_level = tag_folder_levels[0]
    remaining_levels = tag_folder_levels[1:]
    chains: list[list[Path]] = []
    for child_dir in _select_matching_child_dirs(root_dir, current_level, exclude_tokens):
        for tail_chain in _iter_tag_folder_chains(child_dir, remaining_levels, exclude_tokens):
            chains.append([child_dir, *tail_chain])
    return chains

def _collect_batch_image_paths(scan_dir: Path,
                               image_patterns: str | Sequence[str] | None,
                               exclude_tokens: Sequence[str],
                               *,
                               collapse_ome_multifile_series: bool = True,
                               verbose: bool = False) -> list[Path]:
    """Collect image files from one folder using one or more glob patterns."""

    patterns = _normalize_batch_patterns(image_patterns)
    matched_paths: dict[Path, None] = {}
    for pattern in patterns:
        for path in scan_dir.glob(pattern):
            if path.is_file() and not _name_is_excluded(path.name, exclude_tokens):
                matched_paths[path] = None
    paths = sorted(matched_paths)
    if not collapse_ome_multifile_series:
        return paths
    collapsed = _collapse_ome_multifile_series([str(path) for path in paths], verbose=verbose)
    return [Path(path) for path in collapsed]

def _collect_batch_folder_stack_groups(scan_dir: Path,
                                       folder_stack_tags: Sequence[str],
                                       exclude_tokens: Sequence[str]) -> list[tuple[str, tuple[Path, ...]]]:
    """Collect tagged child-folder groups that should be read as OMIO folder stacks."""

    if not folder_stack_tags or not scan_dir.is_dir():
        return []
    child_dirs = sorted(
        path
        for path in scan_dir.iterdir()
        if path.is_dir() and not _name_is_excluded(path.name, exclude_tokens))
    groups: list[tuple[str, tuple[Path, ...]]] = []
    seen_first_paths: set[Path] = set()
    for tag in folder_stack_tags:
        matching_dirs = tuple(path for path in child_dirs if path.name.startswith(tag))
        if not matching_dirs:
            continue
        first_path = matching_dirs[0]
        if first_path in seen_first_paths:
            continue
        seen_first_paths.add(first_path)
        groups.append((tag, matching_dirs))
    return groups

def _output_scope_for_chain(subject_dir: Path, folder_chain: Sequence[Path]) -> Path:
    """Choose the folder that receives the OMIO batch output folder."""

    return folder_chain[-1] if folder_chain else subject_dir

def _strip_ome_suffix(path: Path) -> str:
    """Return an image basename without common OME-TIFF compound suffixes."""

    name = path.name
    lower = name.lower()
    if lower.endswith(".ome.tiff"):
        return name[:-9]
    if lower.endswith(".ome.tif"):
        return name[:-8]
    return path.stem

def _expected_batch_output_path(output_dir: Path, image_path: Path, output_suffix: str) -> Path:
    """Return the default OMIO batch output path for one input image."""

    return output_dir / f"{_strip_ome_suffix(image_path)}{output_suffix}.ome.tif"

def _batch_output_name_source(record: BatchImageRecord) -> Path:
    """Return the path-like object used to derive the default output basename."""

    if record.input_kind == "folder_stack" and record.folder_stack_tag:
        return record.scan_dir / record.folder_stack_tag
    return record.image_path

def _resolve_batch_output_dir(output_scope_dir: Path, output_folder_name: str | Path) -> Path:
    """Return the output directory for one batch image record."""

    output_folder = Path(output_folder_name)
    if output_folder.is_absolute():
        return output_folder
    return output_scope_dir / output_folder

def _relative_report_path(path: Path, root: Path) -> str:
    """Return a POSIX-style path relative to ``root`` when possible."""

    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()

def _metadata_for_batch_output(metadata: dict | None, output_path: Path) -> dict | None:
    """Return metadata whose OMIO output annotations point at ``output_path``."""

    if metadata is None:
        return None
    output_metadata = deepcopy(metadata)
    annotations = output_metadata.get("Annotations", {})
    if not isinstance(annotations, dict):
        annotations = {}
    annotations = dict(annotations)
    annotations["original_filename"] = output_path.name
    annotations["original_filetype"] = "ome.tif"
    annotations["original_parentfolder"] = str(output_path.parent)
    output_metadata["Annotations"] = annotations
    return output_metadata

def _error_record_to_dict(error: BatchErrorRecord, root: Path) -> dict[str, Any]:
    """Serialize a batch error record for persistent reports."""

    return {
        "timestamp": error.timestamp,
        "input_path": str(error.input_path),
        "relative_path": error.relative_path,
        "stage": error.stage,
        "exception_type": error.exception_type,
        "message": error.message,
        "subject_id": error.subject_id,
        "tag_folders": tuple(error.tag_folders),
        "output_dir": None if error.output_dir is None else str(error.output_dir),
        "output_path": None if error.output_path is None else str(error.output_path),
        "extra": dict(error.extra)}

def _write_template_metadata_block(handle, metadata: dict[str, Any]) -> None:
    """Write one formatted ``template_metadata`` block into an error report."""

    handle.write("        'template_metadata': {\n")
    for key, value in metadata.items():
        handle.write(f"            {key!r}: {value!r},\n")
    handle.write("        },\n")

def _write_local_batch_error_report(report_dir: Path,
                                    *,
                                    timestamp: str,
                                    error: BatchErrorRecord,
                                    raw_template_metadata: dict[str, Any]) -> Path:
    """Append one short OMIO error report next to the affected image."""

    report_path = report_dir / f"omio_batch_error_report_{timestamp}.txt"
    report_dir.mkdir(parents=True, exist_ok=True)
    with report_path.open("a", encoding="utf-8") as handle:
        handle.write(f"OMIO batch error report: {timestamp}\n")
        handle.write(f"Input image: {error.input_path}\n")
        handle.write(f"Stage: {error.stage}\n")
        handle.write(f"Error: {error.exception_type}: {error.message}\n")
        if error.output_path is not None:
            handle.write(f"Output path: {error.output_path}\n")
        if error.input_path.suffix.lower() == ".raw":
            handle.write(f"Template metadata defaults: {raw_template_metadata!r}\n")
        handle.write("\n")
    return report_path

def _write_root_batch_error_report(project_root: Path,
                                   *,
                                   timestamp: str,
                                   errors: Sequence[BatchErrorRecord],
                                   raw_template_metadata: dict[str, Any]) -> Path | None:
    """Write one structured root-level OMIO batch error report."""

    if not errors:
        return None
    report_path = project_root / f"omio_batch_error_report_{timestamp}.txt"
    raw_errors = [error for error in errors if error.input_path.suffix.lower() == ".raw"]
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# OMIO batch error report: {timestamp}\n")
        handle.write("# OMIO_BATCH_ERRORS is a valid Python list of structured error records.\n")
        handle.write("# OMIO_BATCH_FAILED_RAW_FILES can be edited and then used with\n")
        handle.write("# omio.batch_create_thorlabs_raw_yaml_templates(...).\n\n")
        handle.write("OMIO_BATCH_ERRORS = ")
        handle.write(repr([_error_record_to_dict(error, project_root) for error in errors]))
        handle.write("\n\n")
        handle.write("OMIO_BATCH_FAILED_RAW_FILES = {\n")
        for error in raw_errors:
            handle.write(f"    {str(error.input_path)!r}: {{\n")
            handle.write(f"        'reason': {error.message!r},\n")
            handle.write(f"        'stage': {error.stage!r},\n")
            handle.write(f"        'subject_id': {error.subject_id!r},\n")
            handle.write(f"        'tag_folders': {tuple(error.tag_folders)!r},\n")
            handle.write(f"        'reported_at': {timestamp!r},\n")
            _write_template_metadata_block(handle, raw_template_metadata)
            handle.write("    },\n")
        handle.write("}\n")
    return report_path

def _run_report_paths(project_root: Path, run_report_name: str) -> tuple[Path, Path]:
    """Return machine-readable and human-readable run report paths."""

    report_base = project_root / run_report_name
    return report_base.with_suffix(".yaml"), report_base.with_suffix(".txt")

def _empty_run_report_payload(project_root: Path) -> dict[str, Any]:
    """Return a new OMIO batch run report payload."""

    return {
        "omio_batch_run_report_version": 1,
        "project_root": str(project_root),
        "last_updated": None,
        "files": {}}

def _load_run_report(path: Path, project_root: Path) -> dict[str, Any]:
    """Load an existing machine-readable run report if present."""

    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return _empty_run_report_payload(project_root)
    text = path.read_text(encoding="utf-8")
    try:
        loaded = yaml.safe_load(text)
    except Exception:
        loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"Run report is not a mapping: {path}")
    loaded.setdefault("omio_batch_run_report_version", 1)
    loaded.setdefault("project_root", str(project_root))
    loaded.setdefault("last_updated", None)
    loaded.setdefault("files", {})
    if not isinstance(loaded["files"], dict):
        raise ValueError(f"Run report 'files' entry is not a mapping: {path}")
    return loaded

def _write_run_report_yaml(path: Path, payload: dict[str, Any]) -> None:
    """Write the machine-readable OMIO batch run report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, default_flow_style=False)
    path.write_text(text, encoding="utf-8")

def _format_run_summary(run: dict[str, Any]) -> str:
    """Return one compact text run-history line."""

    status = str(run.get("status", "unknown"))
    if status == "already_processed":
        return f"{run.get('timestamp', 'unknown')} | skipped/already converted"
    if status == "processed":
        parts = [str(run.get("timestamp", "unknown")), "processed"]
        method = run.get("method")
        if method:
            parts.append(str(method))
        details = run.get("details")
        if details:
            parts.append(str(details))
        return " | ".join(parts)
    stage = run.get("stage", status)
    exception_type = run.get("exception_type", "Error")
    message = run.get("message", "")
    return f"{run.get('timestamp', 'unknown')} | failed | {stage} | {exception_type}: {message}"

def _status_label_from_entry(file_entry: dict[str, Any]) -> str:
    """Return the visible text-report status label for one file entry."""

    latest_status = str(file_entry.get("latest_status", "unknown"))
    if latest_status in {"processed", "already_processed"}:
        return "CONVERTED"
    if latest_status == "failed":
        return "FAILED"
    return latest_status.upper()

def _add_tree_path(tree: dict[str, Any], parts: Sequence[str], file_key: str) -> None:
    """Add one file key to a nested folder tree."""

    node = tree
    for part in parts:
        node = node.setdefault(part, {})
    node.setdefault("__files__", []).append(file_key)

def _render_tree_node(lines: list[str],
                      tree: dict[str, Any],
                      files: dict[str, Any],
                      *,
                      indent: str = "") -> None:
    """Render one nested folder node into ``lines``."""

    folder_names = sorted(key for key in tree if key != "__files__")
    file_keys = sorted(tree.get("__files__", []))
    entries = [(name, "folder") for name in folder_names] + [(key, "file") for key in file_keys]
    for index, (name, entry_type) in enumerate(entries):
        is_last = index == len(entries) - 1
        branch = "└─ " if is_last else "├─ "
        child_indent = indent + ("   " if is_last else "│  ")
        if entry_type == "folder":
            lines.append(f"{indent}{branch}{name}/")
            _render_tree_node(lines, tree[name], files, indent=child_indent)
            continue
        file_entry = files[name]
        lines.append(f"{indent}{branch}{Path(name).name} [{_status_label_from_entry(file_entry)}]")
        output_path = file_entry.get("latest_output_path")
        if output_path:
            lines.append(f"{child_indent}output: {output_path}")
        runs = list(file_entry.get("runs", []))
        if runs:
            lines.append(f"{child_indent}runs:")
            for run in runs:
                lines.append(f"{child_indent}  - {_format_run_summary(run)}")

def _render_run_report_text(payload: dict[str, Any], path: Path) -> None:
    """Write the human-readable OMIO batch run report."""

    files = payload.get("files", {})
    tree: dict[str, Any] = {}
    for file_key in files:
        parts = Path(file_key).parts
        if parts:
            _add_tree_path(tree, parts[:-1], file_key)
    lines = [
        "OMIO batch run report",
        f"Project root: {payload.get('project_root', '')}",
        f"Last updated: {payload.get('last_updated', '')}",
        ""]
    if files:
        _render_tree_node(lines, tree, files)
    else:
        lines.append("No batch image files have been recorded yet.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")

def _append_run_report_entry(payload: dict[str, Any],
                             record: BatchImageRecord,
                             *,
                             root: Path,
                             timestamp: str,
                             status: str,
                             method_name: str,
                             output_path: Path | None = None,
                             error: BatchErrorRecord | None = None,
                             details: str | None = None) -> None:
    """Append one run-history entry for a discovered image."""

    file_key = _relative_report_path(record.image_path, root)
    file_entry = payload.setdefault("files", {}).setdefault(
        file_key,
        {
            "subject_id": record.subject_id,
            "tag_folders": list(record.tag_folders),
            "input_path": file_key,
            "runs": []})
    file_entry["subject_id"] = record.subject_id
    file_entry["tag_folders"] = list(record.tag_folders)
    file_entry["input_path"] = file_key
    run_entry: dict[str, Any] = {
        "timestamp": timestamp,
        "status": status}
    if status == "processed":
        run_entry["method"] = method_name
        if details:
            run_entry["details"] = details
    if output_path is not None:
        relative_output = _relative_report_path(output_path, root)
        run_entry["output_path"] = relative_output
        file_entry["latest_output_path"] = relative_output
    if error is not None:
        run_entry["stage"] = error.stage
        run_entry["exception_type"] = error.exception_type
        run_entry["message"] = error.message
    file_entry.setdefault("runs", []).append(run_entry)
    file_entry["latest_status"] = "failed" if status == "failed" else status

def _extract_named_dict(report_text: str, variable_name: str) -> dict[str, Any]:
    """Extract a Python dictionary assigned to ``variable_name`` from text."""

    assignment_index = report_text.find(variable_name)
    if assignment_index < 0:
        return {}
    brace_start = report_text.find("{", assignment_index)
    if brace_start < 0:
        return {}
    depth = 0
    for index in range(brace_start, len(report_text)):
        character = report_text[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                parsed = ast.literal_eval(report_text[brace_start:index + 1])
                if not isinstance(parsed, dict):
                    raise ValueError(f"{variable_name} is not a dictionary.")
                return parsed
    raise ValueError(f"Could not find the end of {variable_name}.")

def _extract_raw_paths_from_report_text(report_text: str) -> list[Path]:
    """Fallback parser for legacy plain-text reports containing RAW paths."""

    raw_path_pattern = re.compile(r"([A-Za-z]:\\[^\n\r'\"]+?\.raw|/[^\n\r'\"]+?\.raw)")
    return [Path(match.group(1).strip()) for match in raw_path_pattern.finditer(report_text)]

def _load_skipped_raw_entries_from_report(report_path: Path,
                                          *,
                                          raw_template_metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Load RAW paths and editable template metadata from one OMIO or ZenReg report."""

    report_text = report_path.read_text(encoding="utf-8")
    skipped_dict = _extract_named_dict(report_text, "OMIO_BATCH_FAILED_RAW_FILES")
    if not skipped_dict:
        skipped_dict = _extract_named_dict(report_text, "ZENREG_BATCH_SKIPPED_RAW_FILES")
    if skipped_dict:
        entries = []
        for raw_path, details in skipped_dict.items():
            details = details if isinstance(details, dict) else {}
            entries.append(
                {
                    "path": Path(raw_path),
                    "template_metadata": dict(details.get("template_metadata", raw_template_metadata))})
        return entries
    return [
        {
            "path": raw_path,
            "template_metadata": dict(raw_template_metadata)}
        for raw_path in _extract_raw_paths_from_report_text(report_text)]

def _expected_raw_yaml_paths(raw_path: Path) -> tuple[Path, ...]:
    """Return likely OMIO Thorlabs RAW YAML sidecar paths."""

    return (
        raw_path.with_name(raw_path.stem + "_metadata.yaml"),
        raw_path.with_name(raw_path.stem + "_metadata.yml"),
        raw_path.with_suffix(".yaml"),
        raw_path.with_suffix(".yml"))

def _find_latest_batch_error_report(project_root: Path) -> Path | None:
    """Return the latest root-level OMIO or ZenReg batch error report if present."""

    reports = sorted(project_root.glob("omio_batch_error_report_*.txt"))
    if reports:
        return reports[-1]
    reports = sorted(project_root.glob("zenreg_batch_error_report_*.txt"))
    return reports[-1] if reports else None

def discover_bids_like_batch_images(project_root: str | Path,
                                    *,
                                    subject_ids: Iterable[str | Path] | None = None,
                                    subject_prefix: str = "ID",
                                    tag_folder_levels: Sequence[Iterable[str | Path] | None] | None = None,
                                    image_patterns: str | Sequence[str] | None = None,
                                    exclude_name_contains: str | Sequence[str] | None = ("Preview",),
                                    folder_stacks: str | Path | Sequence[str | Path] | None = None,
                                    collapse_ome_multifile_series: bool = True,
                                    verbose: bool = False) -> list[BatchImageRecord]:
    """
    Discover microscopy image files in a flexible BIDS-like project tree.

    Parameters
    ----------
    project_root : str or pathlib.Path
        Root folder that contains subject folders.
    subject_ids : iterable of str or None, optional
        Explicit subject folders to process. If None, all child folders whose
        names start with ``subject_prefix`` are used.
    subject_prefix : str, optional
        Prefix used for automatic subject discovery. Default is ``"ID"``.
    tag_folder_levels : sequence, optional
        Folder-token levels below each subject. Each level can contain one or
        more name tokens matched by containment. Empty levels, ``None``, ``()``,
        and ``[]`` are skipped. If the complete value is None, OMIO uses one
        default experiment level matching ``"TP"``.
    image_patterns : str, sequence[str], or None, optional
        Glob pattern(s) used to find images in the final folder level. If None,
        OMIO uses sensible microscopy defaults.
    exclude_name_contains : str, sequence[str], or None, optional
        Tokens used to exclude file or folder names from discovery.
    folder_stacks : str, sequence[str], or None, optional
        Optional folder-stack tag(s) searched below each final folder-token level.
        For example, ``folder_stacks=("time",)`` detects child folders such as
        ``time01`` and ``time02`` and adds one batch record that is loaded through
        OMIO's folder-stack merge path.
    collapse_ome_multifile_series : bool, optional
        If True, detect OME multi-file TIFF series in each final image folder
        and keep only one representative file per series. Default is True.
    verbose : bool, optional
        If True, print OME multi-file series collapse messages.

    Returns
    -------
    list[BatchImageRecord]
        Sorted image records with subject ID, folder chain, input path, scan
        folder, and output-scope folder.
    """

    root = Path(project_root)
    if not root.exists():
        raise FileNotFoundError(f"project_root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"project_root is not a directory: {root}")

    exclude_tokens = _normalize_exclude_tokens(exclude_name_contains)
    folder_stack_tags = _normalize_folder_stack_tags(folder_stacks)
    requested_subjects = _normalize_subject_ids(subject_ids)
    if requested_subjects is None:
        subject_dirs = sorted(
            path
            for path in root.iterdir()
            if path.is_dir()
            and path.name.startswith(subject_prefix)
            and not _name_is_excluded(path.name, exclude_tokens))
    else:
        subject_dirs = [root / subject_id for subject_id in requested_subjects]

    levels = _normalize_tag_folder_levels(tag_folder_levels)
    records: list[BatchImageRecord] = []
    for subject_dir in subject_dirs:
        if not subject_dir.is_dir() or _name_is_excluded(subject_dir.name, exclude_tokens):
            continue
        folder_chains = _iter_tag_folder_chains(subject_dir, levels, exclude_tokens)
        for folder_chain in folder_chains:
            scan_dir = folder_chain[-1] if folder_chain else subject_dir
            image_paths = _collect_batch_image_paths(
                scan_dir,
                image_patterns,
                exclude_tokens,
                collapse_ome_multifile_series=collapse_ome_multifile_series,
                verbose=verbose)
            output_scope_dir = _output_scope_for_chain(subject_dir, folder_chain)
            tag_folders = tuple(path.name for path in folder_chain)
            for image_path in image_paths:
                records.append(
                    BatchImageRecord(
                        subject_id=subject_dir.name,
                        tag_folders=tag_folders,
                        image_path=image_path,
                        scan_dir=scan_dir,
                        output_scope_dir=output_scope_dir))
            for folder_stack_tag, folder_stack_paths in _collect_batch_folder_stack_groups(
                    scan_dir,
                    folder_stack_tags,
                    exclude_tokens):
                records.append(
                    BatchImageRecord(
                        subject_id=subject_dir.name,
                        tag_folders=tag_folders,
                        image_path=folder_stack_paths[0],
                        scan_dir=scan_dir,
                        output_scope_dir=output_scope_dir,
                        input_kind="folder_stack",
                        folder_stack_tag=folder_stack_tag,
                        folder_stack_paths=folder_stack_paths))
    return records

# OMIO's legacy BIDS-like batch converter function:
def bids_batch_convert(
    fname: str, # must be a directory
    sub: str,   # e.g. "ID" (subject folder detection)
    exp: str,   # e.g. "TP000" (experiment folder detection)
    exp_match_mode: str = "startswith",      # "startswith" | "exact" | "regex"
    tagfolder: str | None = None,            # e.g. "TAG_" (if set: only tagged folders inside exp)
    merge_multiple_files_in_folder: bool = False,
    merge_tagfolders: bool = False,          # if tagfolder is not None: merge TAGFOLDER_01..N into one output
    merge_along_axis: str = "T",
    collapse_ome_multifile_series: bool = True,
    zeropadding: bool = True,
    zarr_store: str | None = None,
    reuse_disk_cache: bool = False,
    recursive: bool = False,
    physicalsize_xyz=None,
    pixelunit: str = "micron",
    compression_level: int = 3,
    relative_path: str | None = "omio_converted",
    overwrite: bool = False,
    cleanup_cache: bool = True,
    return_fnames: bool = False,
    verbose: bool = True):
    """
    Batch converter for a BIDS-like directory tree.

    DEPRECATED since OMIO v0.3.0: use ``bids_batch_process`` instead.

    This legacy converter is kept as ``bids_batch_convert`` for projects
    that rely on the pre-v0.3.0 batch-conversion API. The new
    ``bids_batch_process`` provides flexible subject discovery, arbitrary
    folder-token levels, file-pattern filtering, skip-if-already-converted
    behavior, persistent run reports, and structured error reports.

    This function traverses a project root folder and converts image files found in a
    subject and experiment hierarchy into OME-TIFF using OMIO’s reader and writer.
    It supports two main discovery modes: direct conversion of image files located
    inside experiment folders, or conversion and optional merging of tagged
    subfolders (folder-stacks) inside experiment folders.
    
    **Abstract expected folder scheme:**
    The converter expects a project root that contains subject folders, which in turn
    contain experiment folders. Depending on whether `tagfolder` is provided, an
    experiment folder either contains image files directly, or contains multiple
    tagfolders which contain the image files.

    The schematic below uses ``<...>`` as placeholders for your chosen naming policy::

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
    

    **Folder discovery and selection:**
    The input ``fname`` must be a directory and is treated as the project root.

    Subject detection:
    
    * Every immediate subdirectory of ``fname`` whose name starts with ``sub`` is treated
      as a subject folder. No additional validation is performed.

    Experiment detection:
    
    * Within each subject folder, every immediate subdirectory whose name matches ``exp``
      under ``exp_match_mode`` is treated as an experiment folder.
      Matching modes are:
      
      * ``"startswith"``: folder name starts with ``exp``
      * ``"exact"``: folder name equals ``exp``
      * ``"regex"``: ``re.match(exp, foldername)`` succeeds

    **Conversion behavior inside each experiment folder:**
    Two mutually exclusive modes exist depending on `tagfolder`.

    Mode A: tagfolder is None (direct file conversion):
    
    * The converter processes image files located directly in the experiment folder.
    * If ``merge_multiple_files_in_folder=False``, every supported image file is
      converted to its own OME-TIFF output.
    * If ``merge_multiple_files_in_folder=True``, all supported image files in the
      experiment folder are read and concatenated along ``merge_along_axis`` (with
      optional ``zeropadding`` on non-merge axes) into one merged output.
    
    Mode B: tagfolder is not None (tagged folder stacks):
    
    * Direct image files in the experiment folder are ignored.
    * The converter searches for tagfolders inside the experiment folder whose name
      starts with ``tagfolder`` (for example ``"TAG_"``).
    * If ``merge_tagfolders=False`` (default), each tagfolder is converted separately
      and produces its own OME-TIFF output.
    * If ``merge_tagfolders=True``, all tagfolders are read and merged into a single
      output by reusing OMIO’s folder-stack logic. To keep output naming stable and
      collision-free when provenance-driven naming is used, a synthetic provenance
      name is injected into ``metadata["Annotations"]["original_filename"]``.

    **Input path semantics:**
    Only directory input is accepted:
    
    * ``fname`` must be an existing directory and is treated as the project root.
    * All outputs are written within the experiment scope determined by traversal.

    **Output placement and naming:**
    Output placement follows OMIO’s writer conventions via ``imconvert()`` and
    ``imwrite()``:

    * If ``relative_path`` is not None, outputs are written into a subfolder named
      ``relative_path`` under the relevant experiment folder (or under the experiment
      folder when writing a merged tagfolder product).
    * If ``relative_path`` is None, outputs are written directly into the experiment
      folder.
    * Per-stack output basenames are preferably derived from metadata provenance via
      ``Annotations["original_filename"]`` when present. Otherwise, a fallback basename
      is derived from the corresponding folder name.
    * If ``overwrite=False``, name collisions are resolved by appending an incrementing
      suffix to the output filename.

    **Merging semantics:**

    * ``merge_along_axis`` must be one of {"T","Z","C"}.
    * In merge operations, the merge axis segments are concatenated in discovery order.
    * If ``zeropadding=True``, non-merge axes may differ between inputs and will be
      padded with zeros to the maximum size across inputs before concatenation.
    * If ``zeropadding=False``, non-merge axes must match exactly or the merge is aborted.

    **Zarr and cache handling:**

    * ``zarr_store`` controls whether intermediate data are represented as NumPy in RAM
      or as Zarr arrays ("memory" or "disk") during reading and merging.
    * If ``reuse_disk_cache=True`` together with ``zarr_store="disk"``, OMIO may reuse
      an already existing validated disk cache instead of rebuilding it from the
      original image file.
    * If ``cleanup_cache=True`` and ``zarr_store`` is not None, the function removes the
      per-input `.omio_cache` artifacts created during conversion once outputs are written.

    Parameters
    ----------
    fname : str
        Project root directory (must exist).
    sub : str
        Prefix used to detect subject folders at the project root level.
    exp : str
        Pattern used to detect experiment folders within each subject.
    exp_match_mode : {"startswith","exact","regex"}
        Matching strategy for experiment folder selection.
    tagfolder : str or None
        If None, convert direct files in experiment folders. If set, only process
        tagged subfolders inside experiment folders whose names start with `tagfolder`.
    merge_multiple_files_in_folder : bool
        If tagfolder is None, optionally merge all image files in an experiment folder
        into a single output.
    merge_tagfolders : bool
        If tagfolder is set, optionally merge all detected tagfolders into a single output.
    merge_along_axis : {"T","Z","C"}
        Axis along which merges are performed.
    collapse_ome_multifile_series : bool
        If True, detect and collapse OME multifile series during reading to avoid
        duplicate loading. 
    zeropadding : bool
        If True, allow mismatched non-merge axes by padding with zeros before merging.
    zarr_store : {None,"memory","disk"}
        Intermediate representation for reading and merging.
    recursive : bool
        Passed through to the underlying folder readers for file discovery.
    physicalsize_xyz : tuple or None
        Optional override for physical voxel sizes.
    pixelunit : str
        Unit string for pixel size fields (default "micron").
    compression_level : int
        zlib compression level for OME-TIFF writing.
    relative_path : str or None
        Subfolder name for outputs under experiment folders. Default "omio_converted".
    overwrite : bool
        If True, existing output files may be overwritten. Otherwise, collision-safe
        suffixing is used.
    cleanup_cache : bool
        If True, remove `.omio_cache` artifacts created during conversion.
    return_fnames : bool
        If True, return a list of all written output filenames.
    verbose : bool
        If True, print progress and diagnostic messages.

    Returns
    -------
    list[str] or None
        If `return_fnames` is True, returns a list of written OME-TIFF file paths.
        Otherwise returns None. The list may be empty if nothing matched or all
        conversions failed.

    Raises
    ------
    ValueError
        If `fname` is not an existing directory, or if `merge_along_axis` is not one
        of {"T", "Z", "C"}.
    """
    if fname is None or not os.path.isdir(str(fname)):
        raise ValueError(f"bids_batch_convert: fname must be an existing directory. Got: {fname!r}\n"
                         "Conversion aborted.")

    if merge_along_axis not in _ALLOWED_MERGE_AXES:
        raise ValueError(
            f"bids_batch_convert: merge_along_axis must be one of {sorted(_ALLOWED_MERGE_AXES)}.\n"
            f"Got: {merge_along_axis!r}\n"
            "Conversion aborted.")

    project = str(fname)
    written_all = []

    # subject folders: startswith(sub) only; OMIO policy: OMIO will treat all folders
    # found here as subjects; thus, if the user is messy with their folder naming,
    # they may get unexpected results.
    subs = []
    subjects_list = []
    for d in sorted(os.listdir(project)):
        full = os.path.join(project, d)
        if os.path.isdir(full) and d.startswith(sub):
            subs.append(full)
            subjects_list.append(d)
    if verbose:
        print(f"OMIO batch processor received BIDS project named={os.path.basename(project)!r}")
        print(f"in given root path={os.path.dirname(project)!r}.")
        print(f"Detected subjects with provided subject tag={sub!r} are:")
        for s in subjects_list:
            print(f"   {s}")
        print(f"⟶ {len(subs)} subject(s)")
        print(f"Will now look for experiment folders matching {exp!r} with mode={exp_match_mode!r} inside each subject.")
        
    if not subs:
        warnings.warn(f"[OMIO batch] No subject folders found in {project!r} starting with {sub!r}.")
        if return_fnames:
            return written_all

    # loop over subjects:
    for sub_path in subs:
        # sub_path = subs[0] # for testing
        sub_name = os.path.basename(sub_path)
        if verbose:
            print(f"\nBatch processing subject {sub_name}...")

        # experiment folders inside subject:
        exp_folders = []
        for d in sorted(os.listdir(sub_path)):
            full = os.path.join(sub_path, d)
            if not os.path.isdir(full):
                if verbose:
                    print(f"  Not a directory: {full!r}. Skipping.")
                continue
            if _match_name(d, exp, exp_match_mode):
                exp_folders.append(full)

        if verbose:
            print(f"  {len(exp_folders)} matched experiment folder(s) with exp-tag {exp!r} found with mode={exp_match_mode!r}:")
            for ef in exp_folders:
                print(f"    {os.path.basename(ef)!r}")

        if not exp_folders:
            if verbose:
                print(f"  No exp folders matched {exp!r} with mode={exp_match_mode!r}. Skipping subject.")
            continue
        
        # loop over experiments:
        for exp_path in exp_folders:
            # exp_path = exp_folders[0]  # for testing
            exp_name = os.path.basename(exp_path)
            if verbose:
                print(f"  Processing '{exp_name}' exp folder...\n")

            # default relative path per case:
            rel_default = relative_path

            # -------------------------
            # Case A: no tagfolder -> direct files in exp_path
            # -------------------------
            if tagfolder is None:
                try:
                    fnames_written = imconvert(
                        fname=exp_path,
                        zarr_store=zarr_store,
                        reuse_disk_cache=reuse_disk_cache,
                        recursive=recursive,
                        folder_stacks=False,
                        merge_folder_stacks=False,
                        merge_multiple_files_in_folder=merge_multiple_files_in_folder,
                        merge_along_axis=merge_along_axis,
                        zeropadding=zeropadding,
                        physicalsize_xyz=physicalsize_xyz,
                        pixelunit=pixelunit,
                        compression_level=compression_level,
                        relative_path=rel_default,
                        overwrite=overwrite,
                        return_fnames=True,
                        cleanup_cache=cleanup_cache,
                        verbose=verbose)
                    if verbose:
                        print("\n")
                    if isinstance(fnames_written, list):
                        written_all.extend(fnames_written)
                except Exception as e:
                    if verbose:
                        print(f"    Conversion failed (direct files). Are there any image files in {exp_path!r}?\n"
                          f"    Or did you forget to set tagfolder=?\n"
                          f"    Error: {type(e).__name__}: {e}")
                continue

            # -------------------------
            # Case B: tagfolder set -> only tagged folders inside exp_path
            # -------------------------
            tagfolders = []
            for d in sorted(os.listdir(exp_path)):
                full = os.path.join(exp_path, d)
                if os.path.isdir(full) and d.startswith(tagfolder):
                    tagfolders.append(full)

            if not tagfolders:
                if verbose:
                    print(f"    tagfolder={tagfolder!r} requested, but no tagfolders found. Skipping exp.")
                continue

            if verbose:
                print(f"    found {len(tagfolders)} tagfolder(s) starting with {tagfolder!r}")
            
            rel_tag = relative_path

            # -------------------------
            # B1: default = each tagfolder gets its own output
            # -------------------------
            if not merge_tagfolders:
                for tf in tagfolders:
                    tf_name = os.path.basename(tf)
                    if verbose:
                        print(f"      {tf_name}: converting tagfolder...\n")

                    try:
                        fnames_written = imconvert(
                            fname=tf,
                            zarr_store=zarr_store,
                            reuse_disk_cache=reuse_disk_cache,
                            recursive=recursive,
                            folder_stacks=False,  # ⟵ important: we are already in a tagfolder!
                            merge_folder_stacks=False,
                            merge_multiple_files_in_folder=merge_multiple_files_in_folder,
                            merge_along_axis=merge_along_axis,
                            zeropadding=zeropadding,
                            physicalsize_xyz=physicalsize_xyz,
                            pixelunit=pixelunit,
                            compression_level=compression_level,
                            relative_path=rel_tag,
                            overwrite=overwrite,
                            return_fnames=True,
                            cleanup_cache=cleanup_cache)
                        if verbose:
                            print("\n")
                        if isinstance(fnames_written, list):
                            written_all.extend(fnames_written)
                    except Exception as e:
                        if verbose:
                            print(f"      conversion failed (tagfolder).\n"
                                  f"      Error: {type(e).__name__}: {e}")
                continue

            # -------------------------
            # B2: merge_tagfolders=True -> merge ALL tagfolders into ONE output
            #     Writer uses original_filename; for a merged product we inject a synthetic
            #     provenance name to avoid collisions and make output self-describing.
            # -------------------------
            try:
                # Read and merge by reusing my imread TAG-folder logic:
                merged_img, merged_md = imread(
                    fname=tagfolders[0],         # imread expects one of the tagfolders; it auto-detects the tag
                    zarr_store=zarr_store,
                    return_list=False,
                    recursive=recursive,
                    folder_stacks=True,
                    merge_folder_stacks=True,    # triggers reading of all tagfolders and merging
                    merge_multiple_files_in_folder=False,
                    merge_along_axis=merge_along_axis,
                    collapse_ome_multifile_series=collapse_ome_multifile_series,
                    zeropadding=zeropadding,
                    physicalsize_xyz=physicalsize_xyz,
                    pixelunit=pixelunit,
                    verbose=verbose)
                if verbose:
                    print("\n")

                if merged_img is None or merged_md is None:
                    if verbose:
                        print(f"    {exp_name}: merge_tagfolders produced None. Skipping.")
                    continue

                # Inject synthetic provenance name so writer can stay "original_filename-driven":
                # This avoids depending on fname basename or exp folder name.
                merged_md = dict(merged_md)
                ann = merged_md.get("Annotations", {})
                if not isinstance(ann, dict):
                    ann = {}
                ann = dict(ann)
                ann["original_filename"] = f"{sub_name}_{exp_name}_{tagfolder}merged.ome.tif"
                merged_md["Annotations"] = ann

                # Write merged output at exp level (not inside a tagfolder).
                # We call writer with fname=exp_path to place output in exp scope.
                fnames_written = imwrite(
                    fname=exp_path,
                    images=merged_img,
                    metadatas=merged_md,
                    compression_level=compression_level,
                    relative_path=relative_path if relative_path is not None else "merged",
                    overwrite=overwrite,
                    return_fnames=True,
                    verbose=verbose,
                    indicate_merged_files=True)
                if isinstance(fnames_written, list):
                    written_all.extend(fnames_written)

                if cleanup_cache and zarr_store is not None:
                    cleanup_omio_cache(exp_path, full_cleanup=False, verbose=verbose)

            except Exception as e:
                if verbose:
                    print(f"    {exp_name}: conversion failed (merge_tagfolders). "
                      f"    Error: {type(e).__name__}: {e}")

    if verbose:
        print(f"\nOMIO batch processing done. Written {len(written_all)} file(s).")
        for f in written_all:
            print(f"  {f}")

    if return_fnames:
        return written_all

def _default_batch_process(image,
                           metadata: dict,
                           *,
                           record: BatchImageRecord,
                           processing_options: dict[str, Any]) -> tuple[Any, dict, dict[str, Any]]:
    """Default OMIO batch processing step: pass image and metadata through."""

    return image, metadata, {}

def _load_batch_folder_stack_record(record: BatchImageRecord,
                                    *,
                                    load_options: dict[str, Any],
                                    merge_along_axis: str,
                                    zeropadding: bool,
                                    verbose: bool) -> tuple[Any, dict]:
    """Load and merge one batch-discovered tagged folder-stack record."""

    images = []
    metadatas = []
    for stack_folder in record.folder_stack_paths:
        first_file = _first_image_file_in_folder(
            str(stack_folder),
            allowed_ext=_BATCH_FOLDER_STACK_ALLOWED_EXTENSIONS)
        if first_file is None:
            if verbose:
                print(f"    No valid image file found in folder stack: {stack_folder!r}. Skipping.")
            continue
        image, metadata = _dispatch_read_file(
            first_file,
            zarr_store=load_options.get("zarr_store"),
            zarr_store_path=load_options.get("zarr_store_path"),
            return_list=False,
            physicalsize_xyz=load_options.get("physicalsize_xyz"),
            pixelunit=load_options.get("pixelunit", "micron"),
            reuse_disk_cache=load_options.get("reuse_disk_cache", False),
            on_error=load_options.get("on_error", "raise"),
            verbose=load_options.get("verbose", verbose))
        if image is None or metadata is None:
            if verbose:
                print(f"    Reader returned None for folder stack file: {first_file!r}. Skipping.")
            continue
        metadata = OME_metadata_checkup(metadata, verbose=load_options.get("verbose", verbose))
        images.append(image)
        metadatas.append(metadata)

    if not images:
        return None, {}

    return _merge_folderstacks_with_padding(
        images,
        metadatas,
        merge_along_axis=merge_along_axis,
        zarr_store=load_options.get("zarr_store"),
        zeropadding=zeropadding,
        verbose=load_options.get("verbose", verbose))

def _normalize_process_result(result) -> tuple[Any, dict, dict[str, Any]]:
    """Normalize flexible process callable return values."""

    if not isinstance(result, tuple):
        raise TypeError("process_func must return (image, metadata) or (image, metadata, info).")
    if len(result) == 2:
        image, metadata = result
        return image, metadata, {}
    if len(result) == 3:
        image, metadata, info = result
        info = info if isinstance(info, dict) else {"details": info}
        return image, metadata, info
    raise TypeError("process_func must return (image, metadata) or (image, metadata, info).")

def _batch_callable_name(func: Callable[..., Any] | None) -> str:
    """Return a compact user-facing name for one batch callable."""

    if func is None:
        return ""
    name = getattr(func, "__name__", "")
    if name:
        return str(name)
    return type(func).__name__

def _compact_batch_option_value(value: Any) -> str:
    """Return a compact string representation for one report option value."""

    if isinstance(value, str):
        return repr(value)
    return repr(value)

def _format_batch_processing_options(options: dict[str, Any]) -> str:
    """Return processing options as a compact run-report detail string."""

    if not options:
        return ""
    parts = [f"{key}={_compact_batch_option_value(value)}" for key, value in sorted(options.items())]
    return "processing_options: " + ", ".join(parts)

def _merge_batch_process_details(process_info: dict[str, Any],
                                 processing_options: dict[str, Any],
                                 *,
                                 custom_process: bool) -> str | None:
    """Combine custom processing details and options for run reports."""

    details = process_info.get("details")
    option_details = _format_batch_processing_options(processing_options) if custom_process else ""
    if details and option_details:
        return f"{details}; {option_details}"
    if details:
        return str(details)
    if option_details:
        return option_details
    return None

def _default_batch_save(*,
                        image,
                        metadata: dict,
                        record: BatchImageRecord,
                        output_dir: Path,
                        output_path: Path,
                        save_options: dict[str, Any]) -> Path:
    """Default OMIO batch save step: write one OME-TIFF via :func:`imwrite`."""

    options = dict(save_options)
    compression_level = options.pop("compression_level", 3)
    overwrite = options.pop("overwrite", False)
    verbose = options.pop("verbose", True)
    metadata_out = _metadata_for_batch_output(metadata, output_path)
    fnames = imwrite(
        fname=str(output_dir),
        images=image,
        metadatas=metadata_out,
        compression_level=compression_level,
        relative_path=None,
        overwrite=overwrite,
        return_fnames=True,
        verbose=verbose,
        **options)
    if isinstance(fnames, list) and fnames:
        return Path(fnames[0])
    if isinstance(fnames, (str, Path)):
        return Path(fnames)
    return output_path

def _normalize_saved_path(saved, output_path: Path) -> Path:
    """Normalize save callable output into one path."""

    if saved is None:
        return output_path
    if isinstance(saved, (str, Path)):
        return Path(saved)
    if isinstance(saved, (list, tuple)) and saved:
        return Path(saved[0])
    raise TypeError("save_func must return a path, a non-empty list/tuple of paths, or None.")

def bids_batch_process(project_root: str | Path,
                       *,
                       subject_ids: Iterable[str | Path] | None = None,
                       subject_prefix: str = "ID",
                       tag_folder_levels: Sequence[Iterable[str | Path] | None] | None = None,
                       image_patterns: str | Sequence[str] | None = None,
                       exclude_name_contains: str | Sequence[str] | None = ("Preview",),
                       folder_stacks: str | Path | Sequence[str | Path] | None = None,
                       merge_along_axis: str = "T",
                       zeropadding: bool = True,
                       collapse_ome_multifile_series: bool = True,
                       output_folder_name: str | Path = "omio_converted",
                       output_suffix: str = "_omio_converted",
                       skip_processed: bool = True,
                       load_func: Callable[..., Any] | None = None,
                       process_func: Callable[..., Any] | None = None,
                       save_func: Callable[..., Any] | None = None,
                       load_options: dict[str, Any] | None = None,
                       processing_options: dict[str, Any] | None = None,
                       save_options: dict[str, Any] | None = None,
                       method_name: str = "omio_batch_process",
                       raw_template_metadata: dict[str, Any] | None = None,
                       write_error_reports: bool = True,
                       write_run_report: bool = True,
                       run_report_name: str = "omio_batch_run_report",
                       run_report_format: str | Sequence[str] = ("yaml", "txt"),
                       continue_on_error: bool = True,
                       verbose: bool = True) -> BatchProcessResult:
    """
    Flexibly process a BIDS-like microscopy image batch with robust reporting.

    This processor discovers image files below a project root, optionally follows
    arbitrary folder-token levels, processes every discovered image independently,
    and writes persistent run/error reports. By default, OMIO loads each image via
    ``imread``, performs an identity processing step, and writes an OME-TIFF via
    ``imwrite``. Advanced callers can replace any of the three stages with custom
    callables.

    Parameters
    ----------
    project_root : str or pathlib.Path
        Root folder containing subject folders.
    subject_ids : iterable of str or None, optional
        Explicit subject folders to process. If None, subjects are discovered by
        ``subject_prefix``.
    subject_prefix : str, optional
        Prefix used for automatic subject discovery. Default is ``"ID"``.
    tag_folder_levels : sequence, optional
        Flexible folder-token levels below each subject. Empty levels are skipped.
        If None, OMIO defaults to one experiment level matching ``"TP"``.
    image_patterns : str, sequence[str], or None, optional
        Glob pattern(s) used to find images. If None, OMIO uses common microscopy
        image patterns.
    exclude_name_contains : str, sequence[str], or None, optional
        Tokens used to exclude files or folders by name.
    folder_stacks : str, sequence[str], or None, optional
        Optional folder-stack tag(s) searched below each final folder-token level.
        If set, OMIO adds one batch input per matching tag group and reads it
        through OMIO's folder-stack merge path.
    merge_along_axis : {"T", "Z", "C"}, optional
        Axis used when folder-stack records are merged. Default is ``"T"``.
    zeropadding : bool, optional
        If True, allow folder-stack members with mismatched non-merge axes by zero
        padding to maxima. Default is True.
    collapse_ome_multifile_series : bool, optional
        If True, detect OME multi-file TIFF series during batch discovery and keep
        only one representative file per series. The representative is then read
        by ``om.imread``, which loads the full OME multi-file series. Default is True.
    output_folder_name : str or pathlib.Path, optional
        Output folder for default OMIO saving. Relative paths are resolved below
        the final discovered image folder. Absolute paths are used directly.
    output_suffix : str, optional
        Suffix used for default output filenames before ``.ome.tif``.
    skip_processed : bool, optional
        If True, skip files whose expected output already exists and record them
        as converted/already processed in the run report.
    load_func, process_func, save_func : callable or None, optional
        Optional custom callables for the load, process, and save stages.
    load_options, processing_options, save_options : dict or None, optional
        Keyword options forwarded to the corresponding stage.
    method_name : str, optional
        Method label written to successful run-report entries.
    raw_template_metadata : dict or None, optional
        Metadata defaults written into RAW-related error reports for later YAML
        sidecar generation.
    write_error_reports : bool, optional
        If True, write per-folder and root-level error reports when failures occur.
    write_run_report : bool, optional
        If True, update persistent root-level run reports.
    run_report_name : str, optional
        Base filename for run reports. OMIO can write ``.yaml`` and ``.txt``.
    run_report_format : str or sequence[str], optional
        Requested run-report formats, using ``"yaml"`` and/or ``"txt"``.
    continue_on_error : bool, optional
        If True, errors are recorded and the batch continues. If False, errors are
        re-raised immediately.
    verbose : bool, optional
        If True, print compact progress information.

    Returns
    -------
    BatchProcessResult
        Discovered, processed, skipped, and failed records plus report paths.
    """

    root = Path(project_root)
    records = discover_bids_like_batch_images(
        root,
        subject_ids=subject_ids,
        subject_prefix=subject_prefix,
        tag_folder_levels=tag_folder_levels,
        image_patterns=image_patterns,
        exclude_name_contains=exclude_name_contains,
        folder_stacks=folder_stacks,
        collapse_ome_multifile_series=collapse_ome_multifile_series,
        verbose=verbose)

    custom_loader = load_func is not None
    loader = load_func or imread
    processor = process_func or _default_batch_process
    saver = save_func or _default_batch_save
    base_load_options = dict(load_options or {})
    base_processing_options = dict(processing_options or {})
    base_save_options = dict(save_options or {})
    base_raw_template_metadata = dict(raw_template_metadata or DEFAULT_BATCH_RAW_TEMPLATE_METADATA)
    custom_process = process_func is not None
    effective_method_name = (
        _batch_callable_name(process_func)
        if custom_process and method_name == "omio_batch_process"
        else method_name)
    timestamp = _batch_timestamp()

    processed: list[BatchProcessedRecord] = []
    skipped: list[BatchSkippedRecord] = []
    failed: list[BatchErrorRecord] = []
    local_error_paths: set[Path] = set()
    run_report_events: list[dict[str, Any]] = []

    if verbose:
        print(f"OMIO flexible batch discovered {len(records)} image file(s) in {root}.")

    def build_error(record: BatchImageRecord,
                    *,
                    stage: str,
                    exc: BaseException | None = None,
                    message: str | None = None,
                    output_dir: Path | None = None,
                    output_path: Path | None = None,
                    extra: dict[str, Any] | None = None) -> BatchErrorRecord:
        exception_type = type(exc).__name__ if exc is not None else "BatchError"
        error_message = str(exc) if exc is not None else str(message or "")
        return BatchErrorRecord(
            timestamp=timestamp,
            input_path=record.image_path,
            relative_path=_relative_report_path(record.image_path, root),
            stage=stage,
            exception_type=exception_type,
            message=error_message,
            subject_id=record.subject_id,
            tag_folders=record.tag_folders,
            output_dir=output_dir,
            output_path=output_path,
            extra=dict(extra or {}))

    for record in records:
        output_dir = _resolve_batch_output_dir(record.output_scope_dir, output_folder_name)
        output_path = _expected_batch_output_path(output_dir, _batch_output_name_source(record), output_suffix)

        if skip_processed and output_path.exists():
            skipped_record = BatchSkippedRecord(
                input_path=record.image_path,
                reason=f"Processed output already exists: {output_path}",
                subject_id=record.subject_id,
                tag_folders=record.tag_folders,
                stage="already_processed",
                output_path=output_path)
            skipped.append(skipped_record)
            run_report_events.append(
                {
                    "record": record,
                    "status": "already_processed",
                    "output_path": output_path})
            if verbose:
                print(f"  skipped/already converted: {record.image_path}")
            continue

        if verbose:
            chain = "/".join(record.tag_folders) if record.tag_folders else "subject_root"
            print(f"  processing {record.subject_id}/{chain}/{record.image_path.name}", flush=True)

        try:
            load_options_for_file = dict(base_load_options)
            if record.input_kind == "folder_stack":
                load_options_for_file.setdefault("folder_stacks", True)
                load_options_for_file.setdefault("merge_folder_stacks", True)
                load_options_for_file.setdefault("merge_along_axis", merge_along_axis)
                load_options_for_file.setdefault("zeropadding", zeropadding)
            if record.input_kind == "folder_stack" and not custom_loader:
                load_result = _load_batch_folder_stack_record(
                    record,
                    load_options=load_options_for_file,
                    merge_along_axis=merge_along_axis,
                    zeropadding=zeropadding,
                    verbose=verbose)
            else:
                load_result = loader(record.image_path, **load_options_for_file)
        except Exception as exc:
            if not continue_on_error:
                raise
            error = build_error(record, stage="load", exc=exc, output_dir=output_dir, output_path=output_path)
            failed.append(error)
            run_report_events.append({"record": record, "status": "failed", "error": error})
            if write_error_reports:
                local_error_paths.add(
                    _write_local_batch_error_report(
                        record.image_path.parent,
                        timestamp=timestamp,
                        error=error,
                        raw_template_metadata=base_raw_template_metadata))
            if verbose:
                print(f"  failed [load]: {type(exc).__name__}: {exc}")
            continue

        if load_result is None:
            image, metadata = None, None
        elif isinstance(load_result, tuple) and len(load_result) >= 2:
            image, metadata = load_result[0], load_result[1]
        else:
            raise TypeError("load_func must return (image, metadata) or None.")

        if image is None or metadata is None:
            error = build_error(
                record,
                stage="load",
                message="Load stage returned None or (None, None).",
                output_dir=output_dir,
                output_path=output_path)
            failed.append(error)
            run_report_events.append({"record": record, "status": "failed", "error": error})
            if write_error_reports:
                local_error_paths.add(
                    _write_local_batch_error_report(
                        record.image_path.parent,
                        timestamp=timestamp,
                        error=error,
                        raw_template_metadata=base_raw_template_metadata))
            if verbose:
                print(f"  failed [load]: {error.message}")
            continue

        try:
            process_result = processor(
                image=image,
                metadata=metadata,
                record=record,
                processing_options=base_processing_options)
            image_out, metadata_out, process_info = _normalize_process_result(process_result)
            process_method_name = str(process_info.get("method_name", effective_method_name))
            process_details = _merge_batch_process_details(
                process_info,
                base_processing_options,
                custom_process=custom_process)
        except Exception as exc:
            if not continue_on_error:
                raise
            error = build_error(record, stage="process", exc=exc, output_dir=output_dir, output_path=output_path)
            failed.append(error)
            run_report_events.append({"record": record, "status": "failed", "error": error})
            if write_error_reports:
                local_error_paths.add(
                    _write_local_batch_error_report(
                        record.image_path.parent,
                        timestamp=timestamp,
                        error=error,
                        raw_template_metadata=base_raw_template_metadata))
            if verbose:
                print(f"  failed [process]: {type(exc).__name__}: {exc}")
            continue

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            save_options_for_file = dict(base_save_options)
            save_options_for_file.setdefault("verbose", verbose)
            saved = saver(
                image=image_out,
                metadata=metadata_out,
                record=record,
                output_dir=output_dir,
                output_path=output_path,
                save_options=save_options_for_file)
            written_path = _normalize_saved_path(saved, output_path)
        except Exception as exc:
            if not continue_on_error:
                raise
            error = build_error(record, stage="save", exc=exc, output_dir=output_dir, output_path=output_path)
            failed.append(error)
            run_report_events.append({"record": record, "status": "failed", "error": error})
            if write_error_reports:
                local_error_paths.add(
                    _write_local_batch_error_report(
                        record.image_path.parent,
                        timestamp=timestamp,
                        error=error,
                        raw_template_metadata=base_raw_template_metadata))
            if verbose:
                print(f"  failed [save]: {type(exc).__name__}: {exc}")
            continue

        processed.append(
            BatchProcessedRecord(
                input_path=record.image_path,
                output_path=written_path,
                subject_id=record.subject_id,
                tag_folders=record.tag_folders))
        run_report_events.append(
            {
                "record": record,
                "status": "processed",
                "output_path": written_path,
                "method_name": process_method_name,
                "details": process_details})
        if verbose:
            print(f"  processed; wrote: {written_path}")

    error_report_path = (
        _write_root_batch_error_report(
            root,
            timestamp=timestamp,
            errors=failed,
            raw_template_metadata=base_raw_template_metadata)
        if write_error_reports
        else None)

    run_report_yaml_path = None
    run_report_txt_path = None
    if write_run_report:
        requested_formats = (
            {str(run_report_format).lower()}
            if isinstance(run_report_format, str)
            else {str(item).lower() for item in run_report_format})
        if not requested_formats <= {"yaml", "txt"}:
            raise ValueError(f"run_report_format must contain only 'yaml' and/or 'txt'. Got {run_report_format!r}.")
        run_report_yaml_path, run_report_txt_path = _run_report_paths(root, run_report_name)
        payload = _load_run_report(run_report_yaml_path, root)
        payload["project_root"] = str(root)
        payload["last_updated"] = timestamp
        for event in run_report_events:
            _append_run_report_entry(
                payload,
                event["record"],
                root=root,
                timestamp=timestamp,
                status=event["status"],
                method_name=event.get("method_name", effective_method_name),
                output_path=event.get("output_path"),
                error=event.get("error"),
                details=event.get("details"))
        if "yaml" in requested_formats:
            _write_run_report_yaml(run_report_yaml_path, payload)
        else:
            run_report_yaml_path = None
        if "txt" in requested_formats:
            _render_run_report_text(payload, run_report_txt_path)
        else:
            run_report_txt_path = None

    if verbose:
        print(
            f"OMIO flexible batch done: {len(processed)} processed, "
            f"{len(skipped)} skipped, {len(failed)} failed.")
        if skipped:
            print("Skipped image files:")
            for record in skipped:
                print(f"  {record.input_path} [{record.stage}]")
        if failed:
            print("Failed image files:")
            for error in failed:
                print(f"  {error.input_path} [{error.stage}] {error.exception_type}: {error.message}")
            if error_report_path is not None:
                print(f"Root error report written to: {error_report_path}")

    return BatchProcessResult(
        processed=tuple(processed),
        skipped=tuple(skipped),
        failed=tuple(failed),
        discovered=tuple(records),
        report_path=run_report_txt_path,
        run_report_yaml_path=run_report_yaml_path,
        error_report_path=error_report_path,
        local_error_report_paths=tuple(sorted(local_error_paths)))

def batch_create_thorlabs_raw_yaml_templates(project_root: str | Path,
                                             *,
                                             report_name: str | Path | None = None,
                                             raw_template_metadata: dict[str, Any] | None = None,
                                             overwrite_existing: bool = False,
                                             verbose: bool = True) -> BatchRawYamlTemplateResult:
    """
    Create Thorlabs RAW YAML sidecars from an OMIO or ZenReg batch error report.

    The root-level error report written by :func:`bids_batch_process` contains a
    valid Python dictionary named ``OMIO_BATCH_FAILED_RAW_FILES``. Users can edit
    the ``template_metadata`` blocks centrally in that report and then call this
    helper to distribute those values into per-RAW OMIO YAML sidecars.

    Parameters
    ----------
    project_root : str or pathlib.Path
        Root folder containing the batch error report.
    report_name : str, pathlib.Path, or None, optional
        Report filename or path. If None, OMIO uses the latest
        ``omio_batch_error_report_*.txt`` in ``project_root`` and falls back to
        ZenReg-style report names.
    raw_template_metadata : dict or None, optional
        Fallback metadata used for reports without per-file ``template_metadata``
        blocks.
    overwrite_existing : bool, optional
        If False, skip RAW files that already have a likely YAML/YML sidecar.
    verbose : bool, optional
        If True, print progress and skip reasons.

    Returns
    -------
    BatchRawYamlTemplateResult
        Per-RAW template creation records.
    """

    root = Path(project_root)
    if report_name is None:
        report_path = _find_latest_batch_error_report(root)
        if report_path is None:
            raise FileNotFoundError(f"No omio_batch_error_report_*.txt found in {root!s}.")
    else:
        report_path = Path(report_name)
        if not report_path.is_absolute():
            report_path = root / report_path
    if not report_path.exists():
        raise FileNotFoundError(f"Batch error report not found: {report_path}")

    fallback_metadata = dict(raw_template_metadata or DEFAULT_BATCH_RAW_TEMPLATE_METADATA)
    entries = _load_skipped_raw_entries_from_report(report_path, raw_template_metadata=fallback_metadata)
    records: list[BatchRawYamlTemplateRecord] = []
    for entry in entries:
        raw_path = Path(entry["path"])
        template_metadata = dict(entry.get("template_metadata", fallback_metadata))
        yaml_paths = _expected_raw_yaml_paths(raw_path)
        existing_yaml_paths = [path for path in yaml_paths if path.exists()]

        if not raw_path.exists():
            reason = "RAW file does not exist."
            records.append(
                BatchRawYamlTemplateRecord(
                    raw_path=raw_path,
                    yaml_path=None,
                    template_metadata=template_metadata,
                    status="missing",
                    reason=reason))
            if verbose:
                print(f"Skipping missing RAW file: {raw_path}")
            continue

        if existing_yaml_paths and not overwrite_existing:
            reason = "YAML/YML sidecar already exists."
            records.append(
                BatchRawYamlTemplateRecord(
                    raw_path=raw_path,
                    yaml_path=existing_yaml_paths[0],
                    template_metadata=template_metadata,
                    status="exists",
                    reason=reason))
            if verbose:
                existing_names = ", ".join(str(path) for path in existing_yaml_paths)
                print(f"Skipping existing YAML for {raw_path}: {existing_names}")
            continue

        if verbose:
            print(f"Creating OMIO YAML template for: {raw_path}")
        create_thorlabs_raw_yaml(str(raw_path), verbose=verbose, **template_metadata)
        created_yaml = next((path for path in yaml_paths if path.exists()), yaml_paths[0])
        records.append(
            BatchRawYamlTemplateRecord(
                raw_path=raw_path,
                yaml_path=created_yaml,
                template_metadata=template_metadata,
                status="created",
                reason=""))

    if verbose:
        print(
            f"OMIO RAW YAML template creation finished: "
            f"{len([record for record in records if record.status == 'created'])} created, "
            f"{len([record for record in records if record.status != 'created'])} skipped.")

    return BatchRawYamlTemplateResult(report_path=report_path, records=tuple(records))

# %% ALL
__all__ = [name for name in globals() if not name.startswith("__")]
# %% END
