"""
Lumen AI integration hooks for xarray data.

Provides the glue code needed to make Lumen AI work with xarray files:
- File extension detection and engine mapping
- Upload handler for the Lumen AI file upload UI
- CLI path resolution for `lumen-ai serve data.nc`
- Source code generation for the AIHandler template

NOTE: These hooks are designed to match Lumen AI's internal API
(lumen.ai.controls.base) based on reading Lumen's source code.
They have NOT been integration-tested against a running Lumen AI
instance, as the full coordinator integration does not exist yet.
Unit tests verify each function in isolation.

Usage in Lumen AI:
    # Register xarray upload handler
    from lumen_xarray.ai import register_xarray_handlers
    register_xarray_handlers(upload_controls)

    # CLI detection
    from lumen_xarray.ai import is_xarray_path, resolve_xarray_source
    if is_xarray_path("data.nc"):
        source = resolve_xarray_source("data.nc")
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from ._base import XARRAY_EXTENSIONS, _detect_engine
from .source import XArraySQLSource

# Extensions that Lumen AI should recognize as xarray-compatible
TABLE_EXTENSIONS_XARRAY = (
    "nc", "nc4", "netcdf", "zarr", "h5", "hdf5", "he5",
    "grib", "grib2", "grb", "grb2",
)


def is_xarray_path(path: str) -> bool:
    """
    Check if a file path or URL points to an xarray-compatible format.

    Handles both local paths and remote URLs (OpenDAP, cloud storage).
    """
    if "://" in path:
        # Remote URL — check extension in the URL path
        url_path = path.split("?")[0].split("#")[0]  # strip query/fragment
        return any(url_path.endswith(ext) for ext in XARRAY_EXTENSIONS)

    p = Path(path)
    return p.suffix.lower() in XARRAY_EXTENSIONS or (
        p.is_dir() and p.suffix == ".zarr"
    )


def resolve_xarray_source(
    path: str,
    chunks: dict | str = "auto",
    **kwargs,
) -> XArraySQLSource:
    """
    Create an XArraySQLSource from a file path or URL.

    Used by the CLI integration to handle `lumen-ai serve data.nc`.

    Parameters
    ----------
    path : str
        Path or URL to the xarray data file.
    chunks : dict or str
        Dask chunking specification.
    **kwargs
        Additional parameters for XArraySQLSource.

    Returns
    -------
    XArraySQLSource
    """
    engine = kwargs.pop("engine", None) or _detect_engine(path)
    return XArraySQLSource(
        uri=path,
        engine=engine,
        chunks=chunks,
        **kwargs,
    )


def handle_xarray_upload(
    file_bytes: bytes,
    filename: str,
    chunks: dict | str = "auto",
) -> dict[str, Any]:
    """
    Process an uploaded xarray file and create a source.

    Writes bytes to a temporary file (xarray needs filesystem access),
    creates an XArraySQLSource, and returns a result dict compatible
    with Lumen AI's upload flow.

    Parameters
    ----------
    file_bytes : bytes
        Raw file content.
    filename : str
        Original filename (used for extension detection).
    chunks : dict or str
        Dask chunking.

    Returns
    -------
    dict with keys: source, tables, metadata, message
    """
    suffix = Path(filename).suffix
    engine = _detect_engine(filename)

    # Write to a persistent temp file (xarray needs the file to stay open)
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(file_bytes)
        tmp.flush()
        tmp_path = tmp.name
    finally:
        tmp.close()

    source = XArraySQLSource(
        uri=tmp_path,
        engine=engine,
        chunks=chunks,
    )

    tables = source.get_tables()
    metadata = source.get_metadata()
    dim_info = source.get_dimension_info()

    return {
        "source": source,
        "tables": tables,
        "metadata": metadata,
        "dimension_info": dim_info,
        "temp_path": tmp_path,
        "message": (
            f"Loaded xarray dataset from '{filename}' with "
            f"{len(tables)} variable(s): {', '.join(tables)}. "
            f"You can now query this data using SQL."
        ),
    }


def build_xarray_source_code(table_path: str, var_name: str = "source") -> str:
    """
    Generate Python source code that creates an XArraySQLSource.

    Used by the AIHandler template to generate the application code
    for `lumen-ai serve data.nc`.

    Parameters
    ----------
    table_path : str
        Path to the xarray file.
    var_name : str
        Variable name for the source in generated code.

    Returns
    -------
    str
        Python source code string.
    """
    engine = _detect_engine(table_path)
    engine_str = f"'{engine}'" if engine else "None"

    return f"""\
from lumen_xarray.source import XArraySQLSource

{var_name} = XArraySQLSource(
    uri={table_path!r},
    engine={engine_str},
    chunks='auto',
)
"""


def register_xarray_handlers(upload_controls=None):
    """
    Register xarray file handlers with Lumen AI's upload system.

    This patches the upload controls to recognize xarray file extensions
    and route them to the xarray upload handler.

    Parameters
    ----------
    upload_controls : UploadControls or None
        If provided, registers the handler on this instance.
        If None, attempts to patch the global TABLE_EXTENSIONS.

    Returns
    -------
    bool
        True if registration succeeded, False if Lumen AI is not available.
    """
    try:
        from lumen.ai.controls.base import TABLE_EXTENSIONS
    except ImportError:
        return False

    # Add xarray extensions to the recognized table extensions
    new_extensions = tuple(
        ext for ext in TABLE_EXTENSIONS_XARRAY
        if ext not in TABLE_EXTENSIONS
    )
    if new_extensions:
        import lumen.ai.controls.base as controls_base
        controls_base.TABLE_EXTENSIONS = TABLE_EXTENSIONS + new_extensions

    # Register custom upload handler if controls instance provided
    if upload_controls is not None:
        for ext in TABLE_EXTENSIONS_XARRAY:
            upload_controls.upload_handlers[f".{ext}"] = _lumen_upload_handler

    return True


def _lumen_upload_handler(context, file_obj, alias, filename):
    """
    Upload handler callback compatible with Lumen AI's upload_handlers API.

    Parameters match the signature expected by UploadControls._process_files().
    """
    file_bytes = file_obj.read() if hasattr(file_obj, "read") else file_obj
    result = handle_xarray_upload(file_bytes, filename)
    return result["source"]


def get_extended_table_extensions() -> tuple:
    """
    Return the full tuple of table extensions including xarray formats.

    Can be used to replace TABLE_EXTENSIONS in Lumen AI config.
    """
    try:
        from lumen.ai.controls.base import TABLE_EXTENSIONS
        return TABLE_EXTENSIONS + tuple(
            ext for ext in TABLE_EXTENSIONS_XARRAY
            if ext not in TABLE_EXTENSIONS
        )
    except ImportError:
        return TABLE_EXTENSIONS_XARRAY
