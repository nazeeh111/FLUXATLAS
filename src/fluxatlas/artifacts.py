"""Portable fixed-grid evidence, without pickles or implicit unit conversions."""
from contextlib import contextmanager
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np

from .config import Experiment, GROUP_EDGES, NX, NY, X_EDGES, Y_EDGES

SHAPE = (2, NY, NX)
UNITS = "cm^-2 per source particle"
MAX_ARCHIVE_BYTES = 2_000_000
NORMALIZATION = "tracklength tally per source particle divided by cell volume in cm^3"
GRID = {"x_edges_cm": X_EDGES.tolist(), "y_edges_cm": Y_EDGES.tolist(),
        "energy_order": ["slow", "fast"], "spatial_order": ["y", "x"]}
# The released schema-1 adapter predates explicit spatial coordinates. Its exact
# source/version fingerprint binds the documented fixed grid and normalization.
LEGACY_GRID_FINGERPRINT = "8007b9edd5de4d12aed4e9fa4cea4cbf93ddf5889cd42a30a2dde5d9c44f3d26"


@contextmanager
def fresh_output(path):
    """Publish a complete file exclusively; a concurrent destination also wins."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Output already exists: {path}; choose a fresh destination")
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+b", dir=path.parent, prefix=".fluxatlas-", delete=False) as stream:
            name = stream.name
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.link(name, path)
    finally:
        if name is not None:
            os.unlink(name)


def _field(value, name):
    value = np.asarray(value)
    if value.shape != SHAPE or value.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a real numeric field with shape {SHAPE}")
    value = value.astype(np.float64)
    if not np.isfinite(value).all() or np.any(value < 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _metadata(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate metadata key: {key}")
            result[key] = value
        return result

    value = json.loads(text, object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError("Artifact metadata must be a JSON object")
    if value.get("schema_version") != 1 or isinstance(value["schema_version"], bool):
        raise ValueError("Unsupported artifact schema")
    if value.get("units") != UNITS or value.get("shape") != list(SHAPE):
        raise ValueError("Artifact units or fixed-grid shape do not match")
    if value.get("energy_edges_eV") != GROUP_EDGES:
        raise ValueError("Artifact energy-group order/edges do not match")
    config = value.get("config")
    if not isinstance(config, dict) or set(config) != set(Experiment.__dataclass_fields__):
        raise ValueError("Saved artifacts require the complete experiment configuration")
    config = Experiment(**config)
    value["config"] = config.to_dict()
    for key, expected in GRID.items():
        if key in value and value[key] != expected:
            raise ValueError(f"Artifact {key} does not match the fixed grid")
    return value


def _load_arrays(path, names):
    """Bound decompressed data before NumPy allocates archive members."""
    path = Path(path)
    with path.open("rb") as stream:
        content = stream.read(MAX_ARCHIVE_BYTES + 1)
    if len(content) > MAX_ARCHIVE_BYTES:
        raise ValueError("Field archive exceeds the 2 MB limit")
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            members = archive.infolist()
            expected = {name + ".npy" for name in names}
            if len(members) != len(expected) or {m.filename for m in members} != expected:
                raise ValueError("Field archive has missing, duplicate or unexpected arrays")
            if any(m.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED) for m in members):
                raise ValueError("Field archives support only stored or deflate compression")
            if sum(m.file_size for m in members) > MAX_ARCHIVE_BYTES:
                raise ValueError("Expanded field archive exceeds the 2 MB limit")
            for member in members:
                with archive.open(member) as stream:
                    version = np.lib.format.read_magic(stream)
                    readers = {(1, 0): np.lib.format.read_array_header_1_0,
                               (2, 0): np.lib.format.read_array_header_2_0}
                    if version not in readers:
                        raise ValueError("Unsupported NPY version in field archive")
                    shape, _, dtype = readers[version](stream)
                    metadata = member.filename == "metadata_json.npy"
                    if metadata:
                        valid = shape == () and dtype.kind == "U" and dtype.itemsize <= 65536
                    else:
                        valid = shape == SHAPE and dtype.kind in "fiu" and dtype.itemsize <= 8
                    if not valid or member.file_size - stream.tell() != math.prod(shape) * dtype.itemsize:
                        raise ValueError("Invalid array shape, type or byte count in field archive")
        with np.load(io.BytesIO(content), allow_pickle=False) as arrays:
            return {name: arrays[name] for name in names}, hashlib.sha256(content).hexdigest()
    except (zipfile.BadZipFile, EOFError) as error:
        raise ValueError("Invalid NPZ field archive") from error


def save_prediction(path, config, prediction, halfwidth, model_sha256, ood):
    prediction = _field(prediction, "prediction")
    halfwidth = _field(halfwidth, "halfwidth_90")
    if not isinstance(model_sha256, str) or len(model_sha256) != 64 or any(c not in "0123456789abcdef" for c in model_sha256):
        raise ValueError("Prediction requires the loaded model's SHA256")
    if bool(ood) != (config.family != "solid"):
        raise ValueError("Prediction geometry-domain declaration is inconsistent")
    metadata = {
        "schema_version": 1, "kind": "new_model_prediction",
        "config": config.to_dict(), "shape": list(SHAPE), "units": UNITS,
        "energy_edges_eV": GROUP_EDGES, **GRID,
        "model_sha256": model_sha256,
        "interval_level": .9, "out_of_domain": bool(ood),
        "interval_scope": "Finite-history reference estimates in the supported solid geometry family; not physical safety bounds.",
    }
    with fresh_output(path) as stream:
        np.savez_compressed(stream, prediction=prediction, halfwidth_90=halfwidth,
                            metadata_json=np.array(json.dumps(metadata, allow_nan=False, sort_keys=True)))
    return metadata


def load_prediction(path):
    arrays, digest = _load_arrays(path, ("prediction", "halfwidth_90", "metadata_json"))
    encoded = arrays["metadata_json"]
    if encoded.shape != () or encoded.dtype.kind != "U" or encoded.nbytes > 65536:
        raise ValueError("Prediction metadata must be a bounded Unicode scalar")
    metadata = _metadata(str(encoded.item()))
    if metadata.get("kind") != "new_model_prediction" or metadata.get("interval_level") != .9:
        raise ValueError("Unsupported prediction/interval contract")
    if metadata.get("x_edges_cm") != X_EDGES.tolist() or metadata.get("y_edges_cm") != Y_EDGES.tolist():
        raise ValueError("Prediction spatial grid does not match")
    model_digest = metadata.get("model_sha256", "")
    if not isinstance(model_digest, str) or len(model_digest) != 64 or any(c not in "0123456789abcdef" for c in model_digest):
        raise ValueError("Prediction requires a model SHA256")
    expected_ood = metadata["config"]["family"] != "solid"
    if type(metadata.get("out_of_domain")) is not bool or metadata["out_of_domain"] != expected_ood:
        raise ValueError("Prediction geometry-domain declaration is inconsistent")
    metadata["artifact_sha256"] = digest
    return (_field(arrays["prediction"], "prediction"),
            _field(arrays["halfwidth_90"], "halfwidth_90"), metadata)


def load_reference(path):
    path = Path(path)
    metadata_path = path.with_name("metadata.json")
    with metadata_path.open("rb") as stream:
        content = stream.read(65537)
    if len(content) > 65536:
        raise ValueError("Reference metadata exceeds the 64 KB limit")
    metadata = _metadata(content.decode("utf-8"))
    if metadata.get("solver") != "OpenMC":
        raise ValueError("Reference must declare its OpenMC origin")
    if metadata.get("normalization") != NORMALIZATION:
        raise ValueError("Reference volume/source normalization does not match")
    if not all(key in metadata for key in GRID):
        source = Path(__file__).parent
        current = hashlib.sha256((source / "transport.py").read_bytes()
                                 + (source / "config.py").read_bytes()
                                 + str(metadata.get("solver_version", "")).encode()).hexdigest()
        if metadata.get("physics_fingerprint") not in {LEGACY_GRID_FINGERPRINT, current}:
            raise ValueError("Reference needs explicit grid coordinates/orders or a known fixed-grid adapter fingerprint")
        metadata["grid_contract"] = "fixed grid verified against adapter fingerprint"
    else:
        metadata["grid_contract"] = "explicit fixed-grid coordinates and orders"
    arrays, digest = _load_arrays(path, ("mean", "stderr"))
    metadata["artifact_sha256"] = digest
    metadata["metadata_sha256"] = hashlib.sha256(content).hexdigest()
    return _field(arrays["mean"], "mean"), _field(arrays["stderr"], "stderr"), metadata


def _relative_norm(error, reference):
    numerator_scale = float(np.max(np.abs(error)))
    denominator_scale = float(np.max(np.abs(reference)))
    if denominator_scale == 0:
        return None
    if numerator_scale == 0:
        return 0.
    numerator = float(np.linalg.norm(error / numerator_scale))
    denominator = float(np.linalg.norm(reference / denominator_scale))
    nm, ne = math.frexp(numerator_scale)
    dm, de = math.frexp(denominator_scale)
    try:
        value = math.ldexp((nm / dm) * (numerator / denominator), ne - de)
    except OverflowError:
        return None
    return value if math.isfinite(value) and value > 0 else None


def compare_fields(prediction_path, reference_path):
    prediction, halfwidth, predicted = load_prediction(prediction_path)
    reference, stderr, observed = load_reference(reference_path)
    if predicted["config"] != observed["config"]:
        raise ValueError("Prediction and reference configurations differ")
    error = prediction - reference
    resolved = reference > 3 * stderr
    covered = np.abs(error) <= halfwidth
    relative = _relative_norm(error, reference)
    result = {
        "schema_version": 1, "kind": "saved_field_comparison", "units": UNITS,
        "config": predicted["config"], "model_sha256": predicted["model_sha256"],
        "prediction_sha256": predicted["artifact_sha256"],
        "reference_sha256": observed["artifact_sha256"],
        "reference_metadata_sha256": observed["metadata_sha256"],
        "reference_grid_contract": observed["grid_contract"],
        "out_of_domain": predicted["out_of_domain"],
        "relative_l2": relative,
        "relative_l2_note": None if relative is not None else "Reference norm is zero or the ratio is not representable.",
        "group_relative_l2": {group: _relative_norm(error[i], reference[i])
                              for i, group in enumerate(("slow", "fast"))},
        "resolved_cell_fraction": float(resolved.mean()),
        "resolved_cell_relative_l2": _relative_norm(error[resolved], reference[resolved]) if resolved.any() else None,
        "cell_interval_coverage": float(covered.mean()),
        "whole_field_interval_covered": bool(covered.all()),
        "reference_relative_standard_error": _relative_norm(stderr, reference),
        "scope": "One saved finite-history OpenMC estimate, not a calibration study or independent physical validation. OOD intervals are not qualified.",
    }
    return result


def save_comparison(path, result):
    content = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
    with fresh_output(path) as stream:
        stream.write(content)
