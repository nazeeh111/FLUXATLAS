import io
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest

from fluxatlas.artifacts import (GRID, NORMALIZATION, SHAPE, UNITS, _relative_norm, compare_fields, fresh_output,
                                load_prediction, load_reference, save_comparison, save_prediction)
from fluxatlas.config import Experiment, GROUP_EDGES


@pytest.fixture
def artifacts(tmp_path):
    model = tmp_path / "model.npz"
    model.write_bytes(b"numeric model identity fixture")
    prediction = tmp_path / "prediction"
    reference = tmp_path / "field.npz"
    config = Experiment()
    metadata = save_prediction(prediction, config, np.ones(SHAPE), np.full(SHAPE, .1),
                               hashlib.sha256(model.read_bytes()).hexdigest(), False)
    np.savez_compressed(reference, mean=np.ones(SHAPE), stderr=np.full(SHAPE, .01))
    reference_metadata = {
        "schema_version": 1, "units": UNITS, "shape": list(SHAPE),
        "energy_edges_eV": GROUP_EDGES, "config": config.to_dict(), "solver": "OpenMC",
        "normalization": NORMALIZATION,
        **GRID,
    }
    (tmp_path / "metadata.json").write_text(json.dumps(reference_metadata))
    return prediction, reference, metadata, reference_metadata


def test_portable_prediction_and_exact_comparison(artifacts):
    prediction, reference, metadata, _ = artifacts
    mean, half, restored = load_prediction(prediction)
    np.testing.assert_array_equal(mean, np.ones(SHAPE))
    np.testing.assert_array_equal(half, np.full(SHAPE, .1))
    assert restored["config"] == metadata["config"]
    assert prediction.exists() and not prediction.with_suffix(".npz").exists()
    result = compare_fields(prediction, reference)
    assert result["relative_l2"] == 0
    assert result["resolved_cell_fraction"] == 1
    assert result["whole_field_interval_covered"]
    assert result["reference_relative_standard_error"] == pytest.approx(.01)
    save_comparison(prediction.parent / "report.json", result)
    assert json.loads((prediction.parent / "report.json").read_text()) == result


@pytest.mark.parametrize("field,value", [
    ("units", "Gy"), ("shape", [2, 24, 20]), ("energy_edges_eV", list(reversed(GROUP_EDGES))),
    ("schema_version", True), ("config", Experiment(source_y=1).to_dict()),
])
def test_reference_contract_mismatch_refuses(artifacts, field, value):
    prediction, reference, _, metadata = artifacts
    metadata[field] = value
    reference.with_name("metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        compare_fields(prediction, reference)


def test_zeros_have_explicit_undefined_relative_metrics(artifacts):
    prediction, reference, _, _ = artifacts
    np.savez_compressed(reference, mean=np.zeros(SHAPE), stderr=np.zeros(SHAPE))
    result = compare_fields(prediction, reference)
    assert result["relative_l2"] is None and result["relative_l2_note"]
    assert result["group_relative_l2"] == {"slow": None, "fast": None}
    assert result["resolved_cell_relative_l2"] is None
    assert result["cell_interval_coverage"] == 0
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1])
def test_invalid_fields_refuse_before_publishing(tmp_path, value):
    field = np.ones(SHAPE)
    field.flat[0] = value
    with pytest.raises(ValueError):
        save_prediction(tmp_path / "out.npz", Experiment(), field, field, "a" * 64, False)
    assert not (tmp_path / "out.npz").exists()


def test_publication_is_exclusive_and_failure_does_not_leave_partial_file(tmp_path):
    output = tmp_path / "out"
    with pytest.raises(RuntimeError):
        with fresh_output(output) as stream:
            stream.write(b"partial")
            raise RuntimeError("disk failure fixture")
    assert not output.exists() and list(tmp_path.iterdir()) == []
    with pytest.raises(FileExistsError):
        with fresh_output(output) as stream:
            stream.write(b"candidate")
            output.write_bytes(b"concurrent winner")
    assert output.read_bytes() == b"concurrent winner"
    link = tmp_path / "link"
    link.symlink_to(output)
    with pytest.raises(FileExistsError):
        with fresh_output(link):
            pytest.fail("must not open existing symlink")
    assert output.read_bytes() == b"concurrent winner"
    dangling = tmp_path / "dangling"
    dangling.symlink_to(tmp_path / "absent")
    with pytest.raises(FileExistsError):
        with fresh_output(dangling):
            pytest.fail("must not replace dangling link")


@pytest.mark.parametrize("kind", ["object", "oversized_header", "duplicate", "wrong_shape", "negative"])
def test_malformed_archives_refuse_without_allocation_or_pickle(artifacts, kind):
    prediction, _, _, _ = artifacts
    with np.load(prediction, allow_pickle=False) as archive:
        arrays = dict(archive)
    if kind == "object":
        arrays["prediction"] = np.array({"dangerous": "object"}, dtype=object)
    elif kind == "wrong_shape":
        arrays["prediction"] = np.ones((2, 480))
    elif kind == "negative":
        arrays["prediction"] = -np.ones(SHAPE)
    if kind in {"duplicate", "oversized_header"}:
        with zipfile.ZipFile(prediction, "a") as archive:
            stream = io.BytesIO()
            np.lib.format.write_array_header_1_0(stream, {
                "descr": "<f8", "fortran_order": False, "shape": (2**50,),
            })
            with pytest.warns(UserWarning, match="Duplicate name"):
                archive.writestr("prediction.npy", stream.getvalue())
        if kind == "oversized_header":
            # Keep exactly one member for each required name, but forge its shape.
            with zipfile.ZipFile(prediction) as archive:
                content = {name: archive.read(name) for name in archive.namelist()}
            with zipfile.ZipFile(prediction, "w") as archive:
                for name, value in content.items():
                    archive.writestr(name, value)
    else:
        with prediction.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
    with pytest.raises(ValueError):
        load_prediction(prediction)


def test_cli_uses_real_released_weights_and_refuses_overwrite(tmp_path):
    output = tmp_path / "prediction.portable"
    command = [sys.executable, "-m", "fluxatlas.cli", "predict", "--output", str(output)]
    first = subprocess.run(command, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    mean, half, metadata = load_prediction(output)
    assert mean.shape == half.shape == SHAPE
    assert metadata["out_of_domain"] is False
    from fluxatlas.cli import project_root
    reference = project_root() / "examples/reference/018aaf1e873bfacdc6f5/field.npz"
    comparison = compare_fields(output, reference)
    assert comparison["reference_grid_contract"] == "fixed grid verified against adapter fingerprint"
    assert comparison["relative_l2"] is not None
    original = output.read_bytes()
    second = subprocess.run(command, capture_output=True, text=True)
    assert second.returncode == 2 and "already exists" in second.stderr
    assert output.read_bytes() == original


def test_cli_comparison_and_refusal_preserve_completed_output(artifacts):
    prediction, reference, _, metadata = artifacts
    output = prediction.parent / "comparison.json"
    command = [sys.executable, "-m", "fluxatlas.cli", "compare", "--prediction", str(prediction),
               "--reference", str(reference), "--output", str(output)]
    first = subprocess.run(command, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    assert json.loads(first.stdout)["relative_l2"] == 0
    original = output.read_bytes()
    metadata["config"] = Experiment(thickness=4).to_dict()
    reference.with_name("metadata.json").write_text(json.dumps(metadata))
    second = subprocess.run(command, capture_output=True, text=True)
    assert second.returncode == 2 and "configurations differ" in second.stderr
    assert output.read_bytes() == original


@pytest.mark.parametrize("field,value", [
    ("config", {}), ("energy_order", ["fast", "slow"]),
    ("spatial_order", ["x", "y"]), ("x_edges_cm", (np.arange(25) + 100).tolist()),
    ("y_edges_cm", (np.arange(21) + 100).tolist()),
])
def test_saved_provenance_must_be_complete_and_consistent(artifacts, field, value):
    prediction, reference, _, metadata = artifacts
    metadata[field] = value
    reference.with_name("metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        compare_fields(prediction, reference)


def test_unknown_legacy_grid_and_wrong_normalization_refuse(artifacts):
    prediction, reference, _, metadata = artifacts
    metadata["normalization"] = "unscaled volume-integrated tally"
    reference.with_name("metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="normalization"):
        compare_fields(prediction, reference)
    metadata["normalization"] = NORMALIZATION
    for key in GRID:
        metadata.pop(key)
    metadata["physics_fingerprint"] = "a" * 64
    reference.with_name("metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="fingerprint"):
        compare_fields(prediction, reference)


@pytest.mark.parametrize("codec", [zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA])
def test_forged_size_compression_cannot_bypass_input_bound(artifacts, codec):
    import struct
    import zlib
    _, reference, _, _ = artifacts
    stream = io.BytesIO()
    np.save(stream, np.ones(SHAPE), allow_pickle=False)
    legitimate = stream.getvalue()
    with zipfile.ZipFile(reference, "w", compression=codec) as archive:
        archive.writestr("mean.npy", legitimate + bytes(2_000_001))
        archive.writestr("stderr.npy", legitimate)
    content = bytearray(reference.read_bytes())
    with zipfile.ZipFile(reference) as archive:
        info = archive.getinfo("mean.npy")
        central = archive.start_dir
    for offset in (info.header_offset + 14, central + 16):
        struct.pack_into("<I", content, offset, zlib.crc32(legitimate))
    for offset in (info.header_offset + 22, central + 24):
        struct.pack_into("<I", content, offset, len(legitimate))
    reference.write_bytes(content)
    with pytest.raises(ValueError, match="compression"):
        load_reference(reference)


def test_extreme_but_representable_norm_ratio():
    assert _relative_norm(np.array([1e70]), np.array([1e-100])) == pytest.approx(1e170)
    assert _relative_norm(np.array([1e-100]), np.array([1e70])) == pytest.approx(1e-170, abs=0)
    assert _relative_norm(np.full(960, 1e308), np.full(960, 1e308)) == 1
    assert _relative_norm(np.array([1e-300]), np.array([1e100])) is None
    assert _relative_norm(np.array([0.]), np.array([1e100])) == 0.


def test_weight_identity_is_the_loaded_snapshot(tmp_path, monkeypatch, capsys):
    from fluxatlas import cli
    from fluxatlas.model import FrozenSurrogate
    project = tmp_path / "project"
    (project / "data").mkdir(parents=True)
    weights = project / "data/model.npz"
    weights.write_bytes((cli.project_root() / "data/model.npz").read_bytes())
    original_hash = hashlib.sha256(weights.read_bytes()).hexdigest()
    original_predict = FrozenSurrogate.predict

    def predict_then_replace(self, *args, **kwargs):
        result = original_predict(self, *args, **kwargs)
        weights.write_bytes(b"concurrent replacement")
        return result

    monkeypatch.setattr(FrozenSurrogate, "predict", predict_then_replace)
    monkeypatch.setattr(cli, "project_root", lambda: project)
    output = tmp_path / "prediction.npz"
    monkeypatch.setattr(sys, "argv", ["fluxatlas", "predict", "--output", str(output)])
    cli.main()
    assert json.loads(capsys.readouterr().out)["model_sha256"] == original_hash
    assert load_prediction(output)[2]["model_sha256"] == original_hash


def test_large_config_integer_refuses_cleanly(artifacts):
    prediction, reference, _, metadata = artifacts
    metadata["config"]["thickness"] = 10 ** 1000
    reference.with_name("metadata.json").write_text(json.dumps(metadata))
    result = subprocess.run([sys.executable, "-m", "fluxatlas.cli", "compare",
                             "--prediction", str(prediction), "--reference", str(reference)],
                            capture_output=True, text=True)
    assert result.returncode == 2 and "Traceback" not in result.stderr


@pytest.mark.parametrize("port", [-1, 65536])
def test_cli_refuses_invalid_serve_port_without_traceback_or_outputs(tmp_path, port):
    result = subprocess.run(
        [sys.executable, "-m", "fluxatlas.cli", "serve", "--port", str(port)],
        cwd=tmp_path, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 2
    assert "port" in result.stderr.lower()
    assert "Traceback" not in result.stderr
    assert not result.stdout
    assert not list(tmp_path.iterdir())


def test_cli_predict_accepts_existing_ui_export_and_preserves_bare_config(tmp_path):
    from fluxatlas.cli import project_root
    archive = json.loads((project_root() / "web/atlas.json").read_text())
    record = archive["cases"][0]
    # Exact schema and fields emitted by the existing Export configuration action.
    exported = {"schema_version": 1, "config": record["config"], "seed": record["seed"],
                "histories": record["histories"], "units": archive["units"],
                "source": "FLUXATLAS educational synthetic multigroup benchmark"}
    envelope = tmp_path / "exported-configuration.json"
    bare = tmp_path / "bare-configuration.json"
    envelope.write_text(json.dumps(exported, indent=2))
    bare.write_text(json.dumps(record["config"]))
    original = envelope.read_bytes()
    predictions = []
    for config, output in ((bare, tmp_path / "bare.npz"), (envelope, tmp_path / "exported.npz")):
        result = subprocess.run([sys.executable, "-m", "fluxatlas.cli", "predict", "--config", str(config),
                                 "--output", str(output)], capture_output=True, text=True, timeout=20)
        assert result.returncode == 0, result.stderr
        predictions.append(load_prediction(output))
    np.testing.assert_array_equal(predictions[0][0], predictions[1][0])
    np.testing.assert_array_equal(predictions[0][1], predictions[1][1])
    assert predictions[0][2]["config"] == predictions[1][2]["config"] == record["config"]
    assert predictions[0][2]["model_sha256"] == predictions[1][2]["model_sha256"]
    assert envelope.read_bytes() == original
    output = tmp_path / "exported.npz"
    before = output.read_bytes()
    repeated = subprocess.run([sys.executable, "-m", "fluxatlas.cli", "predict", "--config", str(envelope),
                               "--output", str(output)], capture_output=True, text=True, timeout=20)
    assert repeated.returncode == 2
    assert output.read_bytes() == before
    assert envelope.read_bytes() == original


@pytest.mark.parametrize("document", [
    {"schema_version": 2, "config": Experiment().to_dict()},
    {"schema_version": True, "config": Experiment().to_dict()},
    {"schema_version": 1.0, "config": Experiment().to_dict()},
    {"schema_version": 1, "config": []},
    {"schema_version": 1, "config": {}},
    {"schema_version": 1, "config": {"thickness": 3.}},
    {"config": Experiment().to_dict()},
    {"schema_version": 1, "config": Experiment().to_dict(), "unknown": "field"},
])
def test_cli_refuses_malformed_ui_export_without_publication(tmp_path, document):
    config = tmp_path / "configuration.json"
    output = tmp_path / "prediction.npz"
    config.write_text(json.dumps(document))
    before = config.read_bytes()
    result = subprocess.run([sys.executable, "-m", "fluxatlas.cli", "predict", "--config", str(config),
                             "--output", str(output)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 2
    assert "configuration" in result.stderr.lower()
    assert "Traceback" not in result.stderr
    assert not output.exists()
    assert config.read_bytes() == before
    output.write_bytes(b"prior output must be preserved")
    repeated = subprocess.run([sys.executable, "-m", "fluxatlas.cli", "predict", "--config", str(config),
                               "--output", str(output)], capture_output=True, text=True, timeout=20)
    assert repeated.returncode == 2
    assert output.read_bytes() == b"prior output must be preserved"
    assert config.read_bytes() == before


def test_cli_bare_partial_configuration_still_uses_documented_defaults(tmp_path):
    config = tmp_path / "configuration.json"
    output = tmp_path / "prediction.npz"
    config.write_text('{"thickness":3.0}')
    result = subprocess.run([sys.executable, "-m", "fluxatlas.cli", "predict", "--config", str(config),
                             "--output", str(output)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert load_prediction(output)[2]["config"] == Experiment().to_dict()
