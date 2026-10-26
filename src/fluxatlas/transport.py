"""Independent OpenMC adapter using original, explicitly synthetic MG data.

No evaluated nuclear-data library is required. Cross sections are macroscopic
cm^-1, synthetic scattering is isotropic, and there is no fission.
"""
from pathlib import Path
import json
import hashlib
import shutil
import time
import numpy as np
from .config import Experiment, GROUP_EDGES, NX, NY, CELL_VOLUME

MATERIALS = {
    "background": {"total_fast_slow": [.025, .04], "scatter_fast_to_fast_slow": [.012, .008], "scatter_slow_to_slow": .025},
    "shield": {"total_fast_slow": [.65, 1.10], "scatter_fast_to_fast_slow": [.28, .22], "scatter_slow_to_slow": .65},
}


def require_openmc():
    try:
        import openmc
    except ImportError as exc:
        raise RuntimeError("OpenMC Python module unavailable. Install the supplied conda environment.") from exc
    if not shutil.which("openmc"):
        raise RuntimeError("OpenMC executable unavailable on PATH; activate the OpenMC environment.")
    return openmc


def build_model(config: Experiment, directory, particles=2000, batches=20, seed=101):
    if isinstance(particles, bool) or not isinstance(particles, int) or not 100 <= particles <= 200000:
        raise ValueError("particles per batch must be an integer in [100, 200000]")
    if isinstance(batches, bool) or not isinstance(batches, int) or not 10 <= batches <= 500:
        raise ValueError("batches must be an integer in [10, 500]")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 1 <= seed < 2**31:
        raise ValueError("seed must be an integer in [1, 2^31)")
    openmc = require_openmc()
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    groups = openmc.mgxs.EnergyGroups(GROUP_EDGES)
    library = openmc.MGXSLibrary(groups)
    materials = []
    for name, spec in MATERIALS.items():
        xs = openmc.XSdata(name, groups)
        xs.order = 0
        xs.set_total(np.array(spec["total_fast_slow"]))
        # OpenMC XSdata uses descending energy order: fast, slow.
        scatter = np.zeros((2, 2, 1))
        scatter[0, 0, 0], scatter[0, 1, 0] = spec["scatter_fast_to_fast_slow"]
        scatter[1, 1, 0] = spec["scatter_slow_to_slow"]
        xs.set_scatter_matrix(scatter)
        xs.set_absorption(np.array(spec["total_fast_slow"]) - scatter.sum(axis=(1, 2)))
        library.add_xsdata(xs)
        material = openmc.Material(name=f"synthetic_{name}")
        material.set_density("macro", config.density_scale if name == "shield" else 1.)
        material.add_macroscopic(name)
        materials.append(material)
    library.export_to_hdf5(directory / "mgxs.h5")
    mats = openmc.Materials(materials)
    mats.cross_sections = str(directory / "mgxs.h5")
    x0, x1 = openmc.XPlane(x0=0, boundary_type="vacuum"), openmc.XPlane(x0=20, boundary_type="vacuum")
    y0, y1 = openmc.YPlane(y0=-8, boundary_type="vacuum"), openmc.YPlane(y0=8, boundary_type="vacuum")
    z0, z1 = openmc.ZPlane(z0=-.5, boundary_type="reflective"), openmc.ZPlane(z0=.5, boundary_type="reflective")
    domain = +x0 & -x1 & +y0 & -y1 & +z0 & -z1
    front, back = openmc.XPlane(x0=6), openmc.XPlane(x0=6 + config.thickness)
    bottom, top = openmc.YPlane(y0=-6), openmc.YPlane(y0=6)
    shield = +front & -back & +bottom & -top
    if config.family == "window":
        a, b = openmc.YPlane(y0=-1), openmc.YPlane(y0=1)
        shield = shield & (-a | +b)
    cells = [openmc.Cell(fill=materials[1], region=domain & shield),
             openmc.Cell(fill=materials[0], region=domain & ~shield)]
    geometry = openmc.Geometry(cells)
    source = openmc.IndependentSource()
    source.space = openmc.stats.CartesianIndependent(
        x=openmc.stats.Discrete([.5], [1]),
        y=openmc.stats.Uniform(config.source_y - .4, config.source_y + .4),
        z=openmc.stats.Uniform(-.49, .49))
    source.angle = openmc.stats.Monodirectional((1., 0., 0.))
    source.energy = openmc.stats.Discrete([1., 1.e6], [1 - config.fast_fraction, config.fast_fraction])
    settings = openmc.Settings()
    settings.run_mode, settings.energy_mode = "fixed source", "multi-group"
    settings.source, settings.particles, settings.batches, settings.seed = source, particles, batches, seed
    settings.output = {"tallies": False, "summary": False}
    mesh = openmc.RegularMesh()
    mesh.dimension, mesh.lower_left, mesh.upper_right = (NX, NY, 1), (0, -8, -.5), (20, 8, .5)
    tally = openmc.Tally(name="mesh_flux")
    tally.filters = [openmc.MeshFilter(mesh), openmc.EnergyFilter(GROUP_EDGES)]
    tally.scores = ["flux"]
    tally.estimator = "tracklength"
    model = openmc.Model(geometry, mats, settings, openmc.Tallies([tally]))
    return model


def run_reference(config, directory, particles=2000, batches=20, seed=101, reuse=True):
    """Return volume-mean scalar flux and batch-based standard error, [2,ny,nx]."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    openmc = require_openmc()
    # Hash actual adapter/config source bytes as well as solver version. Changes to
    # cross sections, mesh, source or normalization cannot silently reuse old runs.
    fingerprint = hashlib.sha256(Path(__file__).read_bytes() + Path(__file__).with_name("config.py").read_bytes() + openmc.__version__.encode()).hexdigest()
    key = hashlib.sha256((config.cache_key(particles, batches, seed) + fingerprint).encode()).hexdigest()[:20]
    run_dir = directory / key
    cache = run_dir / "field.npz"
    if reuse and cache.exists():
        data = dict(np.load(cache, allow_pickle=False))
        metadata = json.loads((run_dir / "metadata.json").read_text())
        if metadata.get("physics_fingerprint") != fingerprint:
            raise RuntimeError("Cached physics fingerprint mismatch")
        return {**data, **metadata, "cached": True}
    start = time.perf_counter()
    model = build_model(config, run_dir, particles, batches, seed)
    openmc = require_openmc()
    statepoint = model.run(cwd=run_dir, threads=1, output=False)
    with openmc.StatePoint(statepoint) as sp:
        tally = sp.get_tally(name="mesh_flux")
        # MeshFilter linearization is x-fastest, then y, then z; energy last.
        mean = tally.mean.reshape(NY, NX, 2).transpose(2, 0, 1) / CELL_VOLUME
        stderr = tally.std_dev.reshape(NY, NX, 2).transpose(2, 0, 1) / CELL_VOLUME
        transport_seconds = float(sp.runtime.get("transport", 0))
    if not np.isfinite(mean).all() or not np.isfinite(stderr).all() or (mean < 0).any() or (stderr < 0).any():
        raise RuntimeError("Nonphysical or nonfinite OpenMC tally")
    metadata = {"config": config.to_dict(), "seed": seed, "particles_per_batch": particles,
                "batches": batches, "histories": particles * batches, "solver": "OpenMC",
                "solver_version": openmc.__version__, "units": "cm^-2 per source particle",
                "energy_edges_eV": GROUP_EDGES, "shape": [2, NY, NX],
                "normalization": "tracklength tally per source particle divided by cell volume in cm^3",
                "seconds": time.perf_counter() - start, "transport_seconds": transport_seconds,
                "cache_key": key, "physics_fingerprint": fingerprint, "schema_version": 1}
    np.savez_compressed(cache, mean=mean, stderr=stderr)
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {"mean": mean, "stderr": stderr, **metadata, "cached": False}
