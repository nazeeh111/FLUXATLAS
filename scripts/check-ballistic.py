#!/usr/bin/env python3
"""Reproduce the independent review of the analytical uncollided beam.

After installing FLUXATLAS and activating its OpenMC environment, run::

    python scripts/check-ballistic.py --output work/ballistic-check

The output directory must be new. The script checks 32 supported configurations
against independent SciPy quadrature, then executes a 100,000-history OpenMC
pure-absorber calculation with off-grid source and shield positions. Its local
OpenMC input library has zero scattering and unchanged total cross sections.
Neither production source nor the transport module's material table is changed.

Use --quadrature-only to perform the deterministic check without OpenMC. This
script verifies the analytical component, not the learned scattering residual.
The original reviewed run is retained under benchmarks/physics/ballistic-review.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad

from fluxatlas.ballistic import ballistic_field
from fluxatlas.config import BOUNDS, CELL_VOLUME, GROUP_EDGES, NX, NY, X_EDGES, Y_EDGES, Experiment


def check_quadrature():
    """Integrate a separately expressed cumulative-optical-path integrand."""
    rng = np.random.default_rng(8129)
    inputs = np.vstack([np.zeros(4), np.ones(4), rng.uniform(size=(30, 4))])
    actual = ballistic_field(inputs).reshape(len(inputs), 2, NY, NX)
    expected = np.zeros_like(actual)
    physical = inputs * np.ptp(BOUNDS, axis=1) + BOUNDS[:, 0]
    for n, (thickness, density, source_y, fast_fraction) in enumerate(physical):
        overlap = np.maximum(
            0., np.minimum(Y_EDGES[1:], source_y + .4)
            - np.maximum(Y_EDGES[:-1], source_y - .4))
        for group, (background, shield, weight) in enumerate((
                (.04, 1.1 * density, 1 - fast_fraction),
                (.025, .65 * density, fast_fraction))):
            def integrand(position):
                if position < .5:
                    return 0.
                shield_path = np.clip(position - 6., 0., thickness)
                return np.exp(-background * (position - .5)
                              - (shield - background) * shield_path)

            for ix, (left, right) in enumerate(zip(X_EDGES[:-1], X_EDGES[1:])):
                points = [p for p in (.5, 6., 6. + thickness) if left < p < right]
                integral = quad(integrand, left, right, points=points,
                                epsabs=1.e-13, epsrel=1.e-13)[0]
                expected[n, group, :, ix] = (
                    weight * overlap / .8 / np.diff(Y_EDGES)
                    * integral / (right - left))
    error = np.abs(expected - actual)
    nonzero = expected > 1.e-13
    result = {
        "configurations": len(inputs), "design_seed": 8129,
        "max_absolute_error": float(error.max()),
        "max_relative_error_nonzero": float(np.max(error[nonzero] / expected[nonzero])),
        "rtol": 1.e-11, "atol": 1.e-13,
        "passed": bool(np.allclose(actual, expected, rtol=1.e-11, atol=1.e-13)),
    }
    if not result["passed"]:
        raise RuntimeError(f"Ballistic quadrature disagreement: {result}")
    return result


def check_openmc(output):
    # Import lazily so the quadrature check needs no OpenMC installation.
    from fluxatlas.transport import MATERIALS, build_model, require_openmc

    openmc = require_openmc()
    config = Experiment(thickness=2.37, density_scale=1.17,
                        source_y=1.13, fast_fraction=.73)
    model = build_model(config, output, particles=5000, batches=20, seed=192031)

    # Replace only this disposable model's on-disk data library. Avoid mutating
    # MATERIALS, replacing module functions, or entering the reference cache.
    groups = openmc.mgxs.EnergyGroups(GROUP_EDGES)
    library = openmc.MGXSLibrary(groups)
    for name, specification in MATERIALS.items():
        total = np.array(specification["total_fast_slow"])
        xs = openmc.XSdata(name, groups)
        xs.order = 0
        xs.set_total(total)
        xs.set_absorption(total)
        xs.set_scatter_matrix(np.zeros((2, 2, 1)))
        library.add_xsdata(xs)
    library.export_to_hdf5(output / "mgxs.h5")
    model.materials.cross_sections = "mgxs.h5"
    statepoint = model.run(cwd=output, threads=1, output=False)
    with openmc.StatePoint(statepoint) as result_file:
        tally = result_file.get_tally(name="mesh_flux")
        mean = tally.mean.reshape(NY, NX, 2).transpose(2, 0, 1) / CELL_VOLUME
        stderr = tally.std_dev.reshape(NY, NX, 2).transpose(2, 0, 1) / CELL_VOLUME
    exact = ballistic_field(config.vector()).reshape(2, NY, NX)
    group_error = [float(np.linalg.norm(mean[g] - exact[g]) / np.linalg.norm(exact[g]))
                   for g in range(2)]
    coverage = [float(np.mean(np.abs(mean[g] - exact[g])[exact[g] > 0]
                              <= 3 * stderr[g][exact[g] > 0])) for g in range(2)]
    result = {
        "purpose": "Independent review: heterogeneous pure-absorption beam check; "
                   "zero scattering in the disposable model's data library only. "
                   "Production source and runtime material table are unchanged.",
        "config": config.to_dict(), "histories": 100000, "seed": 192031,
        "solver_version": openmc.__version__, "units": "cm^-2 per source particle",
        "relative_l2": float(np.linalg.norm(mean - exact) / np.linalg.norm(exact)),
        "group_relative_l2": group_error,
        "group_order": ["slow", "fast"],
        "fraction_within_3se_nonzero": coverage,
        "outside_beam_max": float(mean[exact == 0].max()),
    }
    # Generous stochastic tolerances catch normalization/group/mesh mistakes.
    # This is not a radiological validation or simultaneous confidence test.
    result["passed"] = bool(max(group_error) < .02 and min(coverage) >= .8
                            and result["outside_beam_max"] == 0.)
    (output / "review.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(output / "comparison.npz", mean=mean, stderr=stderr, analytic=exact)
    if not result["passed"]:
        raise RuntimeError(f"OpenMC ballistic comparison failed: {result}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("work/ballistic-check"),
                        help="New output folder; existing folders are preserved")
    parser.add_argument("--quadrature-only", action="store_true")
    args = parser.parse_args()
    quadrature = check_quadrature()
    print(json.dumps({"quadrature": quadrature}, indent=2), flush=True)
    if args.quadrature_only:
        return
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "quadrature.json").write_text(json.dumps(quadrature, indent=2) + "\n")
    print(json.dumps({"openmc": check_openmc(output)}, indent=2))


if __name__ == "__main__":
    main()
