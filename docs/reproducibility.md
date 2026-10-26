# Reproducing FLUXATLAS

## Requirements

Use a free local conda/micromamba environment with OpenMC 0.16.0 and Python 3.12. `environment.yml` declares the portable environment. `environment-osx-arm64-explicit.txt` records exact packages used for this release on Apple arm64; explicit platform locks do not apply to Linux or Intel Macs. The Python-only package can view cached results/query exported weights without OpenMC. The reference and training workflows need the solver and science dependencies.

The observed native environment occupied about 0.8 GB. The final numerical data, trained weights and web archive require only several MB; the video adds release size. The benchmark contains 7,008,000 final dataset histories before analytical/convergence checks and exploratory work. Solver runs use one thread. Training is deliberately small. Wall time depends strongly on machine load and numerical libraries; this Mac's OpenMP OpenBLAS default was slower for small neural fits, so use the limits below.

## Install and verify

```bash
git clone https://github.com/nazeeh111/FLUXATLAS.git
cd FLUXATLAS
micromamba create -f environment.yml
micromamba activate fluxatlas
pip install -e .
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
pytest -q
fluxatlas example --output example-run --particles 1000
fluxatlas serve --port 8765
```

No `OPENMC_CROSS_SECTIONS` setting or external nuclear-data download is needed: the adapter writes original synthetic multigroup HDF5 data. The CLI supplies the library path in its model. Open `http://127.0.0.1:8765`. The banner labels preserved archive fields; model query and solver buttons explicitly create new local results.

## Regenerate the scientific record

Run from the source checkout. These commands replace the generated numerical artifacts in the specified destinations; preserve the released reference files if comparing versions.

```bash
fluxatlas generate --output data --cache .cache/fluxatlas
fluxatlas train --project .
fluxatlas validate --output benchmarks/physics --cache .cache/fluxatlas
python scripts/build-report.py
pytest -q
```

The deterministic design is persisted before simulation. Each high/low configuration uses its own documented seed. The final source includes both original training-pool settings and new reserved holdouts from protocol 2. Cache identities include config, particle counts, batches, seed, exact adapter/config source bytes and solver version. Metadata/field arrays preserve mean, standard error, energy edges, units, normalization and computational costs. Reuse validates the physics fingerprint. Editing the solver adapter invalidates earlier cache identities.

`fluxatlas train` fits from the supplied independent OpenMC references, writes actual numeric network weights and the study/observatory exports. The fixed model hyperparameters live in `model.py`; the acquisition schedule and seeds live in `study.py`. Acquisition is retrospective and reads only selected high labels during fitting; its low-pool cost is included. Full precomputation cost is disclosed separately. Calibration and test rows never enter model fitting or acquisition. Tests check numeric-only inference against actual optimized network predictions.

Numerical floating-point reductions and optimization may vary between platforms/libraries, and bit-identical training is not promised. Record package versions, rerun the independent checks and compare the reported tolerances/metrics. The small 24-configuration final test should not be treated as a comprehensive validation suite. The exploratory failure archive remains separate; its inspected tests are not part of final evaluation.

## Numerical files

`data/fields.npz` can be read using `np.load(path, allow_pickle=False)`:

| Array | Shape / meaning |
|---|---|
| `x` | 156×4 normalized parameters in thickness, scale, source_y, fast_fraction order |
| `high` | 156×960 independent 40k-history flux estimates |
| `stderr` | Same shape, estimated standard error of the mean |
| `low`, `low_stderr` | 96×960 independent 8k-history pool estimates and SE |
| `seconds`, `low_seconds` | Measured setup+execution+extraction wall seconds |

A field reshapes to `[2,20,24]`: slow then fast; y then x, with x increasing fastest. y increases from −8 to +8 cm. Plotting may reverse display rows to put +y upward. `data/design.json` identifies the first 96 pool, next 24 calibration, next 24 test, and final 12 OOD configurations. `data/reference-metadata.json` provides solver version, exact seeds, histories, normalization and physics fingerprints. `data/model.npz` contains only numeric optimized arrays. The browser archive has 36 independent comparison cases and actual predictions/intervals.

Use the observatory export buttons for configuration JSON, per-cell numerical CSV or the complete research JSON archive. The CSV includes group energy boundaries and coordinates. New local predictions are not added to the released study, and a reference remains absent until OpenMC actually runs.

## Optional higher-fidelity work

Continuous-energy neutron/photon cases require suitable evaluated cross sections under their own licenses. This release intentionally makes no such result. A defensible extension must document the library/version, material compositions, source spectrum, temperatures, energy/mesh convergence and normalization, then independently compare to the synthetic benchmark and rerun training/calibration for the new domain. CAD/DAGMC requires separate validated geometry preparation. Neither feature is silently activated by the existing interface.

Dose-rate output also requires source strength, irradiation convention, validated conversion and a separate quantity definition. Do not multiply an arbitrary synthetic tally by a dose coefficient and present it as a validated safety quantity.
