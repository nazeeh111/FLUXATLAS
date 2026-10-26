"""Predeclared independent geometry partitions and cached OpenMC data."""
from pathlib import Path
import json
import numpy as np
from scipy.stats import qmc
from .config import BOUNDS, PARAMETERS, Experiment
from .transport import run_reference


def design(seed=713, pool=96, calibration=24, test=24, ood=12):
    sampler = qmc.LatinHypercube(4, seed=seed)
    values = qmc.scale(sampler.random(pool + calibration + test + ood), BOUNDS[:, 0], BOUNDS[:, 1])
    partitions = ["pool"] * pool + ["calibration"] * calibration + ["test"] * test + ["ood"] * ood
    # Version 2: retain the training pool, replace all previously inspected
    # calibration/test/OOD settings using a predeclared independent design.
    fresh=qmc.scale(qmc.LatinHypercube(4,seed=271828).random(calibration+test+ood),BOUNDS[:,0],BOUNDS[:,1])
    values[pool:]=fresh
    records = []
    for i, (v, split) in enumerate(zip(values, partitions)):
        config = Experiment(**dict(zip(PARAMETERS, map(float, v))), family="window" if split == "ood" else "solid")
        records.append({"id": f"{split}-{i:03d}", "split": split, "config": config.to_dict(),
                        "high_seed": 100003 + 101 * i if split == "pool" else 3000001 + 107 * i, "low_seed": 700001 + 103 * i})
    return records


def generate(root, cache, high_particles=2000, low_particles=400, batches=20):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = design()
    # Persist partitions before the first reference run. Acquisition never sees test labels.
    (root / "design.json").write_text(json.dumps(manifest, indent=2) + "\n")
    xs, high, se, low, low_se, times, low_times = [], [], [], [], [], [], []
    metadata = []
    for i, rec in enumerate(manifest):
        config = Experiment(**rec["config"])
        result = run_reference(config, cache, high_particles, batches, rec["high_seed"])
        xs.append(config.vector()); high.append(result["mean"].ravel()); se.append(result["stderr"].ravel())
        times.append(result["seconds"])
        metadata.append({k: v for k, v in result.items() if k not in {"mean", "stderr", "cached"}})
        if rec["split"] == "pool":
            cheap = run_reference(config, cache, low_particles, batches, rec["low_seed"])
            low.append(cheap["mean"].ravel()); low_se.append(cheap["stderr"].ravel()); low_times.append(cheap["seconds"])
        print(f"[{i+1}/{len(manifest)}] {rec['id']} {result['seconds']:.3f}s {'cached' if result['cached'] else 'new'}", flush=True)
    np.savez_compressed(root / "fields.npz", x=xs, high=high, stderr=se, low=low, low_stderr=low_se,
                        seconds=times, low_seconds=low_times)
    (root / "reference-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (root / "dataset-info.json").write_text(json.dumps({"schema_version": 2, "seed": 713, "fresh_holdout_seed": 271828,
        "high_histories_each": high_particles * batches, "low_histories_each": low_particles * batches,
        "pool": 96, "calibration": 24, "test": 24, "ood": 12,
        "generation_high_histories": len(manifest) * high_particles * batches,
        "generation_low_histories": 96 * low_particles * batches,
        "note": "All high pool labels precomputed for retrospective acquisition studies; only selected labels are exposed to each fit."}, indent=2) + "\n")
