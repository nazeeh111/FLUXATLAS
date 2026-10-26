"""Validated experiment definitions. All lengths are cm, energies eV."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import numpy as np

GROUP_EDGES = [1.e-5, 1.e3, 2.e7]  # tally output is slow then fast
NX, NY = 24, 20
X_EDGES = np.linspace(0., 20., NX + 1)
Y_EDGES = np.linspace(-8., 8., NY + 1)
CELL_VOLUME = (20 / NX) * (16 / NY) * 1.0
PARAMETERS = ("thickness", "density_scale", "source_y", "fast_fraction")
BOUNDS = np.array([[1, 5], [.6, 1.4], [-3, 3], [.6, 1.]])

@dataclass(frozen=True)
class Experiment:
    thickness: float = 3.0
    density_scale: float = 1.0
    source_y: float = 0.0
    fast_fraction: float = .8
    family: str = "solid"

    def __post_init__(self):
        for name, (lo, hi) in zip(PARAMETERS, BOUNDS):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            if not lo <= value <= hi:
                raise ValueError(f"{name} must lie in [{lo}, {hi}]")
        if self.family not in {"solid", "window"}:
            raise ValueError("family must be solid or window")

    def vector(self):
        return (np.array([getattr(self, n) for n in PARAMETERS]) - BOUNDS[:, 0]) / np.ptp(BOUNDS, axis=1)

    def to_dict(self):
        return asdict(self)

    def cache_key(self, particles, batches, seed):
        raw = json.dumps({"schema": 1, "config": asdict(self), "particles": particles,
                          "batches": batches, "seed": seed}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:20]
