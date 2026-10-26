# Licensing and provenance

FLUXATLAS original code, branding, diagrams, synthetic macroscopic cross sections and generated numerical examples are distributed under the repository MIT license. Synthetic coefficients were authored for this educational benchmark. They are not measured or evaluated material data. No proprietary nuclear dataset is downloaded, copied or redistributed.

OpenMC is a separate MIT-licensed dependency, credited to its developers and Romano et al. (2015), [doi:10.1016/j.anucene.2014.07.048](https://doi.org/10.1016/j.anucene.2014.07.048). Install it from conda-forge; FLUXATLAS does not relicense OpenMC. Runtime distributions retain their own notices. NumPy, SciPy, scikit-learn and h5py use BSD-family licenses; Matplotlib uses its own PSF-based license. The optional environment may install DAGMC and other OpenMC dependencies under their respective licenses, although FLUXATLAS does not use CAD transport.

NeuralOperator and neural-operator papers are research foundations, not copied implementation or bundled dependencies. The model here is a fixed-grid POD spatial neural surrogate. It does not demonstrate resolution-independent operator learning.

All scientific figures are generated from saved numerical results. Screenshots depict this application and its genuine cached OpenMC/model fields. Video assets and any rendering-library notices are documented in `media/`; no image-generation system supplies numerical fields.

Continuous-energy nuclear libraries are intentionally absent. Users adding one must verify that library's independent license, provenance, processing conditions, temperature grid, physical suitability and normalization. Installing OpenMC does not grant blanket redistribution rights for third-party nuclear data.
