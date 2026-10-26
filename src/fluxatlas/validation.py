"""Analytical absorption and independent-seed convergence checks."""
from pathlib import Path
import json
import time
import numpy as np
from .transport import require_openmc, run_reference
from .config import Experiment, GROUP_EDGES


def analytical_absorption(directory, fast=True, particles=5000, seed=991):
    openmc = require_openmc()
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    groups = openmc.mgxs.EnergyGroups(GROUP_EDGES)
    xs = openmc.XSdata("absorber", groups)
    xs.order = 0
    xs.set_total([.2, .4]); xs.set_absorption([.2, .4])
    xs.set_scatter_matrix(np.zeros((2, 2, 1)))
    lib = openmc.MGXSLibrary(groups); lib.add_xsdata(xs)
    lib.export_to_hdf5(directory / "mgxs.h5")
    mat = openmc.Material(); mat.set_density("macro", 1.); mat.add_macroscopic("absorber")
    mats = openmc.Materials([mat]); mats.cross_sections = str(directory / "mgxs.h5")
    a, b = openmc.XPlane(x0=0,boundary_type="vacuum"), openmc.XPlane(x0=12,boundary_type="vacuum")
    c, d = openmc.YPlane(y0=0,boundary_type="reflective"), openmc.YPlane(y0=1,boundary_type="reflective")
    e, f = openmc.ZPlane(z0=0,boundary_type="reflective"), openmc.ZPlane(z0=1,boundary_type="reflective")
    geometry = openmc.Geometry([openmc.Cell(fill=mat,region=+a&-b&+c&-d&+e&-f)])
    s = openmc.Settings(); s.run_mode="fixed source"; s.energy_mode="multi-group"; s.batches=20; s.particles=particles; s.seed=seed
    source = openmc.IndependentSource(space=openmc.stats.Point((1.e-8,.5,.5)),angle=openmc.stats.Monodirectional((1,0,0)),
                                     energy=openmc.stats.Discrete([1.e6 if fast else 1.],[1.]))
    s.source=source; s.output={"summary":False,"tallies":False}
    mesh=openmc.RegularMesh(); mesh.dimension=(30,1,1);mesh.lower_left=(0,0,0);mesh.upper_right=(12,1,1)
    tally=openmc.Tally(name="attenuation");tally.filters=[openmc.MeshFilter(mesh),openmc.EnergyFilter(GROUP_EDGES)];tally.scores=["flux"]
    model=openmc.Model(geometry,mats,s,openmc.Tallies([tally]))
    statepoint=model.run(cwd=directory,threads=1,output=False)
    with openmc.StatePoint(statepoint) as sp:
        t=sp.get_tally(name="attenuation")
        mean=t.mean.reshape(30,2).T/.4; se=t.std_dev.reshape(30,2).T/.4
    edges=np.linspace(0,12,31); sigma=.2 if fast else .4
    analytic=(np.exp(-sigma*edges[:-1])-np.exp(-sigma*edges[1:]))/(sigma*.4)
    g=1 if fast else 0
    result={"group":"fast" if fast else "slow","sigma_cm_inverse":sigma,"histories":particles*20,
            "relative_l2":float(np.linalg.norm(mean[g]-analytic)/np.linalg.norm(analytic)),
            "fraction_within_3_standard_errors":float(np.mean(np.abs(mean[g]-analytic)<=3*se[g])),
            "wrong_group_max":float(mean[1-g].max()),"mean":mean[g].tolist(),"stderr":se[g].tolist(),
            "analytic":analytic.tolist(),"x_centers":((edges[:-1]+edges[1:])/2).tolist()}
    (directory/"comparison.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def validate(root, cache):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    analytical=[analytical_absorption(root/"absorption-fast",True),analytical_absorption(root/"absorption-slow",False,seed=992)]
    config=Experiment()
    anchor=run_reference(config,cache,particles=20000,batches=20,seed=900019)
    convergence=[]
    for particles in (400,1600,6400):
        runs=[run_reference(config,cache,particles,20,seed=800021+101*j+particles) for j in range(3)]
        means=np.array([r["mean"] for r in runs]); errors=np.linalg.norm((means-anchor["mean"]).reshape(3,-1),axis=1)/np.linalg.norm(anchor["mean"])
        convergence.append({"histories":particles*20,"seeds":[r["seed"] for r in runs],"mean_relative_l2_to_independent_anchor":float(errors.mean()),
            "std_relative_l2":float(errors.std(ddof=1)),"mean_reported_relative_se":float(np.mean([np.linalg.norm(r["stderr"])/np.linalg.norm(r["mean"]) for r in runs])),
            "mean_seconds":float(np.mean([r["seconds"] for r in runs]))})
    result={"analytical":analytical,"convergence":convergence,"anchor_histories":anchor["histories"],
            "anchor_seed":anchor["seed"],"limitations":"Analytical check is a pure absorber with a monodirectional source, not total scattering-field attenuation. Convergence uses an independent finite-history reference, not exact truth."}
    result["passed"] = all(r["relative_l2"] < .025 and r["wrong_group_max"] == 0 and r["fraction_within_3_standard_errors"] >= .8 for r in analytical)
    (root/"validation.json").write_text(json.dumps(result,indent=2)+"\n")
    if not result["passed"]:
        raise RuntimeError("Analytical transport validation failed; inspect validation.json")
    return result
