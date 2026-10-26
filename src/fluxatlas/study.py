"""Reproducible, fixed-protocol model comparison and retrospective acquisition."""
from datetime import datetime, timezone
from pathlib import Path
import json
import time
import numpy as np
from .config import Experiment, X_EDGES,Y_EDGES,NX,NY
from .model import FieldSurrogate,PODRidge,FrozenSurrogate,save_model,interpolate,field_metrics
from .transport import MATERIALS
from .ballistic import ballistic_field


def study(project):
    project=Path(project);data=np.load(project/"data/fields.npz",allow_pickle=False)
    design=json.loads((project/"data/design.json").read_text())
    info=json.loads((project/"data/dataset-info.json").read_text())
    references=json.loads((project/"data/reference-metadata.json").read_text())
    x,y,se=data["x"],data["high"],data["stderr"]
    pool=np.array([i for i,r in enumerate(design) if r["split"]=="pool"])
    cal=np.array([i for i,r in enumerate(design) if r["split"]=="calibration"])
    test=np.array([i for i,r in enumerate(design) if r["split"]=="test"])
    ood=np.array([i for i,r in enumerate(design) if r["split"]=="ood"])
    low=data["low"];start=time.perf_counter()
    model=FieldSurrogate().fit(x[pool],y[pool],x[pool],low)
    training_seconds=time.perf_counter()-start
    factor=model.calibrate(x[cal],y[cal])
    t=time.perf_counter();pred,half=model.predict(x[test]);prediction_seconds=(time.perf_counter()-t)/len(test)
    report={"protocol_version":2,"created_utc":datetime.now(timezone.utc).isoformat(),
        "training_seconds":training_seconds,"prediction_seconds_each":prediction_seconds,
        "calibration":{"target_simultaneous_coverage":.9,"n":len(cal),"factor":factor,
          "scores":model.calibration_scores.tolist(),"interpretation":"Intervals cover independent finite-history OpenMC estimates under exchangeable in-domain geometry sampling; not exact physical truth or OOD fields."},
        "models":[],"active_learning":[],"ablation":{},"optimization":{
            "members":len(model.nets),"iterations":[int(n.n_iter_) for n in model.nets],
            "warnings":[w for n in model.nets for w in n.fluxatlas_convergence_warnings_]}}
    mf_metrics=field_metrics(pred,y[test],se[test],half)
    report["models"].append({"name":"Hybrid multi-fidelity neural field",**mf_metrics})
    report["models"].append({"name":"Analytical uncollided beam only",**field_metrics(ballistic_field(x[test]),y[test],se[test])})
    baseline=PODRidge().fit(x[pool],y[pool]);p=baseline.predict(x[test])
    report["models"].append({"name":"POD + kernel ridge",**field_metrics(p,y[test],se[test])})
    hybrid_pod=PODRidge().fit(x[pool],y[pool]-ballistic_field(x[pool]))
    hp=hybrid_pod.predict(x[test])+ballistic_field(x[test])
    report["models"].append({"name":"Hybrid POD + kernel ridge",**field_metrics(hp,y[test],se[test])})
    hybrid_high=FieldSurrogate(multifidelity=False,hybrid=True).fit(x[pool],y[pool])
    hybrid_high.calibrate(x[cal],y[cal]);hp,hh=hybrid_high.predict(x[test])
    report["models"].append({"name":"Hybrid high-only neural field",**field_metrics(hp,y[test],se[test],hh)})
    high=FieldSurrogate(multifidelity=False,hybrid=False).fit(x[pool],y[pool]);high.calibrate(x[cal],y[cal])
    p,h=high.predict(x[test]);report["models"].append({"name":"High-only neural field",**field_metrics(p,y[test],se[test],h)})
    p=interpolate(x[pool],y[pool],x[test]);report["models"].append({"name":"Inverse-distance interpolation",**field_metrics(p,y[test],se[test])})
    low_pred=np.maximum(model.pca.inverse_transform(model.low.predict(x[test]))*model.scale,0.) + ballistic_field(x[test])
    report["ablation"]["low_fidelity_only"]=field_metrics(low_pred,y[test],se[test])
    _,raw=model.predict(x[test],calibrated=False)
    report["ablation"]["uncalibrated_ensemble_scale"]=field_metrics(pred,y[test],se[test],raw)
    op,oh=model.predict(x[ood]);report["ood"]=field_metrics(op,y[ood],se[ood],oh)
    report["ood"].update({"detector":"Explicit unsupported geometry-family gate and parameter bounds", "detected_fraction":float(np.mean([model.out_of_domain(x[i],design[i]["config"]["family"])[0] for i in ood])),
        "in_domain_false_positive_fraction":float(np.mean([model.out_of_domain(x[i],design[i]["config"]["family"])[0] for i in test])),
        "note":"Window family is excluded from training/calibration. Detection follows declared support, not learned anomaly recognition."})
    # Fixed budgets, identical initial selections and low-fidelity pool for each strategy.
    # Test fields are used only for scoring, never selecting a candidate.
    budgets=(12,24,40,56)
    acquisition_start=time.perf_counter()
    for repeat in range(3):
        for strategy in ("active","random"):
            rng=np.random.default_rng(501+repeat)
            selected=list(map(int,rng.choice(len(pool),budgets[0],replace=False)))
            strategy_start=time.perf_counter()
            for budget in budgets:
                while len(selected)<budget:
                    estimator=FieldSurrogate(members=3,seed=901+repeat).fit(x[pool[selected]],y[pool[selected]],x[pool],low)
                    remaining=np.array([i for i in range(len(pool)) if i not in selected])
                    if strategy=="random":
                        chosen=rng.choice(remaining,min(8,budget-len(selected)),replace=False)
                    else:
                        pm,pu=estimator.predict(x[pool[remaining]],calibrated=False)
                        score=np.linalg.norm(pu,axis=1)/np.maximum(np.linalg.norm(pm,axis=1),1.e-10)
                        chosen=remaining[np.argsort(score)[-min(8,budget-len(selected)):]]
                    selected.extend(map(int,chosen))
                estimator=FieldSurrogate(members=3,seed=901+repeat).fit(x[pool[selected]],y[pool[selected]],x[pool],low)
                pm,_=estimator.predict(x[test]);metrics=field_metrics(pm,y[test])
                row={"strategy":strategy,"repeat":repeat,"selected":budget,
                    "budget_histories":int(len(pool)*info["low_histories_each"]+budget*info["high_histories_each"]),
                    "low_pool_histories":int(len(pool)*info["low_histories_each"]),
                    "cumulative_fit_selection_evaluation_seconds":time.perf_counter()-strategy_start,
                    "selected_reference_seconds":float(data["seconds"][pool[selected]].sum()+data["low_seconds"].sum()),
                    "selection":[design[pool[i]]["id"] for i in selected],**metrics}
                report["active_learning"].append(row)
                print(f"{strategy} repeat={repeat} high={budget} error={metrics['relative_l2']:.4f}",flush=True)
    # Equal wall-time caps are a separate retrospective comparison. Each policy
    # may use only completed checkpoints whose measured solver+model cost fits.
    report["compute_budget_comparison"]=[]
    for repeat in range(3):
        policies={s:[r for r in report["active_learning"] if r["repeat"]==repeat and r["strategy"]==s] for s in ("active","random")}
        cost=lambda r:r["selected_reference_seconds"]+r["cumulative_fit_selection_evaluation_seconds"]
        cap=min(max(map(cost,rows)) for rows in policies.values())
        for strategy,rows in policies.items():
            eligible=[r for r in rows if cost(r)<=cap]
            chosen=eligible[-1]
            report["compute_budget_comparison"].append({"repeat":repeat,"strategy":strategy,"wall_time_cap_seconds":cap,"charged_seconds":cost(chosen),"selected":chosen["selected"],"budget_histories":chosen["budget_histories"],"relative_l2":chosen["relative_l2"]})
    report["acquisition_study_seconds"]=time.perf_counter()-acquisition_start
    report["budget_note"]="History-matched retrospective study. All low-pool histories are charged. Surrogate fitting/acquisition overhead is reported separately, not assumed equal. Precomputed unselected high labels were never exposed to the fit/selector."
    out=project/"benchmarks";out.mkdir(exist_ok=True)
    (out/"study.json").write_text(json.dumps(report,indent=2)+"\n")
    save_model(model,project/"data/model.npz")
    frozen=FrozenSurrogate(project/"data/model.npz")
    assert np.allclose(frozen.predict(x[test])[0],pred,rtol=1.e-10,atol=1.e-12)
    cases=[]
    for i in np.concatenate([test,ood]):
        cfg=Experiment(**design[i]["config"]);t=time.perf_counter();p,h=frozen.predict(x[i]);elapsed=time.perf_counter()-t
        metrics=field_metrics(p,y[i:i+1],se[i:i+1],h)
        cases.append({"id":design[i]["id"],"split":design[i]["split"],"family":cfg.family,"config":cfg.to_dict(),
            "reference":y[i].reshape(2,-1).tolist(),"stderr":se[i].reshape(2,-1).tolist(),
            "prediction":p.reshape(2,-1).tolist(),"uncertainty":h.reshape(2,-1).tolist(),
            "metrics":{**metrics,"reference_seconds":float(data["seconds"][i]),"prediction_seconds":elapsed},
            "seed":design[i]["high_seed"],"histories":info["high_histories_each"]})
    atlas={"schema_version":1,"title":"FLUXATLAS","generated_utc":report["created_utc"],"units":"cm^-2 per source particle",
        "grid":{"nx":NX,"ny":NY,"x_edges":X_EDGES.tolist(),"y_edges":Y_EDGES.tolist(),"z_depth":1,"shape":[2,NY,NX]},
        "groups":[{"label":"Slow","lower_eV":1.e-5,"upper_eV":1.e3},{"label":"Fast","lower_eV":1.e3,"upper_eV":2.e7}],
        "materials":MATERIALS,"cases":cases,"benchmark":report,
        "provenance":{"solver":"OpenMC","version":references[0]["solver_version"],"description":"Independent OpenMC fixed-source simulations with original synthetic 2-group cross sections; analytical uncollided beam plus separately trained fixed-grid POD neural scattered-field residual ensemble.",
          "interval_level":.9,"limitations":["Educational synthetic materials; not evaluated nuclear data.","No dose, dose rate, facility measurement or safety certification.","Window geometry is an unsupported held-out family.","Intervals calibrated against independent Monte Carlo estimates, not exact physical truth.","Original implementation and experimental study; no first-in-literature claim."]}}
    (project/"web/atlas.json").write_text(json.dumps(atlas,separators=(",",":"))+"\n")
    return report
