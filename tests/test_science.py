import json
from pathlib import Path
import numpy as np
import pytest
from fluxatlas.config import Experiment,CELL_VOLUME,NX,NY
from fluxatlas.dataset import design
from fluxatlas.model import FieldSurrogate,FrozenSurrogate,save_model,interpolate


def test_config_rejects_invalid_physical_inputs():
    for key,value in [("thickness",0),("density_scale",float("nan")),("source_y",10),("fast_fraction",True),("family","reactor")]:
        with pytest.raises(ValueError):Experiment(**{key:value})
    assert np.isclose(CELL_VOLUME*NX*NY,320.)


def test_split_and_seed_independence():
    records=design();assert records==design()
    high={r["high_seed"] for r in records};low={r["low_seed"] for r in records}
    assert len(high)==len(records) and high.isdisjoint(low)
    assert {r["config"]["family"] for r in records if r["split"]=="ood"}=={"window"}
    assert all(r["config"]["family"]=="solid" for r in records if r["split"]!="ood")
    assert len({json.dumps(r["config"],sort_keys=True) for r in records})==len(records)


def test_numeric_weight_export_matches_trained_network(tmp_path):
    rng=np.random.default_rng(90);x=rng.random((35,4));positions=np.linspace(0,1,16)
    y=np.exp(-x[:,0,None]*positions)*(1+x[:,1,None])
    model=FieldSurrogate(members=2,rank=6).fit(x[:20],y[:20],x[:20],y[:20]*.98)
    factor=model.calibrate(x[20:30],y[20:30],alpha=.2)
    path=tmp_path/"weights.npz";save_model(model,path);loaded=FrozenSurrogate(path)
    for original,restored in zip(model.predict(x[30:]),loaded.predict(x[30:])):
        np.testing.assert_allclose(original,restored,rtol=1.e-10,atol=1.e-12)
    assert factor>0 and np.isfinite(factor)
    scores=model.calibration_scores
    assert factor==np.sort(scores)[int(np.ceil((len(scores)+1)*.8))-1]
    assert loaded.out_of_domain([[.5]*4],"window")[0]


def test_interpolator_reproduces_training_points():
    x=np.eye(4);y=np.arange(32).reshape(4,8)
    np.testing.assert_allclose(interpolate(x,y,x),y,atol=1.e-10)


@pytest.mark.openmc
def test_live_openmc_energy_order_and_normalization(tmp_path):
    pytest.importorskip("openmc")
    from fluxatlas.validation import analytical_absorption
    for fast in (False,True):
        result=analytical_absorption(tmp_path/str(fast),fast,particles=3000)
        assert result["relative_l2"]<.03
        assert result["wrong_group_max"]==0


def test_ballistic_beam_integrates_source_fraction_and_attenuation():
    from fluxatlas.ballistic import ballistic_field
    from fluxatlas.config import X_EDGES,Y_EDGES
    cfg=Experiment(fast_fraction=1.)
    field=ballistic_field(cfg.vector()).reshape(2,NY,NX)
    assert np.all(field[0]==0)
    assert np.all(field>=0)
    ix=2;left,right=X_EDGES[ix:ix+2]
    # Upstream of the shield: integrating across y recovers exact 1-D beam.
    actual=np.sum(field[1,:,ix]*np.diff(Y_EDGES))
    exact=(np.exp(-.025*(left-.5))-np.exp(-.025*(right-.5)))/(.025*(right-left))
    assert np.isclose(actual,exact,rtol=1.e-12)
    shifted=ballistic_field(Experiment(source_y=2.,fast_fraction=1.).vector()).reshape(2,NY,NX)
    np.testing.assert_allclose(shifted.sum(axis=1),field.sum(axis=1),rtol=1.e-12,atol=1.e-12)


def test_released_numeric_model_reproduces_observatory_fields():
    from fluxatlas.cli import project_root
    root=project_root()
    if not (root/'data/model.npz').exists():pytest.skip('release dataset unavailable')
    archive=json.loads((root/'web/atlas.json').read_text())
    model=FrozenSurrogate(root/'data/model.npz')
    for record in archive['cases']:
        config=Experiment(**record['config']);pred,half=model.predict(config.vector())
        np.testing.assert_allclose(pred.reshape(2,480),record['prediction'],rtol=1.e-9,atol=1.e-11)
        np.testing.assert_allclose(half.reshape(2,480),record['uncertainty'],rtol=1.e-9,atol=1.e-11)
