"""A fixed-grid POD neural field surrogate with low-fidelity residual learning.

This is an original compact research implementation, not a demonstrated
resolution-independent neural operator. Neural weights are genuinely optimized.
Physical-space fields, never logarithmic noisy labels, define the POD targets.
"""
import warnings
import numpy as np
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
from .ballistic import ballistic_field


class FieldSurrogate:
    def __init__(self, multifidelity=True, members=5, seed=41, rank=18, hybrid=True):
        self.multifidelity, self.members, self.seed, self.rank = multifidelity, members, seed, rank
        self.calibration_factor = 1.0
        self.hybrid = hybrid

    def fit(self, x, y, low_x=None, low_y=None):
        self.use_ballistic = self.hybrid and y.shape[1] == 960
        if self.use_ballistic:
            y = y - ballistic_field(x)
            if low_y is not None: low_y = low_y - ballistic_field(low_x)
        # Bases/scales depend only on available training labels; never calibration/test.
        basis_data = low_y if self.multifidelity else y
        self.scale = np.maximum(np.mean(basis_data, axis=0), np.max(basis_data) * .01)
        self.pca = PCA(n_components=min(self.rank, len(basis_data) - 1), svd_solver="full")
        self.pca.fit(basis_data / self.scale)
        target = self.pca.transform(y / self.scale)
        self.low = None
        if self.multifidelity:
            self.low = KernelRidge(alpha=.04, kernel="rbf", gamma=1.5)
            self.low.fit(low_x, self.pca.transform(low_y / self.scale))
            target = target - self.low.predict(x)
        self.target_scale = np.maximum(target.std(axis=0), .05)
        self.nets = []
        rng = np.random.default_rng(self.seed)
        for i in range(self.members):
            indices = rng.choice(len(x), len(x), replace=True) if self.members > 1 else np.arange(len(x))
            net = MLPRegressor(hidden_layer_sizes=(32, 32), activation="tanh", solver="lbfgs",
                alpha=.15, max_iter=600, max_fun=20000, tol=1.e-5, random_state=self.seed + i)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                net.fit(x[indices], (target / self.target_scale)[indices])
            net.fluxatlas_convergence_warnings_ = [str(w.message) for w in caught]
            self.nets.append(net)
        self.x_train = np.array(x)
        return self

    def predict(self, x, calibrated=True):
        x = np.atleast_2d(x)
        baseline = self.low.predict(x) if self.low is not None else 0.
        fields = np.array([np.maximum(self.pca.inverse_transform(net.predict(x) * self.target_scale + baseline) * self.scale, 0)
                           for net in self.nets])
        if self.use_ballistic:
            fields = fields + ballistic_field(x)[None]
        mean = fields.mean(axis=0)
        # Ensemble spread alone can collapse. A documented magnitude-dependent floor
        # is part of the score; it is calibrated, not claimed as Bayesian variance.
        spread = fields.std(axis=0, ddof=1) if len(fields) > 1 else np.zeros_like(mean)
        raw = np.sqrt(spread**2 + (.03 * mean + .002 * mean.max(axis=1, keepdims=True))**2)
        return mean, raw * (self.calibration_factor if calibrated else 1.)

    def calibrate(self, x, y, alpha=.10):
        pred, raw = self.predict(x, calibrated=False)
        scores = np.max(np.abs(y - pred) / raw, axis=1)
        k = int(np.ceil((len(scores) + 1) * (1 - alpha)))
        if k > len(scores):
            raise ValueError("Insufficient calibration geometries for requested coverage")
        self.calibration_factor = float(np.sort(scores)[k - 1])
        self.calibration_scores = scores
        self.alpha = alpha
        return self.calibration_factor

    def out_of_domain(self, x, family="solid"):
        x = np.atleast_2d(x)
        return np.any((x < 0) | (x > 1), axis=1) | (family != "solid")


class PODRidge:
    def fit(self, x, y):
        self.scale = np.maximum(y.mean(axis=0), y.max() * .01)
        self.pca = PCA(n_components=min(18, len(y)-1), svd_solver="full").fit(y / self.scale)
        self.regressor = KernelRidge(alpha=.04, kernel="rbf", gamma=1.5).fit(x, self.pca.transform(y / self.scale))
        return self
    def predict(self, x):
        return np.maximum(self.pca.inverse_transform(self.regressor.predict(x)) * self.scale, 0.)


def interpolate(x_train, y_train, x):
    d = np.linalg.norm(x[:, None] - x_train[None], axis=-1)
    w = 1 / np.maximum(d, 1.e-10)**3
    return w @ y_train / w.sum(axis=1, keepdims=True)


def field_metrics(pred, truth, stderr=None, halfwidth=None):
    errors = np.linalg.norm(pred - truth, axis=1) / np.maximum(np.linalg.norm(truth, axis=1), 1.e-12)
    result = {"relative_l2": float(errors.mean()), "median_relative_l2": float(np.median(errors)),
              "per_case_relative_l2": errors.tolist()}
    if truth.shape[1] == 960:
        pg, tg = pred.reshape(-1,2,480), truth.reshape(-1,2,480)
        result["group_relative_l2"] = {name:float(np.mean(np.linalg.norm(pg[:,g]-tg[:,g],axis=1)/np.maximum(np.linalg.norm(tg[:,g],axis=1),1.e-12))) for g,name in enumerate(("slow","fast"))}
        result["integrated_flux_relative_error"] = float(np.mean(np.abs(pred.sum(axis=1)-truth.sum(axis=1))/np.maximum(truth.sum(axis=1),1.e-12)))
    if stderr is not None:
        result["reference_relative_standard_error"] = float(np.mean(np.linalg.norm(stderr, axis=1) / np.linalg.norm(truth, axis=1)))
        result["fraction_beyond_1_96_reference_se"] = float(np.mean(np.abs(pred-truth) > 1.96*stderr))
        valid = truth > 3 * stderr
        result["resolved_cell_relative_l2"] = float(np.linalg.norm((pred-truth)[valid]) / np.linalg.norm(truth[valid]))
        result["resolved_cell_fraction"] = float(valid.mean())
    if halfwidth is not None:
        covered = np.abs(pred - truth) <= halfwidth
        result["coverage"] = float(covered.mean())
        result["simultaneous_field_coverage"] = float(covered.all(axis=1).mean())
        result["mean_interval_halfwidth_over_mean_flux"] = float(halfwidth.mean() / truth.mean())
    return result

    # (module serialization helpers are intentionally pickle-free below)


def save_model(model, path):
    state={"scale":model.scale,"pca_mean":model.pca.mean_,"pca_components":model.pca.components_,
           "target_scale":model.target_scale,"calibration_factor":np.array(model.calibration_factor),
           "members":np.array(len(model.nets)),"use_ballistic":np.array(model.use_ballistic),"x_train":model.x_train,
           "multifidelity":np.array(model.low is not None),"alpha":np.array(getattr(model,"alpha",.1))}
    if model.low is not None:
        state.update(low_x=model.low.X_fit_,low_dual=model.low.dual_coef_,low_gamma=np.array(model.low.gamma))
    for i,net in enumerate(model.nets):
        state[f"layers_{i}"]=np.array(len(net.coefs_))
        for j,(weight,bias) in enumerate(zip(net.coefs_,net.intercepts_)):
            state[f"w_{i}_{j}"]=weight; state[f"b_{i}_{j}"]=bias
    np.savez_compressed(path,**state)


class FrozenSurrogate:
    """Read numeric weights only. No deserialization of executable Python objects."""
    def __init__(self,path):
        self.state=dict(np.load(path,allow_pickle=False))
        self.calibration_factor=float(self.state["calibration_factor"])
    def predict(self,x,calibrated=True):
        s=self.state;x=np.atleast_2d(x);baseline=0.
        if bool(s["multifidelity"]):
            distance=((x[:,None]-s["low_x"][None])**2).sum(axis=-1)
            baseline=np.exp(-float(s["low_gamma"])*distance)@s["low_dual"]
        fields=[]
        for i in range(int(s["members"])):
            h=x
            for j in range(int(s[f"layers_{i}"])):
                h=h@s[f"w_{i}_{j}"]+s[f"b_{i}_{j}"]
                if j<int(s[f"layers_{i}"])-1:h=np.tanh(h)
            fields.append(np.maximum(((h*s["target_scale"]+baseline)@s["pca_components"]+s["pca_mean"])*s["scale"],0.))
        fields=np.array(fields)
        if bool(s.get("use_ballistic",False)): fields=fields+ballistic_field(x)[None]
        mean=fields.mean(axis=0)
        spread=fields.std(axis=0,ddof=1) if len(fields)>1 else np.zeros_like(mean)
        raw=np.sqrt(spread**2+(.03*mean+.002*mean.max(axis=1,keepdims=True))**2)
        return mean,raw*(self.calibration_factor if calibrated else 1.)
    def out_of_domain(self,x,family="solid"):
        return FieldSurrogate.out_of_domain(self,x,family)
