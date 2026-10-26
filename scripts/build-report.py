#!/usr/bin/env python3
"""Generate publication charts and the technical report from recorded results."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]
s=json.loads((root/'benchmarks/study.json').read_text())
v=json.loads((root/'benchmarks/physics/validation.json').read_text())
info=json.loads((root/'data/dataset-info.json').read_text())
old=json.loads((root/'benchmarks/exploratory-v1/study.json').read_text())
figs=root/'docs/figures';figs.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'figure.facecolor':'#f4f2e9','axes.facecolor':'#f4f2e9','text.color':'#172d36','axes.labelcolor':'#172d36','axes.edgecolor':'#c5ceca','xtick.color':'#69747a','ytick.color':'#69747a','font.size':10,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'#f4f2e9'})
fig,ax=plt.subplots(figsize=(9,4.4));models=s['models'];names=[m['name'] for m in models];values=np.array([m['relative_l2'] for m in models])*100
ax.barh(names[::-1],values[::-1],color=['#d86435' if 'Multi-fidelity' in n or 'multi-fidelity' in n else '#3a7383' for n in names[::-1]])
for i,val in enumerate(values[::-1]):ax.text(val+.5,i,f'{val:.1f}%',va='center',fontsize=9)
ax.set_xlabel('Mean per-configuration relative L₂ field error (%)');ax.set_xlim(0,values.max()*1.16);ax.set_title('Fresh held-out configurations · independent OpenMC references',loc='left',pad=18);fig.tight_layout();fig.savefig(figs/'baselines.png',dpi=170);fig.savefig(figs/'baselines.svg');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(10.5,4))
for strategy,color in [('active','#d86435'),('random','#3a7383')]:
 rows=[r for r in s['active_learning'] if r['strategy']==strategy];budgets=sorted({r['budget_histories'] for r in rows});errors=np.array([[r['relative_l2']*100 for r in rows if r['budget_histories']==b] for b in budgets]);mean=errors.mean(axis=1);std=errors.std(axis=1,ddof=1)
 axs[0].plot(np.array(budgets)/1e6,mean,'o-',label=strategy,color=color);axs[0].fill_between(np.array(budgets)/1e6,mean-std,mean+std,color=color,alpha=.12)
 for repeat in range(3):
  cr=[r for r in rows if r['repeat']==repeat];axs[1].plot([r['selected_reference_seconds']+r['cumulative_fit_selection_evaluation_seconds'] for r in cr],[r['relative_l2']*100 for r in cr],'.-',color=color,alpha=.45,label=strategy if repeat==0 else None)
axs[0].set_xlabel('Charged particle histories (millions)');axs[1].set_xlabel('Measured solver + fitting + selection time (s)')
for ax in axs:ax.set_ylabel('Held-out field error (%)');ax.legend(frameon=False)
axs[0].set_title('Mean ± one SD across 3 acquisition seeds',loc='left',fontsize=10);axs[1].set_title('Individual trials, including model overhead',loc='left',fontsize=10);fig.tight_layout();fig.savefig(figs/'active-learning.png',dpi=170);fig.savefig(figs/'active-learning.svg');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(10,3.6))
for a in v['analytical']:
 ax=axs[0];ax.plot(a['x_centers'],a['analytic'],label=f"{a['group']} analytic");ax.errorbar(a['x_centers'][::3],np.array(a['mean'])[::3],yerr=np.array(a['stderr'])[::3]*3,fmt='.',capsize=2,color='#1f77b4' if a['group']=='fast' else '#ff7f0e')
axs[0].set_xlabel('x (cm)');axs[0].set_ylabel('Volume-mean flux (cm⁻² / source)');axs[0].legend(frameon=False,fontsize=8);axs[0].set_title('Pure absorber · OpenMC points ± 3 SE',loc='left',fontsize=10)
c=v['convergence'];axs[1].loglog([r['histories'] for r in c],[r['mean_relative_l2_to_independent_anchor'] for r in c],'o-',color='#3a7383');axs[1].set_xlabel('Particle histories');axs[1].set_ylabel('Relative L₂ error to reference');axs[1].set_title('Three seeds per budget · 400k-history reference',loc='left',fontsize=10);fig.tight_layout();fig.savefig(figs/'physics-checks.png',dpi=170);fig.savefig(figs/'physics-checks.svg');plt.close(fig)
rows='\n'.join(f"| {m['name']} | {100*m['relative_l2']:.2f}% | {100*m['group_relative_l2']['slow']:.2f}% | {100*m['group_relative_l2']['fast']:.2f}% |" for m in models)
m=models[0];latency=s['prediction_seconds_each'];ref=np.load(root/'data/fields.npz')['seconds'][120:144].mean()
active={strategy:np.array([r['relative_l2'] for r in s['active_learning'] if r['strategy']==strategy and r['selected']==56]) for strategy in ('active','random')}
acq='\n'.join(f"| {strategy} | {100*a.mean():.2f}% | {100*a.std(ddof=1):.2f} percentage points |" for strategy,a in active.items())
compute='\n'.join(f"| {r['repeat']+1} | {r['strategy']} | {r['wall_time_cap_seconds']:.2f} | {r['charged_seconds']:.2f} | {r['selected']} | {100*r['relative_l2']:.2f}% |" for r in s['compute_budget_comparison'])
text=f'''# FLUXATLAS technical report

Version 0.1.0 · Protocol {s['protocol_version']} · Original educational computational experiment

## What ran

OpenMC 0.16.0 independently simulated a non-fissioning fixed-source neutron benchmark with original synthetic two-group cross sections. The released hybrid surrogate combines an exact uncollided-beam component with a genuinely trained POD neural residual ensemble for the scattered field. It is a fixed-grid spatial model, not a demonstrated resolution-independent neural operator. No physical measurements, real facility model, dose prediction or professional safety validation are claimed.

The final primary model's mean relative L₂ error is **{100*m['relative_l2']:.2f}%** on 24 fresh held-out configurations. The best baseline in this experiment is **{min(models,key=lambda r:r['relative_l2'])['name']}** at **{100*min(r['relative_l2'] for r in models):.2f}%**. Read these results together; the hybrid neural component is not assumed to beat every conventional method.

## Experiment and units

Domain: x = 0–20 cm, y = −8–8 cm, z = −0.5–0.5 cm. Vacuum boundaries in x/y and reflective z boundaries make a thin extruded transport problem, not a realistic unrestricted 3D shield. Mesh: 24 × 20 × 1, cell volume 2/3 cm³. The shield occupies x = 6 to 6 + thickness and y = −6 to 6 cm. The source is uniform over y = source_y ±0.4 cm and z = ±0.49 cm at x = 0.5 cm, with monodirectional +x emission. It mixes 1 eV and 1 MeV neutrons.

Supported parameter box: thickness 1–5 cm, synthetic shield macroscopic cross-section scale 0.6–1.4, source_y −3–3 cm, fast fraction 0.6–1.0. OOD window shields remove |y| < 1 cm from the shield; these are excluded from training and calibration. The ballistic component deliberately assumes the supported solid shield, so the window family tests a real model limitation.

Energy edges in ascending tally order are [10⁻⁵, 10³, 2×10⁷] eV: slow then fast. OpenMC XSdata matrices use descending group order: fast then slow. Isotropic order-zero scattering, no fission, and synthetic macroscopic coefficients in cm⁻¹ are specified in `transport.py`. Background totals fast/slow: 0.025/0.04; fast→fast/slow scattering: 0.012/0.008; slow→slow: 0.025. Shield totals: 0.65/1.10; fast→fast/slow: 0.28/0.22; slow→slow: 0.65. Absorption is total minus row scattering sum. The shield scale multiplies all coefficients equally.

OpenMC track-length flux scores first have units cm per source particle. Dividing both mean and standard error by mesh-cell volume gives the volume-mean scalar-flux field in **cm⁻² per source particle**. Groups are energy-integrated, not per-eV densities. No source strength is assumed; flux-per-source must not be relabeled dose or dose rate. Batches provide an estimated standard error of each tally mean, not an exact confidence guarantee.

## Numerical verification

![Physics checks](figures/physics-checks.png)

Pure homogeneous absorber runs use a collimated source and an exactly integrated exponential mesh average. Fast/slow relative L₂ errors are {100*v['analytical'][0]['relative_l2']:.3f}% / {100*v['analytical'][1]['relative_l2']:.3f}%, with zero wrong-group tally. The executable validation gate requires <2.5% field error, zero wrong-group tally and ≥80% of cells within three reported standard errors. This verifies the executed normalization and group mapping in the checked regime; exponential attenuation is not the full scattering solution.

The repeated-seed convergence study uses 8k, 32k and 128k histories with three independent seeds each against an independent 400k-history reference. Mean field errors are {', '.join(f"{100*r['mean_relative_l2_to_independent_anchor']:.3f}%" for r in c)}. The anchor itself remains noisy. These validation runs are distinct from model training/test cases.

## Training and independence

There are 96 training-pool configurations, 24 reserved calibration configurations, 24 final in-domain test configurations and 12 held-out window geometries. Each high reference uses 40,000 histories in 20 batches. Each of 96 low-fidelity references uses 8,000 histories in 20 batches, independently seeded from its high reference. Here fidelity means statistical precision at fixed physics and mesh, not a different nuclear-data accuracy level.

The training pool uses Latin-hypercube seed 713. After the exploratory failure described below, final calibration/test/OOD settings were replaced with a predeclared fresh Latin-hypercube design (seed 271828) and fresh reference seeds. No final test geometry or label enters basis construction, optimization, calibration or acquisition. Hyperparameters were frozen before reading final results. The model is small enough to retrain locally with `fluxatlas train`.

The deterministic beam integrates attenuation across source and shield interfaces and transverse subcell source overlap. Subtracting it from low/high labels leaves a smoother scattered field, still carrying Monte Carlo noise. No logarithm of noisy training labels is taken. A training-only scale and 18-mode POD basis compress these fields. A low-fidelity RBF kernel-ridge map (gamma 1.5, alpha 0.04) predicts coefficients from normalized parameters. Five bootstrapped MLPs, each with 32/32 tanh units, learn high-fidelity residual coefficients using L-BFGS and L2 regularization alpha 0.15. A nonnegative reconstructed scattered field is added to the beam. Clipping, rank truncation, regularization, finite samples and noisy targets can introduce error. Neural optimization warnings are preserved in `study.json`; the primary fit reported {len(s['optimization']['warnings'])} warning(s). Total primary training time was {s['training_seconds']:.3f} s on this run.

Numeric-only NPZ exports store the actual optimized weights. A separate NumPy inference implementation is tested against the trained model. No executable pickle artifact is loaded.

## Held-out model comparison

![Baselines](figures/baselines.png)

| Model | Both-group L₂ | Slow-group L₂ | Fast-group L₂ |
|---|---:|---:|---:|
{rows}

Errors are computed per configuration then averaged. Both-group L₂ weights energetic/high-flux cells strongly. `study.json` additionally records integrated-flux error, resolved-cell error (reference mean > 3 SE), per-case error and the fraction of errors beyond 1.96 reference SE. This last fraction is a diagnostic, not a statistical test of correct physics. It excludes neither model bias nor correlated-bin uncertainty.

For the hybrid model, mean reference relative L₂ standard error is {100*m['reference_relative_standard_error']:.2f}%. Model error can substantially exceed Monte Carlo noise. Zero-count bins are not exact zero physical flux; the observatory masks error/SE ratios where SE is zero. Total-group SE uses a conservative sum of group SEs because cross-group covariance is not stored.

## Uncertainty calibration

The raw scale combines bootstrapped-ensemble spread with a declared magnitude floor: sqrt(spread² + (0.03×prediction + 0.002×case_peak)²). It is an acquisition/interval scale, not Bayesian posterior variance. Reserved calibration uses one maximum standardized absolute error over all 960 entries per geometry. For alpha=0.10 and n=24, the finite-sample conformal order statistic is ceil(25×0.90)=23. Its fitted scale multiplier is {s['calibration']['factor']:.3f}.

Final empirical simultaneous whole-field coverage is **{100*m['simultaneous_field_coverage']:.1f}%** ({round(m['simultaneous_field_coverage']*24)}/24 fields). Marginal cell coverage is {100*m['coverage']:.2f}%; these are different quantities. Mean half-width divided by mean reference flux is {m['mean_interval_halfwidth_over_mean_flux']:.2f}, which exposes the cost of wide fieldwise intervals. The sample is small; no per-configuration conditional guarantee is established.

The calibration target is an independently simulated finite-history OpenMC estimate under the same in-domain sampling design. It does not establish coverage of exact physical truth, another source/mesh/material family, an evaluated continuous-energy result or an actual facility. OOD interval coverage is reported only as a failure diagnostic ({100*s['ood']['simultaneous_field_coverage']:.1f}% whole-field); it is not calibrated. The OOD detector is an explicit family/bounds gate, with measured window detection {100*s['ood']['detected_fraction']:.1f}% and in-domain false positives {100*s['ood']['in_domain_false_positive_fraction']:.1f}%. It is not learned discovery of arbitrary anomalies.

## Active learning and computational costs

![Active learning](figures/active-learning.png)

The retrospective acquisition study uses three seeds with identical initial 12 high-history configurations for each active/random pair. Checkpoints contain 12, 24, 40 and 56 high-history labels. Both policies receive the same full low-history candidate pool, and all 96 × 8,000 low histories are charged at every checkpoint. Only selected high labels enter model fits. The selector never sees test references; they are used only to score fixed checkpoints. Active acquisition ranks candidate relative ensemble-scale norms and acquires batches of at most eight. This is a compact heuristic, not a claimed optimal information gain method. All high pool labels were precomputed for reproducible policy replay; the release reports that full dataset-generation cost separately.

At 56 high labels (3,008,000 charged histories):

| Strategy | Mean test L₂ | SD across acquisition seeds |
|---|---:|---:|
{acq}

Three repeats support a small empirical comparison, not universal superiority or a precise significance claim. `active_learning` rows contain acquisition identifiers, per-case test errors, solver costs and cumulative fitting/selection/evaluation overhead. Mean test predictions take {latency*1000:.3f} ms per case in a 24-case batch, compared with mean reference setup+execution+extraction time {ref:.3f} s. A single interactive query has different overhead. These are local measured times on shared Apple-arm64 resources, not a controlled hardware performance claim; simulation startup/import, current load and numerical libraries affect timing. No reference simulation is required for model inference. Training, low-pool generation and acquisition costs must be paid before inference savings can accumulate.

To compare under equal measured computational caps as well as equal histories, each repetition uses a shared cap that both policies can afford, then selects its last completed checkpoint within that cap. Unused budget is explicit; this is retrospective checkpoint accounting, not a live equal-time scheduler. Charged cost includes measured low+selected-high solver wall time and cumulative model overhead. Cross-run load affects these costs.

| Repeat | Policy | Equal cap (s) | Charged (s) | High labels | Test L₂ |
|---|---|---:|---:|---:|---:|
{compute}

Full final dataset generation includes {info['generation_high_histories']:,} high and {info['generation_low_histories']:,} low histories, including validation/test references needed for the research evaluation. Analytical and convergence runs are additional. The separately preserved exploratory dataset also has its own real cost; it is not part of the final policy's selected-label budget.

## Failure case and research iteration

The initial direct full-field POD neural model had **{100*old['models'][0]['relative_l2']:.2f}%** mean field error on the exploratory test set; the narrow moving beam was poorly represented by that small model. Its weights, design, fields, source snapshots and metrics remain in `benchmarks/exploratory-v1/`. Those inspected tests were not reused as final evaluation. The revision adds known uncollided physics, learns only the smoother scattered field, and uses fresh calibration/test/OOD configurations. This is an observed engineering/research improvement, not a pre-registered single-attempt success. The original source snapshots predate `hybrid` and contain all required v1 model code; they are reference records, not installed modules.

Other limits include a fixed 24×20 mesh, one non-fissioning synthetic material family, a thin reflective-z experiment, a small training set, limited geometry variation, no CAD/DAGMC execution, no continuous-energy validation and no physical dose-rate conversion. Better interpolation or reduced-order methods may outperform the neural model. The interface preserves individual failure cases instead of showing only favorable maps.

## Reproduction and extension

Follow `docs/reproducibility.md`. `data/design.json`, `data/reference-metadata.json`, `data/fields.npz`, `data/model.npz`, `benchmarks/study.json` and `benchmarks/physics/validation.json` are the numerical record. `web/atlas.json` derives from those fields and weights. Cached runs include source/config+solver fingerprints so changing physics cannot silently reuse stale bytes. Cross-platform Monte Carlo reductions and math libraries may differ slightly; validate tolerances and report hardware.

For continuous energy, obtain a separately licensed, suitable evaluated nuclear library and configure OpenMC explicitly, then implement and revalidate a separate adapter. No continuous-energy results are included. DAGMC is an optional future extension, not a supported UI capability. Method foundations, recent related literature and the PaperVizAgent decision are linked in [research-foundations.md](research-foundations.md); licenses are in [licensing.md](licensing.md). Editable conceptual architecture: [SVG](architecture.svg) and [Mermaid](architecture.mmd). Numerical figure generation is `scripts/build-report.py`.
'''
(root/'docs/technical-report.md').write_text(text)
print('Generated charts and technical report from',s['created_utc'])

# Keep machine-generated SVG path formatting clean in source control.
for svg in figs.glob("*.svg"):
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")
