# Sterile-neutrino sensitivity projections with levitated nanoparticles

Reproduces the calculations and figures of **Carney, Leach & Moore,
"Searches for Massive Neutrinos with Mechanical Quantum Sensors",
PRX Quantum 4, 010315 (2023)** (`refs/sterile_paper.pdf`): searches for a
heavy neutrino mass state m4 mixing with the electron flavor, via momentum
reconstruction of nuclear decays inside an optically levitated nanosphere
monitored at the standard quantum limit (SQL).

## Quick start

```bash
pip install -r requirements.txt
python run_all.py                 # all four figures -> results/  (~30 s)
python run_all.py fig4 fig5       # only the sensitivity figures
```

Or open **`sterile_neutrino_demo.ipynb`** for a guided walkthrough of the
calculation (SQL noise -> kinematics -> Monte Carlo reconstruction ->
profile-likelihood limit -> sensitivity curves), with all outputs pre-run.

Change experimental parameters from the command line (applied to every
detector):

```bash
python run_all.py fig2 --iso 51Cr --m4 500 --ue4sq 5e-4
python run_all.py fig4 --exposure 90            # sphere-days
python run_all.py fig4 --diameter 150 --loading 0.02 --trap-khz 50
python run_all.py fig5 --sub-sql 0.1            # 20 dB below the SQL
```

or from Python, where everything is exposed:

```python
from sterile_sens import Detector, sensitivity_curve, get_isotope

det = Detector(diameter_nm=75, loading=0.05, f_trap_hz=50e3,
               trigger_eff=0.5, eta=(0.4, 0.4, 0.6))
m4, ue4sq_95 = sensitivity_curve(get_isotope("32P"), det,
                                 exposure_sphere_days=10 * 365)
```

New isotopes are plain entries in `sterile_sens/isotopes.py`
(`EC_ISOTOPES` / `BETA_ISOTOPES`).

## What is computed (mapping to the paper)

| Output | Paper | Content |
|---|---|---|
| `results/fig2_*.png` | Fig. 2 | Reconstructed neutrino momentum spectrum for 37Ar EC (30 days, 100-nm sphere, 1% loading) with a sterile peak at m4 = 750 keV, \|Ue4\|^2 = 2e-4 |
| `results/fig3_*.png` | Fig. 3 | Reconstructed \|p_nu\| vs measured T_e for 32P beta decay; 2-sigma contours for several m4 |
| `results/fig4_*.png` | Fig. 4 | Median 95% C.L. sensitivity to \|Ue4\|^2 vs m4, EC (37Ar, 49V, 51Cr, 68Ge, 72Se) and beta (32P, 90Y), 1 nanosphere x 1 month (7Be available but, as in the paper's figure, not shown) |
| `results/fig5_*.png` | Fig. 5 | 3H (25 nm polymer, 1 kHz, 20% loading), 35S (50 nm, 10 kHz), 32P (100 nm, 100 kHz) for 1 sphere-month, 10 sphere-yr, 1000 sphere-yr |

Physics implemented (Secs. II-III):

* **SQL momentum noise** (Eq. 5): sigma_p = sqrt(hbar m_s omega_s)/sqrt(eta_i)
  per axis, eta = (0.4, 0.4, 0.6).
* **EC kinematics** (Eq. 7): two-body decay, p_nu = sqrt(E_nu^2 - m4^2), with
  per-branch neutrino energies and secondary Auger e-/x-ray/gamma emission.
* **Beta kinematics** (Eq. 9): allowed spectrum with massive-neutrino phase
  space; observables (\|p_nu\|, T_e) fitted in 2D.
* **Monte Carlo detector model**: decay position uniform in the sphere;
  nuclear recoil always stops; gammas escape freely; electrons transported
  with CSDA ranges (approximate ESTAR / Ashley-Anderson tables in
  `materials.py`); secondary angle resolution 0.02 rad; electron energy
  resolution sigma_E/E = 0.01 sqrt(1 MeV/E); 40% secondary tagging
  efficiency; spheres replaced every half-life (steady activity).
* **Statistics** (Sec. III C): binned extended likelihood, light + heavy
  component with the light yield profiled; median expected 95% C.L. limit via
  the Asimov data set with q(s95) = 2.706 (a toy-MC cross-check,
  `sterile_sens.sensitivity.toy_limit`, agrees with the Asimov limit).
  \|Ue4\|^2 = s95 / (N_decays x phase-space factor x efficiency).

## Phase-space convention (Figs. 4-5)

The decay rate into a heavy state is suppressed by the phase-space factor
r(m4) = Gamma(m4)/Gamma(0) (Shrock 1980); a physically correct limit is
\|Ue4\|^2 = s95 / (N_decays x r(m4) x efficiency).  The paper's Figs. 4-5,
however, show sensitivity floors that stay *flat* up to the kinematic
endpoint -- with r(m4) included those curves would correspond to < 1 detected
signal event at high m4, which no 95% C.L. limit can do.  The paper's curves
are therefore reproduced with the conversion *omitting* r(m4)
(`include_phase_space=False`, the default of `plotting.fig4/fig5`, noted on
the figures).  Pass `include_phase_space=True` (or `run_all.py --phase-space`,
saved with the `_physical` suffix) for the physical limit, which is weaker by
1/r(m4) at high m4 (up to ~30x near the endpoints).

## Validation against the paper

* SQL: 1.05 fg, 100 kHz -> p_SQL = 15.6 keV/c (paper: "15 keV x ...").
* Live time for 1e4 detected events (paper Tables I-II, sphere-days):
  7Be 20 (24), 37Ar 8.3 (9), 49V 105 (119), 51Cr 9.1 (9), 68Ge 122 (147),
  72Se 3.5 (4), 32P 2.6 (2.8), 35S 142 (189), 90Y 1.4 (1.7), 3H 522 (252).
  Differences reflect assumptions about density and the atomic-relaxation
  model that are not fully specified in the paper.
* Fig. 4 (paper convention, 1 sphere x 1 month): 32P floor 1.2e-5 flat to
  the 1710-keV endpoint (paper ~1.2e-5), 90Y ~7e-6 (paper ~8e-6), 37Ar
  ~3.8e-5 (paper ~6e-5), 49V ~5e-4 (paper ~6e-4), 51Cr reproduces the paper's
  characteristic sensitivity spike near m4 ~ 620 keV (heavy peak of the
  ground-state branch hiding under the 320-keV excited-branch peak), 72Se and
  68Ge walls at 361 / 107 keV.  Residual offsets (<~1.7x, e.g. 37Ar) are at
  the level of the statistical convention (asymptotic 1.35-event floor here
  vs the paper's unspecified toy-fit convention).
* Fig. 5: 32P at 1000 spheres x 1 yr reaches ~1e-9, as in the paper; the 3H
  curves are ~3-5x weaker than the paper's, consistent with the paper's more
  optimistic (unspecified) low-energy electron resolution.

## Known simplifications

* Atomic relaxation after EC is reduced to a single secondary per decay with
  exclusive probabilities (no full Auger cascades).
* Beta-neutrino angular correlation neglected (isotropic emission), as the
  paper's spectral-shape-independence argument suggests.
* Backgrounds are assumed negligible (Sec. III D assumption of the paper);
  nuclear-recoil escape from the sphere (<~5% worst case) is not simulated.
* `data/existing_limits.csv` is a *schematic* envelope of existing limits for
  plot context; replace it with a proper digitization for quantitative use.
* The asymptotic (Asimov) limit is used; in the fully background-free regime
  the exact Poisson limit would be ~2x weaker (s95 = 3.0 vs 1.35 events).

## Layout

```
sterile_sens/
  constants.py     units, SQL formula (Eq. 5)
  materials.py     electron CSDA range tables (silica, polymer)
  isotopes.py      Tables I & II isotope data  <-- add isotopes here
  detector.py      Detector dataclass           <-- experimental parameters
  spectra.py       beta spectra, phase-space factors
  montecarlo.py    event generation + reconstruction
  sensitivity.py   NLL / Asimov limit machinery
  plotting.py      figure functions (fig2..fig5)
run_all.py         CLI driver
data/              schematic existing limits
results/           output figures
```
