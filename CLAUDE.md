# sterile_calc

Reproduces Figs. 2-5 of Carney, Leach & Moore, PRX Quantum 4, 010315 (2023)
— sterile-neutrino sensitivity projections for levitated-nanosphere
detectors. Paper PDF in `refs/sterile_paper.pdf`, extracted text in
`refs/paper_text.txt`.

## Layout

- `sterile_sens/` — the package
  - `isotopes.py` — isotope registry dicts (decay Q-values, branching, etc.)
  - `detector.py` — experiment parameters as a `Detector` dataclass
  - `materials.py` — approximate CSDA range tables
  - `spectra.py`, `montecarlo.py` — beta/EC spectrum generation, MC sampling
  - `sensitivity.py` — binned Asimov profile-likelihood sensitivity
    (q95 = 2.706); `toy_limit` is a Monte Carlo cross-check
  - `plotting.py` — figure generation
- `run_all.py` — entry point, e.g. `python run_all.py fig2` (or `fig3`,
  `fig4`, `fig5`), with overrides `--diameter`, `--loading`, `--trap-khz`,
  `--sub-sql`, `--exposure`. Outputs go to `results/`.
- `data/existing_limits.csv` — schematic overlay of existing experimental
  limits (approximate, not authoritative)

## Environment

Runs in the `nanospheres` conda env (numpy 2.2, scipy 1.15, matplotlib
3.10); see `requirements.txt`.

## Known approximations

- CSDA range tables in `materials.py` are approximate
- Existing-limits overlay (`data/existing_limits.csv`) is schematic
- EC de-excitation models a single secondary only

Validated against the paper's Table I-II live times: within ~20% except
3H (factor ~2 off).

## Phase-space factor finding

The paper's Figs. 4-5 appear to omit the heavy sterile-state phase-space
factor r(m4) when converting event counts to |Ue4|^2 — their sensitivity
floors are flat out to the kinematic endpoint, which would otherwise imply
<1 signal event. `sensitivity.py` exposes an `include_phase_space` switch:
- `False` (paper convention, default for plotting) reproduces their curves
  quantitatively (32P floor 1.2e-5, 90Y 7e-6)
- `True` is the physically correct default of `sensitivity_curve`
