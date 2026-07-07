---
name: sterile-calc-project
description: "sterile_calc reproduces Carney/Leach/Moore PRX Quantum 4, 010315 (2023) sterile-neutrino sensitivity projections for levitated nanospheres"
metadata: 
  node_type: memory
  type: project
  originSessionId: 66dbbb01-d33e-416c-a5a9-42465fab4f26
---

`C:\Users\yuhan\sterile_calc` (as of 2026-07-06) is a Python package
(`sterile_sens/`) reproducing Figs. 2-5 of Carney, Leach & Moore, PRX Quantum
4, 010315 (2023) — the paper PDF is in `refs/`, extracted text in
`refs/paper_text.txt`. Entry point: `python run_all.py [fig2..fig5]` with CLI
overrides (--diameter, --loading, --trap-khz, --sub-sql, --exposure); outputs
to `results/`. Isotopes are registry dicts in `isotopes.py`; experiment
parameters in the `Detector` dataclass (`detector.py`). Sensitivity uses a
binned Asimov profile likelihood (q95=2.706); `toy_limit` cross-checks it.
Known approximations documented in README: schematic existing-limits overlay
(`data/existing_limits.csv`), approximate CSDA tables in `materials.py`,
single-secondary EC relaxation. Validated against paper Tables I-II live
times (all within ~20% except 3H, factor ~2). Important finding: the paper's
Figs. 4-5 evidently omit the heavy-state phase-space factor r(m4) in the
count-to-|Ue4|^2 conversion (their floors are flat to the kinematic endpoint,
implying <1 signal event otherwise); `include_phase_space` switch added —
False (paper convention) is the plotting default and reproduces their curves
quantitatively (32P floor 1.2e-5, 90Y 7e-6), True is the physical default of
`sensitivity_curve`. Runs in the user's
`nanospheres` conda env (numpy 2.2, scipy 1.15, matplotlib 3.10).
