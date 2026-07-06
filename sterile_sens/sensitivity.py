"""Projected sensitivity to |U_e4|^2 vs m4.

Method (Sec. III C of the paper): PDFs of the reconstructed observables are
built from the MC for the light-neutrino-only case and for each heavy mass
m4.  The median expected 95% C.L. upper limit is obtained from a binned
extended likelihood with a light + heavy component, profiling the light-state
normalization.  We use the Asimov data set (expected counts for the
light-only hypothesis), which reproduces the median of toy experiments; a
toy-MC cross-check is provided in `toy_limit`.

Test statistic: q(s) = 2 [NLL_prof(s) - NLL_min];  q(s95) = 2.706 (one-sided
95% C.L.).
"""

import numpy as np
from scipy.optimize import brentq

from .detector import Detector, detector_for
from .isotopes import ECIsotope, BetaIsotope, EC_ISOTOPES, BETA_ISOTOPES
from .montecarlo import simulate_ec, simulate_beta

SPHERE_MONTH = 30.0     # days; the paper's "1 nanosphere x 1 month"
Q95 = 2.7055            # (1.6449)^2, one-sided 95% CL


# --------------------------------------------------------------------------
# binned PDFs
# --------------------------------------------------------------------------
def _histogram(sample, iso, det, bins):
    """Normalized histogram of the MC observables (1D for EC, 2D for beta)."""
    if sample.Te is None:
        h, _ = np.histogram(sample.p_nu, bins=bins[0])
    else:
        h, _, _ = np.histogram2d(sample.p_nu, sample.Te, bins=bins)
        h = h.ravel()
    n = h.sum()
    return h / n if n > 0 else h


def _default_bins(iso, det):
    if isinstance(iso, ECIsotope):
        pmax = 1.35 * iso.max_E_nu
        width = max(det.sigma_p_axes.min() / 2.0, pmax / 400)
        return (np.arange(0.0, pmax, width),)
    pmax = 1.15 * iso.Q
    return (np.linspace(0, pmax, 60), np.linspace(0, 1.02 * iso.Q, 60))


# --------------------------------------------------------------------------
# Asimov profile-likelihood upper limit
# --------------------------------------------------------------------------
def _profiled_nll(s, n_obs, p0, p4):
    """min over nu of NLL(nu, s) for expected mu = nu*p0 + s*p4."""
    # Newton iterations on d(NLL)/d(nu) = 1 - sum n*p0/mu = 0
    nu = max(n_obs.sum() - s, 1.0)
    live = (p0 > 0) | (p4 > 0)
    p0l, p4l, nl = p0[live], p4[live], n_obs[live]
    for _ in range(200):
        mu = np.clip(nu * p0l + s * p4l, 1e-300, None)
        f = 1.0 - np.sum(nl * p0l / mu)
        fp = np.sum(nl * (p0l / mu) ** 2)
        step = -f / fp if fp > 0 else 0.0
        nu_new = max(nu + step, 1e-9)
        if abs(nu_new - nu) < 1e-10 * max(nu, 1.0):
            nu = nu_new
            break
        nu = nu_new
    mu = np.clip(nu * p0l + s * p4l, 1e-300, None)
    return (nu + s) - np.sum(nl * np.log(mu))


def nll_upper_limit(n_exp_light, p0, p4, n_obs=None, q_crit=Q95):
    """95% CL upper limit on the number of detected heavy-state events.

    n_obs defaults to the Asimov data (n_exp_light * p0), giving the median
    expected limit.
    """
    if n_obs is None:
        n_obs = n_exp_light * p0
    # distinguishable part of the signal PDF -- if p4 == p0 no limit exists
    nll0 = _profiled_nll(0.0, n_obs, p0, p4)

    def q(s):
        return 2.0 * (_profiled_nll(s, n_obs, p0, p4) - nll0)

    s_hi = max(10.0, 3 * np.sqrt(n_exp_light))
    for _ in range(60):
        if q(s_hi) > q_crit:
            break
        s_hi *= 2.0
    else:
        return np.inf
    return brentq(lambda s: q(s) - q_crit, 0.0, s_hi, xtol=1e-3, rtol=1e-6)


def toy_limit(rng, n_exp_light, p0, p4, n_toys=200, q_crit=Q95):
    """Median 95% CL limit from toy experiments (cross-check of Asimov)."""
    lims = []
    for _ in range(n_toys):
        n_obs = rng.poisson(n_exp_light * p0)
        lims.append(nll_upper_limit(n_exp_light, p0, p4, n_obs=n_obs, q_crit=q_crit))
    return np.median(lims)


# --------------------------------------------------------------------------
# |U_e4|^2 sensitivity curve
# --------------------------------------------------------------------------
def _simulate(rng, iso, det, m4, n):
    if isinstance(iso, ECIsotope):
        return simulate_ec(rng, iso, det, m4=m4, n=n)
    return simulate_beta(rng, iso, det, m4=m4, n=n)


def sensitivity_curve(iso, det: Detector = None, exposure_sphere_days=SPHERE_MONTH,
                      m4_grid=None, n_mc=200_000, seed=1,
                      include_phase_space=True):
    """Median expected 95% CL upper limit on |U_e4|^2 vs m4 [keV].

    Returns (m4_grid, limits).  `exposure_sphere_days` may be an array; then
    `limits` has shape (n_exposures, n_masses) -- the MC PDFs are reused.

    include_phase_space: if True (physical), the heavy-state decay rate is
    suppressed by r(m4) = Gamma(m4)/Gamma(0) when converting the fitted signal
    count to |U_e4|^2.  Set False to reproduce the paper's Figs. 4-5, whose
    flat sensitivity floors up to the kinematic endpoint indicate this factor
    was not applied there (a correct 95% CL limit can never correspond to
    fewer than ~1.4 detected signal events, which the paper's curves would
    otherwise imply at high m4).
    """
    rng = np.random.default_rng(seed)
    det = det or detector_for(iso.name)
    Q_max = iso.max_E_nu if isinstance(iso, ECIsotope) else iso.Q
    if m4_grid is None:
        m4_grid = Q_max * np.concatenate([
            np.linspace(0.03, 0.9, 14), [0.93, 0.96, 0.98, 0.995]])

    exposures = np.atleast_1d(np.asarray(exposure_sphere_days, dtype=float))
    n_decays = np.array([det.decays(iso.A, iso.halflife_days, T) for T in exposures])

    bins = _default_bins(iso, det)
    light = _simulate(rng, iso, det, 0.0, n_mc)
    p0 = _histogram(light, iso, det, bins)
    det_rate_light = light.rate_factor * light.efficiency   # detected / decay

    limits = np.full((len(exposures), len(m4_grid)), np.inf)
    for j, m4 in enumerate(m4_grid):
        heavy = _simulate(rng, iso, det, m4, n_mc)
        factor = heavy.rate_factor if include_phase_space else heavy.branch_factor
        det_rate_heavy = factor * heavy.efficiency
        if det_rate_heavy <= 0:
            continue
        p4 = _histogram(heavy, iso, det, bins)
        for i, N_dec in enumerate(n_decays):
            s95 = nll_upper_limit(N_dec * det_rate_light, p0, p4)
            limits[i, j] = s95 / (N_dec * det_rate_heavy)

    if np.isscalar(exposure_sphere_days):
        return m4_grid, limits[0]
    return m4_grid, limits


def get_isotope(name):
    if name in EC_ISOTOPES:
        return EC_ISOTOPES[name]
    if name in BETA_ISOTOPES:
        return BETA_ISOTOPES[name]
    raise KeyError(f"unknown isotope {name!r}")
