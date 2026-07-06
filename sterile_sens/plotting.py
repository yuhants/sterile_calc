"""Reproduction of Figs. 2-5 of the paper. Each function returns the Figure
so it can be customized further; parameters are exposed as arguments."""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from .detector import Detector, detector_for
from .isotopes import EC_ISOTOPES, BETA_ISOTOPES
from .montecarlo import simulate_ec, simulate_beta
from .sensitivity import sensitivity_curve, SPHERE_MONTH
from .spectra import ec_phase_space_ratio

# --- style ---------------------------------------------------------------
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
# categorical slots (fixed order, validated palette)
C = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948",
     "#e87ba4", "#eb6834"]

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "text.color": INK,
    "font.family": "sans-serif", "font.size": 10,
    "axes.titlesize": 10.5, "legend.frameon": False,
    "lines.linewidth": 2.0,
    "figure.dpi": 130,
})

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def _save(fig, name, outdir=None):
    outdir = outdir or RESULTS_DIR
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.savefig(path, bbox_inches="tight")
    print(f"saved {path}")
    return path


def existing_limits():
    path = os.path.join(DATA_DIR, "existing_limits.csv")
    m, u = np.loadtxt(path, delimiter=",", comments="#", unpack=True)
    return m, u


def _shade_existing(ax):
    m, u = existing_limits()
    ax.fill_between(m, u, 1.0, color="#d9d8d2", zorder=0,
                    label="existing limits (approx.)")
    ax.plot(m, u, color=MUTED, lw=1.0, zorder=1)


# --------------------------------------------------------------------------
# Fig. 2: reconstructed |p_nu| spectrum for 37Ar + sterile peak
# --------------------------------------------------------------------------
def fig2(iso_name="37Ar", det: Detector = None, exposure_days=30.0,
         m4=750.0, ue4sq=2e-4, bin_kev=20.0, n_mc=400_000, seed=2,
         outdir=None):
    rng = np.random.default_rng(seed)
    iso = EC_ISOTOPES[iso_name]
    det = det or detector_for(iso_name)

    light = simulate_ec(rng, iso, det, 0.0, n_mc)
    heavy = simulate_ec(rng, iso, det, m4, n_mc)

    n_dec = det.decays(iso.A, iso.halflife_days, exposure_days)
    N_light = n_dec * light.rate_factor * light.efficiency
    N_heavy = n_dec * ue4sq * heavy.rate_factor * heavy.efficiency

    bins = np.arange(0, 1.25 * iso.max_E_nu, bin_kev)
    ctr = 0.5 * (bins[1:] + bins[:-1])
    h0, _ = np.histogram(light.p_nu, bins=bins)
    h4, _ = np.histogram(heavy.p_nu, bins=bins)
    mu0 = h0 / h0.sum() * N_light                 # expected light spectrum
    mu4 = h4 / max(h4.sum(), 1) * N_heavy         # expected sterile spectrum
    data = rng.poisson(mu0 + mu4)                 # one toy realization

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    nz = data > 0
    ax.errorbar(ctr[nz], data[nz], yerr=np.sqrt(data[nz]), fmt="o", ms=3.5,
                color=INK, ecolor=INK2, elinewidth=1, capsize=0, zorder=5,
                label="simulated data")
    ax.plot(ctr, mu0, color=C[0], label=r"light $\nu$ fit ($m_i \approx 0$)")
    ax.plot(ctr, mu4, color=C[5],
            label=rf"sterile $\nu$: $m_4$={m4:.0f} keV, "
                  rf"$|U_{{e4}}|^2$={ue4sq:g}")
    ax.set_xlabel(r"reconstructed $|p_\nu|$ (keV/$c$)")
    ax.set_ylabel(f"counts per {bin_kev:.0f} keV")
    ax.set_title(
        f"{iso_name} EC, {det.diameter_nm:.0f}-nm sphere, "
        f"{100*det.loading:.0f}% loading, {exposure_days:.0f} days "
        f"({N_light:,.0f} detected decays)", color=INK2)
    ax.set_xlim(0, bins[-1])
    ax.set_ylim(0, None)
    ax.legend(loc="upper left")

    # inset: zoom on the sterile peak
    p4 = np.sqrt(iso.max_E_nu**2 - m4**2)
    axi = ax.inset_axes([0.08, 0.28, 0.42, 0.38])
    axi.set_facecolor(SURFACE)
    lo, hi = max(p4 - 120, 0), p4 + 120
    sel = (ctr > lo) & (ctr < hi)
    nzs = sel & (data > 0)
    axi.errorbar(ctr[nzs], data[nzs], yerr=np.sqrt(data[nzs]),
                 fmt="o", ms=2.5, color=INK, ecolor=INK2, elinewidth=0.8)
    axi.plot(ctr[sel], mu0[sel], color=C[0], lw=1.5)
    axi.plot(ctr[sel], (mu0 + mu4)[sel], color=C[5], lw=1.5)
    axi.set_xlim(lo, hi)
    axi.set_ylim(0, max(1.3 * (mu0 + mu4)[sel].max(), data[sel].max() + 1.5))
    axi.tick_params(labelsize=7)
    axi.set_title(rf"$m_4$ = {m4:.0f} keV region", fontsize=7.5, color=INK2)

    _save(fig, f"fig2_{iso_name}_spectrum.png", outdir)
    return fig


# --------------------------------------------------------------------------
# Fig. 3: |p_nu| vs Te for a beta emitter, 2-sigma contours for various m4
# --------------------------------------------------------------------------
def _contour_level(h, frac=0.95):
    """Density level enclosing `frac` of the probability."""
    hs = np.sort(h.ravel())[::-1]
    cum = np.cumsum(hs)
    return hs[np.searchsorted(cum, frac * cum[-1])]


def fig3(iso_name="32P", det: Detector = None,
         m4_list=(250.0, 500.0, 750.0, 1000.0, 1250.0, 1500.0),
         n_mc=400_000, seed=3, outdir=None):
    rng = np.random.default_rng(seed)
    iso = BETA_ISOTOPES[iso_name]
    det = det or detector_for(iso_name)

    bins = (np.linspace(0, 1.12 * iso.Q, 120), np.linspace(0, 1.05 * iso.Q, 120))
    xc = 0.5 * (bins[1][1:] + bins[1][:-1])   # Te
    yc = 0.5 * (bins[0][1:] + bins[0][:-1])   # p_nu

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    light = simulate_beta(rng, iso, det, 0.0, n_mc)
    h0, _, _ = np.histogram2d(light.p_nu, light.Te, bins=bins)
    ax.pcolormesh(xc, yc, np.ma.masked_equal(h0, 0), cmap="Blues",
                  norm=mpl.colors.PowerNorm(0.4), rasterized=True)
    ax.contour(xc, yc, h0, levels=[_contour_level(h0)], colors=INK,
               linewidths=1.2)

    handles = [mpl.lines.Line2D([], [], color=INK, lw=1.2,
                                label=r"$m_i \approx 0$ (2$\sigma$)")]
    for k, m4 in enumerate(m4_list):
        heavy = simulate_beta(rng, iso, det, m4, n_mc)
        h, _, _ = np.histogram2d(heavy.p_nu, heavy.Te, bins=bins)
        col = C[k % len(C)]
        ax.contour(xc, yc, h, levels=[_contour_level(h)], colors=col,
                   linewidths=1.6)
        handles.append(mpl.lines.Line2D([], [], color=col,
                                        label=rf"$m_4$ = {m4:.0f} keV"))

    ax.set_xlabel(r"electron kinetic energy $T_e$ (keV)")
    ax.set_ylabel(r"reconstructed $|p_\nu|$ (keV/$c$)")
    ax.set_xlim(0, 1.02 * iso.Q)
    ax.set_ylim(0, 1.05 * iso.Q)
    ax.set_title(f"{iso_name} $\\beta^-$ decay, {det.diameter_nm:.0f}-nm sphere "
                 f"(2$\\sigma$ contours per $m_4$)", color=INK2)
    ax.legend(handles=handles, loc="upper right", fontsize=8.5)
    _save(fig, f"fig3_{iso_name}_pnu_vs_Te.png", outdir)
    return fig


# --------------------------------------------------------------------------
# Fig. 4: sensitivity vs m4, EC and beta panels, 1 sphere-month
# --------------------------------------------------------------------------
def _plot_curves(ax, results, title, fill=True):
    for k, (name, (m4, lim)) in enumerate(results.items()):
        ok = np.isfinite(lim)
        col = C[k % len(C)]
        ax.plot(m4[ok], lim[ok], color=col, label=name)
        if fill:
            ax.fill_between(m4[ok], lim[ok], 1.0, color=col, alpha=0.12, lw=0)
    _shade_existing(ax)
    ax.set_yscale("log")
    ax.set_xlabel(r"sterile-$\nu$ mass $m_4$ (keV)")
    ax.set_ylabel(r"$|U_{e4}|^2$ (95% C.L.)")
    ax.set_title(title, color=INK2)
    ax.legend(fontsize=8.5, loc="upper right")


def _ps_note(ax, include_phase_space):
    if not include_phase_space:
        ax.text(0.98, 0.02, "signal rate excludes phase-space factor "
                            "$r(m_4)$ (paper convention)",
                transform=ax.transAxes, fontsize=7, color=MUTED,
                ha="right", va="bottom")


def fig4(exposure_sphere_days=SPHERE_MONTH,
         ec_names=("37Ar", "49V", "51Cr", "68Ge", "72Se"),
         beta_names=("32P", "90Y"), n_mc=200_000, seed=4, outdir=None,
         detectors=None, include_phase_space=False, suffix=""):
    """Paper Fig. 4.  As in the paper, `include_phase_space` defaults to
    False (see `sensitivity_curve`); pass True for the physical limit."""
    detectors = detectors or {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)

    for ax, names, xmax, sim_label in ((axes[0], ec_names, 1100, "EC"),
                                       (axes[1], beta_names, 2500, r"$\beta^-$")):
        results = {}
        for name in names:
            iso = EC_ISOTOPES.get(name) or BETA_ISOTOPES[name]
            det = detectors.get(name) or detector_for(name)
            print(f"  fig4: {name} ...")
            results[name] = sensitivity_curve(
                iso, det, exposure_sphere_days, n_mc=n_mc, seed=seed,
                include_phase_space=include_phase_space)
        _plot_curves(ax, results,
                     f"{sim_label} isotopes, 1 nanosphere x "
                     f"{exposure_sphere_days:.0f} days")
        ax.set_xlim(0, xmax)
        _ps_note(ax, include_phase_space)
    axes[0].set_ylim(1e-6, 1e0)
    axes[1].set_ylabel("")
    _save(fig, f"fig4_sensitivity{suffix}.png", outdir)
    return fig


# --------------------------------------------------------------------------
# Fig. 5: small spheres / low-endpoint isotopes, several exposures
# --------------------------------------------------------------------------
def fig5(names=("32P", "35S", "3H"),
         exposures=((r"1 sphere $\times$ 1 month", 30.0, "-"),
                    (r"10 spheres $\times$ 1 yr", 10 * 365.0, "--"),
                    (r"1000 spheres $\times$ 1 yr", 1000 * 365.0, ":")),
         n_mc=200_000, seed=5, outdir=None, detectors=None,
         include_phase_space=False, suffix=""):
    detectors = detectors or {}
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    exp_days = np.array([e[1] for e in exposures])

    for k, name in enumerate(names):
        iso = BETA_ISOTOPES[name]
        det = detectors.get(name) or detector_for(name)
        print(f"  fig5: {name} ({det.diameter_nm:.0f} nm, "
              f"{det.f_trap_hz/1e3:.0f} kHz) ...")
        m4, lims = sensitivity_curve(iso, det, exp_days, n_mc=n_mc, seed=seed,
                                     include_phase_space=include_phase_space)
        for (label, _, ls), lim in zip(exposures, lims):
            ok = np.isfinite(lim)
            ax.plot(m4[ok], lim[ok], ls=ls, color=C[k % len(C)],
                    label=name if ls == "-" else None)
    _shade_existing(ax)

    for label, _, ls in exposures:
        ax.plot([], [], ls=ls, color=INK2, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 3000)
    ax.set_ylim(1e-10, 1e-1)
    _ps_note(ax, include_phase_space)
    ax.set_xlabel(r"$m_4$ (keV)")
    ax.set_ylabel(r"$|U_{e4}|^2$ (95% C.L.)")
    ax.set_title(r"$\beta^-$ isotopes, background-free exposures", color=INK2)
    ax.legend(fontsize=8.5, loc="lower left", ncol=2)
    _save(fig, f"fig5_exposures{suffix}.png", outdir)
    return fig
