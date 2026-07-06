"""Vectorized Monte Carlo of decay, particle transport, and reconstruction.

Follows Sec. III of the paper:
 * the nuclear recoil always stops in the nanosphere;
 * gammas / x rays leave with no energy loss;
 * electrons (beta or Auger) travel in a straight line and lose energy
   according to the CSDA range of the sphere material; if they stop, their
   momentum is absorbed by the sphere and no charged secondary exits;
 * the sphere momentum kick is measured with per-axis Gaussian noise at the
   SQL scaled by the information-collection efficiency;
 * an exiting secondary triggers the event with probability `trigger_eff`;
   its direction is measured to `angle_res_rad` and (for electrons) its
   energy to sigma_E/E = c sqrt(1 MeV / E);
 * the neutrino momentum is reconstructed as
   p_nu = -(Delta p_sphere,meas + p_secondary,meas).
"""

from dataclasses import dataclass
import numpy as np

from .constants import M_E, electron_momentum
from .detector import Detector
from .isotopes import ECIsotope, BetaIsotope
from .spectra import sample_beta_Te


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def _unit_vectors(rng, n):
    v = rng.standard_normal((n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _positions_in_sphere(rng, n, R):
    return _unit_vectors(rng, n) * R * rng.random(n)[:, None] ** (1 / 3)


def _chord_to_surface(pos, direction, R):
    """Distance from `pos` along `direction` to the sphere surface."""
    b = np.einsum("ij,ij->i", pos, direction)
    r2 = np.einsum("ij,ij->i", pos, pos)
    return -b + np.sqrt(np.clip(R**2 - r2 + b**2, 0.0, None))


def _smear_direction(rng, v, sigma_rad):
    """Add Gaussian angular noise of width sigma_rad to vectors v."""
    n = v / np.linalg.norm(v, axis=1, keepdims=True)
    # orthonormal basis perpendicular to n
    a = np.where(np.abs(n[:, [0]]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
    e1 = np.cross(n, a)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(n, e1)
    g1, g2 = rng.standard_normal((2, len(v), 1)) * sigma_rad
    out = n + g1 * e1 + g2 * e2
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    return out * np.linalg.norm(v, axis=1, keepdims=True)


# --------------------------------------------------------------------------
@dataclass
class EventSample:
    """Reconstructed observables of triggered events.

    The expected number of *detected* events is
        N_det = N_decays * |U|^2 * rate_factor * efficiency
    (with |U|^2 = 1 for the light states), where `rate_factor` is the
    branching fraction to the modeled decay branches including the
    heavy-state phase-space suppression, and `efficiency` is the triggered
    fraction from the MC (secondary escape x tagging acceptance).
    """
    p_nu: np.ndarray            # |p_nu| reconstructed [keV/c]
    Te: np.ndarray = None       # measured beta kinetic energy [keV] (beta only)
    efficiency: float = 1.0     # triggered fraction of simulated decays
    rate_factor: float = 1.0    # branching x phase-space factor
    branch_factor: float = 1.0  # branching only (kinematically open branches)


def _measure_kick(rng, dp_true, det: Detector):
    return dp_true + rng.standard_normal(dp_true.shape) * det.sigma_p_axes


# --------------------------------------------------------------------------
def simulate_ec(rng, iso: ECIsotope, det: Detector, m4=0.0, n=100_000) -> EventSample:
    """EC decays to a state of mass m4. Returns triggered events only."""
    from .spectra import ec_phase_space_ratio
    R = det.radius_nm

    # branch weights: branching fraction x heavy-state phase space
    weights = np.array([b.fraction * ec_phase_space_ratio(b, m4)
                        for b in iso.branches])
    rate_factor = weights.sum()
    branch_factor = sum(b.fraction for b, w in zip(iso.branches, weights) if w > 0)
    if rate_factor == 0.0:              # all branches kinematically closed
        return EventSample(p_nu=np.array([]), efficiency=0.0,
                           rate_factor=0.0, branch_factor=0.0)
    branch_idx = rng.choice(len(iso.branches), size=n, p=weights / rate_factor)

    p_nu_true = np.zeros(n)
    for i, b in enumerate(iso.branches):
        sel = branch_idx == i
        if m4 < b.E_nu:
            p_nu_true[sel] = np.sqrt(b.E_nu**2 - m4**2)

    nu_vec = _unit_vectors(rng, n) * p_nu_true[:, None]

    # secondary: sample one (or none) per event from the branch table
    sec_kind = np.full(n, -1)           # -1 none, 0 electron, 1 gamma
    sec_E = np.zeros(n)
    u = rng.random(n)
    for i, b in enumerate(iso.branches):
        sel = branch_idx == i
        lo = np.zeros(n)
        for kind, E, prob in b.secondaries:
            pick = sel & (u >= lo) & (u < lo + prob)
            sec_kind[pick] = 0 if kind == "e" else 1
            sec_E[pick] = E
            lo += prob

    # transport electrons through the sphere
    is_e = sec_kind == 0
    pos = _positions_in_sphere(rng, n, R)
    dirs = _unit_vectors(rng, n)
    path = _chord_to_surface(pos, dirs, R)
    E_exit = np.where(is_e, det.mat.energy_after(sec_E, path), sec_E)
    escapes = (sec_kind >= 0) & (E_exit > 0)

    p_sec = np.where(sec_kind == 0, electron_momentum(E_exit), E_exit)
    sec_vec = dirs * np.where(escapes, p_sec, 0.0)[:, None]

    # sphere kick and measurement
    dp_true = -(nu_vec + sec_vec)
    dp_meas = _measure_kick(rng, dp_true, det)

    # trigger: an escaping secondary above threshold, within acceptance
    triggered = escapes & (E_exit > det.e_detect_threshold_kev) \
        & (rng.random(n) < det.trigger_eff)

    # measured secondary momentum (energy assumed known to line precision for
    # EC secondaries; direction smeared)
    sec_meas = _smear_direction(rng, sec_vec[triggered], det.angle_res_rad)

    p_nu_rec = -(dp_meas[triggered] + sec_meas)
    return EventSample(p_nu=np.linalg.norm(p_nu_rec, axis=1),
                       efficiency=triggered.mean(), rate_factor=rate_factor,
                       branch_factor=branch_factor)


# --------------------------------------------------------------------------
def simulate_beta(rng, iso: BetaIsotope, det: Detector, m4=0.0, n=100_000) -> EventSample:
    """Allowed beta- decay with neutrino mass m4. Returns triggered events."""
    from .spectra import beta_phase_space_ratio
    if m4 >= iso.Q:
        return EventSample(p_nu=np.array([]), Te=np.array([]),
                           efficiency=0.0, rate_factor=0.0, branch_factor=0.0)
    R = det.radius_nm

    Te = sample_beta_Te(rng, n, iso, m4)
    E_nu = iso.Q - Te
    p_nu_true = np.sqrt(np.clip(E_nu**2 - m4**2, 0.0, None))

    # directions: isotropic, uncorrelated (beta-nu angular correlation neglected)
    nu_vec = _unit_vectors(rng, n) * p_nu_true[:, None]
    e_dir = _unit_vectors(rng, n)

    # electron transport
    pos = _positions_in_sphere(rng, n, R)
    path = _chord_to_surface(pos, e_dir, R)
    Te_exit = det.mat.energy_after(Te, path)
    escapes = Te_exit > 0
    e_vec = e_dir * np.where(escapes, electron_momentum(Te_exit), 0.0)[:, None]

    dp_true = -(nu_vec + e_vec)
    dp_meas = _measure_kick(rng, dp_true, det)

    triggered = escapes & (Te_exit > det.e_detect_threshold_kev) \
        & (rng.random(n) < det.trigger_eff)

    # measured electron: energy resolution + angular resolution
    Te_meas = Te_exit[triggered] + rng.standard_normal(triggered.sum()) \
        * det.sigma_E(Te_exit[triggered])
    Te_meas = np.clip(Te_meas, 0.0, None)
    e_meas_dir = _smear_direction(rng, e_vec[triggered], det.angle_res_rad)
    e_meas_dir /= np.linalg.norm(e_meas_dir, axis=1, keepdims=True)
    e_meas = e_meas_dir * electron_momentum(Te_meas)[:, None]

    p_nu_rec = -(dp_meas[triggered] + e_meas)
    return EventSample(p_nu=np.linalg.norm(p_nu_rec, axis=1),
                       Te=Te_meas,
                       efficiency=triggered.mean(),
                       rate_factor=beta_phase_space_ratio(iso, m4))
