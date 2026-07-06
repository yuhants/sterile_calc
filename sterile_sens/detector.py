"""Experimental configuration: nanosphere, trap, and detection parameters.

All figure/sensitivity functions take a `Detector`; change any field to study
a different experiment.  Defaults follow Sec. III of the paper.
"""

from dataclasses import dataclass, field, replace
import numpy as np

from .constants import N_AVOGADRO, p_sql_kev
from .materials import MATERIALS, Material


@dataclass
class Detector:
    diameter_nm: float = 100.0          # nanosphere diameter
    material: str = "silica"            # key into materials.MATERIALS
    loading: float = 0.01               # isotope mass fraction
    f_trap_hz: float = 100e3            # trap frequency
    eta: tuple = (0.4, 0.4, 0.6)        # information collection eff. (x, y, z)
    angle_res_rad: float = 0.02         # secondary emission-angle resolution
    energy_res_1mev: float = 0.01       # sigma_E/E = c * sqrt(1 MeV / E)
    trigger_eff: float = 0.4            # secondary tagging eff. (solid angle)
    e_detect_threshold_kev: float = 1.0 # min secondary energy to trigger
    sub_sql_factor: float = 1.0         # <1 for squeezing / back-action evasion

    # --- derived -----------------------------------------------------------
    @property
    def mat(self) -> Material:
        return MATERIALS[self.material]

    @property
    def radius_nm(self) -> float:
        return self.diameter_nm / 2

    @property
    def mass_g(self) -> float:
        r_cm = self.radius_nm * 1e-7
        return 4 / 3 * np.pi * r_cm**3 * self.mat.density_g_cm3

    @property
    def p_sql(self) -> float:
        """SQL momentum resolution [keV/c], Eq. (5)."""
        return p_sql_kev(self.mass_g, self.f_trap_hz) * self.sub_sql_factor

    @property
    def sigma_p_axes(self) -> np.ndarray:
        """Per-axis momentum measurement noise [keV/c] incl. collection eff."""
        return self.p_sql / np.sqrt(np.asarray(self.eta))

    def sigma_E(self, E_kev):
        """Secondary-electron energy resolution [keV]."""
        E = np.asarray(E_kev, dtype=float)
        return self.energy_res_1mev * np.sqrt(np.clip(E, 1e-9, None) * 1000.0)

    def n_isotope_atoms(self, A_gmol: float) -> float:
        return self.mass_g * self.loading / A_gmol * N_AVOGADRO

    def decays(self, A_gmol, halflife_days, exposure_sphere_days) -> float:
        """Total decays in an exposure [nanosphere x days].

        Assumes spheres are replaced every half-life so the initial activity
        is roughly maintained (as in the paper).
        """
        rate_per_day = self.n_isotope_atoms(A_gmol) * np.log(2) / halflife_days
        return rate_per_day * exposure_sphere_days

    def clone(self, **kw):
        return replace(self, **kw)


# Paper defaults for the low-endpoint isotopes of Fig. 5
DETECTOR_100NM = Detector()                                        # 32P, 90Y, EC
DETECTOR_35S = Detector(diameter_nm=50.0, f_trap_hz=10e3)          # 35S
DETECTOR_3H = Detector(diameter_nm=25.0, material="polymer",
                       loading=0.20, f_trap_hz=1e3)                # 3H

DEFAULT_DETECTORS = {
    "3H": DETECTOR_3H,
    "35S": DETECTOR_35S,
}


def detector_for(isotope_name: str) -> Detector:
    return DEFAULT_DETECTORS.get(isotope_name, DETECTOR_100NM)
