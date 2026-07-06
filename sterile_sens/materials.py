"""Electron stopping in the nanosphere material.

We use the continuous-slowing-down approximation (CSDA): an electron created
inside the sphere travels in a straight line and exits with the energy of an
electron whose residual CSDA range equals R(E0) - path_length.

The range tables below are approximate composites of NIST ESTAR (>= 1 keV)
and Ashley & Anderson, J. Electron Spectrosc. 24, 127 (1981) (< 1 keV, SiO2),
the same references used in the paper.  They are deliberately kept as plain
data so a user can substitute more accurate tables.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Material:
    name: str
    density_g_cm3: float
    # CSDA range table: energy [keV] -> range [g/cm^2]
    csda_E_kev: np.ndarray = field(repr=False, default=None)
    csda_R_gcm2: np.ndarray = field(repr=False, default=None)

    def range_nm(self, E_kev):
        """CSDA range [nm] at kinetic energy E [keV] (log-log interpolation)."""
        E = np.clip(np.asarray(E_kev, dtype=float), self.csda_E_kev[0], None)
        logR = np.interp(np.log(E), np.log(self.csda_E_kev), np.log(self.csda_R_gcm2))
        return np.exp(logR) / self.density_g_cm3 * 1e7  # cm -> nm

    def energy_after(self, E0_kev, path_nm):
        """Exit kinetic energy [keV] after a straight path [nm]; 0 if stopped."""
        E0 = np.asarray(E0_kev, dtype=float)
        residual = self.range_nm(E0) - np.asarray(path_nm, dtype=float)
        stopped = residual <= self.range_nm(self.csda_E_kev[0])
        logE = np.interp(
            np.log(np.clip(residual, 1e-6, None)),
            np.log(self.range_nm(self.csda_E_kev)),
            np.log(self.csda_E_kev),
        )
        E_exit = np.exp(logE)
        return np.where(stopped, 0.0, np.minimum(E_exit, E0))


# --- approximate CSDA range tables (energy keV, range g/cm^2) -------------

_SIO2_TABLE = np.array([
    #  E [keV]   R [g/cm^2]
    [2.0e-2, 2.2e-7],
    [5.0e-2, 6.6e-7],
    [1.0e-1, 1.1e-6],
    [3.0e-1, 3.0e-6],
    [1.0e+0, 8.5e-6],
    [2.0e+0, 2.5e-5],
    [3.0e+0, 5.0e-5],
    [5.0e+0, 1.2e-4],
    [1.0e+1, 3.0e-4],
    [2.0e+1, 1.0e-3],
    [3.0e+1, 2.0e-3],
    [5.0e+1, 5.0e-3],
    [1.0e+2, 1.6e-2],
    [2.0e+2, 5.0e-2],
    [3.0e+2, 1.0e-1],
    [5.0e+2, 2.0e-1],
    [1.0e+3, 5.0e-1],
    [2.0e+3, 1.05e+0],
    [3.0e+3, 1.6e+0],
])

# hydrogen-rich polymer (e.g. polystyrene/acrylic): slightly longer ranges
# per g/cm^2 (larger Z/A); same shape.
_POLYMER_TABLE = _SIO2_TABLE * np.array([1.0, 1.1])

SILICA = Material("SiO2", 2.0, _SIO2_TABLE[:, 0], _SIO2_TABLE[:, 1])
POLYMER = Material("polymer", 1.05, _POLYMER_TABLE[:, 0], _POLYMER_TABLE[:, 1])

MATERIALS = {"silica": SILICA, "polymer": POLYMER}
