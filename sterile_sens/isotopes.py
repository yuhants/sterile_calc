"""Isotope data (Tables I and II of Carney, Leach & Moore, PRX Quantum 4,
010315 (2023); underlying data from ENSDF).

The registries are plain dicts: add or edit entries to study other isotopes.

Conventions
-----------
* Energies in keV, half-lives in days.
* An EC isotope carries one or more `ECBranch`es. Each branch has a fixed
  neutrino energy E_nu = Q_EC - E_level and a list of possible secondary
  particles, exactly one of which (or none) is emitted per decay:
  `secondaries = [(kind, energy_keV, probability), ...]` with kind in
  {"e", "gamma"}.  ("gamma" also covers x rays.)  This is a simplification
  of the full atomic relaxation cascade; probabilities are exclusive.
* Beta isotopes decay to the daughter ground state with an allowed spectral
  shape (rare excited branches, e.g. in 90Y, are neglected -- the paper notes
  they do not affect the high-m4 sensitivity).
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ECBranch:
    fraction: float                 # branching fraction of all decays
    E_nu: float                     # neutrino total energy for this branch [keV]
    # exclusive list of possible secondaries: (kind, energy [keV], probability)
    secondaries: List[Tuple[str, float, float]] = field(default_factory=list)


@dataclass
class ECIsotope:
    name: str
    A: float                        # atomic mass [g/mol]
    Q_ec: float                     # total decay energy [keV]
    halflife_days: float
    branches: List[ECBranch] = field(default_factory=list)

    @property
    def max_E_nu(self):
        return max(b.E_nu for b in self.branches)


@dataclass
class BetaIsotope:
    name: str
    A: float
    Q: float                        # beta endpoint [keV]
    halflife_days: float
    Z_daughter: int


# ---------------------------------------------------------------------------
# Table I: EC isotopes.  Sub-keV Auger electrons are omitted (they stop in the
# sphere and carry negligible momentum).
# ---------------------------------------------------------------------------
EC_ISOTOPES = {
    # ~10.6% of 7Be decays go through the 478-keV level of 7Li; only this
    # branch has a coincident secondary (the 478-keV gamma) usable as a
    # trigger, so only it is modeled (as in the paper).  The 57-eV Auger
    # always stops in the sphere.
    "7Be": ECIsotope("7Be", 7.017, 861.8, 53.2, [
        ECBranch(0.106, 861.8 - 477.6, [("gamma", 477.6, 1.0)]),
    ]),
    "37Ar": ECIsotope("37Ar", 36.967, 813.9, 35.0, [
        ECBranch(1.0, 813.9, [("e", 2.4, 0.81),
                              ("gamma", 2.6, 0.082), ("gamma", 2.8, 0.005)]),
    ]),
    "49V": ECIsotope("49V", 48.948, 601.9, 330.0, [
        ECBranch(1.0, 601.9, [("e", 4.0, 0.69),
                              ("gamma", 4.5, 0.171), ("gamma", 4.9, 0.019)]),
    ]),
    "51Cr": ECIsotope("51Cr", 50.945, 752.4, 27.7, [
        ECBranch(0.901, 752.4, [("e", 4.4, 0.66),
                                ("gamma", 4.9, 0.194), ("gamma", 5.4, 0.022)]),
        ECBranch(0.099, 752.4 - 320.1, [("gamma", 320.1, 1.0)]),
    ]),
    "68Ge": ECIsotope("68Ge", 67.928, 107.0, 271.0, [
        ECBranch(1.0, 107.0, [("e", 8.0, 0.42),
                              ("gamma", 9.2, 0.39), ("gamma", 10.3, 0.05)]),
    ]),
    "72Se": ECIsotope("72Se", 71.927, 361.0, 8.4, [
        # ~57% captures to the 46-keV level of 72As (tagged by the gamma),
        # remainder to the ground state (tagged by Auger e-/x rays)
        ECBranch(0.57, 361.0 - 45.9, [("gamma", 45.9, 1.0)]),
        ECBranch(0.43, 361.0, [("e", 9.1, 0.53), ("gamma", 10.5, 0.35)]),
    ]),
}

# ---------------------------------------------------------------------------
# Table II: beta- isotopes
# ---------------------------------------------------------------------------
BETA_ISOTOPES = {
    "3H":  BetaIsotope("3H",  3.016, 18.6,   4500.0, 2),
    "32P": BetaIsotope("32P", 31.974, 1710.7, 14.3,  16),
    "35S": BetaIsotope("35S", 34.969, 167.3,  87.4,  17),
    "90Y": BetaIsotope("90Y", 89.907, 2278.7, 2.7,   40),
}
