"""Regenerate the paper figures.

Examples
--------
python run_all.py                       # all figures, default parameters
python run_all.py fig2 --m4 500 --ue4sq 5e-4
python run_all.py fig4 --exposure 90 --n-mc 100000
python run_all.py fig2 --diameter 150 --loading 0.02 --trap-khz 50
"""

import argparse
import time

import matplotlib
matplotlib.use("Agg")

from sterile_sens import plotting
from sterile_sens.detector import Detector, detector_for


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("figures", nargs="*", choices=["fig2", "fig3", "fig4", "fig5"],
                    help="which figures to produce (default: all)")
    ap.add_argument("--n-mc", type=int, default=None,
                    help="MC events per PDF (lower = faster)")
    ap.add_argument("--outdir", default=None, help="output directory")
    # detector overrides (applied to the default 100-nm detector)
    ap.add_argument("--diameter", type=float, help="sphere diameter [nm]")
    ap.add_argument("--loading", type=float, help="isotope mass fraction")
    ap.add_argument("--trap-khz", type=float, help="trap frequency [kHz]")
    ap.add_argument("--trigger-eff", type=float, help="secondary tagging efficiency")
    ap.add_argument("--sub-sql", type=float, help="momentum noise / SQL (<1: squeezed)")
    ap.add_argument("--phase-space", action="store_true",
                    help="include the heavy-state phase-space suppression r(m4) "
                         "in fig4/fig5 (physical; the paper's figures omit it)")
    # fig2 options
    ap.add_argument("--iso", default="37Ar", help="isotope for fig2/fig3")
    ap.add_argument("--m4", type=float, default=750.0, help="fig2 sterile mass [keV]")
    ap.add_argument("--ue4sq", type=float, default=2e-4, help="fig2 mixing |Ue4|^2")
    ap.add_argument("--exposure", type=float, default=30.0,
                    help="exposure [sphere-days] for fig2/fig4")
    args = ap.parse_args()
    if not args.figures:
        args.figures = ["fig2", "fig3", "fig4", "fig5"]

    overrides = {k: v for k, v in {
        "diameter_nm": args.diameter, "loading": args.loading,
        "f_trap_hz": args.trap_khz * 1e3 if args.trap_khz else None,
        "trigger_eff": args.trigger_eff, "sub_sql_factor": args.sub_sql,
    }.items() if v is not None}

    def det_for(name):
        return detector_for(name).clone(**overrides) if overrides else None

    t0 = time.time()
    if "fig2" in args.figures:
        print("fig2 ...")
        plotting.fig2(iso_name=args.iso if args.iso in
                      __import__("sterile_sens").EC_ISOTOPES else "37Ar",
                      det=det_for(args.iso), exposure_days=args.exposure,
                      m4=args.m4, ue4sq=args.ue4sq,
                      n_mc=args.n_mc or 400_000, outdir=args.outdir)
    if "fig3" in args.figures:
        print("fig3 ...")
        plotting.fig3(det=det_for("32P"), n_mc=args.n_mc or 400_000,
                      outdir=args.outdir)
    ps = dict(include_phase_space=args.phase_space,
              suffix="_physical" if args.phase_space else "")
    if "fig4" in args.figures:
        print("fig4 ...")
        dets = {n: det_for(n) for n in
                ("7Be", "37Ar", "49V", "51Cr", "68Ge", "72Se", "32P", "90Y")} \
            if overrides else None
        plotting.fig4(exposure_sphere_days=args.exposure, detectors=dets,
                      n_mc=args.n_mc or 200_000, outdir=args.outdir, **ps)
    if "fig5" in args.figures:
        print("fig5 ...")
        dets = {n: det_for(n) for n in ("32P", "35S", "3H")} if overrides else None
        plotting.fig5(detectors=dets, n_mc=args.n_mc or 200_000,
                      outdir=args.outdir, **ps)
    print(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
