"""Sanity tests on synthetic data. Run: python pipeline/tests/test_metrics.py
(or with pytest). No network / real data needed.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bscco import metrics  # noqa: E402


def _rsj_iv(Ic=0.05, Rn=15.0, npts=200, Imax=0.1):
    """Ideal RSJ branch: V = Rn*sqrt(I^2 - Ic^2) above Ic, 0 below (ascending |I|)."""
    I = np.linspace(1e-4, Imax, npts)
    V = np.where(I > Ic, Rn * np.sqrt(np.clip(I ** 2 - Ic ** 2, 0, None)) * 1e-3, 0.0)
    return I, V


def test_ic_crossing_recovers_known_ic():
    # With a *small* threshold the crossing lands near the true Ic (within grid spacing).
    Ic_true = 0.05
    I, V = _rsj_iv(Ic=Ic_true, npts=400)
    ic = metrics.ic_crossing(I, V, vth=1e-5)
    assert abs(ic - Ic_true) < 1e-3, f"recovered Ic={ic}, expected ~{Ic_true}"


def test_threshold_overestimates_true_ic():
    # Physics: a finite voltage criterion always reads Ic ABOVE the true Ic, and the
    # extracted value solves Rn*sqrt(ic^2 - Ic^2)*1e-3 = vth. This is why the pipeline
    # reports a threshold-sensitivity band.
    Ic_true, Rn, vth = 0.05, 15.0, 1e-3
    I, V = _rsj_iv(Ic=Ic_true, Rn=Rn, npts=2000)
    ic = metrics.ic_crossing(I, V, vth=vth)
    expected = np.sqrt(Ic_true ** 2 + (vth / (Rn * 1e-3)) ** 2)
    assert ic > Ic_true
    assert abs(ic - expected) < 2e-3, f"got {ic}, analytic {expected}"


def test_ic_crossing_monotone_in_threshold():
    I, V = _rsj_iv(Ic=0.05)
    ics = [metrics.ic_crossing(I, V, vth=v) for v in (1e-3, 5e-3, 2e-2)]
    assert ics == sorted(ics), f"Ic should grow with threshold, got {ics}"


def test_ic_crossing_returns_nan_when_never_dissipative():
    I = np.linspace(1e-4, 0.05, 100)
    V = np.zeros_like(I)  # fully superconducting, never crosses
    assert np.isnan(metrics.ic_crossing(I, V, vth=1e-3))


def test_runs_z_flags_structured_residual():
    # perfectly structured (all-negative then all-positive) -> large |z|
    resid = np.concatenate([-np.ones(20), np.ones(20)])
    z = metrics._runs_z(resid)
    assert abs(z) > 3, f"structured residual should give |z|>3, got {z}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
