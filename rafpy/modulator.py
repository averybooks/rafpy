"""
modulator.py
Fringe modulator (MF) calculations for measuring source angular diameter.

Implements §9–10 of the lab handout.

For a uniformly-bright circular disk (the "MUN") the theoretical fringe
modulator is (eq. 22-23):

    MF_theory(ff, R) = delta_h * sum_{n=-N}^{N}
                           sqrt(1 - (n/N)^2) * cos(2*pi*ff*R*n/N)

where ff is the local fringe frequency (cycles/radian) and R is the angular
radius of the source in radians.

The observed fringe modulator MF_obs(ff) is obtained by fitting the fringe
model to the data and extracting the amplitude envelope as a function of ff.
Comparing the zero-crossings of MF_theory and MF_obs gives R.
"""

import numpy as np
from scipy.optimize import curve_fit, brentq
from .fringe import local_fringe_freq


# ── Theoretical fringe modulator (eq. 22-23) ─────────────────────────────────

def mf_theory(ff_R, N=1000):
    """
    Theoretical fringe modulator for a uniformly-bright circular disk.

    MF_theory is evaluated as a function of the dimensionless product ff * R
    (fringe frequency × angular radius, both in consistent units).

    Parameters
    ----------
    ff_R : array_like  Values of ff*R (dimensionless; cycles)
    N    : int         Number of half-slices (more = higher accuracy)

    Returns
    -------
    MF : ndarray  Modulator values (normalised to MF(0) = pi/4)
    """
    ff_R = np.asarray(ff_R, dtype=float)
    n    = np.arange(-N, N + 1, dtype=float)
    w    = np.sqrt(np.maximum(1.0 - (n / N)**2, 0.0))   # shape weights

    # MF(ff_R) = (1/N) * sum_n w_n * cos(2*pi * ff_R * n/N)
    # Broadcast: ff_R is (M,), n/N is (2N+1,)  → outer product
    phase = 2 * np.pi * np.outer(ff_R, n / N)   # (M, 2N+1)
    MF    = (w * np.cos(phase)).mean(axis=1)     # (M,)
    return MF


def mf_theory_zero_crossings(n_zeros=5, N=1000):
    """
    Find the first n_zeros zero-crossings of MF_theory as a function of ff*R.

    Returns
    -------
    zeros : list of float  Values of ff*R at which MF_theory = 0
    """
    ff_R_dense = np.linspace(0.01, n_zeros + 1, 20000)
    mf         = mf_theory(ff_R_dense, N=N)

    zeros = []
    for i in range(len(mf) - 1):
        if mf[i] * mf[i+1] < 0:
            root = brentq(lambda x: mf_theory(np.array([x]), N=N)[0],
                          ff_R_dense[i], ff_R_dense[i+1])
            zeros.append(root)
            if len(zeros) == n_zeros:
                break
    return zeros


# ── Observed fringe modulator ─────────────────────────────────────────────────

def mf_observed(h_s, F_obs, nonlin_result, delta, b_ew, b_ns, lat, lam,
                bin_width_rad=None, n_bins=50):
    """
    Estimate the observed fringe modulator as a function of local fringe
    frequency by binning the fringe amplitude in ff bins.

    The observed amplitude at each time step is sqrt(A^2 + B^2) * MF_obs(ff),
    so we extract MF_obs by dividing the local fringe amplitude by the
    point-source amplitude (A^2 + B^2) from the nonlinear fit.

    Parameters
    ----------
    h_s           : (N,)  hour angles, radians
    F_obs         : (N,)  observed fringe
    nonlin_result : dict  Output of nonlinear_fit (contains A, B, F_model)
    delta         : float Source declination, radians
    b_ew, b_ns    : float Baseline components, metres
    lat           : float Observatory latitude, radians
    lam           : float Wavelength, metres
    bin_width_rad : float, optional  Bin width in radians of hour angle
    n_bins        : int   Number of ff bins (used if bin_width_rad is None)

    Returns
    -------
    result : dict
        'ff_bins'   : (n_bins,)  centre local fringe frequency of each bin, Hz
        'ff_rad_bins': (n_bins,) same in cycles/radian
        'mf_obs'    : (n_bins,)  observed modulator values
        'mf_err'    : (n_bins,)  standard error per bin
        'n_in_bin'  : (n_bins,)  number of samples per bin
    """
    from numpy import hypot
    A    = nonlin_result['A']
    B    = nonlin_result['B']
    pt_amp = hypot(A, B)    # point-source amplitude

    # Residuals relative to best-fit point-source fringe
    residual_amp = F_obs / (pt_amp + 1e-30)

    # Local fringe frequency at each hour angle
    ff_hz  = local_fringe_freq(h_s, delta, b_ew, b_ns, lat, lam, in_hz=True)
    ff_rad = local_fringe_freq(h_s, delta, b_ew, b_ns, lat, lam, in_hz=False)

    # Bin by ff
    ff_min, ff_max = ff_hz.min(), ff_hz.max()
    bins   = np.linspace(ff_min, ff_max, n_bins + 1)
    bin_c  = 0.5 * (bins[:-1] + bins[1:])

    mf_obs  = np.full(n_bins, np.nan)
    mf_err  = np.full(n_bins, np.nan)
    n_bin   = np.zeros(n_bins, dtype=int)

    for k in range(n_bins):
        mask = (ff_hz >= bins[k]) & (ff_hz < bins[k+1])
        if mask.sum() > 0:
            vals         = residual_amp[mask]
            mf_obs[k]    = vals.mean()
            mf_err[k]    = vals.std() / np.sqrt(mask.sum())
            n_bin[k]     = mask.sum()

    # Corresponding ff_rad bin centres
    ff_rad_bins = bin_c * (ff_rad.mean() / (ff_hz.mean() + 1e-30))

    return {
        'ff_bins'    : bin_c,
        'ff_rad_bins': ff_rad_bins,
        'mf_obs'     : mf_obs,
        'mf_err'     : mf_err,
        'n_in_bin'   : n_bin,
    }


# ── Diameter fitting ──────────────────────────────────────────────────────────

def fit_diameter(mf_obs_result, R_range_rad=None, n_grid=2000, N=500):
    """
    Fit the angular radius R of a uniform disk by matching the zero-crossings
    (and full profile) of MF_theory to MF_obs.

    For each trial R, compute MF_theory(ff * R) and compare with MF_obs
    by minimising chi-squared.

    Parameters
    ----------
    mf_obs_result : dict  Output of mf_observed()
    R_range_rad   : (lo, hi) search range for R in radians.
                    Default: corresponds to 0.1 to 1.0 degrees.
    n_grid        : int  Number of R values to try
    N             : int  Accuracy of MF_theory summation

    Returns
    -------
    result : dict
        'R_rad'       : best-fit radius, radians
        'R_deg'       : best-fit radius, degrees
        'R_arcmin'    : best-fit radius, arcminutes
        'diameter_deg': best-fit diameter, degrees
        'chi2_grid'   : (n_grid,)  chi-squared values
        'R_grid'      : (n_grid,)  radius values tried
    """
    if R_range_rad is None:
        R_range_rad = (np.radians(0.1), np.radians(1.0))

    ff_rad = mf_obs_result['ff_rad_bins']
    mf_obs = mf_obs_result['mf_obs']
    mf_err = mf_obs_result['mf_err']

    # Only use bins with valid data
    good = np.isfinite(mf_obs) & np.isfinite(mf_err) & (mf_err > 0)
    if good.sum() < 3:
        raise ValueError("Too few valid MF_obs bins for diameter fitting.")

    ff_g  = ff_rad[good]
    mf_g  = mf_obs[good]
    err_g = mf_err[good]

    R_arr  = np.linspace(R_range_rad[0], R_range_rad[1], n_grid)
    chi2   = np.zeros(n_grid)

    for i, R in enumerate(R_arr):
        mf_th   = mf_theory(ff_g * R, N=N)
        # Normalise to MF at smallest ff
        norm    = mf_th[0] / (mf_g[0] + 1e-30) if mf_g[0] != 0 else 1.0
        chi2[i] = np.sum(((mf_g - mf_th / norm) / err_g)**2)

    i_best = np.argmin(chi2)
    R_best = R_arr[i_best]

    return {
        'R_rad'        : R_best,
        'R_deg'        : np.degrees(R_best),
        'R_arcmin'     : np.degrees(R_best) * 60,
        'diameter_deg' : 2 * np.degrees(R_best),
        'chi2_grid'    : chi2,
        'R_grid'       : R_arr,
    }
