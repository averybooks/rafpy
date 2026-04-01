"""
fitting.py
Least-squares fringe fitting to determine baseline components Q_ew and Q_ns.

Implements both techniques from §8.4 of the lab:

  Brute-force (§8.4.1):
    For a grid of (Q_ew, Q_ns) guesses, solve for A and B linearly,
    record the sum-of-squares residual S, find the minimum.

  Nonlinear (§8.4.2):
    Starting from the brute-force result, use scipy.optimize.curve_fit
    (Levenberg-Marquardt) to refine and obtain the covariance matrix.

Both methods operate on the *linearised* fringe model (eq. 12):

    F(h_s) = A * cos(2*pi * tau'_g(Q_ew, Q_ns))
           + B * sin(2*pi * tau'_g(Q_ew, Q_ns))

For a fixed (Q_ew, Q_ns) the design matrix is linear in A and B.
"""

import numpy as np
from scipy.optimize import curve_fit
from .fringe import fringe_model, geometric_delay
from .params  import C, OMEGA_EARTH


# ── Helper: tau'_g in wavelengths from Q_ew, Q_ns ────────────────────────────

def _tau_g_from_Q(h_s, Q_ew, Q_ns):
    """
    tau'_g in wavelengths (eq. 11):
        nu*tau'_g = Q_ew * sin(h) + Q_ns * cos(h)

    Parameters
    ----------
    h_s  : (N,)  hour angles, radians
    Q_ew : float  (b_ew/lambda)*cos(delta)
    Q_ns : float  (b_ns/lambda)*sin(L)*cos(delta)

    Returns
    -------
    nu_tau_g : (N,)  dimensionless geometric delay
    """
    return Q_ew * np.sin(h_s) + Q_ns * np.cos(h_s)


def _linear_fit_AB(h_s, F_obs, Q_ew, Q_ns):
    """
    For fixed Q_ew, Q_ns solve for A and B by linear least squares.

    Returns A, B, sum_of_squares_residual S.
    """
    phi  = 2 * np.pi * _tau_g_from_Q(h_s, Q_ew, Q_ns)
    cos_ = np.cos(phi)
    sin_ = np.sin(phi)

    # Design matrix [cos, sin]
    X    = np.column_stack([cos_, sin_])
    # Normal equations: (X^T X) [A, B]^T = X^T F
    XtX  = X.T @ X
    XtF  = X.T @ F_obs

    try:
        AB   = np.linalg.solve(XtX, XtF)
    except np.linalg.LinAlgError:
        return 0.0, 0.0, np.inf

    A, B = AB
    resid = F_obs - (A * cos_ + B * sin_)
    S     = np.dot(resid, resid)
    return A, B, S


# ── Brute-force fit (§8.4.1) ─────────────────────────────────────────────────

def brute_force_fit(h_s, F_obs,
                    Q_ew_range=None, Q_ns_range=None,
                    n_ew=400, n_ns=1,
                    b_ew_approx=20.0, lam=None):
    """
    Brute-force 2-D least-squares to find Q_ew and Q_ns.

    For each (Q_ew, Q_ns) on the grid, solves for A, B and records S.
    The best-fit is the grid point that minimises S.

    Parameters
    ----------
    h_s       : (N,)  hour angles, radians
    F_obs     : (N,)  observed fringe (real part of band-averaged visibility)
    Q_ew_range : (lo, hi) search range for Q_ew.
                  Defaults to ±20 % around b_ew_approx/lam.
    Q_ns_range : (lo, hi) search range for Q_ns.  Default: (-50, 50) for bns~0.
    n_ew, n_ns : int  number of grid points along each axis
    b_ew_approx : float  approximate b_ew, metres (for default Q_ew range)
    lam        : float  wavelength, metres (required for default Q_ew range)

    Returns
    -------
    result : dict
        'Q_ew'   : best-fit Q_ew
        'Q_ns'   : best-fit Q_ns
        'A'      : best-fit fringe cosine amplitude
        'B'      : best-fit fringe sine amplitude
        'S_grid' : (n_ew, n_ns) sum-of-squares surface
        'Q_ew_grid' : (n_ew,)
        'Q_ns_grid' : (n_ns,)
        'S_min'  : minimum S value
    """
    if lam is None:
        from .params import C, FREQ_RF
        lam = C / FREQ_RF

    if Q_ew_range is None:
        Q0 = b_ew_approx / lam
        Q_ew_range = (0.7 * Q0, 1.3 * Q0)

    if Q_ns_range is None:
        Q_ns_range = (-100.0, 100.0)

    Q_ew_arr = np.linspace(Q_ew_range[0], Q_ew_range[1], n_ew)
    Q_ns_arr = np.linspace(Q_ns_range[0], Q_ns_range[1], n_ns) if n_ns > 1 \
               else np.array([0.0])

    S_grid = np.full((n_ew, len(Q_ns_arr)), np.inf)
    A_grid = np.zeros_like(S_grid)
    B_grid = np.zeros_like(S_grid)

    for i, Qe in enumerate(Q_ew_arr):
        for j, Qn in enumerate(Q_ns_arr):
            A, B, S = _linear_fit_AB(h_s, F_obs, Qe, Qn)
            S_grid[i, j] = S
            A_grid[i, j] = A
            B_grid[i, j] = B

    idx_flat = np.argmin(S_grid)
    i_best, j_best = np.unravel_index(idx_flat, S_grid.shape)

    return {
        'Q_ew'      : Q_ew_arr[i_best],
        'Q_ns'      : Q_ns_arr[j_best],
        'A'         : A_grid[i_best, j_best],
        'B'         : B_grid[i_best, j_best],
        'S_grid'    : S_grid,
        'Q_ew_grid' : Q_ew_arr,
        'Q_ns_grid' : Q_ns_arr,
        'S_min'     : S_grid[i_best, j_best],
    }


def brute_force_uncertainties(bf_result):
    """
    Estimate uncertainty in Q_ew and Q_ns from the curvature of S near minimum.

    sigma_Q ~ 1 / sqrt(alpha_ii) where alpha_ii = 0.5 * d²S/dQ²

    Returns
    -------
    sigma_Q_ew, sigma_Q_ns : floats
    """
    S      = bf_result['S_grid']
    Q_ew   = bf_result['Q_ew_grid']
    Q_ns   = bf_result['Q_ns_grid']

    i_best = np.argmin(S[:, 0]) if S.shape[1] == 1 else \
             np.unravel_index(np.argmin(S), S.shape)[0]

    # Curvature along Q_ew axis
    if i_best > 0 and i_best < len(Q_ew) - 1:
        d2S_ew = (S[i_best+1, 0] - 2*S[i_best, 0] + S[i_best-1, 0]) \
                 / (Q_ew[1] - Q_ew[0])**2
        sigma_Q_ew = 1.0 / np.sqrt(max(0.5 * d2S_ew, 1e-30))
    else:
        sigma_Q_ew = np.nan

    # Curvature along Q_ns axis (only meaningful if n_ns > 3)
    j_best = np.unravel_index(np.argmin(S), S.shape)[1]
    if S.shape[1] > 3 and 0 < j_best < S.shape[1] - 1:
        d2S_ns = (S[i_best, j_best+1] - 2*S[i_best, j_best] + S[i_best, j_best-1]) \
                 / (Q_ns[1] - Q_ns[0])**2
        sigma_Q_ns = 1.0 / np.sqrt(max(0.5 * d2S_ns, 1e-30))
    else:
        sigma_Q_ns = np.nan

    return sigma_Q_ew, sigma_Q_ns


# ── Nonlinear fit (§8.4.2) ───────────────────────────────────────────────────

def _fringe_model_flat(h_s, A, B, Q_ew, Q_ns):
    """Fringe model parameterised by (A, B, Q_ew, Q_ns) for curve_fit."""
    phi = 2 * np.pi * _tau_g_from_Q(h_s, Q_ew, Q_ns)
    return A * np.cos(phi) + B * np.sin(phi)


def nonlinear_fit(h_s, F_obs, p0_dict, F_sigma=None):
    """
    Nonlinear least-squares fringe fit using Levenberg-Marquardt (§8.4.2).

    Start from brute-force result (p0_dict) and refine to get the
    covariance matrix and hence formal uncertainties.

    Parameters
    ----------
    h_s      : (N,)  hour angles, radians
    F_obs    : (N,)  observed fringe
    p0_dict  : dict  Initial guess with keys 'A', 'B', 'Q_ew', 'Q_ns'
               (use output from brute_force_fit)
    F_sigma  : (N,) or float, optional  Per-point uncertainty on F_obs

    Returns
    -------
    result : dict
        'A', 'B', 'Q_ew', 'Q_ns'   : best-fit parameters
        'sigma_A', 'sigma_B',
        'sigma_Q_ew', 'sigma_Q_ns' : 1-sigma uncertainties from covariance
        'cov'                       : full 4x4 covariance matrix
        'F_model'                   : model evaluated at h_s
        'residuals'                 : F_obs - F_model
    """
    p0 = [p0_dict['A'], p0_dict['B'], p0_dict['Q_ew'], p0_dict['Q_ns']]

    kwargs = dict(p0=p0, method='lm', maxfev=10000)
    if F_sigma is not None:
        kwargs['sigma']          = F_sigma
        kwargs['absolute_sigma'] = True

    try:
        popt, pcov = curve_fit(_fringe_model_flat, h_s, F_obs, **kwargs)
    except RuntimeError as e:
        raise RuntimeError(f"Nonlinear fit did not converge: {e}")

    A, B, Q_ew, Q_ns = popt
    perr = np.sqrt(np.diag(pcov))
    F_model = _fringe_model_flat(h_s, *popt)

    return {
        'A'         : A,
        'B'         : B,
        'Q_ew'      : Q_ew,
        'Q_ns'      : Q_ns,
        'sigma_A'   : perr[0],
        'sigma_B'   : perr[1],
        'sigma_Q_ew': perr[2],
        'sigma_Q_ns': perr[3],
        'cov'       : pcov,
        'F_model'   : F_model,
        'residuals' : F_obs - F_model,
    }


def recover_baseline(Q_ew, Q_ns, delta, lat, lam):
    """
    Recover physical baseline components from fitted Q values.

        b_ew = Q_ew * lam / cos(delta)
        b_ns = Q_ns * lam / (sin(lat) * cos(delta))

    Parameters
    ----------
    Q_ew, Q_ns : floats  Fitted baseline-in-wavelengths parameters
    delta      : float   Source declination, radians
    lat        : float   Observatory latitude, radians
    lam        : float   Wavelength, metres

    Returns
    -------
    b_ew, b_ns : floats  Baseline components, metres
    """
    cos_d = np.cos(delta)
    b_ew = Q_ew * lam / cos_d
    b_ns = Q_ns * lam / (np.sin(lat) * cos_d)
    return b_ew, b_ns
