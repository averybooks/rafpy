"""
plotting.py
Plotting utilities for all stages of interferometry analysis.

All functions return (fig, ax) or (fig, axes) so you can further customise.
Call plt.show() or fig.savefig() after the returned objects.
"""

import numpy as np
import matplotlib.pyplot as plt
from .fringe    import local_fringe_freq, fringe_period
from .modulator import mf_theory


# ── Style helper ──────────────────────────────────────────────────────────────

def _style():
    plt.rcParams.update({
        'figure.dpi'    : 120,
        'axes.grid'     : True,
        'grid.alpha'    : 0.3,
        'lines.linewidth': 1.4,
        'font.size'     : 11,
    })


# ── 1. Raw visibility (amplitude & phase vs time) ────────────────────────────

def plot_visibility(vis_dict, title='Visibility vs Time'):
    """
    Plot band-averaged amplitude and phase vs Unix time.

    Parameters
    ----------
    vis_dict : output of process_visibilities()
    """
    _style()
    t   = vis_dict['times']
    t0  = t[0]
    dt  = (t - t0) / 60.0   # minutes from start

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(title)

    ax1.plot(dt, vis_dict['band_amp'], color='steelblue', label='|V12|')
    ax1.set_ylabel('Amplitude (arb.)')
    ax1.legend()

    ax2.plot(dt, np.degrees(vis_dict['band_phase']), color='coral', label='Phase')
    ax2.set_ylabel('Phase (deg)')
    ax2.set_xlabel('Time from start (min)')
    ax2.legend()

    fig.tight_layout()
    return fig, (ax1, ax2)


# ── 2. Fourier power spectrum of the fringe ──────────────────────────────────

def plot_fringe_spectrum(freqs, power, ff_expected_range=None,
                         title='Fringe Power Spectrum'):
    """
    Plot power spectrum of the real fringe and optionally overlay the
    expected fringe frequency range.

    Parameters
    ----------
    freqs             : output of fourier_fringe_spectrum()
    power             : output of fourier_fringe_spectrum()
    ff_expected_range : (f_lo, f_hi) Hz, optional  Expected fringe freq band
    """
    _style()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogy(freqs * 1e3, power, color='navy', lw=1.2, label='Power spectrum')

    if ff_expected_range is not None:
        ax.axvspan(ff_expected_range[0] * 1e3, ff_expected_range[1] * 1e3,
                   alpha=0.2, color='orange', label='Expected fringe band')

    ax.set_xlabel('Frequency (mHz)')
    ax.set_ylabel('Power')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig, ax


# ── 3. Observed fringe with model overlay ────────────────────────────────────

def plot_fringe_fit(h_s, F_obs, fit_result, delta_deg=None,
                   title='Fringe: Data vs Model'):
    """
    Plot observed fringe and least-squares model.

    Parameters
    ----------
    h_s        : (N,)  hour angles, radians
    F_obs      : (N,)  observed real fringe
    fit_result : dict  from nonlinear_fit() or brute_force_fit()
    delta_deg  : float, optional  source declination in degrees (for label)
    """
    from .fitting import _fringe_model_flat
    _style()

    h_plot = np.degrees(h_s)
    A   = fit_result['A']
    B   = fit_result['B']
    Qew = fit_result['Q_ew']
    Qns = fit_result['Q_ns']

    h_fine  = np.linspace(h_s.min(), h_s.max(), 5000)
    F_model = _fringe_model_flat(h_fine, A, B, Qew, Qns)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6),
                                   gridspec_kw={'height_ratios': [3, 1]},
                                   sharex=True)
    fig.suptitle(title)

    ax1.plot(np.degrees(h_fine), F_model, 'r-', lw=1.5, label='Model', zorder=3)
    ax1.scatter(h_plot, F_obs, s=8, c='steelblue', alpha=0.6, label='Data', zorder=2)
    if delta_deg is not None:
        ax1.set_title(f'δ = {delta_deg:.2f}°,  Q_ew = {Qew:.1f},  Q_ns = {Qns:.1f}')
    ax1.set_ylabel('Fringe amplitude')
    ax1.legend(loc='upper right')

    # Residuals
    F_at_data = _fringe_model_flat(h_s, A, B, Qew, Qns)
    ax2.axhline(0, color='k', lw=0.8)
    ax2.scatter(h_plot, F_obs - F_at_data, s=6, c='grey', alpha=0.6)
    ax2.set_ylabel('Residual')
    ax2.set_xlabel('Hour angle (deg)')

    fig.tight_layout()
    return fig, (ax1, ax2)


# ── 4. Brute-force S surface ──────────────────────────────────────────────────

def plot_S_surface(bf_result, title='Sum-of-Squares vs Q_ew'):
    """
    Plot sum-of-squares S vs Q_ew (and optionally Q_ns) from brute-force fit.
    """
    _style()
    S      = bf_result['S_grid']
    Q_ew   = bf_result['Q_ew_grid']
    Q_ns   = bf_result['Q_ns_grid']
    Q_best = bf_result['Q_ew']

    if len(Q_ns) == 1:
        # 1-D plot
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(Q_ew, S[:, 0], color='darkgreen')
        ax.axvline(Q_best, color='red', ls='--', label=f'Best Q_ew = {Q_best:.2f}')
        ax.set_xlabel('Q_ew  (b_ew cos δ / λ)')
        ax.set_ylabel('Sum of squares S')
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        return fig, ax
    else:
        # 2-D colourmap
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.pcolormesh(Q_ns, Q_ew, S, cmap='viridis_r', shading='auto')
        plt.colorbar(im, ax=ax, label='S')
        ax.plot(bf_result['Q_ns'], Q_best, 'r+', ms=12, mew=2, label='Minimum')
        ax.set_xlabel('Q_ns')
        ax.set_ylabel('Q_ew')
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        return fig, ax


# ── 5. Local fringe frequency vs time ────────────────────────────────────────

def plot_fringe_freq(h_s, unix_times, delta, b_ew, b_ns, lat, lam,
                    title='Local Fringe Frequency vs Time'):
    """
    Plot the local fringe frequency prediction over the observation.
    """
    _style()
    ff_hz   = local_fringe_freq(h_s, delta, b_ew, b_ns, lat, lam, in_hz=True)
    t0      = unix_times[0]
    dt_min  = (unix_times - t0) / 60.0

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(dt_min, ff_hz * 1e3, color='darkorange', label='f_f (mHz)')
    ax.set_xlabel('Time from start (min)')
    ax.set_ylabel('Local fringe frequency (mHz)')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig, ax


# ── 6. MF_theory curve ────────────────────────────────────────────────────────

def plot_mf_theory(ff_R_max=6.0, N=1000, title='Theoretical Fringe Modulator'):
    """
    Plot MF_theory vs ff*R (Source Diameter × f_f in lab Fig. 2 notation).
    Reproduces Fig. 2 of the lab handout.
    """
    _style()
    x  = np.linspace(0, ff_R_max, 2000)
    mf = mf_theory(x, N=N)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(2 * x, mf, color='navy', label='Flat disk (uniform brightness)')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('Source Diameter × f_f  (= 2R × f_f)')
    ax.set_ylabel('Fringe modulator MF_theory')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig, ax


# ── 7. Observed vs theoretical MF + diameter fit ─────────────────────────────

def plot_mf_fit(mf_obs_result, diameter_result, N=500,
                title='Fringe Modulator: Observed vs Theory'):
    """
    Overlay MF_obs data with MF_theory at the best-fit radius.

    Parameters
    ----------
    mf_obs_result    : output of mf_observed()
    diameter_result  : output of fit_diameter()
    """
    _style()
    ff_rad  = mf_obs_result['ff_rad_bins']
    mf_obs  = mf_obs_result['mf_obs']
    mf_err  = mf_obs_result['mf_err']
    R_best  = diameter_result['R_rad']

    x_theory = np.linspace(0, ff_rad.max() * R_best * 1.3, 1000)
    mf_th    = mf_theory(x_theory, N=N)
    # Convert x_theory back to ff units
    ff_th    = x_theory / R_best

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(ff_rad, mf_obs, yerr=mf_err, fmt='o', ms=5,
                color='steelblue', label='MF_observed', capsize=3)
    ax.plot(ff_th, mf_th, 'r-', lw=2,
            label=f'MF_theory  (R = {np.degrees(R_best)*60:.1f} arcmin,'
                  f' d = {diameter_result["diameter_deg"]:.2f}°)')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('Local fringe frequency f_f (cycles/radian)')
    ax.set_ylabel('Fringe modulator')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig, ax


# ── 8. Waterfall plot of visibility vs channel and time ──────────────────────

def plot_waterfall(vis_dict, component='amp', title=None):
    """
    Waterfall (time × channel) plot of visibility amplitude, phase, real, or imag.

    Parameters
    ----------
    component : str  One of 'amp', 'phase', 'real', 'imag'
    """
    _style()
    t   = vis_dict['times']
    t0  = t[0]
    dt  = (t - t0) / 60.0
    fq  = vis_dict['chan_freqs'] / 1e6   # MHz

    if component == 'amp':
        data  = vis_dict['amp']
        label = '|V12|'
        cmap  = 'viridis'
    elif component == 'phase':
        data  = np.degrees(vis_dict['phase'])
        label = 'Phase (deg)'
        cmap  = 'hsv'
    elif component == 'real':
        data  = vis_dict['vis_cube'].real
        label = 'Re(V12)'
        cmap  = 'RdBu_r'
    elif component == 'imag':
        data  = vis_dict['vis_cube'].imag
        label = 'Im(V12)'
        cmap  = 'RdBu_r'
    else:
        raise ValueError(f"Unknown component '{component}'")

    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.pcolormesh(fq, dt, data, cmap=cmap, shading='auto')
    plt.colorbar(im, ax=ax, label=label)
    ax.set_xlabel('Baseband frequency (MHz)')
    ax.set_ylabel('Time from start (min)')
    ax.set_title(title or f'Visibility waterfall — {label}')
    fig.tight_layout()
    return fig, ax
