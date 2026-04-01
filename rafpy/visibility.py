"""
visibility.py
Process raw SNAP complex visibilities into fringe amplitudes, phases,
and per-channel or band-averaged quantities ready for plotting and fitting.

The SNAP delivers:
    data['corr01'] : complex ndarray, shape (N_CHAN,)
        V12(nu) = <v1(nu) * conj(v2(nu))> averaged over one integration

Each call to snap.read_data() gives one such spectrum at a given Unix time.
Your collect_data loop accumulates these into a list; this module takes that
list and prepares it for analysis.
"""

import numpy as np
from .params import N_CHAN, CHAN_BW


def process_visibilities(results, chan_range=None):
    """
    Convert a list of result dicts (from collect_data) into analysis-ready arrays.

    Parameters
    ----------
    results : list of dict
        Each dict must have keys:
            'timestamp'  : float  Unix time at start of averaging window
            'power_spec' : complex ndarray, shape (N_CHAN,)  -- averaged V12
            'n_acc'      : int
        Note: your collect_data computes quad_avg_power which takes |V12|^2.
        If you stored the raw complex average instead, pass that here and set
        take_abs=False; see note below.
    chan_range : tuple (lo, hi), optional
        Channel indices to keep (inclusive).  Defaults to (100, 924) to
        avoid band edges and DC.

    Returns
    -------
    vis : dict with keys:
        'times'      : (N_t,)        Unix timestamps
        'vis_cube'   : (N_t, N_chan) complex visibilities
        'amp'        : (N_t, N_chan) |V12|
        'phase'      : (N_t, N_chan) angle(V12), radians
        'band_amp'   : (N_t,)        band-averaged |V12|
        'band_phase' : (N_t,)        band-averaged phase (via mean phasor)
        'band_real'  : (N_t,)        Re(band-averaged V12)
        'band_imag'  : (N_t,)        Im(band-averaged V12)
        'chan_freqs'  : (N_chan,)     channel baseband frequencies, Hz
        'n_acc'      : (N_t,)        number of accumulations per window
    """
    if chan_range is None:
        chan_lo, chan_hi = 100, 924
    else:
        chan_lo, chan_hi = chan_range

    times    = np.array([r['timestamp']  for r in results])
    n_acc    = np.array([r['n_acc']      for r in results])
    spectra  = np.array([r['power_spec'] for r in results])   # (N_t, N_CHAN)

    # Slice to good channels
    spectra  = spectra[:, chan_lo:chan_hi + 1]
    chan_idx  = np.arange(chan_lo, chan_hi + 1)
    chan_freqs = chan_idx * CHAN_BW

    amp   = np.abs(spectra)
    phase = np.angle(spectra)

    # Band-averaged quantities
    # Average phasors coherently so phase is preserved
    mean_phasor  = spectra.mean(axis=1)          # (N_t,)
    band_amp     = np.abs(mean_phasor)
    band_phase   = np.angle(mean_phasor)
    band_real    = mean_phasor.real
    band_imag    = mean_phasor.imag

    return {
        'times'      : times,
        'vis_cube'   : spectra,
        'amp'        : amp,
        'phase'      : phase,
        'band_amp'   : band_amp,
        'band_phase' : band_phase,
        'band_real'  : band_real,
        'band_imag'  : band_imag,
        'chan_freqs'  : chan_freqs,
        'n_acc'      : n_acc,
    }


def visibility_snr(vis_dict):
    """
    Rough per-channel SNR: mean amplitude / std of amplitude over time.

    Returns
    -------
    snr : (N_chan,)
    """
    amp = vis_dict['amp']           # (N_t, N_chan)
    return amp.mean(axis=0) / (amp.std(axis=0) + 1e-30)


def select_best_channel(vis_dict, n_best=10):
    """
    Return the indices of the n_best channels by SNR.
    Useful for single-channel fringe fitting.
    """
    snr = visibility_snr(vis_dict)
    return np.argsort(snr)[::-1][:n_best]


def fourier_fringe_spectrum(band_real, dt):
    """
    Compute the power spectrum of the band-averaged real fringe to identify
    the dominant fringe frequency.  Compare with local_fringe_freq() predictions.

    Parameters
    ----------
    band_real : (N_t,) array  Real part of band-averaged visibility vs time
    dt        : float         Sampling interval in seconds

    Returns
    -------
    freqs : (N_t//2,)  Frequencies, Hz
    power : (N_t//2,)  Power spectrum
    """
    N     = len(band_real)
    win   = np.hanning(N)
    fft   = np.fft.rfft(band_real * win)
    freqs = np.fft.rfftfreq(N, d=dt)
    power = np.abs(fft) ** 2
    return freqs, power


def fourier_filter_fringe(band_real, dt, f_lo, f_hi):
    """
    Band-pass filter the real fringe to isolate the fringe frequency band.

    Parameters
    ----------
    band_real : (N_t,) array  Real fringe signal
    dt        : float         Sampling interval, seconds
    f_lo, f_hi : float        Pass-band edges, Hz

    Returns
    -------
    filtered : (N_t,) array  Filtered fringe
    """
    N     = len(band_real)
    freqs = np.fft.rfftfreq(N, d=dt)
    fft   = np.fft.rfft(band_real)

    mask       = (np.abs(freqs) >= f_lo) & (np.abs(freqs) <= f_hi)
    fft_filt   = fft * mask

    return np.fft.irfft(fft_filt, n=N)
