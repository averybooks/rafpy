"""
fringe.py
Fringe model and local fringe frequency calculations.

Implements equations 9–15 from the lab handout.  All angles are in radians
unless otherwise noted.  Hour angle increases with time (west is positive).

The SNAP board delivers complex visibilities V12(nu, t).  The *model* for a
point source at declination delta observed at hour angle h_s is (eq. 12):

    F(h_s) = A * cos(2*pi*nu*tau'_g)  +  B * sin(2*pi*nu*tau'_g)

where tau'_g is the geometric delay (eq. 4/11) and A, B absorb the cable
delay / phase (eq. 10).
"""

import numpy as np
from .params import OMEGA_EARTH, InterfParams


# ── Geometric delay ───────────────────────────────────────────────────────────

def geometric_delay(h_s, delta, b_ew, b_ns, lat, lam=None, freq=None):
    """
    Compute the modified geometric delay τ'_g (eq. 4 / 11).

    If lam is given, returns delay in wavelengths (ν τ'_g, dimensionless).
    If freq is given instead, lam = c/freq is used.
    If neither is given, returns delay in seconds.

    Parameters
    ----------
    h_s   : array_like  Hour angle of source, radians
    delta : float       Declination of source, radians
    b_ew  : float       East-west baseline, metres
    b_ns  : float       North-south baseline, metres
    lat   : float       Observatory latitude, radians
    lam   : float, optional  Wavelength, metres
    freq  : float, optional  Frequency, Hz (used if lam is None)

    Returns
    -------
    tau_g : ndarray   Geometric delay (seconds or wavelengths)
    """
    from .params import C
    h_s   = np.asarray(h_s, dtype=float)

    # eq. 4: tau'_g = (b_ew/c)*cos(d)*sin(h) + (b_ns/c)*sin(L)*cos(d)*cos(h)
    tau_g_sec = ((b_ew / C) * np.cos(delta) * np.sin(h_s)
               + (b_ns / C) * np.sin(lat) * np.cos(delta) * np.cos(h_s))

    if lam is not None:
        return tau_g_sec * (C / lam)   # dimensionless: ν τ'_g
    elif freq is not None:
        return tau_g_sec * freq
    return tau_g_sec

def geometric_delay_wavelengths(h_s, delta, p: InterfParams, chan_idx=None):
    """Returns ν τ'_g (dimensionless), the geometric delay in wavelengths.
    This is what goes into the fringe model phase argument 2*pi*nu*tau_g."""
    if chan_idx is not None and p.chan_sky_freq is not None:
        lam = p.chan_lam[chan_idx]
    else:
        lam = p.lam
    return geometric_delay(h_s, delta, p.b_ew, p.b_ns, p.lat, lam=lam)


def geometric_delay_seconds(h_s, delta, p: InterfParams):
    """Returns τ'_g in seconds. Useful for cable delay estimation or debugging.
    Note: frequency-independent, so no chan_idx needed."""
    return geometric_delay(h_s, delta, p.b_ew, p.b_ns, p.lat)

def geometric_delay_from_params(h_s, delta, p: InterfParams, chan_idx=None):
    """
    Convenience wrapper using an InterfParams object.

    Parameters
    ----------
    h_s      : array_like  Hour angles, radians
    delta    : float       Declination, radians
    p        : InterfParams
    chan_idx : int, optional  If given, use the sky frequency of that channel.
                              If None, use the centre RF frequency.

    Returns
    -------
    nu_tau_g : ndarray  Geometric delay in wavelengths (ν τ'_g)
    """
    if chan_idx is not None and p.chan_sky_freq is not None:
        freq = p.chan_sky_freq[chan_idx]
        lam  = p.chan_lam[chan_idx]
    else:
        freq = p.freq_rf
        lam  = p.lam

    return geometric_delay(h_s, delta, p.b_ew, p.b_ns, p.lat, lam=lam)


# ── Local fringe frequency ────────────────────────────────────────────────────

def local_fringe_freq(h_s, delta, b_ew, b_ns, lat, lam, in_hz=True):
    """
    Local fringe frequency f_f (eq. 15).

    f_{f,Hz} / omega_Earth = (b_ew/lam)*cos(d)*cos(h)
                           - (b_ns/lam)*sin(L)*cos(d)*sin(h)

    Parameters
    ----------
    h_s    : array_like  Hour angle(s), radians
    delta  : float       Declination, radians
    b_ew   : float       East-west baseline, metres
    b_ns   : float       North-south baseline, metres
    lat    : float       Observatory latitude, radians
    lam    : float       Wavelength, metres
    in_hz  : bool        If True (default) return Hz; if False return cycles/radian

    Returns
    -------
    f_f : ndarray  Local fringe frequency
    """
    h_s = np.asarray(h_s, dtype=float)

    # cycles per radian (eq. 15 without omega)
    ff_rad = ((b_ew / lam) * np.cos(delta) * np.cos(h_s)
             - (b_ns / lam) * np.sin(lat) * np.cos(delta) * np.sin(h_s))

    if in_hz:
        return ff_rad * OMEGA_EARTH   # Hz
    return ff_rad


def fringe_period(h_s, delta, b_ew, b_ns, lat, lam):
    """
    Fringe period in seconds (= 1 / f_f) at each hour angle.
    """
    ff = local_fringe_freq(h_s, delta, b_ew, b_ns, lat, lam, in_hz=True)
    with np.errstate(divide='ignore'):
        return np.where(ff != 0, 1.0 / ff, np.inf)


# ── Fringe model ──────────────────────────────────────────────────────────────

def fringe_model(h_s, A, B, delta, b_ew, b_ns, lat, lam):
    """
    Point-source fringe model (eq. 12):

        F(h_s) = A * cos(2*pi * nu*tau'_g)  +  B * sin(2*pi * nu*tau'_g)

    A absorbs  cos(2*pi*nu*tau'_c)  (cable delay cosine term)
    B absorbs -sin(2*pi*nu*tau'_c)  (cable delay sine term)

    Parameters
    ----------
    h_s   : array_like  Hour angles, radians
    A, B  : float       Amplitude coefficients (absorb cable phase)
    delta : float       Source declination, radians
    b_ew  : float       East-west baseline, metres
    b_ns  : float       North-south baseline, metres
    lat   : float       Observatory latitude, radians
    lam   : float       Wavelength, metres

    Returns
    -------
    F : ndarray  Model fringe (real)
    """
    phi = 2 * np.pi * geometric_delay(h_s, delta, b_ew, b_ns, lat, lam=lam)
    return A * np.cos(phi) + B * np.sin(phi)


def fringe_amplitude(A, B):
    """Fringe amplitude from least-squares coefficients A, B."""
    return np.hypot(A, B)


def fringe_phase(A, B):
    """Fringe phase (cable delay term) in radians."""
    return np.arctan2(-B, A)


# ── Hour angle utilities ──────────────────────────────────────────────────────

def ha_from_lst_ra(lst_rad, ra_rad):
    """
    Hour angle h_s = LST - RA, wrapped to [-pi, pi].

    Parameters
    ----------
    lst_rad : array_like  Local Sidereal Time, radians
    ra_rad  : float       Source right ascension, radians

    Returns
    -------
    h_s : ndarray  Hour angle, radians
    """
    h_s = np.asarray(lst_rad, dtype=float) - ra_rad
    # Wrap to [-pi, pi]
    h_s = (h_s + np.pi) % (2 * np.pi) - np.pi
    return h_s


def lst_from_unix(unix_times, lon_rad):
    """
    Compute Local Sidereal Time from Unix timestamps.

    Uses ugradio.timing if available; otherwise falls back to a simple
    approximation good to ~0.1 s.

    Parameters
    ----------
    unix_times : array_like  Unix timestamps, seconds
    lon_rad    : float       Observatory longitude, radians (east positive)

    Returns
    -------
    lst : ndarray  LST, radians in [0, 2*pi)
    """
    try:
        import ugradio.timing as timing
        lst = np.array([timing.lst(t, lon_rad) for t in np.atleast_1d(unix_times)])
    except Exception:
        # Fallback: approximate LST from J2000 epoch
        # JD of Unix epoch: 2440587.5
        unix_times = np.asarray(unix_times, dtype=float)
        jd = unix_times / 86400.0 + 2440587.5
        T  = (jd - 2451545.0) / 36525.0   # Julian centuries from J2000
        # Greenwich Mean Sidereal Time (degrees)
        gmst_deg = (280.46061837
                    + 360.98564736629 * (jd - 2451545.0)
                    + 0.000387933 * T**2
                    - T**3 / 38710000.0) % 360.0
        lst = np.radians(gmst_deg) + lon_rad
        lst = lst % (2 * np.pi)
    return lst
