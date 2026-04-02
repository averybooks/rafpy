"""
params.py
System constants and observation parameters for the UCB X-Band interferometer.

The interferometer operates at ~10.5 GHz with an approximately east-west baseline
of ~20m. The SNAP board outputs complex cross-correlation visibilities V12(nu, t).
"""

import numpy as np
import ugradio.timing

# ── Physical / site constants ────────────────────────────────────────────────
C          = 2.99792458e8        # speed of light, m/s
OMEGA_EARTH = 2 * np.pi / 86164.0905  # Earth sidereal rotation rate, rad/s

# UCB NCH Infomration
LAT_DEG  =  37.8731          # degrees N
LON_DEG  = -122.2571         # degrees E (negative = west)
ALT_M    =  95            # metres

LAT  = np.radians(LAT_DEG)
LON  = np.radians(LON_DEG)

# ── Interferometer baseline (approximate; refined by least-squares) ──────────
# East-West and North-South components, metres
B_EW_DEFAULT = 20.0   # ~20 m east-west
B_NS_DEFAULT =  0.0   # approximately east-west baseline; bns ~ 0

# ── RF / signal chain ────────────────────────────────────────────────────────
FREQ_RF_GHz = 10.674          # nominal centre RF frequency, GHz
FREQ_RF     = FREQ_RF_GHz * 1e9   # Hz

# SNAP board digitises at 500 Msps, 1024 channels spanning 0–250 MHz baseband.
# The second LO mixes IF → baseband; the channel frequencies are:
#   nu_chan[k] = k * (sample_rate / 2) / n_chan   for k = 0 … n_chan-1
SAMPLE_RATE = 500e6           # samples per second
N_CHAN      = 1024            # number of spectral channels
CHAN_BW     = SAMPLE_RATE / 2 / N_CHAN   # Hz per channel (~244 kHz)

# Wavelength at centre RF frequency
LAMBDA      = C / FREQ_RF     # metres


class InterfParams:
    """
    Container for all interferometer and observation parameters.

    Parameters
    ----------
    b_ew : float
        East-west baseline component, metres.  (default: B_EW_DEFAULT)
    b_ns : float
        North-south baseline component, metres. (default: B_NS_DEFAULT)
    freq_rf : float
        Centre RF observing frequency, Hz. (default: FREQ_RF)
    freq_lo2 : float
        Second LO frequency used to mix IF → baseband, Hz.
        Required to recover the true sky frequency of each channel.
    lat : float
        Observatory geodetic latitude, radians. (default: LAT)
    lon : float
        Observatory geodetic longitude, radians. (default: LON)
    alt : float
        Observatory altitude, metres. (default: ALT_M)
    """

    def __init__(self,
                 b_ew    = B_EW_DEFAULT,
                 b_ns    = B_NS_DEFAULT,
                 freq_rf = FREQ_RF,
                 freq_lo2 = None,
                 lat     = LAT,
                 lon     = LON,
                 alt     = ALT_M):

        self.b_ew     = b_ew
        self.b_ns     = b_ns
        self.freq_rf  = freq_rf
        self.lam      = C / freq_rf
        self.freq_lo2 = freq_lo2
        self.lat      = lat
        self.lon      = lon
        self.alt      = alt

        # Each SNAP channel k corresponds to baseband frequency k * CHAN_BW
        # The sky frequency is freq_rf - IF_centre + baseband_freq
        # we only need relative frequencies for most calculations so store both
        self.chan_baseband_freq = np.arange(N_CHAN) * CHAN_BW   # Hz, 0–250 MHz
        if freq_lo2 is not None:
            # Sky frequency = LO1 + LO2 + baseband  (SSB, upper sideband assumed)
            # Adjust sign depending on your mixer chain.
            self.chan_sky_freq = freq_lo2 + self.chan_baseband_freq
        else:
            self.chan_sky_freq = None

        # Wavelengths per channel (for baseline-in-wavelengths calculations)
        self.chan_lam = C / (self.chan_baseband_freq + freq_rf) if freq_lo2 is None \
                        else C / self.chan_sky_freq

    # ── Derived quantities ────────────────────────────────────────────────────

    def q_ew(self, delta_rad):
        """
        Q_ew = (b_ew / lambda) * cos(delta)
        Baseline in wavelengths projected east-west. (eq. 11)
        """
        return (self.b_ew / self.lam) * np.cos(delta_rad)

    def q_ns(self, delta_rad):
        """
        Q_ns = (b_ns / lambda) * sin(L) * cos(delta)
        Baseline in wavelengths projected north-south. (eq. 11)
        """
        return (self.b_ns / self.lam) * np.sin(self.lat) * np.cos(delta_rad)

    def integration_time(self):
        """
        SNAP integration time per accumulation.
        From the lab: 305200 spectra × 2048 samples × 2 ns = 1.25 s
        """
        n_spec   = 305200
        n_samp   = 2048
        dt_samp  = 1.0 / SAMPLE_RATE   # 2 ns
        return n_spec * n_samp * dt_samp

    def __repr__(self):
        return (f"InterfParams(b_ew={self.b_ew:.2f} m, b_ns={self.b_ns:.2f} m, "
                f"freq={self.freq_rf/1e9:.3f} GHz, lambda={self.lam*100:.2f} cm, "
                f"lat={np.degrees(self.lat):.4f} deg)")
    import numpy as np
from rafpy.params import N_CHAN, CHAN_BW

def chan_to_sky_freq(lo1_hz, lo2_hz, chan_range=None):
    """
    Convert SNAP channel indices to sky frequencies in GHz.
    
    lo1_hz : float  First LO frequency (RF → IF), Hz
    lo2_hz : float  Second LO frequency (IF → baseband), Hz
    """
    k = np.arange(N_CHAN)
    baseband_hz = k * CHAN_BW                        # 0 to 250 MHz
    sky_hz      = lo1_hz + lo2_hz + baseband_hz      # sky frequency
    sky_ghz     = sky_hz / 1e9

    if chan_range is not None:
        sky_ghz = sky_ghz[chan_range[0]:chan_range[1]+1]

    return sky_ghz
