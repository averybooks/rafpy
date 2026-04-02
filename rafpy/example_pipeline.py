"""
example_pipeline.py
End-to-end example: load saved results, run full analysis, produce all plots.

Assumes you have run collect_data and saved results to a .npz file via writefile.py.
Adapt paths and parameters as needed.
"""

import numpy as np
import matplotlib.pyplot as plt

# ── Package imports ──────────────────────────────────────────────────────────
from rafpy import (
    InterfParams,
    fringe_model, local_fringe_freq, geometric_delay,
    brute_force_fit, nonlinear_fit,
    mf_theory, mf_observed, fit_diameter,
    process_visibilities,
)
from rafpy.fringe   import ha_from_lst_ra, lst_from_unix, fringe_period
from rafpy.fitting  import recover_baseline, brute_force_uncertainties
from rafpy.visibility import fourier_fringe_spectrum, fourier_filter_fringe
from rafpy.plotting import (
    plot_visibility, plot_fringe_spectrum, plot_fringe_fit,
    plot_S_surface, plot_fringe_freq, plot_mf_theory,
    plot_mf_fit, plot_waterfall,
)

# ────────────────────────────────────────────────────────────────────────────
# 0.  CONFIGURATION  (edit these)
# ────────────────────────────────────────────────────────────────────────────
DATA_FILE  = 'sun_obs.npz'   # your saved data
FREQ_RF    = 10.674e9        # your RF LO1 frequency, Hz
FREQ_LO2   = 1.540e9        # your LO2 frequency, Hz
B_EW_APPROX = 20.0          # approximate east-west baseline, metres

# ────────────────────────────────────────────────────────────────────────────
# 1.  LOAD DATA
#     Expected .npz keys: 'timestamps', 'visibilities', 'n_acc'
#     visibilities shape: (N_windows, N_chan) complex
# ────────────────────────────────────────────────────────────────────────────
d = np.load(DATA_FILE, allow_pickle=True)

# Build list-of-dicts that process_visibilities() expects
results = []
for i in range(len(d['timestamps'])):
    results.append({
        'timestamp'  : float(d['timestamps'][i]),
        'power_spec' : d['visibilities'][i],   # complex (N_chan,)
        'n_acc'      : int(d['n_acc'][i]),
    })

# ────────────────────────────────────────────────────────────────────────────
# 2.  SYSTEM PARAMETERS
# ────────────────────────────────────────────────────────────────────────────
p = InterfParams(
    b_ew    = B_EW_APPROX,
    b_ns    = 0.0,
    freq_rf = FREQ_RF,
    freq_lo2= FREQ_LO2,
)
print(p)

# ────────────────────────────────────────────────────────────────────────────
# 3.  PROCESS VISIBILITIES
# ────────────────────────────────────────────────────────────────────────────
vis = process_visibilities(results, chan_range=(100, 924))

# Integration time (spacing between windows)
dt_int = np.median(np.diff(vis['times']))  # seconds
print(f"Median integration interval: {dt_int:.1f} s")

# Plot raw amplitude & phase
fig, _ = plot_visibility(vis, title='Band-averaged visibility vs time')
fig.savefig('plot_1_visibility.png', dpi=150)

# Waterfall
fig, _ = plot_waterfall(vis, component='amp', title='|V12| waterfall')
fig.savefig('plot_2_waterfall_amp.png', dpi=150)

# ────────────────────────────────────────────────────────────────────────────
# 4.  GET SUN POSITION  → hour angles
# ────────────────────────────────────────────────────────────────────────────
import ugradio.coord as coord

times = vis['times']
jds   = times / 86400.0 + 2440587.5   # Unix → JD

# Sun RA/Dec at each time
ra_sun_deg  = np.array([coord.sunpos(jd)[0] for jd in jds])
dec_sun_deg = np.array([coord.sunpos(jd)[1] for jd in jds])

# Use median declination for analysis (slowly varying)
delta = np.radians(np.median(dec_sun_deg))
ra    = np.radians(np.median(ra_sun_deg))
print(f"Sun: RA = {np.degrees(ra):.2f} deg, Dec = {np.degrees(delta):.2f} deg")

# LST → hour angle
from interf_analysis.params import LON
lst  = lst_from_unix(times, LON)
h_s  = ha_from_lst_ra(lst, ra)
print(f"Hour angle range: {np.degrees(h_s.min()):.1f} to {np.degrees(h_s.max()):.1f} deg")

# ────────────────────────────────────────────────────────────────────────────
# 5.  EXPECTED FRINGE FREQUENCY RANGE  (sanity check with §8.3)
# ────────────────────────────────────────────────────────────────────────────
ff_hz  = local_fringe_freq(h_s, delta, p.b_ew, p.b_ns, p.lat, p.lam, in_hz=True)
print(f"Expected fringe freq range: {ff_hz.min()*1e3:.2f} – {ff_hz.max()*1e3:.2f} mHz")
print(f"Fringe period at meridian:  {fringe_period(0.0, delta, p.b_ew, p.b_ns, p.lat, p.lam):.1f} s")

fig, _ = plot_fringe_freq(h_s, times, delta, p.b_ew, p.b_ns, p.lat, p.lam)
fig.savefig('plot_3_fringe_freq.png', dpi=150)

# ────────────────────────────────────────────────────────────────────────────
# 6.  FOURIER SPECTRUM OF THE FRINGE  (§8.3 sanity check)
# ────────────────────────────────────────────────────────────────────────────
band_real = vis['band_real']
freqs, power = fourier_fringe_spectrum(band_real, dt_int)

ff_range = (ff_hz.min(), ff_hz.max())
fig, _ = plot_fringe_spectrum(freqs, power, ff_expected_range=ff_range,
                              title='Fourier spectrum of real fringe')
fig.savefig('plot_4_fringe_spectrum.png', dpi=150)

# Optional: Fourier filter to isolate the fringe
band_real_filt = fourier_filter_fringe(band_real, dt_int,
                                        f_lo=max(ff_range[0]*0.5, 1e-5),
                                        f_hi=ff_range[1]*1.5)

# ────────────────────────────────────────────────────────────────────────────
# 7.  BRUTE-FORCE LEAST-SQUARES FIT  (§8.4.1)
# ────────────────────────────────────────────────────────────────────────────
print("\n--- Brute-force fit ---")
bf = brute_force_fit(h_s, band_real_filt,
                     n_ew=600, n_ns=1,
                     b_ew_approx=B_EW_APPROX, lam=p.lam)

print(f"  Q_ew = {bf['Q_ew']:.4f}  (S_min = {bf['S_min']:.4e})")
sig_ew, _ = brute_force_uncertainties(bf)
print(f"  σ(Q_ew) = {sig_ew:.4f}")

fig, _ = plot_S_surface(bf, title='Brute-force: S vs Q_ew')
fig.savefig('plot_5_S_surface.png', dpi=150)

# Recover physical baseline
b_ew_fit, _ = recover_baseline(bf['Q_ew'], 0.0, delta, p.lat, p.lam)
print(f"  b_ew (fit) = {b_ew_fit:.3f} m")

# ────────────────────────────────────────────────────────────────────────────
# 8.  NONLINEAR REFINEMENT  (§8.4.2)
# ────────────────────────────────────────────────────────────────────────────
print("\n--- Nonlinear fit ---")
nl = nonlinear_fit(h_s, band_real_filt, p0_dict=bf)

print(f"  Q_ew = {nl['Q_ew']:.6f} ± {nl['sigma_Q_ew']:.6f}")
print(f"  Q_ns = {nl['Q_ns']:.6f} ± {nl['sigma_Q_ns']:.6f}")
print(f"  A = {nl['A']:.4f},  B = {nl['B']:.4f}")

b_ew_nl, b_ns_nl = recover_baseline(nl['Q_ew'], nl['Q_ns'], delta, p.lat, p.lam)
print(f"  b_ew = {b_ew_nl:.4f} ± {nl['sigma_Q_ew']*p.lam/np.cos(delta):.4f} m")
print(f"  b_ns = {b_ns_nl:.4f} m")

fig, _ = plot_fringe_fit(h_s, band_real_filt, nl,
                         delta_deg=np.degrees(delta),
                         title='Fringe fit (nonlinear least-squares)')
fig.savefig('plot_6_fringe_fit.png', dpi=150)

# ────────────────────────────────────────────────────────────────────────────
# 9.  SOURCE DIAMETER  (§9–10)
# ────────────────────────────────────────────────────────────────────────────
print("\n--- Source diameter ---")

# Theoretical MF
fig, _ = plot_mf_theory(title='MF_theory for uniform disk')
fig.savefig('plot_7_mf_theory.png', dpi=150)

# Observed MF
mf_obs_r = mf_observed(
    h_s, band_real_filt, nl,
    delta=delta, b_ew=b_ew_nl, b_ns=b_ns_nl,
    lat=p.lat, lam=p.lam, n_bins=40,
)

# Fit diameter
diam_r = fit_diameter(mf_obs_r, R_range_rad=(np.radians(0.1), np.radians(1.0)))
print(f"  Best-fit radius   R = {diam_r['R_arcmin']:.2f} arcmin")
print(f"  Best-fit diameter d = {diam_r['diameter_deg']:.3f} deg")
print(f"  (Sun true diameter ≈ 0.53 deg)")

fig, _ = plot_mf_fit(mf_obs_r, diam_r, title='Fringe modulator: observed vs theory')
fig.savefig('plot_8_mf_fit.png', dpi=150)

# ────────────────────────────────────────────────────────────────────────────
print("\nAll plots saved.  Analysis complete.")
plt.show()
