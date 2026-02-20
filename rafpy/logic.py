import numpy as np
import matplotlib.pyplot as plt
def id_signal_candidates(fig_num, observed, current_fs, filter_range=None, show_graph=True):
    #identifies potential true frequencies and highlights the one within the analog filter range
    #filter_range: tuple (f_min, f_max)
    
    candidates =[]
    f_nyquist = current_fs /2 

    #logic for finding frequency within filter range
    match = None
    nyquist_zone = None
    if filter_range:
        f_min, f_max = filter_range
        num_candidates = int(np.ceil(f_max / current_fs))
    else:
        num_candidates = 5
        
        
                
    for N in range(num_candidates+1):
        f_possible_1 = N*current_fs + observed
        f_possible_2 = N*current_fs - observed
        if f_possible_1 > 0: candidates.append(f_possible_1)
        if f_possible_2 > 0: candidates.append(f_possible_2)  
        #plt.figure(figsize=(12, 6))
    unique_candidates = sorted(list(set(candidates)))

    if filter_range:
        for f in unique_candidates:
            if f_min <= f <= f_max:
                match = f
                nyquist_zone = int(np.ceil(f/(current_fs/2)))
                break
        
    if show_graph:
        fig, ax = plt.subplots(figsize=(12, 5))
        plt.subplots_adjust(bottom=0.3)

        #plots all candidates
        markerline, stemlines, baseline = ax.stem(unique_candidates, np.ones(len(unique_candidates)))
        plt.setp(markerline, color='red', marker='D', markersize=6, alpha=0.5)
        plt.setp(stemlines, color='red', linestyle='--', alpha = 0.3)

        #highlight filter range
        if filter_range:
            ax.axvspan(filter_range[0], filter_range[1], color='orange', alpha=0.15, label='Filter Passband') 
        #highlight frequency
        if match:
            ax.stem([match], [1], linefmt='r-', markerfmt='rD', basefmt=' ')
            ax.annotate(f'MATCH: {match} Hz\nZone {nyquist_zone}', xy=(match, 1), xytext=(match, 1.4), arrowprops=dict(facecolor='green', shrink=0.05), ha='center', fontweight='bold', color='green')
        if observed:
            ax.stem([observed], [1], linefmt='r-', markerfmt='rD', basefmt=' ')
            ax.annotate(f'OBSERVED: {observed} Hz', xy=(observed, 1), xytext=(observed, 1.3), arrowprops=dict(facecolor='blue', shrink=0.05), ha='center', fontweight='bold', color='blue')
        #ax.annotate('Observed\n(Alias)', xy=(observed, 1), xytext=(observed, 1.3),
                    #arrowprops=dict(facecolor='black', shrink=0.05), ha='center')
        ax.set_yticks([]) 
        ax.set_title(f"Fig {fig_num}. Signal Identification (Observed: {observed} Hz | $f_s$: {current_fs} Hz | Bandpass {np.min(filter_range)}-{np.max(filter_range)} Hz)", fontweight='bold')
        ax.set_xlabel("Frequency (Hz)", fontweight='bold')
        ax.set_ylim(0, 1.6)
        plt.legend(loc='upper right')
        print("-" * 30)
        print(f"ANALYSIS FOR {observed} Hz (Fs = {current_fs} Hz)")
        print(f"Potential Frequencies: {unique_candidates}")
        if match:
            print(f"IDENTIFIED SIGNAL: {match} Hz (Nyquist Zone {nyquist_zone})")
        print("-" * 30)
        plt.show()
    return {"all_candidates": unique_candidates, "identified_f": match, "nyquist_zone":nyquist_zone}, match

def plot_complex_spectra(data, fs, freqlimbot=0, freqlimup=500000, title="Complex Voltage Analysis"):
    if data.ndim > 1:
        data = data[0, :]
        
    N = len(data)
    window = np.hanning(N)
    fft_val = np.fft.rfft(data * window)
    freqs = np.fft.rfftfreq(N, 1/fs)
    
    mag = np.abs(fft_val) 
    real = np.real(fft_val) 
    imag = np.imag(fft_val)

    plt.figure(figsize=(12, 6))
    
    # Plotting
    plt.plot(freqs, real, color='tab:cyan', label='Real (Cosine)', zorder=2)
    plt.plot(freqs, imag, color='tab:red', label='Imaginary (Sine)', zorder=1)

    # Styling
    plt.title(title, fontweight='bold')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Voltage Amplitude (Arbitrary Units)")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Zoom to your 50kHz - 500kHz interest area
    plt.xlim(freqlimbot, freqlimup) 
    
    plt.show()
    return fft_val, freqs

def plot_power_and_autocorr_padded(data, fs=3000000, title="SSB Analysis"):
    # 1. Ensure 1D
    if data.ndim > 1:
        data = data.flatten()

    N = len(data)
    # Apply Hanning window to force the autocorrelation into a single diamond shape
    windowed_data = data * np.hanning(N)

    # 2. Linear Padding (Power of 2 is faster for FFT)
    N_padded = N * 2  

    # 3. Calculate FFT
    fft_full = np.fft.fft(windowed_data, n=N_padded)
    fft_shifted = np.fft.fftshift(fft_full)
    
    # Convert Hz to kHz for the x-axis
    freqs_khz = np.fft.fftshift(np.fft.fftfreq(N_padded, 1/fs)) / 1000

    # 4. Correct Power Spectrum Calculation
    # We use Magnitude first, then normalize to 0 dB to avoid the "flat line" floor issue
    mag = np.abs(fft_shifted)
    power_log = 20 * np.log10(mag / (np.max(mag) + 1e-12) + 1e-12)

    # 5. Autocorrelation (using Magnitude to handle Complex SSB data)
    # Power raw must be unshifted for the IFFT
    power_raw = np.abs(fft_full)**2
    autocorr = np.abs(np.fft.ifft(power_raw))
    autocorr_centered = np.fft.fftshift(autocorr)

    # 6. Time Lags in microseconds (better for 3MHz sampling)
    lags_us = np.fft.fftshift(np.fft.fftfreq(N_padded, 1/(fs * 1e6))) 

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Top Plot: Power Spectrum
    ax1.plot(freqs_khz, power_log, color='purple')
    ax1.set_title(f"{title} - Power Spectrum")
    ax1.set_ylabel("Relative Power (dB)")
    ax1.set_xlabel("Frequency (kHz)")
    # Set x-limits to show a broader view (e.g., +/- 1 MHz)
    ax1.set_xlim(-1000, 1000) 
    ax1.grid(True, alpha=0.3)

    # Bottom Plot: Autocorrelation
    ax2.plot(lags_us, autocorr_centered, color='darkgreen')
    ax2.set_title("Linear Autocorrelation (Diamond Envelope)")
    ax2.set_ylabel("Magnitude")
    ax2.set_xlabel("Lag Time (µs)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
   

def calculate_peak_to_valley(freqs, fft_mag, f1_target, f2_target):
    """
    Calculates the Peak-to-Valley ratio for two signals.
    f1_target, f2_target: the expected frequencies (e.g., 100000 and 105000)
    """
    # 1. Find indices for the two peaks within a small range (+/- 1kHz)
    idx1 = np.where((freqs > f1_target - 1000) & (freqs < f1_target + 1000))[0]
    idx2 = np.where((freqs > f2_target - 1000) & (freqs < f2_target + 1000))[0]
    
    peak1_val = np.max(fft_mag[idx1])
    peak2_val = np.max(fft_mag[idx2])
    
    # 2. Find the "Valley" (minimum) between the two peak frequencies
    f1_actual = freqs[idx1[np.argmax(fft_mag[idx1])]]
    f2_actual = freqs[idx2[np.argmax(fft_mag[idx2])]]
    
    valley_idx = np.where((freqs > f1_actual) & (freqs < f2_actual))[0]
    valley_val = np.min(fft_mag[valley_idx])
    
    # 3. Calculate Ratio
    avg_peak = (peak1_val + peak2_val) / 2
    ptv_ratio = avg_peak / valley_val
    
    # 4. Convert to dB (optional but recommended for your report)
    ptv_db = 20 * np.log10(ptv_ratio)
    
    return ptv_ratio, ptv_db, f1_actual, f2_actual

def plot_2d_waterfall(file_path, array_key='samples'):
    # 1. Load Data
    with np.load(file_path) as data:
        matrix = np.abs(data[array_key])

    # 2. Fix Shape / Rank (Prevents "sequence argument" error)
    matrix = np.squeeze(matrix) # Remove empty dimensions
    
    if matrix.ndim == 1:
        # If it's just one row, we can't do a waterfall, so we expand it
        matrix = matrix[np.newaxis, :]
    elif matrix.ndim == 3:
        # If it has a channel dim (Time, Freq, 1), take the first slice
        matrix = matrix[:, :, 0]

    # 3. Baseline Subtraction (Removes the vertical lines/static)
    baseline = np.median(matrix, axis=0)
    diff_waterfall = matrix - baseline

    # 4. Apply Gaussian Smoothing
    # We use a sigma of (0.5, 1) to keep it looking crisp like your image
    smoothed = gaussian_filter(diff_waterfall, sigma=(0.5, 1))

    # 5. Plotting
    plt.figure(figsize=(10, 6))
    
    # Define frequency range based on your image (1419.5 to 1421.5)
    f_start, f_end = 1419.5, 1421.5
    
    im = plt.imshow(smoothed, 
                    aspect='auto', 
                    extent=[f_start, f_end, matrix.shape[0], 0],
                    cmap='viridis', # Matches your image
                    interpolation='bilinear')

    # 6. Color Scaling (The "Secret Sauce")
    # Setting the limits to +/- a few standard deviations makes faint signals pop
    v_limit = np.std(smoothed) * 2
    im.set_clim(-v_limit, v_limit)

    plt.colorbar(im, label='Relative Power')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Measurement Number (Time)')
    plt.title(f'Waterfall Plot (NPZ): {file_path}')
    
    plt.show()