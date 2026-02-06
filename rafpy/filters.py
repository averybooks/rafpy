def RF_filter(signal_array, filter_kernel, caption_text1, caption_text2, sampling_rate=1.0, show_graph=True):
    normalized_kernel = filter_kernel / np.sum(filter_kernel)
    filtered_signal = signal.fftconvolve(signal_array, normalized_kernel, mode='same')
    if show_graph:
        N = len(signal_array)
        t = np.arange(N) / sampling_rate
        M = len(filter_kernel)
        t_kernel = np.arange(M) / sampling_rate
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        plt.subplots_adjust(bottom=0.1, hspace=0.6)
        ax1.set_title("Fig 1. Input 1: Signal & Filter Kernel", fontweight='bold')
        ax1.plot(t, signal_array, color='steelblue', alpha=0.6, label='Input Signal (Left Axis)')
        ax1.tick_params(axis='y', labelcolor='steelblue')
        ax1_twin = ax1.twinx()
        ax1_twin.plot(t_kernel, normalized_kernel, color='#D62728', linewidth=2, label='Filter Kernal (Right Axis)')
        ax1_twin.fill_between(t_kernel, 0, normalized_kernel, color='#D62728', alpha=0.3)
        ax1_twin.tick_params(axis='y', labelcolor='#D62728')
        ax1_twin.set_ylim(bottom=0)
        ax1_twin.set_ylabel("Normalized Amplitude", color='#D62728')
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
        ax1.set_xlabel(f"Time (s)if Fs={sampling_rate} Hz")
        ax1.set_ylabel("Amplitude", color = 'steelblue')
        desc_1 = caption_text1 
        desc_2 = caption_text2
        ax1.text(0.5, -0.25, textwrap.fill(desc_1, width = 100), transform=ax1.transAxes, ha='center', va='top', fontsize=10, color='darkred', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.5))
        ax2.set_title("Fig 2. Output: Convolved Result", fontweight='bold')
        ax2.plot(t, signal_array, color='gray', alpha=0.3, label='(Original Input)')
        ax2.plot(t, filtered_signal, color='green', linewidth=2, label='Filtered Output')
        ax2.set_xlabel(f"Time (seconds) if Fs={sampling_rate}Hz")
        ax2.set_ylabel("Amplitude")
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.text(0.5, -0.3, textwrap.fill(desc_2, width = 100), transform=ax2.transAxes, ha='center', va='top', fontsize=10, color='darkgreen', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.5))
       
        plt.show()
    return filtered_signal

def get_acf_amplitude(signal_array):
    # 1. Remove DC offset (Center the signal at 0)
    centered_signal = signal_array - np.mean(signal_array)
    
    # 2. Calculate the Autocorrelation
    # 'same' mode keeps the output the same length as the input
    acf = np.correlate(centered_signal, centered_signal, mode='full')
    
    # 3. Get the value at zero lag (the center of the 'full' correlation)
    zero_lag_index = len(acf) // 2
    r_0 = acf[zero_lag_index] / len(centered_signal) # Normalize by length
    
    # 4. Calculate RMS and Peak Amplitude
    rms = np.sqrt(r_0)
    peak_amplitude = rms * np.sqrt(2)
    
    return peak_amplitude

def create_bpf_kernel(low_kb, high_kb, fs, num_taps=101):
    """Utility to generate a Band-Pass Filter kernel."""
    return sp_signal.firwin(num_taps, [low_kb, high_kb], fs=fs, pass_zero=False)

def plot_frequency_response(data_list, use_acf=True):
    """
    Plots Max Voltage vs Frequency, handling 1.5MHz and 3MHz separately.
    data_list: list of (freq_hz, filename, fs_hz)
    """
    data_1_5 = []
    data_3_0 = []

    for freq, filename, fs in data_list:
        try:
            data_file = np.load(filename)
            signal_data = data_file["arr_0"][2]
            
            # Choose amplitude method
            val = get_acf_amplitude(signal_data) if use_acf else np.max(np.abs(signal_data))
            
            if fs == 1500000:
                data_1_5.append((freq / 1000, val))
            else:
                data_3_0.append((freq / 1000, val))
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    # Sort to prevent the 'sawtooth' lines
    data_1_5.sort()
    data_3_0.sort()

    plt.figure(figsize=(10, 6))
    if data_1_5:
        x15, y15 = zip(*data_1_5)
        plt.plot(x15, y15, 'bo-', label='1.5 MHz Fs', alpha=0.8)
    if data_3_0:
        x30, y30 = zip(*data_3_0)
        plt.plot(x30, y30, 'ro-', label='3.0 MHz Fs', alpha=0.8)

    plt.title(f"Frequency Response ({'ACF' if use_acf else 'Raw'} Peak)")
    plt.xlabel("Signal Frequency (kHz)")
    plt.ylabel("Amplitude (V)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()