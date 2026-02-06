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