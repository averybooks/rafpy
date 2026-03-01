A package currently being iterated on for use in UC Berkeley Astron 121 Radio Frequency Undergraduate Lab.

I plan to continue to update this package with new functions, QoL, RF, and graphing functions relevant to data analysis of RF astronomical data.

Currently the functions in this package are:
1. RF_filter(fignum, fignum2, signal_array, filter_kernel, caption_text1, caption_text2, xlimitlower=0, xlimitupper=100, ylimitlower=-50, ylimitupper=50, sampling_rate=1.0, show_graph=True)

This function acts as a convolution filter taking array data and a digitally generated filter kernal. The user must generate the filter kernel themselves before input, however the nature of the algorithm should be able to take any function as the kernel. This code will return a filtered array of data, and two graphs displafying the data array and kernel on one graph, and the filtered result in another graph.

2. get_acf_amplitude(signal_array)

This function computed the autocorrelation function and returns the amplitude.

3. def bpf_kernel(low_kb, high_kb, fs, num_taps=101)

This function generates a bandpass filter kernel for use in the filter function.

4. def plot_frequency_response(data_list, use_acf=True)

This function plots frequency vs max voltage.

5. def id_signal_candidates(fig_num, observed, current_fs, filter_range=None, show_graph=True)

This function takes an input frequency, sampling rate, and any filter ranges and determines the possible alias frequencies. It will consider the physical filter range and determine the original signal frequency before aliasing. It will also determine which Nyquist zone the original frequency exists in. This function is designed for alias troublshooting and physical system debugging.

6. def plot_complex_spectra(data, fs, freqlimbot=0, freqlimup=500000, title="Complex Voltage Analysis")

This function plots the complex voltage spectra of a given data array and a sampling rate.

There are a few other functions nested inside this source code, however their use outside of Lab 1 of the course may not be as often as those listed here. If functions are updated for increased frequency use cases they will be added to the readME.
