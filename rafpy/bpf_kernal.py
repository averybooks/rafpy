def create_bpf_kernel(low_kb, high_kb, fs, num_taps=101):
    """Utility to generate a Band-Pass Filter kernel."""
    return sp_signal.firwin(num_taps, [low_kb, high_kb], fs=fs, pass_zero=False)