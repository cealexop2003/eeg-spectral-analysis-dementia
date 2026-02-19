"""
Spectral Leakage Analysis on Synthetic Sinusoidal Signals
==========================================================

This script demonstrates the phenomenon of spectral leakage in the DFT/FFT
by comparing different windowing functions (rectangular, Hanning, Hamming)
on synthetic sinusoidal signals.

Key concepts demonstrated:
- Frequency resolution (Δf = fs/N)
- Effect of windowing on spectral leakage
- "Perfect" periodicity vs. imperfect periodicity in the observation window
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
from pathlib import Path


def compute_spectrum(x, fs, window_type='rectangular'):
    """
    Compute the magnitude spectrum of a signal using different window types.
    
    Parameters
    ----------
    x : array_like
        Input signal
    fs : float
        Sampling frequency in Hz
    window_type : str
        Type of window: 'rectangular', 'hann', or 'hamming'
    
    Returns
    -------
    f : ndarray
        Frequency array (Hz)
    magnitude : ndarray
        Linear magnitude spectrum
    magnitude_db : ndarray
        Magnitude spectrum in dB, normalized to maximum (0 dB)
    """
    N = len(x)
    
    # Apply window function
    if window_type == 'rectangular':
        window = np.ones(N)
    elif window_type == 'hann':
        window = np.hanning(N)
    elif window_type == 'hamming':
        window = np.hamming(N)
    else:
        raise ValueError(f"Unknown window type: {window_type}")
    
    # Apply window to signal
    x_windowed = x * window
    
    # Compute real FFT (efficient for real-valued signals)
    X = np.fft.rfft(x_windowed)
    
    # Compute frequency array
    f = np.fft.rfftfreq(N, d=1/fs)
    
    # Compute magnitude spectrum (linear)
    magnitude = np.abs(X)
    
    # Normalize magnitude for window energy loss
    # For proper amplitude representation
    if window_type == 'rectangular':
        magnitude = magnitude / (N / 2)
    else:
        # For other windows, normalize by the sum of window coefficients
        magnitude = magnitude / (np.sum(window) / 2)
    
    # DC component (index 0) should not be divided by 2
    magnitude[0] = magnitude[0] / 2
    
    # Convert to dB, normalized to maximum (max = 0 dB)
    magnitude_db = 20 * np.log10(magnitude + 1e-12)  # Add small value to avoid log(0)
    magnitude_db = magnitude_db - np.max(magnitude_db)  # Normalize to 0 dB max
    
    return f, magnitude, magnitude_db


def main():
    """Main function to execute the spectral leakage analysis."""
    
    # =========================================================================
    # 1. ΟΡΙΣΜΟΣ ΠΑΡΑΜΕΤΡΩΝ (Parameter Definition)
    # =========================================================================
    fs = 256  # Sampling frequency (Hz)
    T = 10    # Duration (seconds)
    N = fs * T  # Total number of samples
    t = np.arange(0, T, 1/fs)  # Time array
    
    print("=" * 70)
    print("SPECTRAL LEAKAGE ANALYSIS - SYNTHETIC SIGNALS")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  Sampling frequency (fs): {fs} Hz")
    print(f"  Duration (T): {T} s")
    print(f"  Number of samples (N): {N}")
    print(f"  Time array length: {len(t)}")
    
    # =========================================================================
    # 2. ΔΗΜΙΟΥΡΓΙΑ ΣΗΜΑΤΩΝ (Signal Generation)
    # =========================================================================
    # f1 = 10.0 Hz: "Perfect" signal - exactly periodic in the window
    # The window contains exactly 100 complete cycles (10 Hz × 10 s = 100 cycles)
    f1 = 10.0  # Hz
    x1 = np.sin(2 * np.pi * f1 * t)
    
    # f2 = 10.33 Hz: "Imperfect" signal - NOT exactly periodic in the window
    # The window contains 103.3 cycles, which is not an integer multiple
    # This leads to spectral leakage
    f2 = 10.33  # Hz
    x2 = np.sin(2 * np.pi * f2 * t)
    
    print(f"\nSignals:")
    print(f"  Signal 1 (x1): {f1} Hz - Perfect periodicity (integer cycles in window)")
    print(f"  Signal 2 (x2): {f2} Hz - Imperfect periodicity (causes spectral leakage)")
    
    # =========================================================================
    # 3. ΥΠΟΛΟΓΙΣΜΟΣ FREQUENCY RESOLUTION
    # =========================================================================
    # Frequency resolution: Δf = fs / N
    # This is the spacing between frequency bins in the DFT
    delta_f = fs / N
    print(f"\nFrequency Resolution:")
    print(f"  Δf = fs/N = {fs}/{N} = {delta_f} Hz")
    print(f"  This means frequency bins are spaced {delta_f} Hz apart")
    
    # =========================================================================
    # 4. COMPUTE SPECTRA FOR ALL WINDOWS
    # =========================================================================
    window_types = ['rectangular', 'hann', 'hamming']
    
    # Store results for each signal and window combination
    results = {}
    for signal_name, signal in [('x1', x1), ('x2', x2)]:
        results[signal_name] = {}
        for window in window_types:
            f, mag, mag_db = compute_spectrum(signal, fs, window)
            results[signal_name][window] = {
                'f': f,
                'magnitude': mag,
                'magnitude_db': mag_db
            }
    
    # =========================================================================
    # 5. CREATE OUTPUT DIRECTORY
    # =========================================================================
    output_dir = Path('results/part1/synthetic_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # =========================================================================
    # 6. TIME-DOMAIN PLOTS
    # =========================================================================
    print("\nGenerating time-domain plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('Time-Domain Signals: Comparison of Perfect vs Imperfect Periodicity', 
                 fontsize=14, fontweight='bold')
    
    # Full signal plots
    axes[0, 0].plot(t, x1, linewidth=0.8, color='blue')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_title(f'Signal x1 (f = {f1} Hz) - Full Duration')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[1, 0].plot(t, x2, linewidth=0.8, color='red')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].set_title(f'Signal x2 (f = {f2} Hz) - Full Duration')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Zoomed plots at the end of the window
    # Show last 0.5 seconds to visualize periodicity
    zoom_duration = 0.5  # seconds
    zoom_samples = int(zoom_duration * fs)
    t_zoom = t[-zoom_samples:]
    
    axes[0, 1].plot(t_zoom, x1[-zoom_samples:], linewidth=1.5, color='blue', marker='o', 
                    markersize=3, markevery=5)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].set_title(f'Signal x1 - Zoom at End of Window (last {zoom_duration} s)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].text(0.05, 0.95, 'Perfect periodicity:\nStarts and ends at same phase', 
                    transform=axes[0, 1].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    axes[1, 1].plot(t_zoom, x2[-zoom_samples:], linewidth=1.5, color='red', marker='o', 
                    markersize=3, markevery=5)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Amplitude')
    axes[1, 1].set_title(f'Signal x2 - Zoom at End of Window (last {zoom_duration} s)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].text(0.05, 0.95, 'Imperfect periodicity:\nPhase discontinuity at boundaries', 
                    transform=axes[1, 1].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'time_domain_signals.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: time_domain_signals.png")
    
    # =========================================================================
    # 7. SPECTRAL PLOTS - LINEAR MAGNITUDE
    # =========================================================================
    print("\nGenerating linear magnitude spectrum plots...")
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Linear Magnitude Spectra: Effect of Windowing on Spectral Leakage', 
                 fontsize=14, fontweight='bold')
    
    # Frequency range for plotting (0-30 Hz)
    f_max = 30
    
    # Signal x1 (perfect periodicity)
    ax = axes[0]
    for window in window_types:
        f = results['x1'][window]['f']
        mag = results['x1'][window]['magnitude']
        mask = f <= f_max
        ax.plot(f[mask], mag[mask], label=window.capitalize(), linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('Frequency (Hz)', fontsize=11)
    ax.set_ylabel('Magnitude (Linear)', fontsize=11)
    ax.set_title(f'Signal x1 (f = {f1} Hz) - Perfect Periodicity', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.axvline(f1, color='black', linestyle='--', alpha=0.5, linewidth=1, label=f'True freq: {f1} Hz')
    ax.text(0.02, 0.98, 
            'Perfect periodicity → No spectral leakage\nAll windows show sharp peak at exact frequency', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7), fontsize=9)
    
    # Signal x2 (imperfect periodicity)
    ax = axes[1]
    for window in window_types:
        f = results['x2'][window]['f']
        mag = results['x2'][window]['magnitude']
        mask = f <= f_max
        ax.plot(f[mask], mag[mask], label=window.capitalize(), linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('Frequency (Hz)', fontsize=11)
    ax.set_ylabel('Magnitude (Linear)', fontsize=11)
    ax.set_title(f'Signal x2 (f = {f2} Hz) - Imperfect Periodicity', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.axvline(f2, color='black', linestyle='--', alpha=0.5, linewidth=1, label=f'True freq: {f2} Hz')
    ax.text(0.02, 0.98, 
            'Imperfect periodicity → Spectral leakage occurs\nHann/Hamming windows reduce side lobes significantly', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7), fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'spectrum_linear_magnitude.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: spectrum_linear_magnitude.png")
    
    # =========================================================================
    # 8. SPECTRAL PLOTS - dB MAGNITUDE (ZOOMED 8-12 Hz)
    # =========================================================================
    print("\nGenerating dB magnitude spectrum plots (zoomed 8-12 Hz)...")
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('dB Magnitude Spectra: Effect of Windowing on Spectral Leakage (Zoomed 8-12 Hz)', 
                 fontsize=14, fontweight='bold')
    
    # Frequency range for dB plots (zoomed)
    f_min_db = 8
    f_max_db = 12
    
    # Signal x1 (perfect periodicity)
    ax = axes[0]
    for window in window_types:
        f = results['x1'][window]['f']
        mag_db = results['x1'][window]['magnitude_db']
        mask = (f >= f_min_db) & (f <= f_max_db)
        ax.plot(f[mask], mag_db[mask], label=window.capitalize(), linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('Frequency (Hz)', fontsize=11)
    ax.set_ylabel('Magnitude (dB)', fontsize=11)
    ax.set_title(f'Signal x1 (f = {f1} Hz) - Perfect Periodicity', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.axvline(f1, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlim([f_min_db, f_max_db])
    ax.set_ylim([-120, 5])
    ax.text(0.02, 0.98, 
            'Perfect periodicity → Minimal side lobes\nEnergy concentrated at single frequency bin', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7), fontsize=9)
    
    # Signal x2 (imperfect periodicity)
    ax = axes[1]
    for window in window_types:
        f = results['x2'][window]['f']
        mag_db = results['x2'][window]['magnitude_db']
        mask = (f >= f_min_db) & (f <= f_max_db)
        ax.plot(f[mask], mag_db[mask], label=window.capitalize(), linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('Frequency (Hz)', fontsize=11)
    ax.set_ylabel('Magnitude (dB)', fontsize=11)
    ax.set_title(f'Signal x2 (f = {f2} Hz) - Imperfect Periodicity', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.axvline(f2, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlim([f_min_db, f_max_db])
    ax.set_ylim([-120, 5])
    ax.text(0.02, 0.98, 
            'Imperfect periodicity → Severe spectral leakage in rectangular window\n' +
            'Hann/Hamming windows suppress side lobes (better frequency selectivity)', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7), fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'spectrum_db_magnitude.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: spectrum_db_magnitude.png")
    
    # =========================================================================
    # 9. SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nKey Findings:")
    print("  1. Frequency Resolution: Δf = {:.4f} Hz".format(delta_f))
    print("     - Determines the finest frequency detail we can resolve")
    print("\n  2. Perfect Periodicity (f = 10.0 Hz):")
    print("     - Signal completes exact integer number of cycles in window")
    print("     - Minimal spectral leakage with all window types")
    print("     - Peak appears at exact frequency bin")
    print("\n  3. Imperfect Periodicity (f = 10.3 Hz):")
    print("     - Signal does NOT complete integer number of cycles")
    print("     - Rectangular window: severe spectral leakage (high side lobes)")
    print("     - Hann/Hamming windows: significant reduction in side lobes")
    print("     - Trade-off: better side lobe suppression vs. wider main lobe")
    print("\n  4. Window Selection Impact:")
    print("     - Rectangular: Narrowest main lobe, but worst side lobe suppression")
    print("     - Hann/Hamming: Wider main lobe, but excellent side lobe suppression")
    print("     - Critical for distinguishing closely-spaced frequency components")
    print("\nAll plots saved to: {}".format(output_dir.absolute()))
    print("=" * 70)


if __name__ == "__main__":
    main()
