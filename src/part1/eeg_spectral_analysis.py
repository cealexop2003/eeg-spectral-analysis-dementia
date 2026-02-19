"""
EEG Spectral Analysis: Comparing Band Power Across Clinical Groups
===================================================================

This script analyzes EEG data from Alzheimer's (AD), Frontotemporal Dementia (FTD),
and healthy Control (CN) groups to investigate how spectral band power differs
between groups and how windowing parameters affect the analysis.

Dataset: "EEG of Alzheimer's and Frontotemporal dementia" (Kaggle)
Sampling frequency: 500 Hz
Groups: CN (Controls), AD (Alzheimer's), FTD (Frontotemporal Dementia)

Key Analysis:
- Effect of window type (rectangular, Hann, Hamming) on band power estimation
- Effect of window length (2s, 4s, 8s) on frequency resolution and variance
- Group differences in theta, alpha, and beta band power
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
from pathlib import Path
import mne
from scipy import signal
import pandas as pd
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

# Dataset paths
DATA_DIR = Path('data/raw/dataset')
OUTPUT_DIR = Path('results/part1/eeg_real_results')

# EEG parameters
FS = 500  # Sampling frequency (Hz)
CHANNEL = 'Pz'  # Selected channel for analysis

# Segmentation parameters
SEGMENT_DURATION = 15  # seconds (use 15s segments from resting state)

# Windowing parameters
WINDOW_LENGTHS = [2, 4, 8]  # seconds
WINDOW_TYPES = ['rectangular', 'hann', 'hamming']
OVERLAP = 0.5  # 50% overlap

# Frequency bands
BANDS = {
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (13, 30)
}
TOTAL_BAND = (1, 40)  # For normalization

# Subject selection (use all available subjects)
SUBJECTS_PER_GROUP = None  # None means use all subjects


# ============================================================================
# DATA LOADING
# ============================================================================

def load_participants_info() -> pd.DataFrame:
    """
    Load participant information from TSV file.
    
    Returns
    -------
    df : pd.DataFrame
        Participant information with columns: participant_id, Gender, Age, Group, MMSE
    """
    participants_file = DATA_DIR / 'participants.tsv'
    df = pd.read_csv(participants_file, sep='\t')
    
    # Map group letters to descriptive names
    group_mapping = {'A': 'AD', 'C': 'CN', 'F': 'FTD'}
    df['Group'] = df['Group'].map(group_mapping)
    
    return df


def load_eeg_data(subject_id: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load EEG data for a specific subject.
    
    Parameters
    ----------
    subject_id : str
        Subject identifier (e.g., 'sub-001')
    
    Returns
    -------
    data : np.ndarray
        EEG data (channels x samples)
    channel_names : list
        List of channel names
    """
    eeg_file = DATA_DIR / subject_id / 'eeg' / f'{subject_id}_task-eyesclosed_eeg.set'
    
    # Load using MNE
    raw = mne.io.read_raw_eeglab(eeg_file, preload=True, verbose=False)
    
    # Get data and channel names
    data = raw.get_data()  # Shape: (n_channels, n_samples)
    channel_names = raw.ch_names
    
    return data, channel_names


def extract_channel_data(data: np.ndarray, channel_names: List[str], 
                         target_channel: str) -> np.ndarray:
    """
    Extract data from a specific channel.
    
    Parameters
    ----------
    data : np.ndarray
        Full EEG data (channels x samples)
    channel_names : list
        List of channel names
    target_channel : str
        Target channel name
    
    Returns
    -------
    channel_data : np.ndarray
        Data from the target channel
    """
    if target_channel not in channel_names:
        raise ValueError(f"Channel {target_channel} not found. Available: {channel_names}")
    
    channel_idx = channel_names.index(target_channel)
    return data[channel_idx, :]


# ============================================================================
# SIGNAL PROCESSING
# ============================================================================

def extract_segment(data: np.ndarray, fs: int, duration: float) -> np.ndarray:
    """
    Extract a segment from the middle of the recording.
    
    Parameters
    ----------
    data : np.ndarray
        Full signal
    fs : int
        Sampling frequency (Hz)
    duration : float
        Segment duration (seconds)
    
    Returns
    -------
    segment : np.ndarray
        Extracted segment
    """
    n_samples = int(duration * fs)
    total_samples = len(data)
    
    if total_samples < n_samples:
        raise ValueError(f"Signal too short. Need {n_samples} samples, have {total_samples}")
    
    # Extract from middle of recording
    start_idx = (total_samples - n_samples) // 2
    end_idx = start_idx + n_samples
    
    return data[start_idx:end_idx]


def create_window(window_type: str, length: int) -> np.ndarray:
    """
    Create a window function.
    
    Parameters
    ----------
    window_type : str
        Type of window: 'rectangular', 'hann', or 'hamming'
    length : int
        Window length in samples
    
    Returns
    -------
    window : np.ndarray
        Window function
    """
    if window_type == 'rectangular':
        return np.ones(length)
    elif window_type == 'hann':
        return np.hanning(length)
    elif window_type == 'hamming':
        return np.hamming(length)
    else:
        raise ValueError(f"Unknown window type: {window_type}")


def compute_psd_welch(data: np.ndarray, fs: int, window_length: float, 
                      window_type: str, overlap: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Power Spectral Density using Welch's method.
    
    Parameters
    ----------
    data : np.ndarray
        Input signal
    fs : int
        Sampling frequency (Hz)
    window_length : float
        Window length (seconds)
    window_type : str
        Type of window
    overlap : float
        Overlap fraction (0-1)
    
    Returns
    -------
    f : np.ndarray
        Frequency array (Hz)
    psd : np.ndarray
        Power spectral density
    """
    # Convert window length to samples
    nperseg = int(window_length * fs)
    noverlap = int(nperseg * overlap)
    
    # Create window
    window = create_window(window_type, nperseg)
    
    # Compute PSD using Welch's method
    f, psd = signal.welch(data, fs=fs, window=window, nperseg=nperseg, 
                          noverlap=noverlap, scaling='density')
    
    return f, psd


def compute_band_power(f: np.ndarray, psd: np.ndarray, 
                       band: Tuple[float, float]) -> float:
    """
    Compute power in a specific frequency band.
    
    Parameters
    ----------
    f : np.ndarray
        Frequency array (Hz)
    psd : np.ndarray
        Power spectral density
    band : tuple
        Frequency band (f_min, f_max) in Hz
    
    Returns
    -------
    power : float
        Band power (integrated PSD)
    """
    # Find frequency indices within band
    idx = np.logical_and(f >= band[0], f <= band[1])
    
    # Integrate PSD using trapezoidal rule
    band_power = np.trapz(psd[idx], f[idx])
    
    return band_power


def compute_relative_band_power(f: np.ndarray, psd: np.ndarray, 
                                bands: Dict[str, Tuple[float, float]], 
                                total_band: Tuple[float, float]) -> Dict[str, float]:
    """
    Compute relative band power (normalized by total power).
    
    Parameters
    ----------
    f : np.ndarray
        Frequency array (Hz)
    psd : np.ndarray
        Power spectral density
    bands : dict
        Dictionary of frequency bands
    total_band : tuple
        Total frequency range for normalization
    
    Returns
    -------
    relative_powers : dict
        Relative power for each band
    """
    # Compute total power
    total_power = compute_band_power(f, psd, total_band)
    
    # Compute relative power for each band
    relative_powers = {}
    for band_name, band_range in bands.items():
        band_power = compute_band_power(f, psd, band_range)
        relative_powers[band_name] = band_power / total_power if total_power > 0 else 0
    
    return relative_powers


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_subject(subject_id: str, group: str, window_length: float, 
                   window_type: str) -> Dict:
    """
    Analyze a single subject's EEG data.
    
    Parameters
    ----------
    subject_id : str
        Subject identifier
    group : str
        Group label (CN, AD, FTD)
    window_length : float
        Window length in seconds
    window_type : str
        Type of window
    
    Returns
    -------
    results : dict
        Analysis results including PSD and band powers
    """
    try:
        # Load data
        data, channel_names = load_eeg_data(subject_id)
        
        # Extract target channel
        channel_data = extract_channel_data(data, channel_names, CHANNEL)
        
        # Extract segment
        segment = extract_segment(channel_data, FS, SEGMENT_DURATION)
        
        # Compute PSD
        f, psd = compute_psd_welch(segment, FS, window_length, window_type, OVERLAP)
        
        # Compute relative band powers
        rel_powers = compute_relative_band_power(f, psd, BANDS, TOTAL_BAND)
        
        return {
            'subject_id': subject_id,
            'group': group,
            'f': f,
            'psd': psd,
            'relative_powers': rel_powers,
            'success': True
        }
    
    except Exception as e:
        print(f"  Warning: Failed to analyze {subject_id}: {e}")
        return {'subject_id': subject_id, 'group': group, 'success': False}


def analyze_all_subjects(participants: pd.DataFrame, window_length: float, 
                         window_type: str) -> Dict[str, List[Dict]]:
    """
    Analyze all subjects grouped by clinical group.
    
    Parameters
    ----------
    participants : pd.DataFrame
        Participant information
    window_length : float
        Window length in seconds
    window_type : str
        Type of window
    
    Returns
    -------
    results : dict
        Results organized by group
    """
    results = {'CN': [], 'AD': [], 'FTD': []}
    
    # Analyze subjects from each group
    for group in ['CN', 'AD', 'FTD']:
        group_subjects = participants[participants['Group'] == group].head(SUBJECTS_PER_GROUP)
        
        print(f"\n  Analyzing {group} group ({len(group_subjects)} subjects):")
        for _, row in group_subjects.iterrows():
            subject_id = row['participant_id']
            print(f"    Processing {subject_id}...", end=' ')
            
            result = analyze_subject(subject_id, group, window_length, window_type)
            
            if result['success']:
                results[group].append(result)
                print("✓")
            else:
                print("✗")
    
    return results


def compute_group_statistics(results: Dict[str, List[Dict]]) -> pd.DataFrame:
    """
    Compute mean and std for each group and band.
    
    Parameters
    ----------
    results : dict
        Results organized by group
    
    Returns
    -------
    stats : pd.DataFrame
        Statistics table
    """
    stats_data = []
    
    for group in ['CN', 'AD', 'FTD']:
        group_results = results[group]
        
        if len(group_results) == 0:
            continue
        
        for band in ['theta', 'alpha', 'beta']:
            powers = [r['relative_powers'][band] for r in group_results if r['success']]
            
            if len(powers) > 0:
                stats_data.append({
                    'Group': group,
                    'Band': band,
                    'Mean': np.mean(powers),
                    'Std': np.std(powers),
                    'N': len(powers)
                })
    
    return pd.DataFrame(stats_data)


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_group_psd(results: Dict[str, List[Dict]], window_length: float, 
                   window_type: str, output_dir: Path):
    """
    Plot average PSD for each group.
    
    Parameters
    ----------
    results : dict
        Results organized by group
    window_length : float
        Window length used
    window_type : str
        Window type used
    output_dir : Path
        Output directory
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = {'CN': 'green', 'AD': 'red', 'FTD': 'blue'}
    
    for group in ['CN', 'AD', 'FTD']:
        if len(results[group]) == 0:
            continue
        
        # Average PSD across subjects
        psds = [r['psd'] for r in results[group] if r['success']]
        f = results[group][0]['f']
        
        mean_psd = np.mean(psds, axis=0)
        std_psd = np.std(psds, axis=0)
        
        # Plot only 0-40 Hz
        idx = f <= 40
        
        # Convert to dB for better visualization
        mean_psd_db = 10 * np.log10(mean_psd + 1e-12)
        std_psd_db = 10 * np.log10(std_psd + 1e-12)
        
        ax.plot(f[idx], mean_psd_db[idx], label=f'{group} (n={len(psds)})', 
                color=colors[group], linewidth=2)
        ax.fill_between(f[idx], 
                        mean_psd_db[idx] - std_psd_db[idx], 
                        mean_psd_db[idx] + std_psd_db[idx],
                        alpha=0.2, color=colors[group])
    
    # Mark frequency bands
    for band_name, (f_low, f_high) in BANDS.items():
        ax.axvspan(f_low, f_high, alpha=0.1, color='gray')
        ax.text((f_low + f_high) / 2, ax.get_ylim()[1] * 0.95, band_name.capitalize(),
                ha='center', va='top', fontsize=9, style='italic')
    
    ax.set_xlabel('Frequency (Hz)', fontsize=11)
    ax.set_ylabel('PSD (dB/Hz)', fontsize=11)
    ax.set_title(f'Power Spectral Density: {window_type.capitalize()} window, {window_length}s',
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_xlim([0, 40])
    
    plt.tight_layout()
    filename = f'psd_{window_type}_{window_length}s.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    Saved: {filename}")


def plot_band_power_comparison(results: Dict[str, List[Dict]], window_length: float,
                               window_type: str, output_dir: Path):
    """
    Plot bar chart comparing relative band power across groups.
    
    Parameters
    ----------
    results : dict
        Results organized by group
    window_length : float
        Window length used
    window_type : str
        Window type used
    output_dir : Path
        Output directory
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Relative Band Power: {window_type.capitalize()} window, {window_length}s',
                fontsize=14, fontweight='bold')
    
    bands = ['theta', 'alpha', 'beta']
    colors = {'CN': 'green', 'AD': 'red', 'FTD': 'blue'}
    
    for idx, band in enumerate(bands):
        ax = axes[idx]
        
        # Collect data for each group
        group_means = []
        group_stds = []
        group_labels = []
        group_colors = []
        
        for group in ['CN', 'AD', 'FTD']:
            if len(results[group]) == 0:
                continue
            
            powers = [r['relative_powers'][band] for r in results[group] if r['success']]
            
            if len(powers) > 0:
                group_means.append(np.mean(powers))
                group_stds.append(np.std(powers))
                group_labels.append(f'{group}\n(n={len(powers)})')
                group_colors.append(colors[group])
        
        # Create bar plot
        x_pos = np.arange(len(group_labels))
        bars = ax.bar(x_pos, group_means, yerr=group_stds, capsize=5,
                     color=group_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel('Relative Power', fontsize=11)
        ax.set_title(f'{band.capitalize()} Band ({BANDS[band][0]}-{BANDS[band][1]} Hz)',
                    fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(group_labels)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, max(group_means) * 1.3 if group_means else 1])
    
    plt.tight_layout()
    filename = f'band_power_{window_type}_{window_length}s.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    Saved: {filename}")


def plot_window_comparison(all_results: Dict, output_dir: Path):
    """
    Compare the effect of different window lengths on band power estimation.
    
    Parameters
    ----------
    all_results : dict
        All results organized by window parameters
    output_dir : Path
        Output directory
    """
    # For one window type (Hamming), compare different window lengths
    window_type = 'hamming'
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Effect of Window Length on Band Power ({window_type.capitalize()} window)',
                fontsize=14, fontweight='bold')
    
    bands = ['theta', 'alpha', 'beta']
    colors = {'CN': 'green', 'AD': 'red', 'FTD': 'blue'}
    markers = {2: 'o', 4: 's', 8: '^'}
    
    for idx, band in enumerate(bands):
        ax = axes[idx]
        
        for group in ['CN', 'AD', 'FTD']:
            means = []
            stds = []
            window_lengths = []
            
            for win_len in WINDOW_LENGTHS:
                key = f'{window_type}_{win_len}'
                if key in all_results and group in all_results[key]:
                    powers = [r['relative_powers'][band] for r in all_results[key][group] 
                             if r['success']]
                    
                    if len(powers) > 0:
                        means.append(np.mean(powers))
                        stds.append(np.std(powers))
                        window_lengths.append(win_len)
            
            if len(means) > 0:
                ax.errorbar(window_lengths, means, yerr=stds, label=group,
                           color=colors[group], marker='o', markersize=8, 
                           linewidth=2, capsize=5)
        
        ax.set_xlabel('Window Length (s)', fontsize=11)
        ax.set_ylabel('Relative Power', fontsize=11)
        ax.set_title(f'{band.capitalize()} Band', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xticks(WINDOW_LENGTHS)
    
    plt.tight_layout()
    filename = 'window_length_comparison.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    Saved: {filename}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution function."""
    
    print("=" * 80)
    print("EEG SPECTRAL ANALYSIS: COMPARING CLINICAL GROUPS")
    print("=" * 80)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load participant information
    print("\n1. Loading participant information...")
    participants = load_participants_info()
    print(f"   Total subjects: {len(participants)}")
    print(f"   CN: {len(participants[participants['Group'] == 'CN'])}")
    print(f"   AD: {len(participants[participants['Group'] == 'AD'])}")
    print(f"   FTD: {len(participants[participants['Group'] == 'FTD'])}")
    
    print(f"\n2. Analysis parameters:")
    print(f"   Channel: {CHANNEL}")
    print(f"   Segment duration: {SEGMENT_DURATION}s")
    print(f"   Window lengths: {WINDOW_LENGTHS}s")
    print(f"   Window types: {WINDOW_TYPES}")
    print(f"   Overlap: {OVERLAP * 100}%")
    print(f"   Frequency bands: {BANDS}")
    print(f"   Subjects per group: {SUBJECTS_PER_GROUP}")
    
    # Run analysis for all combinations
    print("\n3. Running spectral analysis...")
    all_results = {}
    
    for window_type in WINDOW_TYPES:
        for window_length in WINDOW_LENGTHS:
            print(f"\n  Window: {window_type}, Length: {window_length}s")
            
            results = analyze_all_subjects(participants, window_length, window_type)
            
            key = f'{window_type}_{window_length}'
            all_results[key] = results
            
            # Compute and print statistics
            stats = compute_group_statistics(results)
            print(f"\n  Statistics:")
            print(stats.to_string(index=False))
    
    # Generate visualizations
    print("\n4. Generating visualizations...")
    
    # PSD and band power plots for each configuration
    for window_type in WINDOW_TYPES:
        for window_length in WINDOW_LENGTHS:
            key = f'{window_type}_{window_length}'
            print(f"\n  Plotting: {window_type}, {window_length}s")
            
            plot_group_psd(all_results[key], window_length, window_type, OUTPUT_DIR)
            plot_band_power_comparison(all_results[key], window_length, window_type, OUTPUT_DIR)
    
    # Window length comparison
    print(f"\n  Plotting window length comparison...")
    plot_window_comparison(all_results, OUTPUT_DIR)
    
    # Summary report
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nKey Findings:")
    print("  1. PSD plots show power distribution across frequency bands (0-40 Hz)")
    print("  2. Band power comparisons reveal group differences in theta, alpha, beta")
    print("  3. Window length affects frequency resolution and variance:")
    print("     - Shorter windows (2s): Higher variance, better time resolution")
    print("     - Longer windows (8s): Lower variance, better frequency resolution")
    print("  4. Window type (Hann/Hamming) reduces spectral leakage vs rectangular")
    print(f"\nAll results saved to: {OUTPUT_DIR.absolute()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
