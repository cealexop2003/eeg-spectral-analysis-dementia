"""
EEG Feature Extraction for Alzheimer's Classification - Part 2
==============================================================

This script implements IIR Butterworth filtering and spectral feature extraction
from multi-channel EEG data (Pz, F3) for discriminating between:
- CN (Healthy Controls)
- AD (Alzheimer's Disease)
- FTD (Frontotemporal Dementia)

Key Components:
1. Butterworth IIR bandpass filtering (1-40 Hz)
2. Welch PSD estimation
3. Multi-channel spectral features (band powers, ratios, PAF, centroid)
4. Descriptive analysis and visualization

Dataset: "EEG of Alzheimer's and Frontotemporal dementia" (Kaggle)
Sampling frequency: 500 Hz
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import signal
import mne
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
DATA_DIR = Path('data/raw/dataset')
OUTPUT_DIR = Path('results/part2')

# EEG Parameters
FS = 500  # Sampling frequency (Hz)
CHANNELS = ['Pz', 'F3']  # Selected channels for analysis
SEGMENT_DURATION = 15  # seconds

# Filtering Parameters (Butterworth IIR Bandpass)
FILTER_LOW = 1.0   # Low cutoff (Hz)
FILTER_HIGH = 40.0  # High cutoff (Hz)
FILTER_ORDER = 4   # Filter order

# Welch PSD Parameters
WELCH_WINDOW_SEC = 4  # Window length in seconds
WELCH_OVERLAP = 0.5   # 50% overlap

# Frequency Bands
BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (13, 30)
}

# Additional frequency ranges
TOTAL_BAND = (1, 40)      # For normalization
ALPHA_BAND = (8, 12)       # For PAF
CENTROID_BAND = (4, 15)    # For spectral centroid


# ============================================================================
# DATA LOADING
# ============================================================================

def load_participants() -> pd.DataFrame:
    """
    Load participant information from TSV file.
    
    Returns
    -------
    df : pd.DataFrame
        Participant metadata with Group labels mapped to CN/AD/FTD
    """
    participants_file = DATA_DIR / 'participants.tsv'
    df = pd.read_csv(participants_file, sep='\t')
    
    # Map group letters to descriptive names
    group_mapping = {'A': 'AD', 'C': 'CN', 'F': 'FTD'}
    df['Group'] = df['Group'].map(group_mapping)
    
    return df


def load_subject_raw(subject_id: str):
    """
    Load raw EEG data for a specific subject using MNE.
    
    Parameters
    ----------
    subject_id : str
        Subject identifier (e.g., 'sub-001')
    
    Returns
    -------
    raw : mne.io.Raw
        Raw EEG data object
    """
    eeg_file = DATA_DIR / subject_id / 'eeg' / f'{subject_id}_task-eyesclosed_eeg.set'
    raw = mne.io.read_raw_eeglab(eeg_file, preload=True, verbose=False)
    return raw


def extract_channel(raw, channel_name: str) -> np.ndarray:
    """
    Extract data from a specific channel.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    channel_name : str
        Target channel name
    
    Returns
    -------
    data : np.ndarray
        Channel data (1D array)
    """
    if channel_name not in raw.ch_names:
        raise ValueError(f"Channel {channel_name} not found. Available: {raw.ch_names}")
    
    channel_idx = raw.ch_names.index(channel_name)
    data = raw.get_data()[channel_idx, :]
    return data


def extract_middle_segment(x: np.ndarray, fs: int, duration_s: float) -> np.ndarray:
    """
    Extract a segment from the middle of the signal.
    
    Parameters
    ----------
    x : np.ndarray
        Input signal
    fs : int
        Sampling frequency (Hz)
    duration_s : float
        Desired segment duration (seconds)
    
    Returns
    -------
    segment : np.ndarray
        Extracted segment
    """
    n_samples = int(duration_s * fs)
    total_samples = len(x)
    
    if total_samples < n_samples:
        raise ValueError(f"Signal too short: need {n_samples}, have {total_samples}")
    
    # Extract from middle
    start_idx = (total_samples - n_samples) // 2
    end_idx = start_idx + n_samples
    
    return x[start_idx:end_idx]


# ============================================================================
# FILTERING (IIR Butterworth Bandpass)
# ============================================================================

def butter_bandpass_sos(low: float, high: float, fs: int, order: int = 4):
    """
    Design an IIR Butterworth bandpass filter (SOS representation).
    
    Butterworth filter characteristics:
    - Maximally flat passband (no ripples)
    - Smooth frequency response
    
    IIR (Infinite Impulse Response):
    - Efficient: requires fewer coefficients than FIR
    - Can be unstable if not designed carefully
    
    SOS (Second-Order Sections):
    - Numerically stable representation
    - Prevents coefficient quantization errors
    
    Parameters
    ----------
    low : float
        Low cutoff frequency (Hz)
    high : float
        High cutoff frequency (Hz)
    fs : int
        Sampling frequency (Hz)
    order : int
        Filter order (default: 4)
    
    Returns
    -------
    sos : ndarray
        Second-order sections representation of the filter
    """
    nyquist = 0.5 * fs
    low_normalized = low / nyquist
    high_normalized = high / nyquist
    
    # Design Butterworth bandpass filter
    sos = signal.butter(order, [low_normalized, high_normalized], 
                       btype='band', output='sos')
    
    return sos


def apply_filter_sos(x: np.ndarray, sos) -> np.ndarray:
    """
    Apply SOS filter with zero-phase filtering.
    
    Zero-phase filtering (sosfiltfilt):
    - Applies filter forward and backward
    - Eliminates phase distortion
    - Doubles the effective filter order
    - Preserves temporal alignment of features
    
    Parameters
    ----------
    x : np.ndarray
        Input signal
    sos : ndarray
        Second-order sections filter coefficients
    
    Returns
    -------
    y : np.ndarray
        Filtered signal
    """
    y = signal.sosfiltfilt(sos, x)
    return y


# ============================================================================
# SPECTRAL ESTIMATION (Welch PSD)
# ============================================================================

def compute_welch_psd(x: np.ndarray, fs: int, 
                     win_s: float, overlap: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Power Spectral Density using Welch's method.
    
    Welch's method:
    - Divides signal into overlapping segments
    - Computes FFT for each segment
    - Averages periodograms to reduce variance
    - Trade-off: better variance reduction vs. frequency resolution
    
    Parameters
    ----------
    x : np.ndarray
        Input signal
    fs : int
        Sampling frequency (Hz)
    win_s : float
        Window length (seconds)
    overlap : float
        Overlap fraction (0-1)
    
    Returns
    -------
    f : np.ndarray
        Frequency array (Hz)
    psd : np.ndarray
        Power spectral density (V²/Hz)
    """
    nperseg = int(win_s * fs)
    noverlap = int(nperseg * overlap)
    
    # Use Hann window (good spectral leakage properties)
    window = signal.get_window('hann', nperseg)
    
    # Compute Welch PSD
    f, psd = signal.welch(x, fs=fs, window=window, nperseg=nperseg,
                         noverlap=noverlap, scaling='density')
    
    return f, psd


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def band_power(f: np.ndarray, psd: np.ndarray, f1: float, f2: float) -> float:
    """
    Compute absolute band power by integrating PSD.
    
    Parameters
    ----------
    f : np.ndarray
        Frequency array (Hz)
    psd : np.ndarray
        Power spectral density
    f1, f2 : float
        Frequency band limits (Hz)
    
    Returns
    -------
    power : float
        Integrated power in the band
    """
    idx = np.logical_and(f >= f1, f <= f2)
    power = np.trapz(psd[idx], f[idx])
    return power


def spectral_centroid(f: np.ndarray, psd: np.ndarray, f1: float, f2: float) -> float:
    """
    Compute spectral centroid (center of mass of spectrum).
    
    Spectral centroid indicates where the "center" of the spectrum is located.
    Lower values indicate more slow-wave activity.
    
    Parameters
    ----------
    f : np.ndarray
        Frequency array (Hz)
    psd : np.ndarray
        Power spectral density
    f1, f2 : float
        Frequency range for centroid calculation
    
    Returns
    -------
    centroid : float
        Spectral centroid (Hz)
    """
    idx = np.logical_and(f >= f1, f <= f2)
    f_band = f[idx]
    psd_band = psd[idx]
    
    if np.sum(psd_band) == 0:
        return np.nan
    
    centroid = np.sum(f_band * psd_band) / np.sum(psd_band)
    return centroid


def peak_alpha_frequency(f: np.ndarray, psd: np.ndarray) -> float:
    """
    Find Peak Alpha Frequency (PAF) - frequency of maximum power in alpha band.
    
    PAF is often reduced in Alzheimer's disease (slowing of alpha rhythm).
    
    Parameters
    ----------
    f : np.ndarray
        Frequency array (Hz)
    psd : np.ndarray
        Power spectral density
    
    Returns
    -------
    paf : float
        Peak alpha frequency (Hz)
    """
    alpha_low, alpha_high = ALPHA_BAND
    idx = np.logical_and(f >= alpha_low, f <= alpha_high)
    
    f_alpha = f[idx]
    psd_alpha = psd[idx]
    
    if len(psd_alpha) == 0:
        return np.nan
    
    peak_idx = np.argmax(psd_alpha)
    paf = f_alpha[peak_idx]
    
    return paf


def extract_features_for_channel(f: np.ndarray, psd: np.ndarray, 
                                 channel_name: str) -> Dict[str, float]:
    """
    Extract all spectral features for a single channel.
    
    Features include:
    - Relative band powers (delta, theta, alpha, beta)
    - Power ratios (theta/alpha, delta/alpha, slowing ratio)
    - Peak alpha frequency (PAF)
    - Spectral centroid (4-15 Hz)
    
    Parameters
    ----------
    f : np.ndarray
        Frequency array (Hz)
    psd : np.ndarray
        Power spectral density
    channel_name : str
        Channel name for feature labeling (e.g., 'Pz', 'F3')
    
    Returns
    -------
    features : dict
        Dictionary of features with channel prefix
    """
    features = {}
    
    # Compute absolute band powers
    band_powers_abs = {}
    for band_name, (f1, f2) in BANDS.items():
        band_powers_abs[band_name] = band_power(f, psd, f1, f2)
    
    # Total power for normalization
    total_power = band_power(f, psd, TOTAL_BAND[0], TOTAL_BAND[1])
    
    # Relative band powers
    for band_name in BANDS.keys():
        rel_power = band_powers_abs[band_name] / total_power if total_power > 0 else 0
        features[f'{channel_name}_{band_name}_rel'] = rel_power
    
    # Power ratios
    alpha_power = band_powers_abs['alpha']
    theta_power = band_powers_abs['theta']
    delta_power = band_powers_abs['delta']
    
    features[f'{channel_name}_theta_alpha'] = theta_power / alpha_power if alpha_power > 0 else np.nan
    features[f'{channel_name}_delta_alpha'] = delta_power / alpha_power if alpha_power > 0 else np.nan
    features[f'{channel_name}_slowing_ratio'] = (theta_power + delta_power) / alpha_power if alpha_power > 0 else np.nan
    
    # Peak alpha frequency
    features[f'{channel_name}_paf'] = peak_alpha_frequency(f, psd)
    
    # Spectral centroid (4-15 Hz)
    features[f'{channel_name}_centroid_4_15'] = spectral_centroid(f, psd, 
                                                                   CENTROID_BAND[0], 
                                                                   CENTROID_BAND[1])
    
    return features


# ============================================================================
# PIPELINE
# ============================================================================

def process_subject(subject_id: str, group: str) -> Dict:
    """
    Complete processing pipeline for one subject.
    
    Steps:
    1. Load raw EEG
    2. Extract channels (Pz, F3)
    3. Extract 15s segment from middle
    4. Apply Butterworth bandpass filter (1-40 Hz)
    5. Compute Welch PSD
    6. Extract features from both channels
    
    Parameters
    ----------
    subject_id : str
        Subject identifier
    group : str
        Group label (CN/AD/FTD)
    
    Returns
    -------
    result : dict
        Features + metadata
    """
    try:
        # Load raw data
        raw = load_subject_raw(subject_id)
        
        # Design filter (once for all channels)
        sos = butter_bandpass_sos(FILTER_LOW, FILTER_HIGH, FS, FILTER_ORDER)
        
        # Process each channel
        all_features = {
            'subject_id': subject_id,
            'label': group
        }
        
        for channel in CHANNELS:
            # Extract channel data
            channel_data = extract_channel(raw, channel)
            
            # Extract middle segment
            segment = extract_middle_segment(channel_data, FS, SEGMENT_DURATION)
            
            # Apply Butterworth bandpass filter (1-40 Hz)
            filtered = apply_filter_sos(segment, sos)
            
            # Compute Welch PSD
            f, psd = compute_welch_psd(filtered, FS, WELCH_WINDOW_SEC, WELCH_OVERLAP)
            
            # Extract features for this channel
            channel_features = extract_features_for_channel(f, psd, channel)
            all_features.update(channel_features)
        
        all_features['success'] = True
        return all_features
    
    except Exception as e:
        print(f"    Error processing {subject_id}: {e}")
        return {'subject_id': subject_id, 'label': group, 'success': False}


def build_feature_dataframe(participants: pd.DataFrame) -> pd.DataFrame:
    """
    Process all subjects and build feature DataFrame.
    
    Parameters
    ----------
    participants : pd.DataFrame
        Participant metadata
    
    Returns
    -------
    df : pd.DataFrame
        Feature matrix with all subjects
    """
    all_results = []
    
    print("\nProcessing all subjects...")
    print("=" * 80)
    
    for group in ['CN', 'AD', 'FTD']:
        group_subjects = participants[participants['Group'] == group]
        print(f"\n{group} group ({len(group_subjects)} subjects):")
        
        for idx, row in group_subjects.iterrows():
            subject_id = row['participant_id']
            print(f"  {subject_id}...", end=' ')
            
            result = process_subject(subject_id, group)
            
            if result['success']:
                all_results.append(result)
                print("✓")
            else:
                print("✗")
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Remove success column
    if 'success' in df.columns:
        df = df.drop(columns=['success'])
    
    return df


# ============================================================================
# DESCRIPTIVE ANALYSIS
# ============================================================================

def descriptive_analysis_and_plots(df: pd.DataFrame, output_dir: Path):
    """
    Generate descriptive statistics and visualization plots.
    
    Outputs:
    1. Summary statistics (mean ± std per group)
    2. Boxplots for key features
    3. Correlation heatmap
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame
    output_dir : Path
        Output directory for plots
    """
    print("\n" + "=" * 80)
    print("DESCRIPTIVE ANALYSIS")
    print("=" * 80)
    
    # ========================================================================
    # 1. Summary Statistics
    # ========================================================================
    print("\n1. Summary Statistics (Mean ± Std per Group):")
    print("-" * 80)
    
    # Select feature columns (exclude metadata)
    feature_cols = [col for col in df.columns if col not in ['subject_id', 'label']]
    
    # Group statistics
    grouped = df.groupby('label')[feature_cols]
    
    print(f"\nNumber of subjects per group:")
    print(df['label'].value_counts().sort_index())
    
    print(f"\nKey features (first few):")
    for group in ['CN', 'AD', 'FTD']:
        print(f"\n{group}:")
        group_data = grouped.get_group(group)
        
        # Show first 5 features
        for col in feature_cols[:5]:
            mean = group_data[col].mean()
            std = group_data[col].std()
            print(f"  {col}: {mean:.4f} ± {std:.4f}")
    
    # ========================================================================
    # 2. Boxplots for Key Features
    # ========================================================================
    print("\n2. Generating boxplots...")
    
    # Key features to visualize
    key_features_pz = ['Pz_alpha_rel', 'Pz_theta_rel', 'Pz_theta_alpha', 
                       'Pz_paf', 'Pz_centroid_4_15']
    key_features_f3 = ['F3_alpha_rel', 'F3_theta_rel', 'F3_theta_alpha',
                       'F3_paf', 'F3_centroid_4_15']
    
    # Plot Pz features
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, feature in enumerate(key_features_pz):
        if feature in df.columns:
            ax = axes[idx]
            df.boxplot(column=feature, by='label', ax=ax)
            ax.set_title(f'{feature}')
            ax.set_xlabel('Group')
            ax.set_ylabel('Value')
            plt.sca(ax)
            plt.xticks(rotation=0)
    
    # Remove empty subplot
    if len(key_features_pz) < len(axes):
        fig.delaxes(axes[-1])
    
    plt.suptitle('Pz Channel - Key Spectral Features by Group', 
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / 'boxplots_pz_features.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: boxplots_pz_features.png")
    
    # Plot F3 features
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, feature in enumerate(key_features_f3):
        if feature in df.columns:
            ax = axes[idx]
            df.boxplot(column=feature, by='label', ax=ax)
            ax.set_title(f'{feature}')
            ax.set_xlabel('Group')
            ax.set_ylabel('Value')
            plt.sca(ax)
            plt.xticks(rotation=0)
    
    # Remove empty subplot
    if len(key_features_f3) < len(axes):
        fig.delaxes(axes[-1])
    
    plt.suptitle('F3 Channel - Key Spectral Features by Group',
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / 'boxplots_f3_features.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: boxplots_f3_features.png")
    
    # ========================================================================
    # 3. Correlation Heatmap
    # ========================================================================
    print("\n3. Generating correlation heatmap...")
    
    # Compute correlation matrix for all features
    corr_matrix = df[feature_cols].corr()
    
    # Plot heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, 
                vmin=-1, vmax=1, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: correlation_heatmap.png")
    
    # ========================================================================
    # 4. Violin Plots for Selected Features
    # ========================================================================
    print("\n4. Generating violin plots...")
    
    selected_features = ['Pz_alpha_rel', 'Pz_theta_alpha', 'Pz_paf',
                        'F3_alpha_rel', 'F3_theta_alpha', 'F3_paf']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, feature in enumerate(selected_features):
        if feature in df.columns:
            ax = axes[idx]
            
            # Prepare data for violin plot
            data_to_plot = [df[df['label'] == group][feature].dropna() 
                           for group in ['CN', 'AD', 'FTD']]
            
            parts = ax.violinplot(data_to_plot, positions=[1, 2, 3], 
                                 showmeans=True, showmedians=True)
            
            ax.set_xticks([1, 2, 3])
            ax.set_xticklabels(['CN', 'AD', 'FTD'])
            ax.set_title(feature)
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('Violin Plots - Feature Distributions by Group',
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / 'violin_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: violin_plots.png")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution function."""
    
    print("=" * 80)
    print("EEG FEATURE EXTRACTION - PART 2")
    print("IIR Butterworth Filtering + Spectral Features")
    print("=" * 80)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load participant information
    print("\n1. Loading participant information...")
    participants = load_participants()
    print(f"   Total subjects: {len(participants)}")
    print(f"   CN: {len(participants[participants['Group'] == 'CN'])}")
    print(f"   AD: {len(participants[participants['Group'] == 'AD'])}")
    print(f"   FTD: {len(participants[participants['Group'] == 'FTD'])}")
    
    print(f"\n2. Processing parameters:")
    print(f"   Channels: {CHANNELS}")
    print(f"   Segment duration: {SEGMENT_DURATION}s")
    print(f"   Butterworth filter: {FILTER_LOW}-{FILTER_HIGH} Hz, Order {FILTER_ORDER}")
    print(f"   Welch PSD: {WELCH_WINDOW_SEC}s window, {WELCH_OVERLAP*100}% overlap")
    print(f"   Frequency bands: {BANDS}")
    
    # Build feature DataFrame
    print("\n3. Extracting features from all subjects...")
    df = build_feature_dataframe(participants)
    
    # Save features
    features_file = OUTPUT_DIR / 'features.csv'
    df.to_csv(features_file, index=False)
    print(f"\n✓ Features saved to: {features_file}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    
    # Descriptive analysis
    print("\n4. Generating descriptive analysis and plots...")
    descriptive_analysis_and_plots(df, OUTPUT_DIR)
    
    # Summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nOutputs:")
    print(f"  1. Feature matrix: {features_file}")
    print(f"  2. Boxplots (Pz): boxplots_pz_features.png")
    print(f"  3. Boxplots (F3): boxplots_f3_features.png")
    print(f"  4. Violin plots: violin_plots.png")
    print(f"  5. Correlation heatmap: correlation_heatmap.png")
    print(f"\nAll results saved to: {OUTPUT_DIR.absolute()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
