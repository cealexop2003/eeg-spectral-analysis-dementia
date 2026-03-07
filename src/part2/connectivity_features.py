"""
Connectivity Feature Extraction: Coherence Analysis
===================================================

Extracts coherence features between channel pairs to capture
"communication" between brain regions.

Hypothesis:
- AD: Long-range connectivity disruption (F3↔Pz, F4↔Pz)
- FTD: Frontal local connectivity disruption (F3↔F4)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt, coherence
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import mne
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path('data/raw/dataset')
OUTPUT_DIR = Path('results/part2/connectivity_features')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Parameters
FS = 500  # Sampling frequency
CHANNELS = ['Pz', 'F3', 'F4', 'O1']
DURATION = 15  # seconds

# Channel pairs for connectivity analysis
CHANNEL_PAIRS = [
    ('F3', 'F4'),  # Frontal interhemispheric (short-range)
    ('F3', 'Pz'),  # Frontal-parietal left (long-range)
    ('F4', 'Pz'),  # Frontal-parietal right (long-range)
    ('F3', 'O1'),  # Frontal-occipital left (very long)
    ('F4', 'O1'),  # Frontal-occipital right (very long)
    ('Pz', 'O1'),  # Parietal-occipital (medium-range)
]

# Frequency bands
BANDS = {
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (13, 30)
}

# Welch parameters
NPERSEG = 2000  # 4s window
NOVERLAP = 1000  # 50% overlap

print("="*80)
print("CONNECTIVITY FEATURE EXTRACTION")
print("="*80)
print(f"\nChannels: {CHANNELS}")
print(f"Pairs: {len(CHANNEL_PAIRS)}")
print(f"Bands: {list(BANDS.keys())}")
print(f"Total connectivity features: {len(CHANNEL_PAIRS)} pairs × {len(BANDS)} bands = {len(CHANNEL_PAIRS) * len(BANDS)}")

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Apply Butterworth bandpass filter
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def extract_coherence_features(eeg_data, channel_pairs, bands, fs, nperseg, noverlap):
    """
    Extract coherence features for all channel pairs and frequency bands
    
    Returns:
        dict: {feature_name: coherence_value}
    """
    features = {}
    
    for ch1, ch2 in channel_pairs:
        # Get signals
        signal1 = eeg_data[ch1]
        signal2 = eeg_data[ch2]
        
        # Compute coherence
        f, Cxy = coherence(signal1, signal2, fs=fs, nperseg=nperseg, noverlap=noverlap, window='hann')
        
        # Extract coherence for each band
        for band_name, (low_freq, high_freq) in bands.items():
            # Find frequency indices for this band
            band_idx = (f >= low_freq) & (f <= high_freq)
            
            # Mean coherence in this band
            mean_coherence = np.mean(Cxy[band_idx])
            
            # Feature name
            feature_name = f'coherence_{ch1}_{ch2}_{band_name}'
            features[feature_name] = mean_coherence
    
    return features

def process_subject(subject_id, label):
    """
    Process one subject: load data, filter, extract connectivity features
    """
    try:
        # Load EEG data using MNE
        eeg_file = DATA_DIR / subject_id / 'eeg' / f'{subject_id}_task-eyesclosed_eeg.set'
        
        if not eeg_file.exists():
            return None
        
        # Load raw EEG
        raw = mne.io.read_raw_eeglab(eeg_file, preload=True, verbose=False)
        
        # Extract segment (middle 15s)
        segment_length = DURATION * FS
        total_samples = raw.n_times
        
        if total_samples < segment_length:
            return None
        
        # Extract from middle
        start_sample = (total_samples - segment_length) // 2
        end_sample = start_sample + segment_length
        
        # Store channel data
        eeg_data = {}
        for channel in CHANNELS:
            if channel not in raw.ch_names:
                return None
            
            # Get channel data
            channel_idx = raw.ch_names.index(channel)
            raw_signal = raw.get_data()[channel_idx, start_sample:end_sample]
            
            # Apply bandpass filter (1-40 Hz)
            filtered_signal = butter_bandpass_filter(raw_signal, 1, 40, FS, order=4)
            
            eeg_data[channel] = filtered_signal
        
        # Extract connectivity features
        conn_features = extract_coherence_features(
            eeg_data, CHANNEL_PAIRS, BANDS, FS, NPERSEG, NOVERLAP
        )
        
        # Add metadata
        conn_features['subject_id'] = subject_id
        conn_features['label'] = label
        
        return conn_features
        
    except Exception as e:
        print(f"\nError processing {subject_id}: {e}")
        return None

# Main processing
print("\n" + "="*80)
print("PROCESSING SUBJECTS")
print("="*80)

# Load participants
participants_file = DATA_DIR / 'participants.tsv'
df_participants = pd.read_csv(participants_file, sep='\t')

# Map group letters to full names
group_mapping = {'A': 'AD', 'C': 'CN', 'F': 'FTD'}
df_participants['label'] = df_participants['Group'].map(group_mapping)

print(f"\nTotal participants: {len(df_participants)}")
print(f"  CN:  {(df_participants['label'] == 'CN').sum()}")
print(f"  AD:  {(df_participants['label'] == 'AD').sum()}")
print(f"  FTD: {(df_participants['label'] == 'FTD').sum()}")

all_features = []

for _, row in tqdm(df_participants.iterrows(), total=len(df_participants), desc="Processing"):
    subject_id = row['participant_id']
    label = row['label']
    
    features = process_subject(subject_id, label)
    if features is not None:
        all_features.append(features)

# Create DataFrame
if len(all_features) == 0:
    print("\n❌ ERROR: No features extracted! Check data paths.")
    import sys
    sys.exit(1)

df_features = pd.DataFrame(all_features)

# Reorder columns: subject_id, label, then features
feature_cols = [col for col in df_features.columns if col not in ['subject_id', 'label']]
df_features = df_features[['subject_id', 'label'] + sorted(feature_cols)]

print(f"\n" + "="*80)
print(f"CONNECTIVITY FEATURES EXTRACTED")
print(f"="*80)
print(f"Total subjects: {len(df_features)}")
print(f"  CN:  {(df_features['label'] == 'CN').sum()}")
print(f"  AD:  {(df_features['label'] == 'AD').sum()}")
print(f"  FTD: {(df_features['label'] == 'FTD').sum()}")
print(f"\nConnectivity features: {len(feature_cols)}")
print(f"Shape: {df_features.shape}")

# Save connectivity features
df_features.to_csv(OUTPUT_DIR / 'connectivity_features.csv', index=False)
print(f"\n✓ Saved: {OUTPUT_DIR / 'connectivity_features.csv'}")

# Descriptive statistics
print("\n" + "="*80)
print("CONNECTIVITY FEATURE STATISTICS")
print("="*80)

for label in ['CN', 'AD', 'FTD']:
    print(f"\n{label} Group (N={df_features[df_features['label']==label].shape[0]}):")
    subset = df_features[df_features['label'] == label][feature_cols]
    print(subset.describe().loc[['mean', 'std']].T.to_string())

# Visualize key connectivity features
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# Plot 1: Coherence heatmap for each group
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, label in enumerate(['CN', 'AD', 'FTD']):
    ax = axes[idx]
    
    # Get mean coherence for this group
    subset = df_features[df_features['label'] == label][feature_cols]
    mean_coherence = subset.mean()
    
    # Reshape into matrix (pairs × bands)
    coherence_matrix = np.zeros((len(CHANNEL_PAIRS), len(BANDS)))
    
    for i, (ch1, ch2) in enumerate(CHANNEL_PAIRS):
        for j, band_name in enumerate(['theta', 'alpha', 'beta']):
            feature_name = f'coherence_{ch1}_{ch2}_{band_name}'
            coherence_matrix[i, j] = mean_coherence[feature_name]
    
    # Plot heatmap
    sns.heatmap(coherence_matrix, 
                annot=True, 
                fmt='.3f',
                cmap='RdYlGn',
                vmin=0, vmax=1,
                xticklabels=['Theta', 'Alpha', 'Beta'],
                yticklabels=[f'{ch1}↔{ch2}' for ch1, ch2 in CHANNEL_PAIRS],
                ax=ax,
                cbar_kws={'label': 'Coherence'})
    
    ax.set_title(f'{label} Group (N={subset.shape[0]})', fontweight='bold', fontsize=12)
    ax.set_xlabel('Frequency Band', fontweight='bold')
    ax.set_ylabel('Channel Pair', fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'coherence_heatmaps_by_group.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR / 'coherence_heatmaps_by_group.png'}")

# Plot 2: Key connectivity pairs comparison
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

key_connections = [
    ('F3', 'F4', 'alpha', 'Frontal Interhemispheric (Alpha)'),
    ('F3', 'Pz', 'alpha', 'Frontal-Parietal Left (Alpha)'),
    ('F4', 'Pz', 'alpha', 'Frontal-Parietal Right (Alpha)'),
    ('F3', 'F4', 'theta', 'Frontal Interhemispheric (Theta)'),
    ('F3', 'Pz', 'theta', 'Frontal-Parietal Left (Theta)'),
    ('Pz', 'O1', 'alpha', 'Parietal-Occipital (Alpha)'),
]

for idx, (ch1, ch2, band, title) in enumerate(key_connections):
    ax = axes[idx]
    feature_name = f'coherence_{ch1}_{ch2}_{band}'
    
    # Box plot
    data_to_plot = [
        df_features[df_features['label'] == 'CN'][feature_name].values,
        df_features[df_features['label'] == 'AD'][feature_name].values,
        df_features[df_features['label'] == 'FTD'][feature_name].values
    ]
    
    bp = ax.boxplot(data_to_plot, labels=['CN', 'AD', 'FTD'], patch_artist=True)
    
    colors = ['#2ecc71', '#e74c3c', '#f39c12']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Coherence', fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add mean values as text
    for i, label in enumerate(['CN', 'AD', 'FTD']):
        mean_val = df_features[df_features['label'] == label][feature_name].mean()
        ax.text(i+1, 0.95, f'{mean_val:.3f}', ha='center', va='top', 
                fontweight='bold', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'key_connectivity_comparisons.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR / 'key_connectivity_comparisons.png'}")

# Plot 3: Connectivity differences (AD vs CN, FTD vs CN)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# AD vs CN
ax = axes[0]
cn_means = df_features[df_features['label'] == 'CN'][feature_cols].mean()
ad_means = df_features[df_features['label'] == 'AD'][feature_cols].mean()
differences_ad = ad_means - cn_means

ax.barh(range(len(differences_ad)), differences_ad.values, color=['red' if x < 0 else 'green' for x in differences_ad.values], alpha=0.7)
ax.set_yticks(range(len(differences_ad)))
ax.set_yticklabels(differences_ad.index, fontsize=8)
ax.set_xlabel('Coherence Difference (AD - CN)', fontweight='bold')
ax.set_title('AD vs CN: Connectivity Changes', fontweight='bold')
ax.axvline(0, color='black', linewidth=1)
ax.grid(True, alpha=0.3, axis='x')

# FTD vs CN
ax = axes[1]
ftd_means = df_features[df_features['label'] == 'FTD'][feature_cols].mean()
differences_ftd = ftd_means - cn_means

ax.barh(range(len(differences_ftd)), differences_ftd.values, color=['red' if x < 0 else 'green' for x in differences_ftd.values], alpha=0.7)
ax.set_yticks(range(len(differences_ftd)))
ax.set_yticklabels(differences_ftd.index, fontsize=8)
ax.set_xlabel('Coherence Difference (FTD - CN)', fontweight='bold')
ax.set_title('FTD vs CN: Connectivity Changes', fontweight='bold')
ax.axvline(0, color='black', linewidth=1)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'connectivity_differences.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR / 'connectivity_differences.png'}")

print("\n" + "="*80)
print("CONNECTIVITY FEATURE EXTRACTION COMPLETE")
print("="*80)
print(f"\nNext step: Combine with spectral features and run classification")
