import pandas as pd
import numpy as np

df = pd.read_csv('results/part2/simple_stats/features.csv')

print("="*80)
print("SIMPLE STATISTICS - KEY FINDINGS")
print("="*80)

print("\n1. GROUP MEANS FOR THETA/ALPHA RATIO:")
print("-" * 60)
for group in ['CN', 'AD', 'FTD']:
    group_data = df[df['label'] == group]
    pz_ta = group_data['Pz_theta_alpha'].mean()
    f3_ta = group_data['F3_theta_alpha'].mean()
    print(f"{group:5s} (N={len(group_data):2d}): Pz = {pz_ta:6.3f},  F3 = {f3_ta:6.3f}")

print("\n2. GROUP MEANS FOR ALPHA RELATIVE POWER:")
print("-" * 60)
for group in ['CN', 'AD', 'FTD']:
    group_data = df[df['label'] == group]
    pz_alpha = group_data['Pz_alpha_rel'].mean()
    f3_alpha = group_data['F3_alpha_rel'].mean()
    print(f"{group:5s} (N={len(group_data):2d}): Pz = {pz_alpha:6.3f},  F3 = {f3_alpha:6.3f}")

print("\n3. GROUP MEANS FOR PEAK ALPHA FREQUENCY (PAF):")
print("-" * 60)
for group in ['CN', 'AD', 'FTD']:
    group_data = df[df['label'] == group]
    pz_paf = group_data['Pz_paf'].mean()
    f3_paf = group_data['F3_paf'].mean()
    print(f"{group:5s} (N={len(group_data):2d}): Pz = {pz_paf:5.2f} Hz,  F3 = {f3_paf:5.2f} Hz")

print("\n4. TOP CORRELATIONS (Pz vs F3 same features):")
print("-" * 60)
feature_pairs = [
    ('Pz_theta_alpha', 'F3_theta_alpha'),
    ('Pz_alpha_rel', 'F3_alpha_rel'),
    ('Pz_paf', 'F3_paf'),
    ('Pz_centroid_4_15', 'F3_centroid_4_15')
]

for feat1, feat2 in feature_pairs:
    corr_val = df[feat1].corr(df[feat2])
    print(f"{feat1:25s} <-> {feat2:25s}: r = {corr_val:.3f}")

print("\n5. WITHIN-CHANNEL CORRELATIONS:")
print("-" * 60)
print("Pz channel:")
print(f"  theta_rel <-> alpha_rel: r = {df['Pz_theta_rel'].corr(df['Pz_alpha_rel']):.3f}")
print(f"  theta_alpha <-> slowing_ratio: r = {df['Pz_theta_alpha'].corr(df['Pz_slowing_ratio']):.3f}")

print("\nF3 channel:")
print(f"  theta_rel <-> alpha_rel: r = {df['F3_theta_rel'].corr(df['F3_alpha_rel']):.3f}")
print(f"  theta_alpha <-> slowing_ratio: r = {df['F3_theta_alpha'].corr(df['F3_slowing_ratio']):.3f}")
