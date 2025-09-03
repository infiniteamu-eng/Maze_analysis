#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 16:38:35 2025

@author: amarpreetdheer
"""

import modules.lib_process_data_to_mat as plib
import modules.lib_plot_mouse_trajectory as pltlib
import modules.calc_latency_distance_speed as calc
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd

# Define mouse groups
KO_mice = ['60', '61', '62', '63']
WT_mice = ['64', '65', '66', '67']

def analyze_mouse_probe1(mouse_id):
    """Analyze time and distance to reach target A in probe1 trial"""
    try:
        # Load probe1 data
        d_probe1 = plib.TrialData()
        d_probe1.Load('2025-08-22', mouse_id, 'probe1')
        
        # Target A is the main target in probe1
        target_A_coords = d_probe1.target
        
        # Find when mouse reaches target A
        try:
            target_A_index = pltlib.coords_to_target(d_probe1.r_nose, target_A_coords)
            target_A_time = d_probe1.time[target_A_index]
            
            # Calculate distance to target A
            target_A_distance = calc.calc_distance(d_probe1.r_nose[:target_A_index+1])
            
            print(f"Mouse {mouse_id} Probe1:")
            print(f"  Time to target A: {target_A_time:.2f}s")
            print(f"  Distance to target A: {target_A_distance:.2f} cm")
            
            return target_A_time, target_A_distance
            
        except:
            print(f"Mouse {mouse_id}: Did not reach target A in probe1")
            return None, None
            
    except Exception as e:
        print(f"Error analyzing mouse {mouse_id}: {e}")
        return None, None

# Collect data for all mice
results = []
for mouse_id in KO_mice + WT_mice:
    time_A, dist_A = analyze_mouse_probe1(mouse_id)
    if time_A is not None and dist_A is not None:
        group = 'KO' if mouse_id in KO_mice else 'WT'
        results.append({
            'mouse_id': mouse_id,
            'group': group,
            'probe1_time': time_A,
            'probe1_distance': dist_A
        })

# Convert to DataFrame
df = pd.DataFrame(results)
print(f"\nData collected for {len(df)} mice")
print(df)

# Separate groups
ko_times = df[df['group'] == 'KO']['probe1_time'].values
wt_times = df[df['group'] == 'WT']['probe1_time'].values
ko_distances = df[df['group'] == 'KO']['probe1_distance'].values
wt_distances = df[df['group'] == 'WT']['probe1_distance'].values

print(f"\nKO group - Time (n={len(ko_times)}): {ko_times}")
print(f"WT group - Time (n={len(wt_times)}): {wt_times}")
print(f"KO group - Distance (n={len(ko_distances)}): {ko_distances}")
print(f"WT group - Distance (n={len(wt_distances)}): {wt_distances}")

# Statistical analysis
if len(ko_times) > 0 and len(wt_times) > 0:
    # Time analysis
    t_stat_time, p_value_time = stats.ttest_ind(ko_times, wt_times)
    ko_time_mean = np.mean(ko_times)
    ko_time_sem = stats.sem(ko_times)
    wt_time_mean = np.mean(wt_times)
    wt_time_sem = stats.sem(wt_times)
    
    # Distance analysis
    t_stat_dist, p_value_dist = stats.ttest_ind(ko_distances, wt_distances)
    ko_dist_mean = np.mean(ko_distances)
    ko_dist_sem = stats.sem(ko_distances)
    wt_dist_mean = np.mean(wt_distances)
    wt_dist_sem = stats.sem(wt_distances)
    
    print(f"\nStatistical Results:")
    print(f"TIME - KO: {ko_time_mean:.2f} ± {ko_time_sem:.2f}s, WT: {wt_time_mean:.2f} ± {wt_time_sem:.2f}s")
    print(f"TIME - T-test: t={t_stat_time:.3f}, p={p_value_time:.3f}")
    print(f"DISTANCE - KO: {ko_dist_mean:.2f} ± {ko_dist_sem:.2f}cm, WT: {wt_dist_mean:.2f} ± {wt_dist_sem:.2f}cm")
    print(f"DISTANCE - T-test: t={t_stat_dist:.3f}, p={p_value_dist:.3f}")
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 4))
    
    # TIME PLOT
    # Individual data points with jitter
    ko_x_time = np.random.normal(1, 0.05, len(ko_times))
    wt_x_time = np.random.normal(2, 0.05, len(wt_times))
    
    ax1.scatter(ko_x_time, ko_times, alpha=0.7, s=60, color='#306ed1', label=f'KO (n={len(ko_times)})')
    ax1.scatter(wt_x_time, wt_times, alpha=0.7, s=60, color='black', label=f'WT (n={len(wt_times)})')
    
    # Plot means with error bars
    ax1.errorbar(1, ko_time_mean, yerr=ko_time_sem, fmt='o', markersize=8, color='#306ed1', 
                capsize=5, capthick=2, elinewidth=2, markeredgecolor='white', markeredgewidth=1)
    ax1.errorbar(2, wt_time_mean, yerr=wt_time_sem, fmt='o', markersize=8, color='black', 
                capsize=5, capthick=2, elinewidth=2, markeredgecolor='white', markeredgewidth=1)
    
    # Statistical annotation for time
    if p_value_time < 0.001:
        sig_text_time = "***"
    elif p_value_time < 0.01:
        sig_text_time = "**"
    elif p_value_time < 0.05:
        sig_text_time = "*"
    else:
        sig_text_time = f"p={p_value_time:.3f}"
    
    y_max_time = max(max(ko_times), max(wt_times))
    y_pos_time = y_max_time * 1.1
    ax1.plot([1, 2], [y_pos_time, y_pos_time], 'k-', linewidth=1)
    ax1.text(1.5, y_pos_time * 1.005, sig_text_time, ha='center', va='bottom', fontsize=12)
    
    ax1.set_xlim(0.5, 2.5)
    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(['KO', 'WT'])
    ax1.set_ylabel('Time to Target A (s)')
    ax1.set_title('Probe1: Latency to \nTarget A Discovery')
    # Remove top and right spines for ax1
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # DISTANCE PLOT
    ko_x_dist = np.random.normal(1, 0.05, len(ko_distances))
    wt_x_dist = np.random.normal(2, 0.05, len(wt_distances))
    
    ax2.scatter(ko_x_dist, ko_distances, alpha=0.7, s=60, color='#306ed1', label=f'KO (n={len(ko_distances)})')
    ax2.scatter(wt_x_dist, wt_distances, alpha=0.7, s=60, color='black', label=f'WT (n={len(wt_distances)})')
    
    ax2.errorbar(1, ko_dist_mean, yerr=ko_dist_sem, fmt='o', markersize=8, color='#306ed1', 
                capsize=5, capthick=2, elinewidth=2, markeredgecolor='white', markeredgewidth=1)
    ax2.errorbar(2, wt_dist_mean, yerr=wt_dist_sem, fmt='o', markersize=8, color='black', 
                capsize=5, capthick=2, elinewidth=2, markeredgecolor='white', markeredgewidth=1)
    
    # Statistical annotation for distance
    if p_value_dist < 0.001:
        sig_text_dist = "***"
    elif p_value_dist < 0.01:
        sig_text_dist = "**"
    elif p_value_dist < 0.05:
        sig_text_dist = "*"
    else:
        sig_text_dist = f"p={p_value_dist:.3f}"
    
    y_max_dist = max(max(ko_distances), max(wt_distances))
    y_pos_dist = y_max_dist * 1.1
    ax2.plot([1, 2], [y_pos_dist, y_pos_dist], 'k-', linewidth=1)
    ax2.text(1.5, y_pos_dist * 1.005, sig_text_dist, ha='center', va='bottom', fontsize=12)
    
    ax2.set_xlim(0.5, 2.5)
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(['KO', 'WT'])
    ax2.set_ylabel('Distance to Target A (cm)')
    ax2.set_title('Probe1: Path Length to \nTarget A Discovery')
    # Remove top and right spines for ax2
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # Print interpretation
    print(f"\nInterpretation (Probe1 - Target A only):")
    print("TIME:")
    if p_value_time < 0.05:
        time_direction = "longer" if ko_time_mean > wt_time_mean else "shorter"
        print(f"  KO mice take significantly {time_direction} time to reach target A (p={p_value_time:.3f})")
    else:
        print(f"  No significant difference in time to target A (p={p_value_time:.3f})")
    
    print("DISTANCE:")
    if p_value_dist < 0.05:
        dist_direction = "longer" if ko_dist_mean > wt_dist_mean else "shorter"
        print(f"  KO mice travel significantly {dist_direction} distances to target A (p={p_value_dist:.3f})")
    else:
        print(f"  No significant difference in distance to target A (p={p_value_dist:.3f})")
        
    # Additional context
    print(f"\nContext:")
    print(f"Probe1 tests memory for the originally learned target A location.")
    print(f"At this point, mice have no knowledge of target B.")
    print(f"Differences here reflect pure spatial memory retention for the trained location.")

else:
    print("Insufficient data for statistical comparison")