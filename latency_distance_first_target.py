#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 16:13:12 2025

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

def analyze_mouse_first_target(mouse_id):
    """Analyze time and distance to reach the first target (A or B)"""
    try:
        # Get target A coordinates from probe1
        d_probe1 = plib.TrialData()
        d_probe1.Load('2025-08-22', mouse_id, 'probe1')
        target_A_coords = d_probe1.target
        
        # Load probe2 data
        d_probe2 = plib.TrialData()
        d_probe2.Load('2025-08-22', mouse_id, 'probe2')
        
        # Find target indices
        try:
            target_A_index = pltlib.coords_to_target(d_probe2.r_nose, target_A_coords)
            target_A_time = d_probe2.time[target_A_index]
        except:
            target_A_index = None
            target_A_time = None
            
        try:
            target_B_index = pltlib.coords_to_target(d_probe2.r_nose, d_probe2.target)
            target_B_time = d_probe2.time[target_B_index]
        except:
            target_B_index = None
            target_B_time = None
        
        # Determine which target was reached first
        if target_A_index is not None and target_B_index is not None:
            if target_A_index < target_B_index:
                # A was reached first
                first_target_index = target_A_index
                first_target_time = target_A_time
                first_target_type = "A"
                first_target_coords = target_A_coords
            else:
                # B was reached first
                first_target_index = target_B_index
                first_target_time = target_B_time
                first_target_type = "B"
                first_target_coords = d_probe2.target
                
        elif target_A_index is not None:
            # Only A was reached
            first_target_index = target_A_index
            first_target_time = target_A_time
            first_target_type = "A_only"
            first_target_coords = target_A_coords
            
        elif target_B_index is not None:
            # Only B was reached
            first_target_index = target_B_index
            first_target_time = target_B_time
            first_target_type = "B_only"
            first_target_coords = d_probe2.target
            
        else:
            print(f"Mouse {mouse_id}: Did not reach any target")
            return None, None, None
        
        # Calculate distance to first target
        first_target_distance = calc.calc_distance(d_probe2.r_nose[:first_target_index+1])
        
        print(f"Mouse {mouse_id}: First target = {first_target_type}")
        print(f"  Time to first target: {first_target_time:.2f}s")
        print(f"  Distance to first target: {first_target_distance:.2f} cm")
        
        return first_target_time, first_target_distance, first_target_type
        
    except Exception as e:
        print(f"Error analyzing mouse {mouse_id}: {e}")
        return None, None, None

# Collect data for all mice
results = []
for mouse_id in KO_mice + WT_mice:
    time_first, dist_first, target_type = analyze_mouse_first_target(mouse_id)
    if time_first is not None and dist_first is not None:
        group = 'KO' if mouse_id in KO_mice else 'WT'
        results.append({
            'mouse_id': mouse_id,
            'group': group,
            'first_target_time': time_first,
            'first_target_distance': dist_first,
            'first_target_type': target_type
        })

# Convert to DataFrame
df = pd.DataFrame(results)
print(f"\nData collected for {len(df)} mice")
print(df)

# Separate groups
ko_times = df[df['group'] == 'KO']['first_target_time'].values
wt_times = df[df['group'] == 'WT']['first_target_time'].values
ko_distances = df[df['group'] == 'KO']['first_target_distance'].values
wt_distances = df[df['group'] == 'WT']['first_target_distance'].values

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
    
    # Add mouse labels for time
    # for i, (x, y, mouse_id) in enumerate(zip(ko_x_time, ko_times, df[df['group'] == 'KO']['mouse_id'])):
    #     ax1.text(x, y + y_max_time*0.02, mouse_id, ha='center', va='bottom', fontsize=8, alpha=0.7)
    # for i, (x, y, mouse_id) in enumerate(zip(wt_x_time, wt_times, df[df['group'] == 'WT']['mouse_id'])):
    #     ax1.text(x, y + y_max_time*0.02, mouse_id, ha='center', va='bottom', fontsize=8, alpha=0.7)
    
    ax1.set_xlim(0.5, 2.5)
    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(['KO', 'WT'])
    ax1.set_ylabel('Time to First Target (s)')
    ax1.set_title('Latency to First \nTarget Discovery')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    # ax1.legend()

    # ax1.grid(True, alpha=0.3)

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
    
    # Add mouse labels for distance
    # for i, (x, y, mouse_id) in enumerate(zip(ko_x_dist, ko_distances, df[df['group'] == 'KO']['mouse_id'])):
    #     ax2.text(x, y + y_max_dist*0.02, mouse_id, ha='center', va='bottom', fontsize=8, alpha=0.7)
    # for i, (x, y, mouse_id) in enumerate(zip(wt_x_dist, wt_distances, df[df['group'] == 'WT']['mouse_id'])):
    #     ax2.text(x, y + y_max_dist*0.02, mouse_id, ha='center', va='bottom', fontsize=8, alpha=0.7)
    
    ax2.set_xlim(0.5, 2.5)
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(['KO', 'WT'])
    ax2.set_ylabel('Distance to First Target (cm)')
    ax2.set_title('Path Length to First \nTarget Discovery')
    # ax2.legend()
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Strategy breakdown
    print(f"\nFirst Target Preference:")
    target_summary = df.groupby(['group', 'first_target_type']).size().unstack(fill_value=0)
    print(target_summary)
    
    # Print interpretation
    print(f"\nInterpretation:")
    print("TIME:")
    if p_value_time < 0.05:
        time_direction = "longer" if ko_time_mean > wt_time_mean else "shorter"
        print(f"  KO mice take significantly {time_direction} time to reach first target (p={p_value_time:.3f})")
    else:
        print(f"  No significant difference in time to first target (p={p_value_time:.3f})")
    
    print("DISTANCE:")
    if p_value_dist < 0.05:
        dist_direction = "longer" if ko_dist_mean > wt_dist_mean else "shorter"
        print(f"  KO mice travel significantly {dist_direction} distances to first target (p={p_value_dist:.3f})")
    else:
        print(f"  No significant difference in distance to first target (p={p_value_dist:.3f})")

else:
    print("Insufficient data for statistical comparison")