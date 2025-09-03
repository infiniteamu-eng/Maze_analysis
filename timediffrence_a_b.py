#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 22:49:24 2025

@author: amarpreetdheer
"""

import modules.lib_process_data_to_mat as plib
import modules.lib_plot_mouse_trajectory as pltlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd

# Define mouse groups
KO_mice = ['60', '61', '62', '63']
WT_mice = ['64', '65', '66', '67']


def analyze_mouse_probe2(mouse_id):
    """Analyze probe2 trial for a single mouse and return time difference"""
    try:
        # Get target A coordinates from probe1
        d_probe1 = plib.TrialData()
        d_probe1.Load('2025-08-22', mouse_id, 'probe1')
        target_A_coords = d_probe1.target
        
        # Load probe2 data
        d_probe2 = plib.TrialData()
        d_probe2.Load('2025-08-22', mouse_id, 'probe2')
        
        # Find target indices and times
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
        
        # Calculate time difference if both targets were visited
        if target_A_time is not None and target_B_time is not None:
            time_diff = abs(target_B_time - target_A_time)
            strategy = "A_first" if target_A_index < target_B_index else "B_first"
            print(f"Mouse {mouse_id}: {strategy}, time difference = {time_diff:.2f}s")
            return time_diff, strategy
        else:
            print(f"Mouse {mouse_id}: Did not visit both targets")
            return None, None
            
    except Exception as e:
        print(f"Error analyzing mouse {mouse_id}: {e}")
        return None, None

# Collect data for all mice
results = []
for mouse_id in KO_mice + WT_mice:
    time_diff, strategy = analyze_mouse_probe2(mouse_id)
    if time_diff is not None:
        group = 'KO' if mouse_id in KO_mice else 'WT'
        results.append({
            'mouse_id': mouse_id,
            'group': group,
            'time_diff': time_diff,
            'strategy': strategy
        })

# Convert to DataFrame for easier analysis
df = pd.DataFrame(results)
print(f"\nData collected for {len(df)} mice")
print(df)

# Separate groups
ko_times = df[df['group'] == 'KO']['time_diff'].values
wt_times = df[df['group'] == 'WT']['time_diff'].values

print(f"\nKO group (n={len(ko_times)}): {ko_times}")
print(f"WT group (n={len(wt_times)}): {wt_times}")

# Statistical analysis
if len(ko_times) > 0 and len(wt_times) > 0:
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(ko_times, wt_times)
    
    # Calculate means and SEM
    ko_mean = np.mean(ko_times)
    ko_sem = stats.sem(ko_times)
    wt_mean = np.mean(wt_times)
    wt_sem = stats.sem(wt_times)
    
    print(f"\nStatistical Results:")
    print(f"KO: {ko_mean:.2f} ± {ko_sem:.2f} seconds")
    print(f"WT: {wt_mean:.2f} ± {wt_sem:.2f} seconds")
    print(f"T-test: t={t_stat:.3f}, p={p_value:.3f}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(2, 4))
    
    # Individual data points with jitter
    ko_x = np.random.normal(1, 0.05, len(ko_times))
    wt_x = np.random.normal(2, 0.05, len(wt_times))
    
    # Plot individual points
    ax.scatter(ko_x, ko_times, alpha=0.7, s=60, color='#306ed1', label=f'KO (n={len(ko_times)})')
    ax.scatter(wt_x, wt_times, alpha=0.7, s=60, color='black', label=f'WT (n={len(wt_times)})')
    
    # Plot means with error bars
    ax.errorbar(1, ko_mean, yerr=ko_sem, fmt='o', markersize=8, color='#306ed1', 
                capsize=5, capthick=2, elinewidth=2, markeredgecolor='white', markeredgewidth=1)
    ax.errorbar(2, wt_mean, yerr=wt_sem, fmt='o', markersize=8, color='black', 
                capsize=5, capthick=2, elinewidth=2, markeredgecolor='white', markeredgewidth=1)
    
    # Formatting
    ax.set_xlim(0.5, 2.5)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['KO', 'WT'])
    ax.set_ylabel('Time Difference Between Targets (s)')
    ax.set_title('Time Between Target Visits\n in Probe 2 Trial')
    
    # Add statistical annotation
    if p_value < 0.001:
        sig_text = "***"
    elif p_value < 0.01:
        sig_text = "**"
    elif p_value < 0.05:
        sig_text = "*"
    else:
        sig_text = f"p={p_value:.3f}"
    
    # Add significance bar
    y_max = max(max(ko_times), max(wt_times))
    y_pos = y_max * 1.1
    ax.plot([1, 2], [y_pos, y_pos], 'k-', linewidth=1)
    ax.text(1.5, y_pos * 1.005, sig_text, ha='center', va='bottom', fontsize=12)
    
    # Add individual mouse labels
    # for i, (x, y, mouse_id) in enumerate(zip(ko_x, ko_times, df[df['group'] == 'KO']['mouse_id'])):
    #     ax.text(x, y + 0.5, mouse_id, ha='center', va='bottom', fontsize=8, alpha=0.7)
    # for i, (x, y, mouse_id) in enumerate(zip(wt_x, wt_times, df[df['group'] == 'WT']['mouse_id'])):
    #     ax.text(x, y + 0.5, mouse_id, ha='center', va='bottom', fontsize=8, alpha=0.7)
    
    # ax.legend()
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()
    
    # Print interpretation
    print(f"\nInterpretation:")
    if p_value < 0.05:
        direction = "longer" if ko_mean > wt_mean else "shorter"
        print(f"KO mice show significantly {direction} time differences than WT mice (p={p_value:.3f})")
    else:
        print(f"No significant difference between groups (p={p_value:.3f})")
        
    if wt_mean < ko_mean:
        print("WT mice show faster transitions between targets, supporting maintained cognitive mapping.")
    else:
        print("Unexpected result: KO mice show faster transitions than WT mice.")

else:
    print("Insufficient data for statistical comparison")