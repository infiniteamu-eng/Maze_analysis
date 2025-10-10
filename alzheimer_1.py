# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 12:31:08 2025

@author: BeiqueLab
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 22:59:18 2025

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
KO_mice = ['46', '47', '48', '49']
WT_mice = ['50', '51', '52', '53']
WT_mice = ['51', '52', '53']

def analyze_mouse_distance(mouse_id):
    """Analyze distance traveled between targets for a single mouse"""
    try:
        # Get target A coordinates from probe1
        d_probe1 = plib.TrialData()
        d_probe1.Load('2025-06-04', mouse_id, 'probe1')
        target_A_coords = d_probe1.target
        
        # Load probe2 data
        d_probe2 = plib.TrialData()
        d_probe2.Load('2025-06-04', mouse_id, 'probe2')
        
        # Find target indices
        try:
            target_A_index = pltlib.coords_to_target(d_probe2.r_nose, target_A_coords)
        except:
            target_A_index = None
            
        try:
            target_B_index = pltlib.coords_to_target(d_probe2.r_nose, d_probe2.target)
        except:
            target_B_index = None
        
        # Calculate distance between targets if both were visited
        if target_A_index is not None and target_B_index is not None:
            # Use the calc_dist_bw_points function from your code
            distance_AB = calc.calc_dist_bw_points(d_probe2.r_nose, target_A_coords, d_probe2.target)
            
            # Determine which target was visited first
            if target_A_index < target_B_index:
                strategy = "A_first"
                print(f"Mouse {mouse_id}: A first → B, distance = {distance_AB:.2f} cm")
            else:
                strategy = "B_first" 
                print(f"Mouse {mouse_id}: B first → A, distance = {distance_AB:.2f} cm")
                
            return distance_AB, strategy
        else:
            print(f"Mouse {mouse_id}: Did not visit both targets")
            return None, None
            
    except Exception as e:
        print(f"Error analyzing mouse {mouse_id}: {e}")
        return None, None

# Collect data for all mice
results = []
for mouse_id in KO_mice + WT_mice:
    distance, strategy = analyze_mouse_distance(mouse_id)
    if distance is not None:
        group = 'KO' if mouse_id in KO_mice else 'WT'
        results.append({
            'mouse_id': mouse_id,
            'group': group,
            'distance_AB': distance,
            'strategy': strategy
        })

# Convert to DataFrame
df = pd.DataFrame(results)
print(f"\nData collected for {len(df)} mice")
print(df)

# Separate groups
ko_distances = df[df['group'] == 'KO']['distance_AB'].values
wt_distances = df[df['group'] == 'WT']['distance_AB'].values

print(f"\nKO group distances (n={len(ko_distances)}): {ko_distances}")
print(f"WT group distances (n={len(wt_distances)}): {wt_distances}")

# Statistical analysis
if len(ko_distances) > 0 and len(wt_distances) > 0:
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(ko_distances, wt_distances)
    
    # Calculate means and SEM
    ko_mean = np.mean(ko_distances)
    ko_sem = stats.sem(ko_distances)
    wt_mean = np.mean(wt_distances)
    wt_sem = stats.sem(wt_distances)
    
    print(f"\nStatistical Results:")
    print(f"KO: {ko_mean:.2f} ± {ko_sem:.2f} cm")
    print(f"WT: {wt_mean:.2f} ± {wt_sem:.2f} cm")
    print(f"T-test: t={t_stat:.3f}, p={p_value:.3f}")
    
    # Calculate theoretical minimum distance (straight line)
    # Using approximate target coordinates based on your data
    theoretical_min = np.sqrt((37.6 - (-18.96))**2 + (13.3 - (-23.8))**2)
    print(f"Theoretical minimum distance (straight line): {theoretical_min:.2f} cm")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(2,4))
    
    # Individual data points with jitter
    ko_x = np.random.normal(1, 0.05, len(ko_distances))
    wt_x = np.random.normal(2, 0.05, len(wt_distances))
    
    # Plot individual points
    ax.scatter(ko_x, ko_distances, alpha=0.7, s=60, color='#306ed1', label=f'KO (n={len(ko_distances)})')
    ax.scatter(wt_x, wt_distances, alpha=0.7, s=60, color='black', label=f'WT (n={len(wt_distances)})')
    
    # Plot means with error bars
    ax.errorbar(1, ko_mean, yerr=ko_sem, fmt='o', markersize=8, color='#306ed1', 
                capsize=5, capthick=2, elinewidth=2, markeredgecolor='white', markeredgewidth=1)
    ax.errorbar(2, wt_mean, yerr=wt_sem, fmt='o', markersize=8, color='black', 
                capsize=5, capthick=2, elinewidth=2, markeredgecolor='white', markeredgewidth=1)
    
    # Add horizontal line for theoretical minimum
    # ax.axhline(y=theoretical_min, color='gray', linestyle='--', alpha=0.7, 
    #            label=f'Theoretical minimum: {theoretical_min:.1f} cm')
    
    # Formatting
    ax.set_xlim(0.5, 2.5)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['KO', 'WT'])
    ax.set_ylabel('Distance Between Targets (cm)')
    ax.set_title('Path Length Between Target \nLocations in Probe 2 Trial')
    
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
    y_max = max(max(ko_distances), max(wt_distances))
    y_pos = y_max * 1.1
    ax.plot([1, 2], [y_pos, y_pos], 'k-', linewidth=1)
    ax.text(1.5, y_pos * 1.002, sig_text, ha='center', va='bottom', fontsize=10.5)
    
    # Add individual mouse labels
    # for i, (x, y, mouse_id) in enumerate(zip(ko_x, ko_distances, df[df['group'] == 'KO']['mouse_id'])):
    #     ax.text(x, y + y_max*0.02, mouse_id, ha='center', va='bottom', fontsize=8, alpha=0.7)
    # for i, (x, y, mouse_id) in enumerate(zip(wt_x, wt_distances, df[df['group'] == 'WT']['mouse_id'])):
    #     ax.text(x, y + y_max*0.02, mouse_id, ha='center', va='bottom', fontsize=8, alpha=0.7)
    
    # ax.legend()
    # ax.grid(True, alpha=0.3)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()
    
    # Calculate path efficiency
    ko_efficiency = theoretical_min / ko_mean * 100
    wt_efficiency = theoretical_min / wt_mean * 100
    
    print(f"\nPath Efficiency:")
    print(f"KO group: {ko_efficiency:.1f}% of optimal")
    print(f"WT group: {wt_efficiency:.1f}% of optimal")
    
    # Additional analysis: strategy breakdown
    print(f"\nSearch Strategy Breakdown:")
    strategy_summary = df.groupby(['group', 'strategy']).size().unstack(fill_value=0)
    print(strategy_summary)
    
    # Print interpretation
    print(f"\nInterpretation:")
    if p_value < 0.05:
        direction = "longer" if ko_mean > wt_mean else "shorter"
        print(f"KO mice show significantly {direction} path lengths between targets (p={p_value:.3f})")
    else:
        print(f"No significant difference in path length between groups (p={p_value:.3f})")
        
    if wt_mean < ko_mean:
        print("WT mice show more direct paths between targets, suggesting better spatial navigation.")
    else:
        print("Unexpected result: KO mice show more direct paths than WT mice.")

else:
    print("Insufficient data for statistical comparison")