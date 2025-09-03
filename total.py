#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 15:51:02 2025

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

def analyze_mouse_total_distance(mouse_id):
    """Analyze total distance traveled until both targets are visited"""
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
        
        # Calculate total distance to visit both targets
        if target_A_index is not None and target_B_index is not None:
            # Find which target is visited last (endpoint for distance calculation)
            final_index = max(target_A_index, target_B_index)
            
            # Calculate total distance from start to when both targets are visited
            total_distance = calc.calc_distance(d_probe2.r_nose[:final_index+1])
            
            # Determine strategy
            if target_A_index < target_B_index:
                strategy = "A_first"
                first_target_dist = calc.calc_distance(d_probe2.r_nose[:target_A_index+1])
                print(f"Mouse {mouse_id}: A first → B")
            else:
                strategy = "B_first"
                first_target_dist = calc.calc_distance(d_probe2.r_nose[:target_B_index+1])
                print(f"Mouse {mouse_id}: B first → A")
                
            print(f"  Distance to first target: {first_target_dist:.2f} cm")
            print(f"  Total distance to both: {total_distance:.2f} cm")
            
            return total_distance, strategy, first_target_dist
        else:
            print(f"Mouse {mouse_id}: Did not visit both targets")
            return None, None, None
            
    except Exception as e:
        print(f"Error analyzing mouse {mouse_id}: {e}")
        return None, None, None

# Collect data for all mice
results = []
for mouse_id in KO_mice + WT_mice:
    total_dist, strategy, first_dist = analyze_mouse_total_distance(mouse_id)
    if total_dist is not None:
        group = 'KO' if mouse_id in KO_mice else 'WT'
        results.append({
            'mouse_id': mouse_id,
            'group': group,
            'total_distance': total_dist,
            'first_target_distance': first_dist,
            'strategy': strategy
        })

# Convert to DataFrame
df = pd.DataFrame(results)
print(f"\nData collected for {len(df)} mice")
print(df)

# Separate groups
ko_distances = df[df['group'] == 'KO']['total_distance'].values
wt_distances = df[df['group'] == 'WT']['total_distance'].values

print(f"\nKO group total distances (n={len(ko_distances)}): {ko_distances}")
print(f"WT group total distances (n={len(wt_distances)}): {wt_distances}")

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
    
    # Create figure
    fig, ax = plt.subplots(figsize=(2, 4))
    
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
    
    # Formatting
    ax.set_xlim(0.5, 2.5)
    ax.set_ylim(0,6500)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['KO', 'WT'])
    ax.set_ylabel('Total Distance Traveled (cm)')
    ax.set_title('Total Path Length to Visit \nBoth Targets in Probe 2 Trial')
    
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
    ax.text(1.5, y_pos * 1.001, sig_text, ha='center', va='bottom', fontsize=12)
    
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
    
    # Additional analysis: first target distances
    ko_first = df[df['group'] == 'KO']['first_target_distance'].values
    wt_first = df[df['group'] == 'WT']['first_target_distance'].values
    
    print(f"\nFirst Target Distance Analysis:")
    print(f"KO first target: {np.mean(ko_first):.2f} ± {stats.sem(ko_first):.2f} cm")
    print(f"WT first target: {np.mean(wt_first):.2f} ± {stats.sem(wt_first):.2f} cm")
    
    # Strategy breakdown
    print(f"\nSearch Strategy Breakdown:")
    strategy_summary = df.groupby(['group', 'strategy']).size().unstack(fill_value=0)
    print(strategy_summary)
    
    # Print interpretation
    print(f"\nInterpretation:")
    if p_value < 0.05:
        direction = "longer" if ko_mean > wt_mean else "shorter"
        print(f"KO mice show significantly {direction} total search distances (p={p_value:.3f})")
    else:
        print(f"No significant difference in total search distance between groups (p={p_value:.3f})")
        
    if wt_mean < ko_mean:
        print("WT mice show more efficient overall search patterns, suggesting better spatial memory.")
    else:
        print("Unexpected result: KO mice show more efficient search than WT mice.")
        
    # Calculate search efficiency metrics
    print(f"\nSearch Efficiency Metrics:")
    print(f"Average KO search distance: {ko_mean:.1f} cm")
    print(f"Average WT search distance: {wt_mean:.1f} cm")
    if ko_mean > wt_mean:
        efficiency_diff = ((ko_mean - wt_mean) / wt_mean) * 100
        print(f"KO mice travel {efficiency_diff:.1f}% more distance than WT mice")

else:
    print("Insufficient data for statistical comparison")