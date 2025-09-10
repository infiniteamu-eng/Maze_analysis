#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 21:29:09 2025

@author: amarpreetdheer
"""

import modules.lib_process_data_to_mat as plib
import modules.lib_plot_mouse_trajectory as pltlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd

# Define mouse groups for Target B analysis - Only mice 60-67
KO_mice_targetb = ['60', '61', '62', '63']  # n=4
WT_mice_targetb = ['64', '65', '66', '67']  # n=4

def calculate_path_efficiency(mouse_id, trials=range(1, 21), exclude_trials=None, date='2025-08-22'):
    """Calculate path efficiency for trials 14-20, with option to exclude specific trials"""
    
    if exclude_trials is None:
        exclude_trials = []
    
    efficiencies = []
    trial_numbers = []
    
    for trial in trials:
        # Skip excluded trials for this mouse
        if trial in exclude_trials:
            continue
            
        try:
            # Load trial data
            d = plib.TrialData()
            d.Load(date, mouse_id, str(trial))
            
            coords = d.r_nose
            valid_idx = ~(np.isnan(coords[:, 0]) | np.isnan(coords[:, 1]))
            coords_clean = coords[valid_idx]
            
            if len(coords_clean) < 2:
                print(f"Insufficient coordinates for mouse {mouse_id}, trial {trial}")
                continue
            
            # Find start and end positions
            start_pos = coords_clean[0]
            try:
                target_idx = pltlib.coords_to_target(coords, d.target)
                end_pos = coords[target_idx]
                coords_to_use = coords_clean[:target_idx+1]  # Only up to target
            except:
                end_pos = coords_clean[-1]
                coords_to_use = coords_clean
            
            # Calculate optimal distance (straight line)
            optimal_distance = np.linalg.norm(end_pos - start_pos)
            
            # Calculate actual path length
            actual_path_length = 0
            for i in range(1, len(coords_to_use)):
                segment_length = np.linalg.norm(coords_to_use[i] - coords_to_use[i-1])
                actual_path_length += segment_length
            
            # Calculate path efficiency
            if actual_path_length > 0:
                efficiency = optimal_distance / actual_path_length
                efficiencies.append(efficiency)
                trial_numbers.append(trial)
            
        except Exception as e:
            print(f"Error processing mouse {mouse_id}, trial {trial}: {e}")
            continue
    
    return trial_numbers, efficiencies

def plot_path_efficiency_comparison():
    """Plot path efficiency comparison for Target B analysis - mice 60-67 only (trials 14-20)"""
    
    # Collect data for mice 60-67 only (Target B has data only for these mice)
    ko_data = []
    wt_data = []
    
    # Define which mice to exclude from specific trials if needed
    wt_exclusions = {
        '65': [16],      # Exclude mouse 65 from trial 16
        '67': [4, 15]    # Exclude mouse 67 from trials 4 and 15
    }
    
    print("Attempting to load Target B path efficiency data for mice 60-67...")
    
    # Process KO mice 60-63
    for mouse_id in KO_mice_targetb:
        print(f"  Trying KO mouse {mouse_id}")
        trials, efficiencies = calculate_path_efficiency(mouse_id, date='2025-08-22')
        if len(efficiencies) > 0:
            ko_data.append({'mouse_id': mouse_id, 'trials': trials, 'efficiencies': efficiencies})
            print(f"    Success: {len(efficiencies)} trials loaded")
        else:
            print(f"    No data found for KO mouse {mouse_id}")
    
    # Process WT mice 64-67
    for mouse_id in WT_mice_targetb:
        print(f"  Trying WT mouse {mouse_id}")
        exclude_trials = wt_exclusions.get(mouse_id, [])
        trials, efficiencies = calculate_path_efficiency(mouse_id, exclude_trials=exclude_trials, date='2025-08-22')
        if len(efficiencies) > 0:
            wt_data.append({'mouse_id': mouse_id, 'trials': trials, 'efficiencies': efficiencies})
            print(f"    Success: {len(efficiencies)} trials loaded")
        else:
            print(f"    No data found for WT mouse {mouse_id}")
    
    print(f"\nFinal summary: Loaded data for {len(ko_data)} KO mice and {len(wt_data)} WT mice")
    
    if len(ko_data) == 0 and len(wt_data) == 0:
        print("ERROR: No data loaded for any mice! Check your file paths and dates.")
        print("Make sure the data files exist and the dates/mouse IDs are correct.")
        return
    
    print(f"Target B analysis - KO n={len(ko_data)}, WT n={len(wt_data)} (mice 60-67 only)")
    
    # First, identify trials where both groups have data
    all_trials_raw = range(14, 21)  # Trials 14-20
    x_labels_raw = [-2, -1, 0, 1, 2, 3, 4]  # Corresponding x-axis labels
    
    valid_trials = []
    valid_x_labels = []
    ko_means = []
    wt_means = []
    ko_sems = []
    wt_sems = []
    
    for trial, x_label in zip(all_trials_raw, x_labels_raw):
        ko_trial_effs = [eff for data in ko_data for t, eff in zip(data['trials'], data['efficiencies']) if t == trial]
        wt_trial_effs = [eff for data in wt_data for t, eff in zip(data['trials'], data['efficiencies']) if t == trial]
        
        # Only include trials where both groups have data
        if len(ko_trial_effs) > 0 and len(wt_trial_effs) > 0:
            valid_trials.append(trial)
            valid_x_labels.append(x_label)
            ko_means.append(np.mean(ko_trial_effs))
            ko_sems.append(stats.sem(ko_trial_effs))
            wt_means.append(np.mean(wt_trial_effs))
            wt_sems.append(stats.sem(wt_trial_effs))
    
    print(f"Valid trials with data for both groups: {valid_trials} (x-labels: {valid_x_labels})")
    
    # Create plot with specified figure size
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    
    # Update sample sizes in labels
    ko_n = len(ko_data)
    wt_n = len(wt_data)
    
    # Plot group means with error bars - WT first for legend order
    wt_line = ax.errorbar(valid_x_labels, wt_means, yerr=wt_sems, color='black', 
                linewidth=3, capsize=3, label=f'WT n={wt_n}', marker='o', markersize=6)
    ko_line = ax.errorbar(valid_x_labels, ko_means, yerr=ko_sems, color='#306ed1', 
                linewidth=3, capsize=3, label=f'KO n={ko_n}', marker='o', markersize=6)
    
    # Add significance testing
    max_y = 0
    if len(ko_means) > 0 and len(wt_means) > 0:
        max_y = max(max(ko_means), max(wt_means))
    
    for i, (trial, x_pos) in enumerate(zip(valid_trials, valid_x_labels)):
        ko_trial_effs = [eff for data in ko_data for t, eff in zip(data['trials'], data['efficiencies']) if t == trial]
        wt_trial_effs = [eff for data in wt_data for t, eff in zip(data['trials'], data['efficiencies']) if t == trial]
        
        # Perform t-test
        t_stat, p_val = stats.ttest_ind(ko_trial_effs, wt_trial_effs)
        
        # Add significance markers
        if p_val < 0.001:
            sig_marker = '***'
        elif p_val < 0.01:
            sig_marker = '**'
        elif p_val < 0.05:
            sig_marker = '*'
        else:
            continue  # No significance marker
        
        # Position significance marker above the higher point
        y_pos = max(ko_means[i] + ko_sems[i], wt_means[i] + wt_sems[i]) + max_y * 0.05
        ax.text(x_pos, y_pos, sig_marker, ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Trials', fontsize=14, fontweight='bold')
    ax.set_ylabel('Path Efficiency', fontsize=14, fontweight='bold')
    ax.set_title('Path Efficiency to Target B', fontsize=16, fontweight='bold')
    
    # Create legend with WT first, KO second
    # ax.legend(handles=[wt_line, ko_line], fontsize=12, loc='upper right', bbox_to_anchor=(0.97, 1.05))
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_ylim(0, 0.90)  # Set y-limit to 0.85
    ax.set_xticks(valid_x_labels)
    
    # Add vertical dotted line at x=0 (start of Target B) if x=0 is in valid trials
    if 0 in valid_x_labels:
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Add horizontal lines and labels for experimental phases
    # Phase A: Target A expertise (from x=-2 to x=-0.2) - separate line
    ax.plot([-2, -0.2], [0.78, 0.78], color='darkgray', linewidth=2)
    ax.text(-1.1, 0.79, 'A', color='#525863',fontsize=13, fontweight='bold', ha='center', va='bottom')
    
    # Phase B: Target B learning (from x=0.2 to x=4) - separate line with gap
    ax.plot([0.2, 4], [0.78, 0.78], color='darkgray', linewidth=2)
    ax.text(2.1, 0.79, 'B', color='#525863', fontsize=13, fontweight='bold', ha='center', va='bottom')
    
    # Set custom y-ticks to show increments of 0.1
    ax.set_yticks(np.arange(0, 0.85, 0.1))
    
    # Increase tick label font size
    ax.tick_params(axis='both', which='major', labelsize=13)
    
    # Add 'C' annotation in top-left corner
    plt.text(-0.25, 1.18, 'D', transform=plt.gca().transAxes, fontsize=30, fontweight='bold', 
         verticalalignment='top', horizontalalignment='left')
    
    plt.tight_layout()
    plt.savefig('path_efficiency_targetb_mice60-67.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics for trials 14-20 (Target B analysis)
    print("\nTarget B Path Efficiency Summary (Valid Trials Only) - Mice 60-67:")
    print("=" * 70)
    if len(ko_means) > 0:
        ko_overall_mean = np.nanmean(ko_means)
        print(f"KO mice - Overall mean: {ko_overall_mean:.3f} (n={ko_n})")
    
    if len(wt_means) > 0:
        wt_overall_mean = np.nanmean(wt_means)
        print(f"WT mice - Overall mean: {wt_overall_mean:.3f} (n={wt_n})")
    
    # Print specific mice included
    print(f"\nTarget B cohort (mice 60-67 only):")
    print(f"KO mice: {[d['mouse_id'] for d in ko_data]}")
    print(f"WT mice: {[d['mouse_id'] for d in wt_data]}")
    
    # Print trial-by-trial data with sample sizes and significance (only valid trials)
    print("\nTrial-by-trial means with significance testing (only trials with data for both groups):")
    for i, (trial, x_pos) in enumerate(zip(valid_trials, valid_x_labels)):
        ko_trial_effs = [eff for data in ko_data for t, eff in zip(data['trials'], data['efficiencies']) if t == trial]
        wt_trial_effs = [eff for data in wt_data for t, eff in zip(data['trials'], data['efficiencies']) if t == trial]
        
        # Perform t-test for significance
        t_stat, p_val = stats.ttest_ind(ko_trial_effs, wt_trial_effs)
        sig_level = ""
        if p_val < 0.001:
            sig_level = " (***)"
        elif p_val < 0.01:
            sig_level = " (**)"
        elif p_val < 0.05:
            sig_level = " (*)"
        
        print(f"Trial {trial} (x={x_pos}): KO = {ko_means[i]:.3f} ± {ko_sems[i]:.3f} (n={len(ko_trial_effs)}), WT = {wt_means[i]:.3f} ± {wt_sems[i]:.3f} (n={len(wt_trial_effs)}), p = {p_val:.4f}{sig_level}")
    
    print("\nSignificance levels: * p<0.05, ** p<0.01, *** p<0.001")

# Run the analysis
if __name__ == "__main__":
    print("Analyzing Target B path efficiency for trials 14-20 (mice 60-67 only)...")
    plot_path_efficiency_comparison()