#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 21:29:09 2025

@author: amarpreetdheer
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 16:47:57 2025

@author: amarpreetdheer
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  7 22:00:25 2025

@author: amarpreetdheer
"""

import modules.lib_process_data_to_mat as plib
import modules.lib_plot_mouse_trajectory as pltlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd

# Define mouse groups - Original mice
KO_mice_original = ['60', '61', '62', '63']
WT_mice_original = ['64', '65', '66', '67']

# Define ATRX mouse groups
KO_mice_atrx = ['85','88','91','92','93','94','98']
WT_mice_atrx = ['86','87','89','90','95','96','97']

# Combined groups
KO_mice = KO_mice_original + KO_mice_atrx
WT_mice = WT_mice_original + WT_mice_atrx

def calculate_path_efficiency(mouse_id, trials=range(1, 21), exclude_trials=None, date='2025-08-22'):
    """Calculate path efficiency for trials 1-5, with option to exclude specific trials"""
    
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
    """Plot path efficiency comparison between groups for trials 1-5 with complete dataset"""
    
    # Collect data for all mice with specific exclusions
    ko_data = []
    wt_data = []
    
    # Define which mice to exclude from specific trials if needed
    wt_exclusions = {
        '67': [4]  # Exclude mouse 67 from trial 4
    }
    
    print("Attempting to load path efficiency data...")
    
    # Process original mice (2025-08-22 data)
    print("Loading original mice data...")
    for mouse_id in KO_mice_original:
        print(f"  Trying KO mouse {mouse_id}")
        trials, efficiencies = calculate_path_efficiency(mouse_id, date='2025-08-22')
        if len(efficiencies) > 0:
            ko_data.append({'mouse_id': mouse_id, 'trials': trials, 'efficiencies': efficiencies, 'cohort': 'original'})
            print(f"    Success: {len(efficiencies)} trials loaded")
        else:
            print(f"    No data found for KO mouse {mouse_id}")
    
    for mouse_id in WT_mice_original:
        print(f"  Trying WT mouse {mouse_id}")
        exclude_trials = wt_exclusions.get(mouse_id, [])
        trials, efficiencies = calculate_path_efficiency(mouse_id, exclude_trials=exclude_trials, date='2025-08-22')
        if len(efficiencies) > 0:
            wt_data.append({'mouse_id': mouse_id, 'trials': trials, 'efficiencies': efficiencies, 'cohort': 'original'})
            print(f"    Success: {len(efficiencies)} trials loaded")
        else:
            print(f"    No data found for WT mouse {mouse_id}")
    
    # Process ATRX mice (need to handle multiple dates)
    atrx_dates = ['2023-07-07', '2023-08-15']
    print("Loading ATRX mice data...")
    
    for mouse_id in KO_mice_atrx:
        print(f"  Trying ATRX KO mouse {mouse_id}")
        all_trials = []
        all_efficiencies = []
        for date in atrx_dates:
            try:
                trials, efficiencies = calculate_path_efficiency(mouse_id, date=date)
                all_trials.extend(trials)
                all_efficiencies.extend(efficiencies)
                if len(efficiencies) > 0:
                    print(f"    Found {len(efficiencies)} trials for date {date}")
            except Exception as e:
                print(f"    No data for date {date}: {e}")
                continue
        if len(all_efficiencies) > 0:
            ko_data.append({'mouse_id': mouse_id, 'trials': all_trials, 'efficiencies': all_efficiencies, 'cohort': 'atrx'})
            print(f"    Total success: {len(all_efficiencies)} trials loaded")
        else:
            print(f"    No data found for ATRX KO mouse {mouse_id}")
    
    for mouse_id in WT_mice_atrx:
        print(f"  Trying ATRX WT mouse {mouse_id}")
        all_trials = []
        all_efficiencies = []
        for date in atrx_dates:
            try:
                trials, efficiencies = calculate_path_efficiency(mouse_id, date=date)
                all_trials.extend(trials)
                all_efficiencies.extend(efficiencies)
                if len(efficiencies) > 0:
                    print(f"    Found {len(efficiencies)} trials for date {date}")
            except Exception as e:
                print(f"    No data for date {date}: {e}")
                continue
        if len(all_efficiencies) > 0:
            wt_data.append({'mouse_id': mouse_id, 'trials': all_trials, 'efficiencies': all_efficiencies, 'cohort': 'atrx'})
            print(f"    Total success: {len(all_efficiencies)} trials loaded")
        else:
            print(f"    No data found for ATRX WT mouse {mouse_id}")
    
    print(f"\nFinal summary: Loaded data for {len(ko_data)} KO mice and {len(wt_data)} WT mice")
    
    if len(ko_data) == 0 and len(wt_data) == 0:
        print("ERROR: No data loaded for any mice! Check your file paths and dates.")
        print("Make sure the data files exist and the dates/mouse IDs are correct.")
        return
    
    print(f"KO breakdown: {len([d for d in ko_data if d['cohort'] == 'original'])} original, {len([d for d in ko_data if d['cohort'] == 'atrx'])} ATRX")
    print(f"WT breakdown: {len([d for d in wt_data if d['cohort'] == 'original'])} original, {len([d for d in wt_data if d['cohort'] == 'atrx'])} ATRX")
    
    # Create plot with specified figure size
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    
    # Calculate and plot group means for trials 1-5
    all_trials = range(1, 6)  # Only trials 1-5
    ko_means = []
    wt_means = []
    ko_sems = []
    wt_sems = []
    
    for trial in all_trials:
        ko_trial_effs = [eff for data in ko_data for t, eff in zip(data['trials'], data['efficiencies']) if t == trial]
        wt_trial_effs = [eff for data in wt_data for t, eff in zip(data['trials'], data['efficiencies']) if t == trial]
        
        if len(ko_trial_effs) > 0:
            ko_means.append(np.mean(ko_trial_effs))
            ko_sems.append(stats.sem(ko_trial_effs))
        else:
            ko_means.append(np.nan)
            ko_sems.append(np.nan)
            
        if len(wt_trial_effs) > 0:
            wt_means.append(np.mean(wt_trial_effs))
            wt_sems.append(stats.sem(wt_trial_effs))
        else:
            wt_means.append(np.nan)
            wt_sems.append(np.nan)
    
    # Update sample sizes in labels
    ko_n = len(ko_data)
    wt_n = len(wt_data)
    
    # Plot group means with error bars - WT first for legend order
    wt_line = ax.errorbar(all_trials, wt_means, yerr=wt_sems, color='black', 
                linewidth=3, capsize=3, label=f'WT n={wt_n}', marker='o', markersize=6)
    ko_line = ax.errorbar(all_trials, ko_means, yerr=ko_sems, color='#306ed1', 
                linewidth=3, capsize=3, label=f'KO n={ko_n}', marker='o', markersize=6)
    
    # Add significance testing
    max_y = 0
    if not all(np.isnan(ko_means + wt_means)):
        max_y = max(max([m for m in ko_means if not np.isnan(m)]), 
                    max([m for m in wt_means if not np.isnan(m)]))
    
    for i, trial in enumerate(all_trials):
        ko_trial_effs = [eff for data in ko_data for t, eff in zip(data['trials'], data['efficiencies']) if t == trial]
        wt_trial_effs = [eff for data in wt_data for t, eff in zip(data['trials'], data['efficiencies']) if t == trial]
        
        if len(ko_trial_effs) > 0 and len(wt_trial_effs) > 0:
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
            ax.text(trial, y_pos, sig_marker, ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Trials', fontsize=14, fontweight='bold', labelpad=2)
    ax.set_ylabel('Path Efficiency', fontsize=14, fontweight='bold')
    ax.set_title('Path Efficiency to Target A', fontsize=16, fontweight='bold')
    
    # Create legend with WT first, KO second
    # ax.legend(handles=[wt_line, ko_line], fontsize=12, loc='upper right', bbox_to_anchor=(0.55, 1.05))
    # Create legend with WT first, KO second - positioned at bottom
  
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if max_y > 0:
        ax.set_ylim(0, 0.58)  # Set y-limit to 0.5
    ax.set_xticks(all_trials)
    
    # Set custom y-ticks to show increments of 0.1
    ax.set_yticks(np.arange(0, 0.6, 0.1))
    
    # Increase tick label font size
    ax.tick_params(axis='both', which='major', labelsize=13)
    
    # Add 'D' annotation in top-left corner
    plt.text(-0.25, 1.18, 'C', transform=plt.gca().transAxes, fontsize=30, fontweight='bold', 
         verticalalignment='top', horizontalalignment='left')
    
    plt.tight_layout()
    
    plt.savefig('path_efficiency_targeta_atrx.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics for trials 1-5
    print("\nPath Efficiency Summary (Trials 1-5) - Combined Dataset:")
    print("=" * 70)
    if len(ko_means) > 0:
        ko_overall_mean = np.nanmean(ko_means)
        print(f"KO mice - Overall mean: {ko_overall_mean:.3f} (n={ko_n})")
    
    if len(wt_means) > 0:
        wt_overall_mean = np.nanmean(wt_means)
        print(f"WT mice - Overall mean: {wt_overall_mean:.3f} (n={wt_n})")
    
    # Print cohort breakdown
    print(f"\nCohort breakdown:")
    print(f"Original cohort: KO n={len([d for d in ko_data if d['cohort'] == 'original'])}, WT n={len([d for d in wt_data if d['cohort'] == 'original'])}")
    print(f"ATRX cohort: KO n={len([d for d in ko_data if d['cohort'] == 'atrx'])}, WT n={len([d for d in wt_data if d['cohort'] == 'atrx'])}")
    
    # Print trial-by-trial data with sample sizes and significance
    print("\nTrial-by-trial means with significance testing:")
    for i, trial in enumerate(all_trials):
        ko_trial_effs = [eff for data in ko_data for t, eff in zip(data['trials'], data['efficiencies']) if t == trial]
        wt_trial_effs = [eff for data in wt_data for t, eff in zip(data['trials'], data['efficiencies']) if t == trial]
        
        if not np.isnan(ko_means[i]) and not np.isnan(wt_means[i]):
            # Perform t-test for significance
            t_stat, p_val = stats.ttest_ind(ko_trial_effs, wt_trial_effs)
            sig_level = ""
            if p_val < 0.001:
                sig_level = " (***)"
            elif p_val < 0.01:
                sig_level = " (**)"
            elif p_val < 0.05:
                sig_level = " (*)"
            
            print(f"Trial {trial}: KO = {ko_means[i]:.3f} ± {ko_sems[i]:.3f} (n={len(ko_trial_effs)}), WT = {wt_means[i]:.3f} ± {wt_sems[i]:.3f} (n={len(wt_trial_effs)}), p = {p_val:.4f}{sig_level}")
        elif not np.isnan(ko_means[i]):
            print(f"Trial {trial}: KO = {ko_means[i]:.3f} ± {ko_sems[i]:.3f} (n={len(ko_trial_effs)}), WT = No data")
        elif not np.isnan(wt_means[i]):
            print(f"Trial {trial}: KO = No data, WT = {wt_means[i]:.3f} ± {wt_sems[i]:.3f} (n={len(wt_trial_effs)})")
        else:
            print(f"Trial {trial}: No data for either group")
    
    print("\nSignificance levels: * p<0.05, ** p<0.01, *** p<0.001")

# Run the analysis
if __name__ == "__main__":
    print("Analyzing path efficiency for trials 1-5 with ATRX mice included...")
    plot_path_efficiency_comparison()