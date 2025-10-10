# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 10:35:15 2025

@author: BeiqueLab
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 10:26:29 2025

@author: BeiqueLab
"""

import modules.lib_process_data_to_mat as plib
import modules.lib_plot_mouse_trajectory as pltlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd

# Define mouse groups
KO_mice = ['46', '47', '48', '49']
WT_mice = ['50', '51', '52', '53']

def calculate_path_efficiency(mouse_id, trials=range(1, 30), exclude_trials=None):
    """Calculate path efficiency for trials 1-15, with option to exclude specific trials"""
    
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
            d.Load('2025-06-04', mouse_id, str(trial))
            
            coords = d.r_nose
            valid_idx = ~(np.isnan(coords[:, 0]) | np.isnan(coords[:, 1]))
            coords_clean = coords[valid_idx]
            
            if len(coords_clean) < 2:
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
    """Plot path efficiency comparison between groups"""
    
    # Collect data for all mice with specific exclusions
    ko_data = []
    wt_data = []
    
    # Define which mice to exclude from trials 4 and 15 (keeping n=3 for WT)
    # Assuming we exclude mouse 67 based on previous data issues
    wt_exclusions = {
        '46':[2,3,5,9,10,11,25],
        '47':[1,3,4,5,10,11],
        '50':[1,3,19],
        '51':[1,2,3,4,5,11]
    }
    
    for mouse_id in KO_mice:
        trials, efficiencies = calculate_path_efficiency(mouse_id)
        if len(efficiencies) > 0:
            ko_data.append({'mouse_id': mouse_id, 'trials': trials, 'efficiencies': efficiencies})
    
    for mouse_id in WT_mice:
        exclude_trials = wt_exclusions.get(mouse_id, [])
        trials, efficiencies = calculate_path_efficiency(mouse_id, exclude_trials=exclude_trials)
        if len(efficiencies) > 0:
            wt_data.append({'mouse_id': mouse_id, 'trials': trials, 'efficiencies': efficiencies})
    
    print(f"Loaded data for {len(ko_data)} KO mice and {len(wt_data)} WT mice")
    print("Note: WT trials 4 and 15 use n=3 (excluding mouse 67)")
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Plot 1: Individual trajectories
    # KO mice
    # for data in ko_data:
    #     ax1.plot(data['trials'], data['efficiencies'], 'o-', alpha=0.6, 
    #             color='#306ed1', linewidth=1, markersize=4)
    
    # # WT mice  
    # for data in wt_data:
    #     ax1.plot(data['trials'], data['efficiencies'], 'o-', alpha=0.6, 
    #             color='black', linewidth=1, markersize=4)
    
    # Calculate and plot group means
    all_trials = range(20, 30)
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
    
    # Plot group means with error bars
    ax1.errorbar(all_trials, ko_means, yerr=ko_sems, color='#306ed1', 
                linewidth=3, capsize=3, label='KO', marker='o', markersize=6)
    ax1.errorbar(all_trials, wt_means, yerr=wt_sems, color='black', 
                linewidth=3, capsize=3, label='WT', marker='s', markersize=6)
    
    ax1.set_xlabel('Trial Number')
    ax1.set_ylabel('Path Efficiency')
    ax1.set_title('Path Efficiency Across Training\n(Trials 1-15)')
    ax1.legend()
    # ax1.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_ylim(0, 0.8)
    
    # Plot 2: Learning curves (early vs late)
    early_trials = range(1, 6)  # Trials 1-5
    late_trials = range(11, 16)  # Trials 11-15
    
    ko_early = []
    ko_late = []
    wt_early = []
    wt_late = []
    
    for data in ko_data:
        early_effs = [eff for t, eff in zip(data['trials'], data['efficiencies']) if t in early_trials]
        late_effs = [eff for t, eff in zip(data['trials'], data['efficiencies']) if t in late_trials]
        if early_effs:
            ko_early.append(np.mean(early_effs))
        if late_effs:
            ko_late.append(np.mean(late_effs))
    
    for data in wt_data:
        early_effs = [eff for t, eff in zip(data['trials'], data['efficiencies']) if t in early_trials]
        late_effs = [eff for t, eff in zip(data['trials'], data['efficiencies']) if t in late_trials]
        if early_effs:
            wt_early.append(np.mean(early_effs))
        if late_effs:
            wt_late.append(np.mean(late_effs))
    
    # Plot early vs late comparison
    positions = [1, 2, 3.5, 4.5]
    ko_early_x = np.random.normal(1, 0.05, len(ko_early))
    ko_late_x = np.random.normal(2, 0.05, len(ko_late))
    wt_early_x = np.random.normal(3.5, 0.05, len(wt_early))
    wt_late_x = np.random.normal(4.5, 0.05, len(wt_late))
    
    ax2.scatter(ko_early_x, ko_early, alpha=0.7, s=60, color='#306ed1', label='KO')
    ax2.scatter(ko_late_x, ko_late, alpha=0.7, s=60, color='#306ed1')
    ax2.scatter(wt_early_x, wt_early, alpha=0.7, s=60, color='black', label='WT')
    ax2.scatter(wt_late_x, wt_late, alpha=0.7, s=60, color='black')
    
    # Add means with error bars
    if len(ko_early) > 0:
        ax2.errorbar(1, np.mean(ko_early), yerr=stats.sem(ko_early), fmt='o', 
                    markersize=8, color='#306ed1', capsize=5, capthick=2, elinewidth=2, 
                    markeredgecolor='white', markeredgewidth=1)
    if len(ko_late) > 0:
        ax2.errorbar(2, np.mean(ko_late), yerr=stats.sem(ko_late), fmt='o', 
                    markersize=8, color='#306ed1', capsize=5, capthick=2, elinewidth=2, 
                    markeredgecolor='white', markeredgewidth=1)
    if len(wt_early) > 0:
        ax2.errorbar(3.5, np.mean(wt_early), yerr=stats.sem(wt_early), fmt='s', 
                    markersize=8, color='black', capsize=5, capthick=2, elinewidth=2, 
                    markeredgecolor='white', markeredgewidth=1)
    if len(wt_late) > 0:
        ax2.errorbar(4.5, np.mean(wt_late), yerr=stats.sem(wt_late), fmt='s', 
                    markersize=8, color='black', capsize=5, capthick=2, elinewidth=2, 
                    markeredgecolor='white', markeredgewidth=1)
    
    ax2.set_xlim(0.5, 5)
    ax2.set_xticks([1.5, 4])
    ax2.set_xticklabels(['KO', 'WT'])
    ax2.set_ylabel('Path Efficiency')
    ax2.set_title('Learning Comparison\nEarly (1-5) vs Late (11-15)')
    
    # Add brackets for early vs late within each group
    if len(ko_early) > 0 and len(ko_late) > 0:
        y_max_ko = max(max(ko_early), max(ko_late))
        ax2.plot([1, 2], [y_max_ko*1.1, y_max_ko*1.1], 'k-', linewidth=1)
        t_stat_ko, p_val_ko = stats.ttest_rel(ko_early[:min(len(ko_early), len(ko_late))], 
                                             ko_late[:min(len(ko_early), len(ko_late))])
        sig_text_ko = "***" if p_val_ko < 0.001 else "**" if p_val_ko < 0.01 else "*" if p_val_ko < 0.05 else f"p={p_val_ko:.3f}"
        ax2.text(1.5, y_max_ko*1.12, sig_text_ko, ha='center', va='bottom', fontsize=10)
    
    if len(wt_early) > 0 and len(wt_late) > 0:
        y_max_wt = max(max(wt_early), max(wt_late))
        ax2.plot([3.5, 4.5], [y_max_wt*1.1, y_max_wt*1.1], 'k-', linewidth=1)
        t_stat_wt, p_val_wt = stats.ttest_rel(wt_early[:min(len(wt_early), len(wt_late))], 
                                             wt_late[:min(len(wt_early), len(wt_late))])
        sig_text_wt = "***" if p_val_wt < 0.001 else "**" if p_val_wt < 0.01 else "*" if p_val_wt < 0.05 else f"p={p_val_wt:.3f}"
        ax2.text(4, y_max_wt*1.12, sig_text_wt, ha='center', va='bottom', fontsize=10)
    
    ax2.legend()
    # ax2.grid(True, alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_ylim(0, 0.8)
    
    # Add text annotations for early vs late
    ax2.text(1.5, -0.1, 'Early(1-5)', ha='center', va='top', transform=ax2.get_xaxis_transform(), fontsize=9)
    ax2.text(4, -0.1, 'Late(11-15)', ha='center', va='top', transform=ax2.get_xaxis_transform(), fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\nPath Efficiency Summary (Trials 1-15):")
    if len(ko_early) > 0 and len(ko_late) > 0:
        print(f"KO mice - Early trials (1-5): {np.mean(ko_early):.3f} ± {stats.sem(ko_early):.3f}")
        print(f"KO mice - Late trials (11-15): {np.mean(ko_late):.3f} ± {stats.sem(ko_late):.3f}")
    
    if len(wt_early) > 0 and len(wt_late) > 0:
        print(f"WT mice - Early trials (1-5): {np.mean(wt_early):.3f} ± {stats.sem(wt_early):.3f}")
        print(f"WT mice - Late trials (11-15): {np.mean(wt_late):.3f} ± {stats.sem(wt_late):.3f}")

# Run the analysis
if __name__ == "__main__":
    print("Analyzing path efficiency across trials 1-15...")
    plot_path_efficiency_comparison()