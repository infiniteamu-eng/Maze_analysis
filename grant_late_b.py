#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 22:28:51 2025

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

def calculate_path_efficiency(mouse_id, trials=range(19, 21), exclude_trials=None):
    """Calculate path efficiency for trials 19-20 (late condition)"""
    
    if exclude_trials is None:
        exclude_trials = []
    
    efficiencies = []
    trial_numbers = []
    
    for trial in trials:
        # Skip excluded trials for this mouse
        if trial in exclude_trials:
            print(f"Excluding trial {trial} for mouse {mouse_id}")
            continue
            
        try:
            # Load trial data
            d = plib.TrialData()
            d.Load('2025-08-22', mouse_id, str(trial))
            
            coords = d.r_nose
            valid_idx = ~(np.isnan(coords[:, 0]) | np.isnan(coords[:, 1]))
            coords_clean = coords[valid_idx]
            
            if len(coords_clean) < 2:
                print(f"Insufficient data for mouse {mouse_id}, trial {trial}")
                continue
            
            # Find start and end positions
            start_pos = coords_clean[0]
            try:
                target_idx = pltlib.coords_to_target(coords, d.target)
                end_pos = coords[target_idx]
                coords_to_use = coords_clean[:target_idx+1]  # Only up to target
            except:
                print(f"Target not reached for mouse {mouse_id}, trial {trial}")
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
            if actual_path_length > 0 and optimal_distance > 0:
                efficiency = optimal_distance / actual_path_length
                if not np.isnan(efficiency) and not np.isinf(efficiency):
                    efficiencies.append(efficiency)
                    trial_numbers.append(trial)
                    print(f"Mouse {mouse_id}, trial {trial}: efficiency = {efficiency:.3f}")
                else:
                    print(f"Invalid efficiency for mouse {mouse_id}, trial {trial}: {efficiency}")
            else:
                print(f"Zero distance for mouse {mouse_id}, trial {trial}")
            
        except Exception as e:
            print(f"Error processing mouse {mouse_id}, trial {trial}: {e}")
            continue
    
    return trial_numbers, efficiencies

def add_stat_annotation(ax, x1, x2, y, h, text, text_offset=0.02):
    """Add statistical significance annotation between two bars"""
    # Draw horizontal line
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='black')
    # Add significance text
    ax.text((x1+x2)*.5, y+h+text_offset, text, ha='center', va='bottom', fontweight='bold', fontsize=12)

def plot_late_condition_bar():
    """Plot bar graph for late condition (trials 19-20) only with statistical significance"""
    
    # Collect data for all mice
    ko_data = []
    wt_data = []
    
    print("Loading data for late condition (trials 19-20)...")
    
    for mouse_id in KO_mice:
        print(f"\nProcessing KO mouse {mouse_id}:")
        trials, efficiencies = calculate_path_efficiency(mouse_id, trials=range(19, 21))
        if len(efficiencies) > 0:
            ko_data.append({'mouse_id': mouse_id, 'trials': trials, 'efficiencies': efficiencies})
    
    for mouse_id in WT_mice:
        print(f"\nProcessing WT mouse {mouse_id}:")
        trials, efficiencies = calculate_path_efficiency(mouse_id, trials=range(19, 21))
        if len(efficiencies) > 0:
            wt_data.append({'mouse_id': mouse_id, 'trials': trials, 'efficiencies': efficiencies})
    
    # Calculate late condition averages for each mouse
    wt_late = []
    ko_late = []
    
    late_trials = range(19, 21)  # Trials 19-20
    
    print(f"\nLate trials analysis (trials {list(late_trials)}):")
    
    for data in wt_data:
        late_effs = [eff for t, eff in zip(data['trials'], data['efficiencies']) if t in late_trials]
        print(f"WT Mouse {data['mouse_id']}: late={late_effs}")
        
        if late_effs and not any(np.isnan(late_effs)):
            wt_late.append(np.mean(late_effs))
    
    for data in ko_data:
        late_effs = [eff for t, eff in zip(data['trials'], data['efficiencies']) if t in late_trials]
        print(f"KO Mouse {data['mouse_id']}: late={late_effs}")
        
        if late_effs and not any(np.isnan(late_effs)):
            ko_late.append(np.mean(late_effs))
    
    print(f"\nFinal late arrays:")
    print(f"WT late: {wt_late}")
    print(f"KO late: {ko_late}")
    
    # Create bar plot with figure size (2,4)
    fig, ax = plt.subplots(1, 1, figsize=(2,4))
    
    # Plot individual data points and means
    positions = [1, 2]  # WT on left (1), KO on right (2)
    
    wt_mean, ko_mean = 0, 0  # Initialize for statistical test
    
    if wt_late:
        # Individual points with jitter
        wt_x = np.random.normal(1, 0.05, len(wt_late))
        ax.scatter(wt_x, wt_late, alpha=0.7, s=60, color='black', zorder=3)
        
        # Mean with error bar
        wt_mean = np.mean(wt_late)
        wt_sem = stats.sem(wt_late)
        ax.bar(1, wt_mean, width=0.6, color='lightgray', alpha=0.7, 
               edgecolor='black', linewidth=1, zorder=1)
        ax.errorbar(1, wt_mean, yerr=wt_sem, fmt='none', 
                   capsize=5, capthick=2, elinewidth=2, color='black', zorder=2)
    
    if ko_late:
        # Individual points with jitter
        ko_x = np.random.normal(2, 0.05, len(ko_late))
        ax.scatter(ko_x, ko_late, alpha=0.7, s=60, color='#306ed1', zorder=3)
        
        # Mean with error bar
        ko_mean = np.mean(ko_late)
        ko_sem = stats.sem(ko_late)
        ax.bar(2, ko_mean, width=0.6, color='lightblue', alpha=0.7, 
               edgecolor='#306ed1', linewidth=1, zorder=1)
        ax.errorbar(2, ko_mean, yerr=ko_sem, fmt='none', 
                   capsize=5, capthick=2, elinewidth=2, color='#306ed1', zorder=2)
    
    # Statistical comparison and annotation
    if len(wt_late) > 0 and len(ko_late) > 0:
        t_stat, p_val = stats.ttest_ind(wt_late, ko_late)
        
        # Determine significance level and symbol
        if p_val < 0.001:
            sig_text = '***'
        elif p_val < 0.01:
            sig_text = '**'
        elif p_val < 0.05:
            sig_text = '*'
        else:
            sig_text = 'ns'  # not significant
        
        # Find the height for statistical annotation
        max_height = max([max(wt_late) if wt_late else 0, max(ko_late) if ko_late else 0])
        # Add some padding above the highest bar/error bar
        if wt_late:
            wt_top = wt_mean + stats.sem(wt_late)
        else:
            wt_top = 0
        if ko_late:
            ko_top = ko_mean + stats.sem(ko_late)
        else:
            ko_top = 0
        
        annotation_height = 0.51
        
        # Add statistical annotation
        add_stat_annotation(ax, 1, 2, annotation_height, 0.01, sig_text)
    
    # Customize plot
    ax.set_xlim(0.5, 2.5)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['WT', 'KO'])
    ax.set_ylabel('Path Efficiency', fontsize=19, fontweight='bold')
    ax.set_title('Target B \n Late Training', fontsize=19, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=13)
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set y-axis limit to accommodate statistical annotation at y=0.65
    ax.set_ylim(0, 0.61)  # Fixed limit to show annotation at y=0.65
    ax.tick_params(axis='both', which='major', labelsize=16)
    plt.savefig('path_targetb.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\nPath Efficiency Summary - Late Condition (Trials 19-20):")
    if len(wt_late) > 0:
        print(f"WT mice: {np.mean(wt_late):.3f} ± {stats.sem(wt_late):.3f} (n={len(wt_late)})")
    else:
        print("WT mice: No valid data")
        
    if len(ko_late) > 0:
        print(f"KO mice: {np.mean(ko_late):.3f} ± {stats.sem(ko_late):.3f} (n={len(ko_late)})")
    else:
        print("KO mice: No valid data")
    
    # Statistical comparison
    if len(wt_late) > 0 and len(ko_late) > 0:
        t_stat, p_val = stats.ttest_ind(wt_late, ko_late)
        print(f"\nStatistical comparison (WT vs KO):")
        print(f"t-statistic: {t_stat:.3f}")
        print(f"p-value: {p_val:.3f}")
        
        # Print significance interpretation
        if p_val < 0.001:
            print("Significance: *** (p < 0.001)")
        elif p_val < 0.01:
            print("Significance: ** (p < 0.01)")
        elif p_val < 0.05:
            print("Significance: * (p < 0.05)")
        else:
            print("Significance: ns (not significant)")

# Run the analysis
if __name__ == "__main__":
    print("Analyzing path efficiency for late condition (trials 19-20)...")
    plot_late_condition_bar()