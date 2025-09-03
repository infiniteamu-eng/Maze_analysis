import modules.lib_process_data_to_mat as plib
import modules.lib_plot_mouse_trajectory as pltlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd

# Define mouse groups
KO_mice = ['60', '61', '62', '63']
WT_mice = ['64', '65', '66', '67']

def calculate_path_efficiency(mouse_id, trials=range(16, 22), exclude_trials=None):
    """Calculate path efficiency for trials 16-21 (target B training)"""
    
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

def plot_path_efficiency_comparison():
    """Plot path efficiency comparison between groups"""
    
    # Collect data for all mice with specific exclusions
    ko_data = []
    wt_data = []
    
    # Define which mice to exclude from specific trials
    # Trial 16: n=3 for WT (excluding mouse 67)
    wt_exclusions = {
        '67': [16]  # Exclude mouse 67 from trial 16
    }
    
    print("Loading data...")
    
    for mouse_id in KO_mice:
        print(f"\nProcessing KO mouse {mouse_id}:")
        trials, efficiencies = calculate_path_efficiency(mouse_id)
        if len(efficiencies) > 0:
            ko_data.append({'mouse_id': mouse_id, 'trials': trials, 'efficiencies': efficiencies})
    
    for mouse_id in WT_mice:
        print(f"\nProcessing WT mouse {mouse_id}:")
        exclude_trials = wt_exclusions.get(mouse_id, [])
        trials, efficiencies = calculate_path_efficiency(mouse_id, exclude_trials=exclude_trials)
        if len(efficiencies) > 0:
            wt_data.append({'mouse_id': mouse_id, 'trials': trials, 'efficiencies': efficiencies})
    
    print(f"\nLoaded data for {len(ko_data)} KO mice and {len(wt_data)} WT mice")
    print("Note: WT trial 16 uses n=3 (excluding mouse 67)")
    
    # Debug - check what data we have
    print("\nDebug - Data summary:")
    for data in ko_data:
        print(f"KO Mouse {data['mouse_id']}: trials {data['trials']}")
    for data in wt_data:
        print(f"WT Mouse {data['mouse_id']}: trials {data['trials']}")
    
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
    all_trials = range(16, 22)
    ko_means = []
    wt_means = []
    ko_sems = []
    wt_sems = []
    
    for trial in all_trials:
        ko_trial_effs = [eff for data in ko_data for t, eff in zip(data['trials'], data['efficiencies']) if t == trial]
        wt_trial_effs = [eff for data in wt_data for t, eff in zip(data['trials'], data['efficiencies']) if t == trial]
        
        print(f"\nTrial {trial}:")
        print(f"  KO efficiencies: {ko_trial_effs}")
        print(f"  WT efficiencies: {wt_trial_effs}")
        
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
    
    # Plot group means with error bars (filter out NaN values)
    valid_ko_trials = [t for t, m in zip(all_trials, ko_means) if not np.isnan(m)]
    valid_ko_means = [m for m in ko_means if not np.isnan(m)]
    valid_ko_sems = [s for s in ko_sems if not np.isnan(s)]
    
    valid_wt_trials = [t for t, m in zip(all_trials, wt_means) if not np.isnan(m)]
    valid_wt_means = [m for m in wt_means if not np.isnan(m)]
    valid_wt_sems = [s for s in wt_sems if not np.isnan(s)]
    
    if valid_ko_trials:
        ax1.errorbar(valid_ko_trials, valid_ko_means, yerr=valid_ko_sems, color='#306ed1', 
                    linewidth=3, capsize=3, label='KO', marker='o', markersize=6)
    if valid_wt_trials:
        ax1.errorbar(valid_wt_trials, valid_wt_means, yerr=valid_wt_sems, color='black', 
                    linewidth=3, capsize=3, label='WT', marker='s', markersize=6)
    
    ax1.set_xlabel('Trial Number')
    ax1.set_ylabel('Path Efficiency')
    ax1.set_title('Path Efficiency Across Target B Training\n(Trials 16-21)')
    ax1.legend()
    # ax1.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_ylim(0, 0.8)
    
    # Plot 2: Learning curves (early vs late)
    early_trials = range(16, 17)  # Trials 16-18
    late_trials = range(20, 22)   # Trials 19-21
    
    ko_early = []
    ko_late = []
    wt_early = []
    wt_late = []
    
    print(f"\nEarly vs Late Analysis:")
    print(f"Early trials: {list(early_trials)}")
    print(f"Late trials: {list(late_trials)}")
    
    for data in ko_data:
        early_effs = [eff for t, eff in zip(data['trials'], data['efficiencies']) if t in early_trials]
        late_effs = [eff for t, eff in zip(data['trials'], data['efficiencies']) if t in late_trials]
        
        print(f"KO Mouse {data['mouse_id']}: early={early_effs}, late={late_effs}")
        
        if early_effs and not any(np.isnan(early_effs)):
            ko_early.append(np.mean(early_effs))
        if late_effs and not any(np.isnan(late_effs)):
            ko_late.append(np.mean(late_effs))
    
    for data in wt_data:
        early_effs = [eff for t, eff in zip(data['trials'], data['efficiencies']) if t in early_trials]
        late_effs = [eff for t, eff in zip(data['trials'], data['efficiencies']) if t in late_trials]
        
        print(f"WT Mouse {data['mouse_id']}: early={early_effs}, late={late_effs}")
        
        if early_effs and not any(np.isnan(early_effs)):
            wt_early.append(np.mean(early_effs))
        if late_effs and not any(np.isnan(late_effs)):
            wt_late.append(np.mean(late_effs))
    
    print(f"\nFinal early/late arrays:")
    print(f"KO early: {ko_early}")
    print(f"KO late: {ko_late}")
    print(f"WT early: {wt_early}")
    print(f"WT late: {wt_late}")
    
    # Plot early vs late comparison
    positions = [1, 2, 3.5, 4.5]
    
    if ko_early:
        ko_early_x = np.random.normal(1, 0.05, len(ko_early))
        ax2.scatter(ko_early_x, ko_early, alpha=0.7, s=60, color='#306ed1', label='KO')
    if ko_late:
        ko_late_x = np.random.normal(2, 0.05, len(ko_late))
        ax2.scatter(ko_late_x, ko_late, alpha=0.7, s=60, color='#306ed1')
    if wt_early:
        wt_early_x = np.random.normal(3.5, 0.05, len(wt_early))
        ax2.scatter(wt_early_x, wt_early, alpha=0.7, s=60, color='black', label='WT')
    if wt_late:
        wt_late_x = np.random.normal(4.5, 0.05, len(wt_late))
        ax2.scatter(wt_late_x, wt_late, alpha=0.7, s=60, color='black')
    
    # Add means with error bars
    if ko_early:
        ax2.errorbar(1, np.mean(ko_early), yerr=stats.sem(ko_early), fmt='o', 
                    markersize=8, color='#306ed1', capsize=5, capthick=2, elinewidth=2, 
                    markeredgecolor='white', markeredgewidth=1)
    if ko_late:
        ax2.errorbar(2, np.mean(ko_late), yerr=stats.sem(ko_late), fmt='o', 
                    markersize=8, color='#306ed1', capsize=5, capthick=2, elinewidth=2, 
                    markeredgecolor='white', markeredgewidth=1)
    if wt_early:
        ax2.errorbar(3.5, np.mean(wt_early), yerr=stats.sem(wt_early), fmt='s', 
                    markersize=8, color='black', capsize=5, capthick=2, elinewidth=2, 
                    markeredgecolor='white', markeredgewidth=1)
    if wt_late:
        ax2.errorbar(4.5, np.mean(wt_late), yerr=stats.sem(wt_late), fmt='s', 
                    markersize=8, color='black', capsize=5, capthick=2, elinewidth=2, 
                    markeredgecolor='white', markeredgewidth=1)
    
    ax2.set_xlim(0.5, 5)
    ax2.set_xticks([1.5, 4])
    ax2.set_xticklabels(['KO', 'WT'])
    ax2.set_ylabel('Path Efficiency')
    ax2.set_title('Learning Comparison\nEarly (16-18) vs Late (19-21)')
    
    # Add brackets for early vs late within each group
    if len(ko_early) > 0 and len(ko_late) > 0:
        y_max_ko = max(max(ko_early), max(ko_late))
        ax2.plot([1, 2], [y_max_ko*1.1, y_max_ko*1.1], 'k-', linewidth=1)
        if len(ko_early) == len(ko_late):
            t_stat_ko, p_val_ko = stats.ttest_rel(ko_early, ko_late)
        else:
            t_stat_ko, p_val_ko = stats.ttest_ind(ko_early, ko_late)
        sig_text_ko = "***" if p_val_ko < 0.001 else "**" if p_val_ko < 0.01 else "*" if p_val_ko < 0.05 else f"p={p_val_ko:.3f}"
        ax2.text(1.5, y_max_ko*1.12, sig_text_ko, ha='center', va='bottom', fontsize=10)
    
    if len(wt_early) > 0 and len(wt_late) > 0:
        y_max_wt = max(max(wt_early), max(wt_late))
        ax2.plot([3.5, 4.5], [y_max_wt*1.1, y_max_wt*1.1], 'k-', linewidth=1)
        if len(wt_early) == len(wt_late):
            t_stat_wt, p_val_wt = stats.ttest_rel(wt_early, wt_late)
        else:
            t_stat_wt, p_val_wt = stats.ttest_ind(wt_early, wt_late)
        sig_text_wt = "***" if p_val_wt < 0.001 else "**" if p_val_wt < 0.01 else "*" if p_val_wt < 0.05 else f"p={p_val_wt:.3f}"
        ax2.text(4, y_max_wt*1.12, sig_text_wt, ha='center', va='bottom', fontsize=10)
    
    ax2.legend()
    # ax2.grid(True, alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_ylim(0, 0.8)
    
    # Add text annotations for early vs late
    ax2.text(1.5, -0.1, 'Early\n(16-18)', ha='center', va='top', transform=ax2.get_xaxis_transform(), fontsize=9)
    ax2.text(4, -0.1, 'Late\n(19-21)', ha='center', va='top', transform=ax2.get_xaxis_transform(), fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\nPath Efficiency Summary (Trials 16-21 - Target B Training):")
    if len(ko_early) > 0:
        print(f"KO mice - Early trials (16-18): {np.mean(ko_early):.3f} ± {stats.sem(ko_early):.3f} (n={len(ko_early)})")
    else:
        print("KO mice - Early trials (16-18): No valid data")
        
    if len(ko_late) > 0:
        print(f"KO mice - Late trials (19-21): {np.mean(ko_late):.3f} ± {stats.sem(ko_late):.3f} (n={len(ko_late)})")
    else:
        print("KO mice - Late trials (19-21): No valid data")
    
    if len(wt_early) > 0:
        print(f"WT mice - Early trials (16-18): {np.mean(wt_early):.3f} ± {stats.sem(wt_early):.3f} (n={len(wt_early)})")
    else:
        print("WT mice - Early trials (16-18): No valid data")
        
    if len(wt_late) > 0:
        print(f"WT mice - Late trials (19-21): {np.mean(wt_late):.3f} ± {stats.sem(wt_late):.3f} (n={len(wt_late)})")
    else:
        print("WT mice - Late trials (19-21): No valid data")

# Run the analysis
if __name__ == "__main__":
    print("Analyzing path efficiency across trials 16-21 (Target B training)...")
    plot_path_efficiency_comparison()