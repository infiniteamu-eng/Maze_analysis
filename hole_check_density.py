import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import modules.lib_process_data_to_mat as plib
import scipy.stats as stats

def calculate_trial_half_hole_check_density(data):
    """
    Calculate hole check density in first half vs second half of trial
    Density = number of hole checks / distance traveled in that half
    
    Parameters:
    - data: TrialData object with time, k_hole_checks, k_reward, and trajectory data
    
    Returns:
    - first_half_density: hole checks per unit distance in first half
    - second_half_density: hole checks per unit distance in second half
    - first_half_distance: distance traveled in first half
    - second_half_distance: distance traveled in second half
    - half_point_time: time at halfway point
    - total_trial_time: total time to find reward
    """
    
    # Get reward time (total trial time)
    if hasattr(data, 'k_reward') and data.k_reward is not None:
        reward_index = data.k_reward
        reward_time = data.time[reward_index]
        trial_start_time = data.time[0]
        total_trial_time = reward_time - trial_start_time
        half_point_time = trial_start_time + (total_trial_time / 2)
        
        # Find the index closest to half_point_time
        half_point_index = np.argmin(np.abs(data.time - half_point_time))
    else:
        print("No reward index found")
        return None, None, None, None, None, None
    
    # Calculate distances for each half
    # Use nose coordinates if available, otherwise use center coordinates
    if hasattr(data, 'r_nose') and len(data.r_nose) > 1 and not np.all(np.isnan(data.r_nose)):
        coords = data.r_nose
    elif hasattr(data, 'r_center') and len(data.r_center) > 1:
        coords = data.r_center
    else:
        print("No coordinate data found")
        return None, None, None, None, None, None
    
    # Calculate distance traveled in first half (from start to half_point_index)
    first_half_coords = coords[0:half_point_index+1]
    first_half_distance = 0
    for i in range(1, len(first_half_coords)):
        if not (np.isnan(first_half_coords[i]).any() or np.isnan(first_half_coords[i-1]).any()):
            dist = np.sqrt((first_half_coords[i,0] - first_half_coords[i-1,0])**2 + 
                          (first_half_coords[i,1] - first_half_coords[i-1,1])**2)
            first_half_distance += dist
    
    # Calculate distance traveled in second half (from half_point_index to reward_index)
    second_half_coords = coords[half_point_index:reward_index+1]
    second_half_distance = 0
    for i in range(1, len(second_half_coords)):
        if not (np.isnan(second_half_coords[i]).any() or np.isnan(second_half_coords[i-1]).any()):
            dist = np.sqrt((second_half_coords[i,0] - second_half_coords[i-1,0])**2 + 
                          (second_half_coords[i,1] - second_half_coords[i-1,1])**2)
            second_half_distance += dist
    
    # Count hole checks in each half
    if hasattr(data, 'k_hole_checks') and data.k_hole_checks is not None:
        hole_checks = data.k_hole_checks
        
        first_half_checks = 0
        second_half_checks = 0
        
        # k_hole_checks is a 2D array: [hole_id, time_index]
        for i in range(hole_checks.shape[0]):
            hole_id = hole_checks[i, 0]  # which hole (0-99)
            time_index = hole_checks[i, 1]  # when it happened (index)
            
            # Convert index to time
            check_time = data.time[time_index]
            
            # Categorize into first or second half
            if trial_start_time <= check_time <= half_point_time:
                first_half_checks += 1
            elif half_point_time < check_time <= reward_time:
                second_half_checks += 1
        
        # Calculate densities (avoid division by zero)
        first_half_density = first_half_checks / first_half_distance if first_half_distance > 0 else 0
        second_half_density = second_half_checks / second_half_distance if second_half_distance > 0 else 0
        
        return (first_half_density, second_half_density, 
                first_half_distance, second_half_distance, 
                half_point_time, total_trial_time)
    else:
        print("No hole check data found")
        return None, None, None, None, None, None

def analyze_trial_halves_density_all_mice(experiment_date, mice_list, trial_range=(1, 22)):
    """
    Analyze first half vs second half hole check density for all mice across all trials
    
    Returns:
    - results_df: DataFrame with results for each mouse-trial combination
    - summary_stats: Summary statistics by group and trial phase
    """
    
    results = []
    
    for mouse_id in mice_list:
        print(f"Analyzing mouse {mouse_id}...")
        
        for trial_num in range(trial_range[0], trial_range[1]):
            try:
                # Load trial data
                data = plib.TrialData()
                data.Load(experiment_date, mouse_id, str(trial_num))
                
                # Calculate trial half hole check densities
                (first_half_density, second_half_density, 
                 first_half_distance, second_half_distance, 
                 half_point, total_time) = calculate_trial_half_hole_check_density(data)
                
                if first_half_density is not None and second_half_density is not None:
                    # Determine training phase and target
                    if trial_num <= 15:
                        phase = "Target_A"
                        target_phase = "Training_A"
                    else:  # trials 16-21
                        phase = "Target_B" 
                        target_phase = "Training_B"
                    
                    # Determine mouse group
                    group = 'KO' if mouse_id in ['60', '61', '62', '63'] else 'WT'
                    
                    results.append({
                        'mouse_id': mouse_id,
                        'trial': trial_num,
                        'group': group,
                        'phase': phase,
                        'target_phase': target_phase,
                        'first_half_density': first_half_density,
                        'second_half_density': second_half_density,
                        'first_half_distance': first_half_distance,
                        'second_half_distance': second_half_distance,
                        'total_trial_time': total_time,
                        'half_point_time': half_point
                    })
                    
                    print(f"  Trial {trial_num}: 1st half density={first_half_density:.3f}, 2nd half density={second_half_density:.3f} checks/cm")
                
            except Exception as e:
                print(f"  Trial {trial_num}: Error - {e}")
                continue
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate summary statistics
    summary_stats = results_df.groupby(['group', 'target_phase']).agg({
        'first_half_density': ['count', 'mean', 'std', 'sem'],
        'second_half_density': ['count', 'mean', 'std', 'sem'],
        'first_half_distance': ['mean', 'std'],
        'second_half_distance': ['mean', 'std'],
        'total_trial_time': ['mean', 'std'],
        'trial': ['min', 'max']
    }).round(4)
    
    return results_df, summary_stats

def plot_trial_halves_density_analysis(results_df, savefig=False):
    """
    Create visualizations for first half and second half hole check density
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Separate data by groups
    ko_data = results_df[results_df['group'] == 'KO']
    wt_data = results_df[results_df['group'] == 'WT']
    
    # Plot 1: First Half Density - All 21 trials
    if len(ko_data) > 0:
        ko_avg_first = ko_data.groupby('trial')['first_half_density'].agg(['mean', 'sem']).reset_index()
        ax1.errorbar(ko_avg_first['trial'], ko_avg_first['mean'], yerr=ko_avg_first['sem'], 
                    label='KO', marker='o', color='#306ed1', alpha=0.7, linewidth=2)
    
    if len(wt_data) > 0:
        wt_avg_first = wt_data.groupby('trial')['first_half_density'].agg(['mean', 'sem']).reset_index()
        ax1.errorbar(wt_avg_first['trial'], wt_avg_first['mean'], yerr=wt_avg_first['sem'], 
                    label='WT', marker='s', color='black', alpha=0.7, linewidth=2)
    
    ax1.axvline(x=15.5, color='gray', linestyle=':', alpha=0.7, label='Target Switch')
    ax1.set_xlabel('Trial Number')
    ax1.set_ylabel('Hole Check Density (checks/cm)')
    ax1.set_title('First Half: Exploration Density')
    ax1.legend()
    ax1.set_xlim(0.5, 21.5)
    ax1.set_ylim(0, 0.07)
    ax1.set_xticks(range(1, 22, 2))  # Show every other trial number
    
    # Plot 2: Second Half Density - All 21 trials
    if len(ko_data) > 0:
        ko_avg_second = ko_data.groupby('trial')['second_half_density'].agg(['mean', 'sem']).reset_index()
        ax2.errorbar(ko_avg_second['trial'], ko_avg_second['mean'], yerr=ko_avg_second['sem'], 
                    label='KO', marker='o', color='#306ed1', alpha=0.7, linewidth=2)
    
    if len(wt_data) > 0:
        wt_avg_second = wt_data.groupby('trial')['second_half_density'].agg(['mean', 'sem']).reset_index()
        ax2.errorbar(wt_avg_second['trial'], wt_avg_second['mean'], yerr=wt_avg_second['sem'], 
                    label='WT', marker='s', color='black', alpha=0.7, linewidth=2)
    
    ax2.axvline(x=15.5, color='gray', linestyle=':', alpha=0.7, label='Target Switch')
    ax2.set_xlabel('Trial Number')
    ax2.set_ylabel('Hole Check Density (checks/cm)')
    ax2.set_title('Second Half: Search Efficiency Density')
    ax2.legend()
    ax2.set_xlim(0.5, 21.5)
    ax2.set_ylim(0, 0.07)
    ax2.set_xticks(range(1, 22, 2))  # Show every other trial number
    
    plt.suptitle('Hole Check Density Analysis: Search Efficiency Across All Trials', fontsize=16, weight='bold')
    plt.tight_layout()
    
    if savefig:
        plt.savefig('trial_halves_density_analysis.png', dpi=300, bbox_inches='tight')
    
    plt.show()

def statistical_analysis_density(results_df):
    """
    Perform statistical analyses comparing groups and trial halves for hole check density
    """
    
    print("\n" + "="*70)
    print("TRIAL HALVES HOLE CHECK DENSITY STATISTICAL ANALYSIS")
    print("="*70)
    
    # Overall group comparison - First Half Density
    ko_first_all = results_df[results_df['group'] == 'KO']['first_half_density']
    wt_first_all = results_df[results_df['group'] == 'WT']['first_half_density']
    
    t_stat_first, p_val_first = stats.ttest_ind(ko_first_all, wt_first_all)
    
    print(f"\nFirst Half Density - Overall Group Comparison:")
    print(f"KO mice: {ko_first_all.mean():.4f} ± {stats.sem(ko_first_all):.4f} checks/cm")
    print(f"WT mice: {wt_first_all.mean():.4f} ± {stats.sem(wt_first_all):.4f} checks/cm")
    print(f"T-test: t={t_stat_first:.3f}, p={p_val_first:.3f}")
    
    # Overall group comparison - Second Half Density
    ko_second_all = results_df[results_df['group'] == 'KO']['second_half_density']
    wt_second_all = results_df[results_df['group'] == 'WT']['second_half_density']
    
    t_stat_second, p_val_second = stats.ttest_ind(ko_second_all, wt_second_all)
    
    print(f"\nSecond Half Density - Overall Group Comparison:")
    print(f"KO mice: {ko_second_all.mean():.4f} ± {stats.sem(ko_second_all):.4f} checks/cm")
    print(f"WT mice: {wt_second_all.mean():.4f} ± {stats.sem(wt_second_all):.4f} checks/cm")
    print(f"T-test: t={t_stat_second:.3f}, p={p_val_second:.3f}")
    
    # Within-group comparison: First half vs Second half density
    print(f"\nWithin-Group Comparisons (First vs Second Half Density):")
    
    # KO: First vs Second half density
    t_stat_ko_halves, p_val_ko_halves = stats.ttest_rel(ko_first_all, ko_second_all)
    print(f"KO mice - First vs Second half density:")
    print(f"  First half: {ko_first_all.mean():.4f} ± {stats.sem(ko_first_all):.4f} checks/cm")
    print(f"  Second half: {ko_second_all.mean():.4f} ± {stats.sem(ko_second_all):.4f} checks/cm")
    print(f"  Paired t-test: t={t_stat_ko_halves:.3f}, p={p_val_ko_halves:.3f}")
    
    # WT: First vs Second half density
    t_stat_wt_halves, p_val_wt_halves = stats.ttest_rel(wt_first_all, wt_second_all)
    print(f"WT mice - First vs Second half density:")
    print(f"  First half: {wt_first_all.mean():.4f} ± {stats.sem(wt_first_all):.4f} checks/cm")
    print(f"  Second half: {wt_second_all.mean():.4f} ± {stats.sem(wt_second_all):.4f} checks/cm")
    print(f"  Paired t-test: t={t_stat_wt_halves:.3f}, p={p_val_wt_halves:.3f}")
    
    # Phase-specific analysis
    print(f"\nPhase-Specific Density Analysis:")
    
    for phase in ['Target_A', 'Target_B']:
        ko_phase_first = results_df[(results_df['group'] == 'KO') & (results_df['phase'] == phase)]['first_half_density']
        wt_phase_first = results_df[(results_df['group'] == 'WT') & (results_df['phase'] == phase)]['first_half_density']
        ko_phase_second = results_df[(results_df['group'] == 'KO') & (results_df['phase'] == phase)]['second_half_density']
        wt_phase_second = results_df[(results_df['group'] == 'WT') & (results_df['phase'] == phase)]['second_half_density']
        
        if len(ko_phase_first) > 0 and len(wt_phase_first) > 0:
            t_stat_phase_first, p_val_phase_first = stats.ttest_ind(ko_phase_first, wt_phase_first)
            t_stat_phase_second, p_val_phase_second = stats.ttest_ind(ko_phase_second, wt_phase_second)
            
            print(f"\n{phase} Phase:")
            print(f"First Half - KO: {ko_phase_first.mean():.4f}±{stats.sem(ko_phase_first):.4f}, WT: {wt_phase_first.mean():.4f}±{stats.sem(wt_phase_first):.4f} (p={p_val_phase_first:.3f})")
            print(f"Second Half - KO: {ko_phase_second.mean():.4f}±{stats.sem(ko_phase_second):.4f}, WT: {wt_phase_second.mean():.4f}±{stats.sem(wt_phase_second):.4f} (p={p_val_phase_second:.3f})")
    
    # Learning trends for density
    print(f"\nLearning Trends Analysis (Density):")
    
    # Early trials (1-5) vs Late trials (17-21)
    early_trials = results_df[results_df['trial'] <= 5]
    late_trials = results_df[results_df['trial'] >= 17]
    
    ko_early_first = early_trials[early_trials['group'] == 'KO']['first_half_density'].mean()
    ko_late_first = late_trials[late_trials['group'] == 'KO']['first_half_density'].mean()
    wt_early_first = early_trials[early_trials['group'] == 'WT']['first_half_density'].mean()
    wt_late_first = late_trials[late_trials['group'] == 'WT']['first_half_density'].mean()
    
    ko_early_second = early_trials[early_trials['group'] == 'KO']['second_half_density'].mean()
    ko_late_second = late_trials[late_trials['group'] == 'KO']['second_half_density'].mean()
    wt_early_second = early_trials[early_trials['group'] == 'WT']['second_half_density'].mean()
    wt_late_second = late_trials[late_trials['group'] == 'WT']['second_half_density'].mean()
    
    print(f"KO First Half Density - Early trials (1-5): {ko_early_first:.4f}, Late trials (17-21): {ko_late_first:.4f}")
    print(f"   Change: {((ko_early_first - ko_late_first) / ko_early_first * 100):+.1f}%")
    print(f"WT First Half Density - Early trials (1-5): {wt_early_first:.4f}, Late trials (17-21): {wt_late_first:.4f}")
    print(f"   Change: {((wt_early_first - wt_late_first) / wt_early_first * 100):+.1f}%")
    
    print(f"KO Second Half Density - Early trials (1-5): {ko_early_second:.4f}, Late trials (17-21): {ko_late_second:.4f}")
    print(f"   Change: {((ko_early_second - ko_late_second) / ko_early_second * 100):+.1f}%")
    print(f"WT Second Half Density - Early trials (1-5): {wt_early_second:.4f}, Late trials (17-21): {wt_late_second:.4f}")
    print(f"   Change: {((wt_early_second - wt_late_second) / wt_early_second * 100):+.1f}%")
    
    # Additional analysis: Distance traveled comparison
    print(f"\nDistance Analysis:")
    ko_first_dist = results_df[results_df['group'] == 'KO']['first_half_distance'].mean()
    ko_second_dist = results_df[results_df['group'] == 'KO']['second_half_distance'].mean()
    wt_first_dist = results_df[results_df['group'] == 'WT']['first_half_distance'].mean()
    wt_second_dist = results_df[results_df['group'] == 'WT']['second_half_distance'].mean()
    
    print(f"Average Distance Traveled:")
    print(f"KO - First half: {ko_first_dist:.1f} cm, Second half: {ko_second_dist:.1f} cm")
    print(f"WT - First half: {wt_first_dist:.1f} cm, Second half: {wt_second_dist:.1f} cm")

def main_trial_halves_density_analysis():
    """
    Main function to run the complete trial halves hole check density analysis
    """
    
    experiment_date = '2025-08-22'
    mice_list = ['60', '61', '62', '63', '64', '65', '66', '67']
    
    print(f"Analyzing trial halves hole check density...")
    print(f"Density = hole checks / distance traveled in each half")
    print(f"Target A training: Trials 1-15")
    print(f"Target B training: Trials 16-21")
    
    # Run analysis
    results_df, summary_stats = analyze_trial_halves_density_all_mice(
        experiment_date, mice_list, (1, 22))
    
    print(f"\nAnalysis complete. Processed {len(results_df)} trials.")
    
    # Display summary statistics
    print(f"\nSummary Statistics:")
    print(summary_stats)
    
    # Create visualizations
    plot_trial_halves_density_analysis(results_df, savefig=True)
    
    # Perform statistical tests
    statistical_analysis_density(results_df)
    
    return results_df, summary_stats

# Run the analysis
if __name__ == "__main__":
    results, summary = main_trial_halves_density_analysis()