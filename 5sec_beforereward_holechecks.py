import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import modules.lib_process_data_to_mat as plib
import scipy.stats as stats

def calculate_pre_reward_hole_checks(data, time_window=5.0):
    """
    Calculate the number of hole checks in the specified time window before reward
    
    Parameters:
    - data: TrialData object with time, k_hole_checks, and k_reward
    - time_window: time window in seconds before reward (default 5.0 seconds)
    
    Returns:
    - count: number of hole checks in the pre-reward window
    - hole_check_times: actual times of hole checks in the window
    - reward_time: time when reward was found
    """
    
    # Get reward time (convert index to seconds)
    if hasattr(data, 'k_reward') and data.k_reward is not None:
        reward_index = data.k_reward
        reward_time = data.time[reward_index]
    else:
        print("No reward index found")
        return None, None, None
    
    # Define time window
    window_start = reward_time - time_window
    
    # Get hole check data
    if hasattr(data, 'k_hole_checks') and data.k_hole_checks is not None:
        hole_checks = data.k_hole_checks
        
        # Count hole checks in the time window
        hole_checks_in_window = []
        hole_check_times = []
        
        # k_hole_checks is a 2D array: [hole_id, time_index]
        for i in range(hole_checks.shape[0]):  # for each hole check event
            hole_id = hole_checks[i, 0]  # which hole (0-99)
            time_index = hole_checks[i, 1]  # when it happened (index)
            
            # Convert index to time
            check_time = data.time[time_index]
            
            # Check if this hole check is in our time window
            if window_start <= check_time <= reward_time:
                hole_checks_in_window.append(hole_id)
                hole_check_times.append(check_time)
        
        return len(hole_checks_in_window), hole_check_times, reward_time
    else:
        print("No hole check data found")
        return None, None, None

def analyze_pre_reward_behavior_all_mice(experiment_date, mice_list, trial_range=(1, 22), time_window=5.0):
    """
    Analyze pre-reward hole checking behavior for all mice across all trials
    
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
                
                # Calculate pre-reward hole checks
                hole_count, check_times, reward_time = calculate_pre_reward_hole_checks(data, time_window)
                
                if hole_count is not None:
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
                        'pre_reward_hole_checks': hole_count,
                        'reward_time': reward_time,
                        'time_window': time_window
                    })
                    
                    print(f"  Trial {trial_num}: {hole_count} hole checks in {time_window}s before reward")
                
            except Exception as e:
                print(f"  Trial {trial_num}: Error - {e}")
                continue
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate summary statistics
    summary_stats = results_df.groupby(['group', 'target_phase']).agg({
        'pre_reward_hole_checks': ['count', 'mean', 'std', 'sem'],
        'trial': ['min', 'max']
    }).round(2)
    
    return results_df, summary_stats

def plot_pre_reward_analysis(results_df, time_window=5.0, savefig=False):
    """
    Create comprehensive visualizations of pre-reward hole checking behavior
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Separate data by groups
    ko_data = results_df[results_df['group'] == 'KO']
    wt_data = results_df[results_df['group'] == 'WT']
    
    # Plot 1: Learning curves - Target A vs Target B
    for phase in ['Target_A', 'Target_B']:
        ko_phase = ko_data[ko_data['phase'] == phase]
        wt_phase = wt_data[wt_data['phase'] == phase]
        
        if len(ko_phase) > 0:
            ko_avg = ko_phase.groupby('trial')['pre_reward_hole_checks'].agg(['mean', 'sem']).reset_index()
            ax1.errorbar(ko_avg['trial'], ko_avg['mean'], yerr=ko_avg['sem'], 
                        label=f'KO {phase}', marker='o', linestyle='-' if phase == 'Target_A' else '--',
                        color='#306ed1', alpha=0.7)
        
        if len(wt_phase) > 0:
            wt_avg = wt_phase.groupby('trial')['pre_reward_hole_checks'].agg(['mean', 'sem']).reset_index()
            ax1.errorbar(wt_avg['trial'], wt_avg['mean'], yerr=wt_avg['sem'], 
                        label=f'WT {phase}', marker='s', linestyle='-' if phase == 'Target_A' else '--',
                        color='black', alpha=0.7)
    
    ax1.axvline(x=15.5, color='gray', linestyle=':', alpha=0.7, label='Target Switch')
    ax1.set_xlabel('Trial Number')
    ax1.set_ylabel(f'Hole Checks in {time_window}s Before Reward')
    ax1.set_title('Learning Curves: Pre-Reward Search Efficiency')
    ax1.legend()
    # ax1.grid(True, alpha=0.3)
    
    # Plot 2: Bar plot comparison showing Target A early/late and Target B early/late
    # Target A Early (trials 1-4), Target A Late (trials 11-15), 
    # Target B Early (trials 16-18), Target B Late (trials 19-21)
    ko_a_early = ko_data[(ko_data['trial'] >= 1) & (ko_data['trial'] <= 4)]['pre_reward_hole_checks']
    ko_a_late = ko_data[(ko_data['trial'] >= 11) & (ko_data['trial'] <= 15)]['pre_reward_hole_checks']
    ko_b_early = ko_data[(ko_data['trial'] >= 16) & (ko_data['trial'] <= 18)]['pre_reward_hole_checks']
    ko_b_late = ko_data[(ko_data['trial'] >= 19) & (ko_data['trial'] <= 21)]['pre_reward_hole_checks']
    
    wt_a_early = wt_data[(wt_data['trial'] >= 1) & (wt_data['trial'] <= 4)]['pre_reward_hole_checks']
    wt_a_late = wt_data[(wt_data['trial'] >= 11) & (wt_data['trial'] <= 15)]['pre_reward_hole_checks']
    wt_b_early = wt_data[(wt_data['trial'] >= 16) & (wt_data['trial'] <= 18)]['pre_reward_hole_checks']
    wt_b_late = wt_data[(wt_data['trial'] >= 19) & (wt_data['trial'] <= 21)]['pre_reward_hole_checks']
    
    ko_means = [ko_a_early.mean(), ko_a_late.mean(), ko_b_early.mean(), ko_b_late.mean()]
    wt_means = [wt_a_early.mean(), wt_a_late.mean(), wt_b_early.mean(), wt_b_late.mean()]
    ko_sems = [stats.sem(ko_a_early), stats.sem(ko_a_late), stats.sem(ko_b_early), stats.sem(ko_b_late)]
    wt_sems = [stats.sem(wt_a_early), stats.sem(wt_a_late), stats.sem(wt_b_early), stats.sem(wt_b_late)]
    
    x_pos = np.arange(4)
    width = 0.35
    
    ax2.bar(x_pos - width/2, ko_means, width, yerr=ko_sems, 
           label='KO', color='#306ed1', alpha=0.7, capsize=5)
    ax2.bar(x_pos + width/2, wt_means, width, yerr=wt_sems, 
           label='WT', color='black', alpha=0.7, capsize=5)
    
    ax2.set_xlabel('Training Phase')
    ax2.set_ylabel(f'Mean Hole Checks ({time_window}s Before Reward)')
    ax2.set_title('Search Efficiency: Learning Progression')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(['Target A Early\n(Trials 1-4)', 'Target A Late\n(Trials 11-15)', 
                        'Target B Early\n(Trials 16-18)', 'Target B Late\n(Trials 19-21)'])
    ax2.legend()
    # ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Pre-Reward Hole-Check Analysis ({time_window}s Window)', fontsize=16, weight='bold')
    plt.tight_layout()
    
    if savefig:
        plt.savefig(f'pre_reward_hole_analysis_{time_window}s.png', dpi=300, bbox_inches='tight')
    
    plt.show()

def statistical_analysis(results_df):
    """
    Perform statistical analyses comparing groups and phases
    """
    
    print("\n" + "="*60)
    print("PRE-REWARD HOLE CHECK STATISTICAL ANALYSIS")
    print("="*60)
    
    # Overall group comparison
    ko_all = results_df[results_df['group'] == 'KO']['pre_reward_hole_checks']
    wt_all = results_df[results_df['group'] == 'WT']['pre_reward_hole_checks']
    
    t_stat, p_val = stats.ttest_ind(ko_all, wt_all)
    
    print(f"\nOverall Group Comparison:")
    print(f"KO mice: {ko_all.mean():.2f} ± {stats.sem(ko_all):.2f} hole checks")
    print(f"WT mice: {wt_all.mean():.2f} ± {stats.sem(wt_all):.2f} hole checks")
    print(f"T-test: t={t_stat:.3f}, p={p_val:.3f}")
    
    # Phase-specific comparisons
    for phase in ['Target_A', 'Target_B']:
        ko_phase = results_df[(results_df['group'] == 'KO') & (results_df['phase'] == phase)]['pre_reward_hole_checks']
        wt_phase = results_df[(results_df['group'] == 'WT') & (results_df['phase'] == phase)]['pre_reward_hole_checks']
        
        if len(ko_phase) > 0 and len(wt_phase) > 0:
            t_stat_phase, p_val_phase = stats.ttest_ind(ko_phase, wt_phase)
            
            print(f"\n{phase} Phase Comparison:")
            print(f"KO: {ko_phase.mean():.2f} ± {stats.sem(ko_phase):.2f}")
            print(f"WT: {wt_phase.mean():.2f} ± {stats.sem(wt_phase):.2f}")
            print(f"T-test: t={t_stat_phase:.3f}, p={p_val_phase:.3f}")
    
    # Learning analysis - early vs late trials within each phase
    print(f"\nLearning Analysis:")
    
    # Target A learning (trials 1-4 vs 11-15)
    ko_a_early = results_df[(results_df['group'] == 'KO') & (results_df['trial'] >= 1) & (results_df['trial'] <= 4)]['pre_reward_hole_checks']
    ko_a_late = results_df[(results_df['group'] == 'KO') & (results_df['trial'] >= 11) & (results_df['trial'] <= 15)]['pre_reward_hole_checks']
    
    if len(ko_a_early) > 0 and len(ko_a_late) > 0:
        t_stat_ko, p_val_ko = stats.ttest_ind(ko_a_early, ko_a_late)
        print(f"KO Target A - Early (1-4): {ko_a_early.mean():.2f} ± {stats.sem(ko_a_early):.2f}")
        print(f"KO Target A - Late (11-15): {ko_a_late.mean():.2f} ± {stats.sem(ko_a_late):.2f}")
        print(f"  Improvement: {((ko_a_early.mean() - ko_a_late.mean()) / ko_a_early.mean() * 100):+.1f}%")
        print(f"  T-test: t={t_stat_ko:.3f}, p={p_val_ko:.3f}")
    
    wt_a_early = results_df[(results_df['group'] == 'WT') & (results_df['trial'] >= 1) & (results_df['trial'] <= 4)]['pre_reward_hole_checks']
    wt_a_late = results_df[(results_df['group'] == 'WT') & (results_df['trial'] >= 11) & (results_df['trial'] <= 15)]['pre_reward_hole_checks']
    
    if len(wt_a_early) > 0 and len(wt_a_late) > 0:
        t_stat_wt, p_val_wt = stats.ttest_ind(wt_a_early, wt_a_late)
        print(f"WT Target A - Early (1-4): {wt_a_early.mean():.2f} ± {stats.sem(wt_a_early):.2f}")
        print(f"WT Target A - Late (11-15): {wt_a_late.mean():.2f} ± {stats.sem(wt_a_late):.2f}")
        print(f"  Improvement: {((wt_a_early.mean() - wt_a_late.mean()) / wt_a_early.mean() * 100):+.1f}%")
        print(f"  T-test: t={t_stat_wt:.3f}, p={p_val_wt:.3f}")
    
    # Target B learning (trials 16-18 vs 19-21)
    ko_b_early = results_df[(results_df['group'] == 'KO') & (results_df['trial'] >= 16) & (results_df['trial'] <= 18)]['pre_reward_hole_checks']
    ko_b_late = results_df[(results_df['group'] == 'KO') & (results_df['trial'] >= 19) & (results_df['trial'] <= 21)]['pre_reward_hole_checks']
    
    if len(ko_b_early) > 0 and len(ko_b_late) > 0:
        t_stat_ko_b, p_val_ko_b = stats.ttest_ind(ko_b_early, ko_b_late)
        print(f"KO Target B - Early (16-18): {ko_b_early.mean():.2f} ± {stats.sem(ko_b_early):.2f}")
        print(f"KO Target B - Late (19-21): {ko_b_late.mean():.2f} ± {stats.sem(ko_b_late):.2f}")
        print(f"  Improvement: {((ko_b_early.mean() - ko_b_late.mean()) / ko_b_early.mean() * 100):+.1f}%")
        print(f"  T-test: t={t_stat_ko_b:.3f}, p={p_val_ko_b:.3f}")
    
    wt_b_early = results_df[(results_df['group'] == 'WT') & (results_df['trial'] >= 16) & (results_df['trial'] <= 18)]['pre_reward_hole_checks']
    wt_b_late = results_df[(results_df['group'] == 'WT') & (results_df['trial'] >= 19) & (results_df['trial'] <= 21)]['pre_reward_hole_checks']
    
    if len(wt_b_early) > 0 and len(wt_b_late) > 0:
        t_stat_wt_b, p_val_wt_b = stats.ttest_ind(wt_b_early, wt_b_late)
        print(f"WT Target B - Early (16-18): {wt_b_early.mean():.2f} ± {stats.sem(wt_b_early):.2f}")
        print(f"WT Target B - Late (19-21): {wt_b_late.mean():.2f} ± {stats.sem(wt_b_late):.2f}")
        print(f"  Improvement: {((wt_b_early.mean() - wt_b_late.mean()) / wt_b_early.mean() * 100):+.1f}%")
        print(f"  T-test: t={t_stat_wt_b:.3f}, p={p_val_wt_b:.3f}")

def main_pre_reward_analysis():
    """
    Main function to run the complete pre-reward hole checking analysis
    """
    
    experiment_date = '2025-08-22'
    mice_list = ['60', '61', '62', '63', '64', '65', '66', '67']
    time_window = 5.0  # 5 seconds before reward
    
    print(f"Analyzing pre-reward hole checking behavior...")
    print(f"Time window: {time_window} seconds before reward")
    print(f"Target A training: Trials 1-15")
    print(f"Target B training: Trials 16-21")
    
    # Run analysis
    results_df, summary_stats = analyze_pre_reward_behavior_all_mice(
        experiment_date, mice_list, (1, 22), time_window)
    
    print(f"\nAnalysis complete. Processed {len(results_df)} trials.")
    
    # Display summary statistics
    print(f"\nSummary Statistics:")
    print(summary_stats)
    
    # Create visualizations
    plot_pre_reward_analysis(results_df, time_window, savefig=True)
    
    # Perform statistical tests
    statistical_analysis(results_df)
    
    return results_df, summary_stats

# Run the analysis
if __name__ == "__main__":
    results, summary = main_pre_reward_analysis()