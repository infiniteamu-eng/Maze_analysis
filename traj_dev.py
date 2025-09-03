#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trajectory Deviation Analysis - Simplified Version
Measures how much mouse trajectories deviate from the optimal straight-line path
Focus on Path Efficiency and Mean Deviation only
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

def calculate_trajectory_deviation_metrics(coords, start_point, target_point):
    """
    Calculate trajectory deviation metrics - simplified version
    
    Parameters:
    coords: trajectory coordinates (Nx2)
    start_point: starting position (2,)
    target_point: target position (2,)
    
    Returns:
    dict with deviation metrics
    """
    # Remove NaN values
    valid_idx = ~(np.isnan(coords[:, 0]) | np.isnan(coords[:, 1]))
    coords_clean = coords[valid_idx]
    
    if len(coords_clean) < 3:
        return {
            'path_efficiency': np.nan,
            'mean_deviation': np.nan,
            'optimal_distance': np.nan,
            'actual_path_length': np.nan
        }
    
    # Calculate optimal (straight-line) distance
    optimal_distance = np.sqrt(np.sum((target_point - start_point)**2))
    
    # Calculate actual path length
    path_segments = np.diff(coords_clean, axis=0)
    segment_lengths = np.sqrt(np.sum(path_segments**2, axis=1))
    actual_path_length = np.sum(segment_lengths)
    
    # Metric 1: Path Efficiency (0-1, where 1 is optimal)
    path_efficiency = optimal_distance / actual_path_length if actual_path_length > 0 else 0
    
    # Metric 2: Point-to-line deviations
    def point_to_line_distance(point, line_start, line_end):
        """Calculate perpendicular distance from point to line"""
        line_vec = line_end - line_start
        line_length = np.sqrt(np.sum(line_vec**2))
        
        if line_length == 0:
            return np.sqrt(np.sum((point - line_start)**2))
        
        line_unit = line_vec / line_length
        point_vec = point - line_start
        projection_length = np.dot(point_vec, line_unit)
        projection_point = line_start + projection_length * line_unit
        deviation = np.sqrt(np.sum((point - projection_point)**2))
        return deviation
    
    # Calculate deviation for each trajectory point
    deviations = []
    for point in coords_clean:
        dev = point_to_line_distance(point, start_point, target_point)
        deviations.append(dev)
    
    deviations = np.array(deviations)
    mean_deviation = np.mean(deviations)
    
    return {
        'path_efficiency': path_efficiency,
        'mean_deviation': mean_deviation,
        'optimal_distance': optimal_distance,
        'actual_path_length': actual_path_length
    }

def analyze_mouse_trajectory_deviation(mouse_id, trials=range(11, 16)):
    """Analyze trajectory deviation for a single mouse across specified trials"""
    
    results = []
    
    for trial in trials:
        try:
            # Load trial data
            d = plib.TrialData()
            d.Load('2025-08-22', mouse_id, str(trial))
            
            # Get coordinates up to target
            coords = d.r_nose
            start_point = coords[0]  # First valid coordinate
            target_point = d.target
            
            try:
                target_idx = pltlib.coords_to_target(coords, target_point)
                coords_to_analyze = coords[:target_idx+1]
            except:
                coords_to_analyze = coords
            
            # Calculate deviation metrics
            metrics = calculate_trajectory_deviation_metrics(coords_to_analyze, start_point, target_point)
            
            # Add mouse and trial info
            metrics['mouse_id'] = mouse_id
            metrics['trial'] = trial
            
            results.append(metrics)
            
        except Exception as e:
            print(f"Error analyzing mouse {mouse_id}, trial {trial}: {e}")
            continue
    
    return results

def plot_deviation_comparison(trials=range(11, 16)):
    """Compare trajectory deviation metrics between KO and WT groups"""
    
    # Collect data for all mice
    all_results = []
    
    for mouse_id in KO_mice + WT_mice:
        mouse_results = analyze_mouse_trajectory_deviation(mouse_id, trials)
        all_results.extend(mouse_results)
    
    if not all_results:
        print("No data available for deviation analysis")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Add group information
    df['group'] = df['mouse_id'].apply(lambda x: 'KO' if x in KO_mice else 'WT')
    
    # Calculate mean metrics per mouse
    mouse_summary = df.groupby(['mouse_id', 'group']).agg({
        'path_efficiency': 'mean',
        'mean_deviation': 'mean'
    }).reset_index()
    
    print("Trajectory Deviation Analysis Results:")
    print(mouse_summary[['mouse_id', 'group', 'path_efficiency', 'mean_deviation']])
    
    # Create comparison plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    
    deviation_metrics = [
        ('path_efficiency', 'Path Efficiency\n(1 = optimal)', 'higher is better', ax1),
        ('mean_deviation', 'Mean Deviation\n(distance from optimal path)', 'lower is better', ax2)
    ]
    
    for i, (metric, title, interpretation, ax) in enumerate(deviation_metrics):
        
        ko_data = mouse_summary[mouse_summary['group'] == 'KO'][metric].dropna().values
        wt_data = mouse_summary[mouse_summary['group'] == 'WT'][metric].dropna().values
        
        if len(ko_data) == 0 or len(wt_data) == 0:
            ax.text(0.5, 0.5, f'No data available\\nfor {title}', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Individual points with jitter
        ko_x = np.random.normal(1, 0.05, len(ko_data))
        wt_x = np.random.normal(2, 0.05, len(wt_data))
        
        ax.scatter(ko_x, ko_data, alpha=0.7, s=80, color='#306ed1', 
                  label=f'KO (n={len(ko_data)})', edgecolor='white', linewidth=1)
        ax.scatter(wt_x, wt_data, alpha=0.7, s=80, color='black', 
                  label=f'WT (n={len(wt_data)})', edgecolor='white', linewidth=1)
        
        # Group means with error bars
        ko_mean, wt_mean = np.mean(ko_data), np.mean(wt_data)
        ko_sem, wt_sem = stats.sem(ko_data), stats.sem(wt_data)
        
        ax.errorbar(1, ko_mean, yerr=ko_sem, fmt='D', markersize=10, color='#306ed1',
                   capsize=6, capthick=3, elinewidth=3, markeredgecolor='white', markeredgewidth=2)
        ax.errorbar(2, wt_mean, yerr=wt_sem, fmt='D', markersize=10, color='black',
                   capsize=6, capthick=3, elinewidth=3, markeredgecolor='white', markeredgewidth=2)
        
        # Statistical comparison (only if sufficient sample sizes)
        if len(ko_data) >= 2 and len(wt_data) >= 2:
            t_stat, p_value = stats.ttest_ind(ko_data, wt_data)
            
            if p_value < 0.001:
                sig_text = "***"
            elif p_value < 0.01:
                sig_text = "**"
            elif p_value < 0.05:
                sig_text = "*"
            else:
                sig_text = f"p={p_value:.3f}"
            
            # Add significance bar
            y_max = max(max(ko_data), max(wt_data))
            y_min = min(min(ko_data), min(wt_data))
            y_range = y_max - y_min if y_max != y_min else 1
            bar_height = y_max + 0.1 * y_range
            
            ax.plot([1, 2], [bar_height, bar_height], 'k-', linewidth=2)
            ax.plot([1, 1], [bar_height, bar_height - 0.03 * y_range], 'k-', linewidth=2)
            ax.plot([2, 2], [bar_height, bar_height - 0.03 * y_range], 'k-', linewidth=2)
            ax.text(1.5, bar_height + 0.02 * y_range, sig_text, ha='center', va='bottom', 
                   fontsize=12, fontweight='bold')
        else:
            y_max = max(max(ko_data) if len(ko_data) > 0 else [0], 
                       max(wt_data) if len(wt_data) > 0 else [0])
            ax.text(1.5, y_max, 'insufficient n', ha='center', va='bottom', 
                   fontsize=10, color='gray')
        
        # Formatting
        ax.set_xlim(0.5, 2.5)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['KO', 'WT'])
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{title}\\n({interpretation})')
        # ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper right')
    
    plt.suptitle(f'Trajectory Deviation Analysis - Navigation Efficiency (Trials {trials[0]}-{trials[-1]})', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Print detailed summary statistics
    print(f"\\nDetailed Trajectory Deviation Summary (Trials {trials[0]}-{trials[-1]}):")
    print("="*80)
    
    for metric, title, interpretation, _ in deviation_metrics:
        ko_data = mouse_summary[mouse_summary['group'] == 'KO'][metric].dropna().values
        wt_data = mouse_summary[mouse_summary['group'] == 'WT'][metric].dropna().values
        
        if len(ko_data) > 0 and len(wt_data) > 0:
            print(f"\\n{title} ({interpretation}):")
            print(f"  KO (n={len(ko_data)}): {np.mean(ko_data):.3f} ± {stats.sem(ko_data):.3f}")
            print(f"  WT (n={len(wt_data)}): {np.mean(wt_data):.3f} ± {stats.sem(wt_data):.3f}")
            
            if len(ko_data) >= 2 and len(wt_data) >= 2:
                t_stat, p_value = stats.ttest_ind(ko_data, wt_data)
                cohen_d = (np.mean(ko_data) - np.mean(wt_data)) / np.sqrt((np.std(ko_data)**2 + np.std(wt_data)**2) / 2)
                
                print(f"  Statistical test: t={t_stat:.3f}, p={p_value:.3f}, Cohen's d={cohen_d:.3f}")
                
                if interpretation == 'higher is better':
                    if np.mean(ko_data) > np.mean(wt_data):
                        print(f"  → KO mice show better performance")
                    else:
                        print(f"  → WT mice show better performance")
                else:  # lower is better
                    if np.mean(ko_data) < np.mean(wt_data):
                        print(f"  → KO mice show better performance")
                    else:
                        print(f"  → WT mice show better performance")
            else:
                print(f"  Statistical test: Not performed (insufficient sample size)")

def plot_learning_progression_comparison():
    """Compare trajectory deviation between early (1-4) and late (11-15) trials"""
    
    early_trials = range(1, 5)
    late_trials = range(11, 16)
    
    # Collect data for both trial phases
    all_results_early = []
    all_results_late = []
    
    print("Collecting early trial data (1-4)...")
    for mouse_id in KO_mice + WT_mice:
        mouse_results_early = analyze_mouse_trajectory_deviation(mouse_id, early_trials)
        all_results_early.extend(mouse_results_early)
    
    print("Collecting late trial data (11-15)...")
    for mouse_id in KO_mice + WT_mice:
        mouse_results_late = analyze_mouse_trajectory_deviation(mouse_id, late_trials)
        all_results_late.extend(mouse_results_late)
    
    if not all_results_early or not all_results_late:
        print("Insufficient data for learning progression analysis")
        return
    
    # Convert to DataFrames
    df_early = pd.DataFrame(all_results_early)
    df_late = pd.DataFrame(all_results_late)
    
    # Add group information
    df_early['group'] = df_early['mouse_id'].apply(lambda x: 'KO' if x in KO_mice else 'WT')
    df_late['group'] = df_late['mouse_id'].apply(lambda x: 'KO' if x in KO_mice else 'WT')
    
    # Calculate mean metrics per mouse for each phase
    early_summary = df_early.groupby(['mouse_id', 'group']).agg({
        'path_efficiency': 'mean',
        'mean_deviation': 'mean'
    }).reset_index()
    
    late_summary = df_late.groupby(['mouse_id', 'group']).agg({
        'path_efficiency': 'mean',
        'mean_deviation': 'mean'
    }).reset_index()
    
    # Create learning progression plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))
    
    metrics_info = [
        ('path_efficiency', 'Path Efficiency','higher is better', ax1, ax3),
        ('mean_deviation', 'Mean Deviation', 'lower is better', ax2, ax4)
    ]
    
    for metric_idx, (metric, title, interpretation, ax_main, ax_curve) in enumerate(metrics_info):
        
        # Get data for main comparison plot
        ko_early = early_summary[early_summary['group'] == 'KO'][metric].dropna().values
        wt_early = early_summary[early_summary['group'] == 'WT'][metric].dropna().values
        ko_late = late_summary[late_summary['group'] == 'KO'][metric].dropna().values
        wt_late = late_summary[late_summary['group'] == 'WT'][metric].dropna().values
        
        if len(ko_early) == 0 or len(wt_early) == 0 or len(ko_late) == 0 or len(wt_late) == 0:
            ax_main.text(0.5, 0.5, f'Insufficient data for\\n{title}', 
                        ha='center', va='center', transform=ax_main.transAxes)
            ax_curve.text(0.5, 0.5, f'Insufficient data for\\n{title}', 
                         ha='center', va='center', transform=ax_curve.transAxes)
            continue
        
        # Main comparison plot: Early vs Late for each group
        x_positions = [1, 1.8, 3, 3.8]  # KO_early, KO_late, WT_early, WT_late
        data_groups = [ko_early, ko_late, wt_early, wt_late]
        colors = ['#306ed1', '#1e4a8c', 'gray', 'black']
        labels = [f'KO Early (n={len(ko_early)})', f'KO Late (n={len(ko_late)})', 
                 f'WT Early (n={len(wt_early)})', f'WT Late (n={len(wt_late)})']
        
        # Plot individual points with jitter
        for i, (x_pos, data, color, label) in enumerate(zip(x_positions, data_groups, colors, labels)):
            if len(data) > 0:
                x_jitter = np.random.normal(x_pos, 0.05, len(data))
                ax_main.scatter(x_jitter, data, alpha=0.6, s=60, color=color, 
                              edgecolor='white', linewidth=1)
                
                # Plot mean with error bar
                mean_val = np.mean(data)
                sem_val = stats.sem(data)
                ax_main.errorbar(x_pos, mean_val, yerr=sem_val, fmt='D', markersize=8, 
                               color=color, capsize=5, capthick=2, elinewidth=2, 
                               markeredgecolor='white', markeredgewidth=1)
        
        # Statistical comparisons
        # KO improvement (early vs late)
        if len(ko_early) >= 2 and len(ko_late) >= 2:
            try:
                t_stat_ko, p_val_ko = stats.ttest_ind(ko_early, ko_late)
                if p_val_ko < 0.05:
                    sig_text_ko = "***" if p_val_ko < 0.001 else "**" if p_val_ko < 0.01 else "*"
                    y_max = max(max(ko_early), max(ko_late))
                    ax_main.plot([1, 1.8], [y_max*1.05, y_max*1.05], 'b-', linewidth=1)
                    ax_main.text(1.4, y_max*1.07, sig_text_ko, ha='center', va='bottom', 
                                color='blue', fontsize=10, fontweight='bold')
            except:
                pass
        
        # WT improvement (early vs late)
        if len(wt_early) >= 2 and len(wt_late) >= 2:
            try:
                t_stat_wt, p_val_wt = stats.ttest_ind(wt_early, wt_late)
                if p_val_wt < 0.05:
                    sig_text_wt = "***" if p_val_wt < 0.001 else "**" if p_val_wt < 0.01 else "*"
                    y_max = max(max(wt_early), max(wt_late))
                    ax_main.plot([3, 3.8], [y_max*1.15, y_max*1.15], 'k-', linewidth=1)
                    ax_main.text(3.4, y_max*1.17, sig_text_wt, ha='center', va='bottom', 
                                color='black', fontsize=10, fontweight='bold')
            except:
                pass
        
        # Group difference in late trials
        if len(ko_late) >= 2 and len(wt_late) >= 2:
            try:
                t_stat_late, p_val_late = stats.ttest_ind(ko_late, wt_late)
                if p_val_late < 0.05:
                    sig_text_late = "***" if p_val_late < 0.001 else "**" if p_val_late < 0.01 else "*"
                    y_max = max(max(ko_late), max(wt_late))
                    ax_main.plot([1.8, 3.8], [y_max*1.25, y_max*1.25], 'r-', linewidth=2)
                    ax_main.text(2.8, y_max*1.27, sig_text_late, ha='center', va='bottom', 
                                color='red', fontsize=12, fontweight='bold')
            except:
                pass
        
        ax_main.set_xlim(0.5, 4.3)
        ax_main.set_xticks(x_positions)
        ax_main.set_xticklabels(['KO\\nEarly', 'KO\\nLate', 'WT\\nEarly', 'WT\\nLate'], fontsize=10)
        ax_main.set_ylabel(title)
        ax_main.set_title(f'{title} - Learning Progression\\n({interpretation})')
        # ax_main.grid(True, alpha=0.3)
        ax_main.spines['top'].set_visible(False)
        ax_main.spines['right'].set_visible(False)
        
        # Learning curves - show individual mouse trajectories
        for mouse_id in KO_mice + WT_mice:
            # Get data for this mouse across both phases
            mouse_early_data = df_early[df_early['mouse_id'] == mouse_id][metric].values
            mouse_late_data = df_late[df_late['mouse_id'] == mouse_id][metric].values
            
            if len(mouse_early_data) > 0 and len(mouse_late_data) > 0:
                early_mean = np.mean(mouse_early_data)
                late_mean = np.mean(mouse_late_data)
                
                color = '#306ed1' if mouse_id in KO_mice else 'black'
                alpha = 0.6
                
                ax_curve.plot([1, 2], [early_mean, late_mean], 'o-', color=color, 
                            alpha=alpha, linewidth=2, markersize=6)
        
        # Add group average lines
        if len(ko_early) > 0 and len(ko_late) > 0:
            ax_curve.plot([1, 2], [np.mean(ko_early), np.mean(ko_late)], 'o-', 
                         color='#306ed1', linewidth=4, markersize=10, 
                         label='KO Average', alpha=0.8)
        
        if len(wt_early) > 0 and len(wt_late) > 0:
            ax_curve.plot([1, 2], [np.mean(wt_early), np.mean(wt_late)], 'o-', 
                         color='black', linewidth=4, markersize=10, 
                         label='WT Average', alpha=0.8)
        
        ax_curve.set_xlim(0.8, 2.2)
        ax_curve.set_xticks([1, 2])
        ax_curve.set_xticklabels(['Early\\n(1-4)', 'Late\\n(11-15)'])
        ax_curve.set_ylabel(title)
        ax_curve.set_title(f'{title} - Individual Learning Curves')
        # ax_curve.grid(True, alpha=0.3)
        ax_curve.spines['top'].set_visible(False)
        ax_curve.spines['right'].set_visible(False)
        
        if metric_idx == 0:
            ax_curve.legend()
    
    plt.suptitle('Trajectory Deviation: Learning Progression Analysis\\n(Early Trials 1-4 vs Late Trials 11-15)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return early_summary, late_summary

def plot_example_trajectories_with_deviation(mouse_ids=['60', '66'], trial=15):
    """Plot example trajectories showing deviation from optimal path"""
    
    print(f"Attempting to plot trajectories for mice {mouse_ids}, trial {trial}")
    
    fig, axes = plt.subplots(1, len(mouse_ids), figsize=(6*len(mouse_ids), 5))
    if len(mouse_ids) == 1:
        axes = [axes]
    
    for i, mouse_id in enumerate(mouse_ids):
        ax = axes[i]
        
        try:
            print(f"Loading data for mouse {mouse_id}, trial {trial}")
            d = plib.TrialData()
            d.Load('2025-08-22', mouse_id, str(trial))
            
            coords = d.r_nose
            valid_idx = ~(np.isnan(coords[:, 0]) | np.isnan(coords[:, 1]))
            coords_clean = coords[valid_idx]
            
            if len(coords_clean) < 3:
                ax.text(0.5, 0.5, f'Insufficient data\\nMouse {mouse_id}\\nTrial {trial}', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            start_point = coords_clean[0]
            target_point = d.target
            group = 'KO' if mouse_id in KO_mice else 'WT'
            
            # Calculate metrics
            metrics = calculate_trajectory_deviation_metrics(coords_clean, start_point, target_point)
            
            # Plot optimal path
            ax.plot([start_point[0], target_point[0]], [start_point[1], target_point[1]], 
                   'r-', linewidth=4, alpha=0.8, label='Optimal Path')
            
            # Plot actual trajectory
            trajectory_color = '#306ed1' if group == 'KO' else 'black'
            ax.plot(coords_clean[:, 0], coords_clean[:, 1], color=trajectory_color, 
                   linewidth=2, alpha=0.8, label='Actual Path')
            
            # Mark start and target
            ax.scatter(start_point[0], start_point[1], color='green', s=150, marker='o', 
                      edgecolor='white', linewidth=2, label='Start', zorder=5)
            ax.scatter(target_point[0], target_point[1], color='red', s=150, marker='s', 
                      edgecolor='white', linewidth=2, label='Target', zorder=5)
            
            # Add deviation visualization
            n_deviation_lines = min(8, len(coords_clean))
            if n_deviation_lines > 2:
                indices = np.linspace(1, len(coords_clean)-2, n_deviation_lines-2, dtype=int)
                
                for idx in indices:
                    point = coords_clean[idx]
                    
                    # Find closest point on optimal line
                    line_vec = target_point - start_point
                    line_length = np.sqrt(np.sum(line_vec**2))
                    
                    if line_length > 0:
                        line_unit = line_vec / line_length
                        point_vec = point - start_point
                        projection_length = np.dot(point_vec, line_unit)
                        projection_point = start_point + projection_length * line_unit
                        
                        # Draw deviation line
                        ax.plot([point[0], projection_point[0]], [point[1], projection_point[1]], 
                               'gray', linewidth=1, alpha=0.6)
            
            ax.set_aspect('equal')
            ax.set_title(f'Mouse {mouse_id} ({group}) - Trial {trial}\\n'
                        f'Path Efficiency: {metrics["path_efficiency"]:.3f}\\n'
                        f'Mean Deviation: {metrics["mean_deviation"]:.1f}')
            ax.legend()
            # ax.grid(True, alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            print(f"Successfully plotted mouse {mouse_id}")
            
        except Exception as e:
            print(f"Error plotting trajectory for mouse {mouse_id}: {e}")
            ax.text(0.5, 0.5, f'Error loading data\\nMouse {mouse_id}\\nTrial {trial}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
            ax.set_title(f'Mouse {mouse_id} - Error')
    
    plt.suptitle('Example Trajectories with Deviation Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_early_vs_late_trajectory_examples():
    """Plot trajectory examples comparing early vs late trials for the same mice"""
    
    example_mice = ['60', '66']  # One KO, one WT
    early_trial = 4
    late_trial = 15
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for mouse_idx, mouse_id in enumerate(example_mice):
        group = 'KO' if mouse_id in KO_mice else 'WT'
        
        for trial_idx, (trial, trial_label) in enumerate([(early_trial, 'Early'), (late_trial, 'Late')]):
            ax = axes[mouse_idx, trial_idx]
            
            try:
                d = plib.TrialData()
                d.Load('2025-08-22', mouse_id, str(trial))
                
                coords = d.r_nose
                valid_idx = ~(np.isnan(coords[:, 0]) | np.isnan(coords[:, 1]))
                coords_clean = coords[valid_idx]
                
                if len(coords_clean) < 3:
                    ax.text(0.5, 0.5, f'Insufficient data\\nMouse {mouse_id}\\nTrial {trial}', 
                           ha='center', va='center', transform=ax.transAxes)
                    continue
                
                start_point = coords_clean[0]
                target_point = d.target
                
                # Calculate metrics
                metrics = calculate_trajectory_deviation_metrics(coords_clean, start_point, target_point)
                
                # Plot optimal path
                ax.plot([start_point[0], target_point[0]], [start_point[1], target_point[1]], 
                       'r-', linewidth=3, alpha=0.8, label='Optimal Path')
                
                # Plot actual trajectory
                trajectory_color = '#306ed1' if group == 'KO' else 'black'
                ax.plot(coords_clean[:, 0], coords_clean[:, 1], color=trajectory_color, 
                       linewidth=2, alpha=0.8, label=f'Actual Path')
                
                # Mark start and target
                ax.scatter(start_point[0], start_point[1], color='green', s=120, marker='o', 
                          edgecolor='white', linewidth=2, label='Start', zorder=5)
                ax.scatter(target_point[0], target_point[1], color='red', s=120, marker='s', 
                          edgecolor='white', linewidth=2, label='Target', zorder=5)
                
                # Add deviation visualization for some points
                n_deviation_lines = min(8, len(coords_clean))
                if n_deviation_lines > 2:
                    indices = np.linspace(1, len(coords_clean)-2, n_deviation_lines-2, dtype=int)
                    
                    for idx in indices:
                        point = coords_clean[idx]
                        
                        # Find closest point on optimal line
                        line_vec = target_point - start_point
                        line_length = np.sqrt(np.sum(line_vec**2))
                        
                        if line_length > 0:
                            line_unit = line_vec / line_length
                            point_vec = point - start_point
                            projection_length = np.dot(point_vec, line_unit)
                            projection_point = start_point + projection_length * line_unit
                            
                            # Draw deviation line
                            ax.plot([point[0], projection_point[0]], [point[1], projection_point[1]], 
                                   'gray', linewidth=1, alpha=0.5)
                
                ax.set_aspect('equal')
                ax.set_title(f'Mouse {mouse_id} ({group}) - {trial_label} Trial {trial}\\n'
                            f'Efficiency: {metrics["path_efficiency"]:.3f}, '
                            f'Mean Dev: {metrics["mean_deviation"]:.1f}')
                
                if mouse_idx == 0 and trial_idx == 0:
                    ax.legend(loc='upper right', fontsize=8)
                
                # ax.grid(True, alpha=0.3)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Error loading data\\nMouse {mouse_id}, Trial {trial}\\n{str(e)[:50]}...', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=8)
                print(f"Error plotting trajectory for mouse {mouse_id}, trial {trial}: {e}")
    
    plt.suptitle('Trajectory Examples: Early vs Late Trials\\nShowing Learning Progression', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Run trajectory deviation analysis
if __name__ == "__main__":
    print("Analyzing trajectory deviation from optimal path...")
    print("Focus: Path Efficiency and Mean Deviation only\\n")
    
    # Compare deviation metrics between groups for trials 11-15
    plot_deviation_comparison(trials=range(11, 16))
    
    # Learning progression analysis
    print("\\nAnalyzing learning progression...")
    early_data, late_data = plot_learning_progression_comparison()
    
    # Show example trajectories with more explicit display
    print("\\nGenerating example trajectory visualizations...")
    try:
        plot_example_trajectories_with_deviation(['60', '66'], trial=15)
        print("Example trajectory plots completed successfully.")
    except Exception as e:
        print(f"Error generating example trajectories: {e}")
        print("Trying alternative mice or trials...")
        try:
            plot_example_trajectories_with_deviation(['61', '64'], trial=12)
        except Exception as e2:
            print(f"Alternative attempt also failed: {e2}")
    
    # Additional summary plot for early vs late comparison
    print("\\nGenerating early vs late trajectory examples...")
    plot_early_vs_late_trajectory_examples()