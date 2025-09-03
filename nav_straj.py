import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import modules.lib_process_data_to_mat as plib
import modules.lib_plot_mouse_trajectory as pltlib
import scipy.stats as stats
from scipy.spatial.distance import euclidean

def calculate_path_metrics(data):
    """
    Calculate various path strategy metrics
    
    Returns:
    - optimal_distance: straight-line distance from start to target
    - actual_distance: total distance traveled to target
    - path_efficiency: optimal/actual distance ratio
    - wall_preference: time spent near walls
    - path_strategy: classification of navigation strategy
    """
    
    # Get coordinates up to target
    if hasattr(data, 'r_nose') and len(data.r_nose) > 1:
        coords = data.r_nose
    else:
        coords = data.r_center
    
    try:
        target_idx = pltlib.coords_to_target(coords, data.target)
        coords_to_target = coords[:target_idx+1]
    except:
        coords_to_target = coords
    
    # Remove NaN values
    valid_idx = ~(np.isnan(coords_to_target[:, 0]) | np.isnan(coords_to_target[:, 1]))
    coords_clean = coords_to_target[valid_idx]
    
    if len(coords_clean) < 2:
        return None
    
    # Calculate optimal (straight-line) distance
    start_point = coords_clean[0]
    target_point = data.target
    optimal_distance = euclidean(start_point, target_point)
    
    # Calculate actual path distance
    actual_distance = 0
    for i in range(1, len(coords_clean)):
        actual_distance += euclidean(coords_clean[i-1], coords_clean[i])
    
    # Path efficiency
    path_efficiency = optimal_distance / actual_distance if actual_distance > 0 else 0
    
    # Wall proximity analysis (assume circular arena)
    arena_center = data.arena_circle[:2]  # x, y center
    arena_radius = data.arena_circle[2]   # radius
    
    # Calculate distance from arena center for each point
    distances_from_center = []
    for coord in coords_clean:
        dist_from_center = euclidean(coord, arena_center)
        distances_from_center.append(dist_from_center)
    
    distances_from_center = np.array(distances_from_center)
    
    # Define "wall zone" as outer 20% of arena radius
    wall_threshold = arena_radius * 0.8
    time_near_wall = np.sum(distances_from_center > wall_threshold) / len(distances_from_center)
    
    # Classify path strategy
    path_strategy = classify_path_strategy(coords_clean, start_point, target_point, 
                                         arena_center, arena_radius)
    
    return {
        'optimal_distance': optimal_distance,
        'actual_distance': actual_distance,
        'path_efficiency': path_efficiency,
        'time_near_wall': time_near_wall,
        'path_strategy': path_strategy,
        'coords_clean': coords_clean
    }

def classify_path_strategy(coords, start_point, target_point, arena_center, arena_radius):
    """
    Classify navigation strategy based on path characteristics
    """
    
    if len(coords) < 3:
        return 'insufficient_data'
    
    # Calculate wall proximity throughout path
    wall_threshold = arena_radius * 0.8
    distances_from_center = np.array([euclidean(coord, arena_center) for coord in coords])
    proportion_near_wall = np.sum(distances_from_center > wall_threshold) / len(distances_from_center)
    
    # Calculate path directness (how often mouse moves toward target)
    target_direction_consistency = calculate_target_direction_consistency(coords, target_point)
    
    # Classify strategy
    if proportion_near_wall > 0.7:  # >70% of time near wall
        if target_direction_consistency > 0.6:
            return 'wall_following_direct'  # Follows wall but generally toward target
        else:
            return 'wall_following_exploratory'  # Follows wall with lots of exploration
    elif proportion_near_wall < 0.3:  # <30% of time near wall
        if target_direction_consistency > 0.7:
            return 'direct_central'  # Direct path through center
        else:
            return 'central_exploratory'  # Explores center area
    else:
        return 'mixed_strategy'  # Uses both wall and center

def calculate_target_direction_consistency(coords, target_point):
    """
    Calculate how consistently the mouse moves toward the target
    """
    if len(coords) < 3:
        return 0
    
    toward_target_moves = 0
    total_moves = 0
    
    for i in range(1, len(coords)):
        # Current distance to target
        current_dist = euclidean(coords[i], target_point)
        # Previous distance to target
        previous_dist = euclidean(coords[i-1], target_point)
        
        # If mouse moved closer to target
        if current_dist < previous_dist:
            toward_target_moves += 1
        
        total_moves += 1
    
    return toward_target_moves / total_moves if total_moves > 0 else 0

def analyze_thigmotaxis_all_mice(experiment_date, mice_list, trial_range=(10, 16)):
    """
    Analyze thigmotaxis and path strategies for all mice (trials 10-15)
    """
    
    results = []
    
    for mouse_id in mice_list:
        print(f"Analyzing mouse {mouse_id}...")
        
        for trial_num in range(trial_range[0], trial_range[1]):
            try:
                # Load trial data
                data = plib.TrialData()
                data.Load(experiment_date, mouse_id, str(trial_num))
                
                # Calculate path metrics
                metrics = calculate_path_metrics(data)
                
                if metrics is not None:
                    # Determine mouse group
                    group = 'KO' if mouse_id in ['60', '61', '62', '63'] else 'WT'
                    
                    results.append({
                        'mouse_id': mouse_id,
                        'trial': trial_num,
                        'group': group,
                        'optimal_distance': metrics['optimal_distance'],
                        'actual_distance': metrics['actual_distance'],
                        'path_efficiency': metrics['path_efficiency'],
                        'time_near_wall': metrics['time_near_wall'],
                        'path_strategy': metrics['path_strategy']
                    })
                    
                    print(f"  Trial {trial_num}: {metrics['path_strategy']}, efficiency={metrics['path_efficiency']:.3f}")
                
            except Exception as e:
                print(f"  Trial {trial_num}: Error - {e}")
                continue
    
    return pd.DataFrame(results)

def plot_thigmotaxis_analysis(results_df):
    """
    Create comprehensive visualizations of thigmotaxis behavior
    """
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
    
    # Separate groups
    ko_data = results_df[results_df['group'] == 'KO']
    wt_data = results_df[results_df['group'] == 'WT']
    
    # Plot 1: Path Efficiency
    ko_efficiency = ko_data['path_efficiency'].values
    wt_efficiency = wt_data['path_efficiency'].values
    
    ko_x = np.random.normal(1, 0.05, len(ko_efficiency))
    wt_x = np.random.normal(2, 0.05, len(wt_efficiency))
    
    ax1.scatter(ko_x, ko_efficiency, alpha=0.7, s=60, color='#306ed1', label='KO')
    ax1.scatter(wt_x, wt_efficiency, alpha=0.7, s=60, color='black', label='WT')
    
    # Group means
    ko_mean = np.mean(ko_efficiency)
    wt_mean = np.mean(wt_efficiency)
    ko_sem = stats.sem(ko_efficiency)
    wt_sem = stats.sem(wt_efficiency)
    
    ax1.errorbar(1, ko_mean, yerr=ko_sem, fmt='o', markersize=10, color='#306ed1',
                capsize=5, capthick=2, elinewidth=2, markeredgecolor='white')
    ax1.errorbar(2, wt_mean, yerr=wt_sem, fmt='o', markersize=10, color='black',
                capsize=5, capthick=2, elinewidth=2, markeredgecolor='white')
    
    # Statistics
    t_stat, p_value = stats.ttest_ind(ko_efficiency, wt_efficiency)
    sig_text = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else f"p={p_value:.3f}"
    
    y_max = max(max(ko_efficiency), max(wt_efficiency))
    ax1.plot([1, 2], [y_max*1.05, y_max*1.05], 'k-', linewidth=1)
    ax1.text(1.5, y_max*1.07, sig_text, ha='center', va='bottom')
    
    ax1.set_xlim(0.5, 2.5)
    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(['KO', 'WT'])
    ax1.set_ylabel('Path Efficiency\n(Optimal Distance / Actual Distance)')
    ax1.set_title('Navigation Efficiency')
    ax1.legend()
    # ax1.grid(True, alpha=0.3)
    
    # Plot 2: Time Near Wall (Thigmotaxis)
    ko_wall = ko_data['time_near_wall'].values
    wt_wall = wt_data['time_near_wall'].values
    
    ko_x = np.random.normal(1, 0.05, len(ko_wall))
    wt_x = np.random.normal(2, 0.05, len(wt_wall))
    
    ax2.scatter(ko_x, ko_wall, alpha=0.7, s=60, color='#306ed1', label='KO')
    ax2.scatter(wt_x, wt_wall, alpha=0.7, s=60, color='black', label='WT')
    
    # Group means
    ko_wall_mean = np.mean(ko_wall)
    wt_wall_mean = np.mean(wt_wall)
    ko_wall_sem = stats.sem(ko_wall)
    wt_wall_sem = stats.sem(wt_wall)
    
    ax2.errorbar(1, ko_wall_mean, yerr=ko_wall_sem, fmt='o', markersize=10, color='#306ed1',
                capsize=5, capthick=2, elinewidth=2, markeredgecolor='white')
    ax2.errorbar(2, wt_wall_mean, yerr=wt_wall_sem, fmt='o', markersize=10, color='black',
                capsize=5, capthick=2, elinewidth=2, markeredgecolor='white')
    
    # Statistics
    t_stat, p_value = stats.ttest_ind(ko_wall, wt_wall)
    sig_text = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else f"p={p_value:.3f}"
    
    y_max = max(max(ko_wall), max(wt_wall))
    ax2.plot([1, 2], [y_max*1.05, y_max*1.05], 'k-', linewidth=1)
    ax2.text(1.5, y_max*1.07, sig_text, ha='center', va='bottom')
    
    ax2.set_xlim(0.5, 2.5)
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(['KO', 'WT'])
    ax2.set_ylabel('Proportion of Time Near Wall\n(Thigmotaxis Index)')
    ax2.set_title('Wall-Seeking Behavior')
    ax2.legend()
    # ax2.grid(True, alpha=0.3)
    
    # Plot 3: Path Strategy Distribution
    strategy_counts = results_df.groupby(['group', 'path_strategy']).size().unstack(fill_value=0)
    strategy_props = strategy_counts.div(strategy_counts.sum(axis=1), axis=0)
    
    strategy_props.plot(kind='bar', ax=ax3, width=0.7)
    ax3.set_title('Navigation Strategy Distribution')
    ax3.set_xlabel('Group')
    ax3.set_ylabel('Proportion of Trials')
    ax3.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.tick_params(axis='x', rotation=0)
    
    # Plot 4: Efficiency vs Wall Time Relationship
    colors = ['#306ed1' if g == 'KO' else 'black' for g in results_df['group']]
    
    ax4.scatter(results_df['time_near_wall'], results_df['path_efficiency'], 
               c=colors, alpha=0.6, s=40)
    
    # Add trendline
    x_vals = results_df['time_near_wall'].values
    y_vals = results_df['path_efficiency'].values
    z = np.polyfit(x_vals, y_vals, 1)
    p = np.poly1d(z)
    ax4.plot(x_vals, p(x_vals), "r--", alpha=0.8)
    
    # Correlation
    r, p_val = stats.pearsonr(x_vals, y_vals)
    ax4.text(0.05, 0.95, f'r = {r:.3f}\np = {p_val:.3f}', 
            transform=ax4.transAxes, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax4.set_xlabel('Time Near Wall')
    ax4.set_ylabel('Path Efficiency')
    ax4.set_title('Thigmotaxis vs Efficiency')
    ax4.grid(True, alpha=0.3)
    
    # Add legend for colors
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#306ed1', markersize=8, label='KO'),
                      Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='WT')]
    ax4.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.show()
    
    return strategy_counts

def plot_example_paths(results_df, experiment_date):
    """
    Plot example paths showing different navigation strategies
    """
    
    # Find examples of different strategies
    strategies_to_show = ['direct_central', 'wall_following_direct', 'wall_following_exploratory', 'mixed_strategy']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, strategy in enumerate(strategies_to_show):
        strategy_examples = results_df[results_df['path_strategy'] == strategy]
        
        if len(strategy_examples) > 0:
            # Pick a random example
            example = strategy_examples.sample(1).iloc[0]
            mouse_id = example['mouse_id']
            trial = example['trial']
            group = example['group']
            
            try:
                # Load and plot the path
                data = plib.TrialData()
                data.Load(experiment_date, mouse_id, str(trial))
                
                coords = data.r_nose if hasattr(data, 'r_nose') else data.r_center
                try:
                    target_idx = pltlib.coords_to_target(coords, data.target)
                    coords_to_plot = coords[:target_idx+1]
                except:
                    coords_to_plot = coords
                
                valid_idx = ~(np.isnan(coords_to_plot[:, 0]) | np.isnan(coords_to_plot[:, 1]))
                coords_clean = coords_to_plot[valid_idx]
                
                # Plot arena
                arena_circle = plt.Circle(data.arena_circle[:2], data.arena_circle[2], 
                                        fill=False, color='black', linewidth=2)
                axes[i].add_patch(arena_circle)
                
                # Plot wall zone
                wall_circle = plt.Circle(data.arena_circle[:2], data.arena_circle[2] * 0.8, 
                                       fill=False, color='gray', linewidth=1, linestyle='--', alpha=0.5)
                axes[i].add_patch(wall_circle)
                
                # Plot path
                axes[i].plot(coords_clean[:, 0], coords_clean[:, 1], 'b-', alpha=0.7, linewidth=2)
                
                # Plot start and target
                axes[i].plot(coords_clean[0, 0], coords_clean[0, 1], 'go', markersize=10, label='Start')
                axes[i].plot(data.target[0], data.target[1], 'rs', markersize=10, label='Target')
                
                # Plot optimal path
                axes[i].plot([coords_clean[0, 0], data.target[0]], 
                           [coords_clean[0, 1], data.target[1]], 
                           'r--', alpha=0.5, linewidth=1, label='Optimal')
                
                axes[i].set_aspect('equal')
                axes[i].set_title(f'{strategy.replace("_", " ").title()}\nMouse {mouse_id} ({group}) - Trial {trial}')
                axes[i].legend()
                
            except Exception as e:
                axes[i].text(0.5, 0.5, f'No example\navailable', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{strategy.replace("_", " ").title()}')
    
    plt.tight_layout()
    plt.show()

def print_thigmotaxis_summary(results_df):
    """
    Print summary statistics for thigmotaxis analysis
    """
    
    print("\n" + "="*60)
    print("THIGMOTAXIS AND PATH STRATEGY ANALYSIS")
    print("="*60)
    
    ko_data = results_df[results_df['group'] == 'KO']
    wt_data = results_df[results_df['group'] == 'WT']
    
    # Path efficiency comparison
    ko_efficiency = ko_data['path_efficiency'].values
    wt_efficiency = wt_data['path_efficiency'].values
    
    print(f"\nPath Efficiency:")
    print(f"KO mice: {np.mean(ko_efficiency):.3f} ± {stats.sem(ko_efficiency):.3f}")
    print(f"WT mice: {np.mean(wt_efficiency):.3f} ± {stats.sem(wt_efficiency):.3f}")
    
    t_stat, p_value = stats.ttest_ind(ko_efficiency, wt_efficiency)
    print(f"T-test: t={t_stat:.3f}, p={p_value:.3f}")
    
    # Thigmotaxis comparison
    ko_wall = ko_data['time_near_wall'].values
    wt_wall = wt_data['time_near_wall'].values
    
    print(f"\nThigmotaxis (Time Near Wall):")
    print(f"KO mice: {np.mean(ko_wall):.3f} ± {stats.sem(ko_wall):.3f}")
    print(f"WT mice: {np.mean(wt_wall):.3f} ± {stats.sem(wt_wall):.3f}")
    
    t_stat, p_value = stats.ttest_ind(ko_wall, wt_wall)
    print(f"T-test: t={t_stat:.3f}, p={p_value:.3f}")
    
    # Strategy distribution
    print(f"\nNavigation Strategy Distribution:")
    strategy_counts = results_df.groupby(['group', 'path_strategy']).size().unstack(fill_value=0)
    strategy_props = strategy_counts.div(strategy_counts.sum(axis=1), axis=0) * 100
    
    for strategy in strategy_props.columns:
        ko_prop = strategy_props.loc['KO', strategy] if 'KO' in strategy_props.index else 0
        wt_prop = strategy_props.loc['WT', strategy] if 'WT' in strategy_props.index else 0
        print(f"  {strategy.replace('_', ' ').title()}: KO {ko_prop:.1f}%, WT {wt_prop:.1f}%")
    
    return strategy_counts

def main_thigmotaxis_analysis():
    """
    Main function to run complete thigmotaxis analysis for trials 10-15
    """
    
    experiment_date = '2025-08-22'
    mice_list = ['60', '61', '62', '63', '64', '65', '66', '67']
    trial_range = (10, 16)  # Trials 10-15
    
    print("Analyzing thigmotaxis and navigation strategies...")
    print(f"Trials: {trial_range[0]}-{trial_range[1]-1}")
    
    # Analyze all mice
    results_df = analyze_thigmotaxis_all_mice(experiment_date, mice_list, trial_range)
    
    print(f"\nAnalysis complete. Processed {len(results_df)} trials.")
    
    # Create visualizations
    strategy_counts = plot_thigmotaxis_analysis(results_df)
    
    # Show example paths
    plot_example_paths(results_df, experiment_date)
    
    # Print summary
    print_thigmotaxis_summary(results_df)
    
    return results_df

# Run the analysis
if __name__ == "__main__":
    results = main_thigmotaxis_analysis()