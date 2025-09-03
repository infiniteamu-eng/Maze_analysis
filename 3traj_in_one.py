import modules.lib_process_data_to_mat as plib
import modules.lib_plot_mouse_trajectory as pltlib
import matplotlib.pyplot as plt
import numpy as np

# Define mouse groups
KO_mice = ['60', '61', '62', '63']
WT_mice = ['64', '65', '66', '67']

def plot_simple_trajectories(mouse_id, trials=[11, 12, 13, 14]):
    """Plot only the trajectory visualization for specified trials"""
    
    fig, ax = plt.subplots(figsize=(6, 4))
    group = 'KO' if mouse_id in KO_mice else 'WT'
    
    # Generate colors for each trial
    colors = plt.cm.viridis(np.linspace(0, 1, len(trials)))
    
    valid_trials = 0
    
    for i, trial in enumerate(trials):
        try:
            # Load trial data
            d = plib.TrialData()
            d.Load('2025-08-22', mouse_id, str(trial))
            
            coords = d.r_nose
            valid_idx = ~(np.isnan(coords[:, 0]) | np.isnan(coords[:, 1]))
            coords_clean = coords[valid_idx]
            
            if len(coords_clean) < 2:
                print(f"Skipping trial {trial} - insufficient data")
                continue
            
            valid_trials += 1
            
            # Plot trajectory
            ax.plot(coords_clean[:, 0], coords_clean[:, 1], 
                   color=colors[i], alpha=0.7, linewidth=2, 
                   label=f'Trial {trial}')
            
            # Plot optimal path (dashed line from start to target)
            start_pos = coords_clean[0]
            try:
                target_idx = pltlib.coords_to_target(coords, d.target)
                end_pos = coords[target_idx]
            except:
                end_pos = coords_clean[-1]  # Use last position if target not found
            
            ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                   '--', color=colors[i], alpha=0.5, linewidth=1)
            
            # Mark start and target for first trial only
            if i == 0:
                ax.scatter(start_pos[0], start_pos[1], s=100, color='green', 
                          marker='o', edgecolor='black', label='Start', zorder=5)
                ax.scatter(end_pos[0], end_pos[1], s=100, color='red', 
                          marker='s', edgecolor='black', label='Target A', zorder=5)
        
        except Exception as e:
            print(f"Error loading trial {trial} for mouse {mouse_id}: {e}")
            continue
    
    if valid_trials == 0:
        print(f"No valid trials found for mouse {mouse_id}")
        plt.close(fig)
        return
    
    # Formatting
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title(f'Mouse {mouse_id} ({group}) - Trials {trials[0]}-{trials[-1]} Trajectories', 
                fontsize=14, fontweight='bold')
    # ax.set_xlabel('X position (cm)')
    # ax.set_ylabel('Y position (cm)')
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)  # Remove x-axis line
    ax.spines['left'].set_visible(False)   
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()

# Run analysis for specific mice
if __name__ == "__main__":
    print("Plotting trajectory visualizations...")
    
    # Plot examples from each group
    plot_simple_trajectories('60', trials=[1, 2, 3])  # KO example
    plot_simple_trajectories('61', trials=[1, 2, 3])  # KO example
    plot_simple_trajectories('62', trials=[1, 2, 3])  # KO example
    plot_simple_trajectories('63', trials=[1, 2, 3])  # KO example
    plot_simple_trajectories('64', trials=[1, 2, 3])  # WT example
    plot_simple_trajectories('67', trials=[1, 2, 3])  # WT example
    plot_simple_trajectories('66', trials=[1, 2, 3])  # WT example
    plot_simple_trajectories('65', trials=[1, 2, 3])  # WT example

    
    # You can also plot individual trials or different trial ranges
    # plot_simple_trajectories('60', trials=[10, 11, 12])
    # plot_simple_trajectories('60', trials=[13, 14, 15])