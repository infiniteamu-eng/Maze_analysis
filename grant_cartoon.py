import modules.lib_process_data_to_mat as plib
import modules.lib_plot_mouse_trajectory as pltlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import os
from modules.config import ROOT_DIR
import numpy as np

# Load trial to get exact coordinates (using your overlay example)
trial = plib.TrialData()
trial.Load('2025-08-22', '66', '4')

print('Mouse %s Trial %s' % (trial.mouse_number, trial.trial))

# Create the plot using your exact layout
fig, ax = plt.subplots(figsize=(6, 4))

# Set up the background and arena exactly like your overlay
if hasattr(trial, 'arena_circle'):
    pltlib.draw_arena(trial, ax)
else:
    print('Missing arena circle coordinates')
    # Import and display background image
    img = mpimg.imread(os.path.join(ROOT_DIR, 'data', 'BackgroundImage', trial.bkgd_img))
    im = ax.imshow(img, extent=trial.img_extent)
    ax.axis('off')

# Get entrance coordinates (first non-NaN coordinate)
for i, coord in enumerate(trial.r_nose):
    if not np.isnan(coord[0]):
        entrance_coord = coord
        break

# Draw straight line from entrance to Target A (blue)
if hasattr(trial, 'target'):
    ax.plot([entrance_coord[0], trial.target[0]], 
            [entrance_coord[1], trial.target[1]], 
            ls='-', color='#1c4c99', linewidth=4, label='Training to Target A')
    
    # Add arrow at the end
    arrow_a = patches.FancyArrowPatch(entrance_coord, trial.target,
                                    arrowstyle='->', mutation_scale=20, 
                                    color='#1c4c99', linewidth=4)
    ax.add_patch(arrow_a)

# Draw straight line from entrance to Target B (red) 
# You'll need to define where Target B is - I'll put it opposite to Target A
if hasattr(trial, 'target'):
    # Calculate Target B position (opposite side of arena from Target A)
    center_x, center_y = trial.arena_circle[0], trial.arena_circle[1]
    target_b_x = center_x + (center_x - trial.target[0])
    target_b_y = center_y + (center_y - trial.target[1])
    target_b = (target_b_x, target_b_y)
    
    ax.plot([entrance_coord[0], target_b[0]], 
            [entrance_coord[1], target_b[1]], 
            ls='-', color='#8f1719', linewidth=4, label='Reversal to Target B')
    
    # Add arrow at the end
    arrow_b = patches.FancyArrowPatch(entrance_coord, target_b,
                                    arrowstyle='->', mutation_scale=20, 
                                    color='#8f1719', linewidth=4)
    ax.add_patch(arrow_b)
    
    # Draw Target B
    target_b_circle = plt.Circle(target_b, 2.5, color='#8f1719', label='Target B (REL)')
    ax.add_artist(target_b_circle)

# Add Target A (using your existing target)
if hasattr(trial, 'target'):
    target_a = plt.Circle((trial.target), 2.5, color='#1c4c99', label='Target A (Food)')
    ax.add_artist(target_a)

# Draw entrance exactly like your overlay
pltlib.draw_entrance(trial, ax)

# Add start label
ax.text(entrance_coord[0]-9, entrance_coord[1] + 9, 'Start', 
        ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add phase labels
if hasattr(trial, 'target'):
    # A phase label
    mid_x_a = (entrance_coord[0] + trial.target[0]) / 2
    mid_y_a = (entrance_coord[1] + trial.target[1]) / 2
    ax.text(mid_x_a+1, mid_y_a + 5, 'A', 
            ha='center', va='center', fontsize=15, color='#1c4c99', fontweight='bold')
    
    # B phase label
    mid_x_b = (entrance_coord[0] + target_b[0]) / 2
    mid_y_b = (entrance_coord[1] + target_b[1]) / 2
    ax.text(mid_x_b+5, mid_y_b + 5, 'B', 
            ha='center', va='center', fontsize=15, color='#8f1719', fontweight='bold')

plt.title('Experiment Setup \n Target A + Target B Training', fontsize=16, fontweight='bold')
# plt.legend(loc='upper right')

# Create legend with just the target dots
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1c4c99', 
               markersize=10, label='Target A Reward', markeredgecolor='black'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#8f1719', 
               markersize=10, label='Target B Reward', markeredgecolor='black')
]

# Position legend at bottom
ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.03), 
          ncol=2, frameon=True, fancybox=True, shadow=True)

# Save figure
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
filename = 'Experiment_Setup_Cartoon.pdf'
plt.savefig(os.path.join(desktop_path, filename), 
            dpi=600, bbox_inches='tight', pad_inches=0)

plt.show()