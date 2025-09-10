#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 22:57:59 2025

@author: amarpreetdheer
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 22:51:14 2025

@author: amarpreetdheer
"""

import modules.lib_process_data_to_mat as plib
import modules.lib_plot_mouse_trajectory as pltlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from modules.config import ROOT_DIR

# Load both trials
trial_4 = plib.TrialData()
trial_4.Load('2025-08-22', '66', '17')

trial_5 = plib.TrialData()
trial_5.Load('2025-08-22', '66', '20')

print('Mouse %s Trial %s' % (trial_4.mouse_number, trial_4.trial))
print('Mouse %s Trial %s' % (trial_5.mouse_number, trial_5.trial))

# Create the plot
fig, ax = plt.subplots(figsize=(5, 3))

# Set up the background and arena
if hasattr(trial_4, 'arena_circle'):
    pltlib.draw_arena(trial_4, ax)
else:
    print('Missing arena circle coordinates')
    # Import and display background image
    img = mpimg.imread(os.path.join(ROOT_DIR, 'data', 'BackgroundImage', trial_4.bkgd_img))
    im = ax.imshow(img, extent=trial_4.img_extent)
    ax.axis('off')

# Plot trial 4 in gray
if hasattr(trial_4, 'target'):
    index_4 = pltlib.coords_to_target(trial_4.r_nose, trial_4.target)
    ax.plot(trial_4.r_nose[:index_4+1, 0], trial_4.r_nose[:index_4+1, 1], 
            ls='-', color='gray', linewidth=1.5, label='Trial 4', alpha=0.8)
else:
    ax.plot(trial_4.r_nose[:, 0], trial_4.r_nose[:, 1], 
            ls='-', color='gray', linewidth=2, label='Trial 4', alpha=0.8)

# Plot trial 5 in red
if hasattr(trial_5, 'target'):
    index_5 = pltlib.coords_to_target(trial_5.r_nose, trial_5.target)
    ax.plot(trial_5.r_nose[:index_5+1, 0], trial_5.r_nose[:index_5+1, 1], 
            ls='-', color='red', linewidth=2, label='Trial 5')
else:
    ax.plot(trial_5.r_nose[:, 0], trial_5.r_nose[:, 1], 
            ls='-', color='red', linewidth=2, label='Trial 5')

# Add target if it exists
if hasattr(trial_4, 'target'):
    # target = plt.Circle((trial_4.target), 2.5, color='#1c4c99', label='Target')
    target = plt.Circle((trial_4.target), 2.5, color='#8f1719', label='Target')
    ax.add_artist(target)

# Draw entrance for reference
pltlib.draw_entrance(trial_4, ax)

# Determine mouse type and add title
mouse_num = int(trial_4.mouse_number)
if 60 <= mouse_num <= 63:
    mouse_type = "KO"
elif 64 <= mouse_num <= 67:
    mouse_type = "WT"
else:
    mouse_type = f"Mouse {trial_4.mouse_number}"

# plt.title(f'{mouse_type} - Trials 4 (Gray) & 5 (Red) Overlay')
# plt.legend()

# Save figure option
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
filename = f'WT_TARGET_B_Overlay.pdf'
plt.savefig(os.path.join(desktop_path, filename), 
            dpi=600, bbox_inches='tight', pad_inches=0)

plt.show()