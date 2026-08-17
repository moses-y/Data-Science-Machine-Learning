import numpy as np
import matplotlib.pyplot as plt
import os

# Use ImageIO v2 explicitly to avoid future deprecation issues
import imageio.v2 as imageio

# Determine the script directory for saving the frames and GIF
script_dir = "c:/Users/moses_y/OneDrive/Desktop/ML Projects/Generating Gifs"
frames_dir = os.path.join(script_dir, 'gif_frames')
os.makedirs(frames_dir, exist_ok=True)  # Create the directory if it doesn't exist


# Function to plot and save each frame
def plot_frame(x, y, filename, color):
    fig, ax = plt.subplots(figsize=(8, 6))
    # Plotting the function with a shadow for a realistic effect
    ax.plot(x, y - 0.1, color='grey', alpha=0.5)  # Shadow effect
    ax.plot(x, y, color=color)
    ax.fill_between(x, 0, y, color=color, alpha=0.3)  # Fill under curve for extra effect
    ax.set_xlim([0, 2*np.pi])
    ax.set_ylim([-1.5, 1.5])
    plt.axis('off')  # Hide the axis for a cleaner look
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)

# Generate frames
x = np.linspace(0, 2*np.pi, 100)
colors = plt.cm.viridis(np.linspace(0, 1, 50))  # Using a colormap for psychedelic effect
frames = []

for i, color in enumerate(colors):
    y = np.sin(x + i*0.1)  # Changing the phase for dynamic effect
    filename = os.path.join(frames_dir, f'frame_{i}.png')
    plot_frame(x, y, filename, color=color)
    frames.append(imageio.imread(filename))

# Create GIF and Save the GIF in the same directory as the script
gif_path = os.path.join(script_dir, 'sine_wave_psychedelic.gif')

duration = 3000 / 10  # 10 fps -> duration of each frame in milliseconds

# Save the GIF using the 'duration' parameter instead of 'fps'
imageio.mimsave(gif_path, frames, duration=duration)

print(f'GIF saved to {gif_path}')
