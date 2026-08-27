import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
# Load the images
img1 = mpimg.imread('real_toxic_0_plot.png')
img2 = mpimg.imread('real_toxic_1_plot.png')
img3 = mpimg.imread('synthetic_toxic_0_plot.png')
img4 = mpimg.imread('synthetic_toxic_1_plot.png')


# Create a figure with 2x2 subplots, styled like the provided example
fig, axs = plt.subplots(2, 2, figsize=(12, 12))

# Load the images again for each subplot
axs[0, 0].imshow(img3)
axs[0, 0].axis('off')
axs[0, 0].set_title('Non-Toxic', fontsize=16, fontweight='bold')

axs[0, 1].imshow(img4)
axs[0, 1].axis('off')
axs[0, 1].set_title('Toxic', fontsize=16, fontweight='bold')

axs[1, 0].imshow(img1)
axs[1, 0].axis('off')

axs[1, 1].imshow(img2)
axs[1, 1].axis('off')

# Add row tit
# les
fig.text(0.04, 0.75, 'Synthetic', ha='center', va='center', rotation='vertical', fontsize=16, fontweight='bold')
fig.text(0.04, 0.35, 'Non-Synthetic', ha='center', va='center', rotation='vertical', fontsize=16, fontweight='bold')

# Adjust layout and display
plt.tight_layout(rect=[0.1, 0.1, 1, 1])
output_file_path = os.path.join('psa_combined_plot.png')
plt.savefig(output_file_path, dpi= 300)