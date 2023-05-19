import numpy as np
import matplotlib.pyplot as plt
import emnist
import matplotlib.cm as cm
np.random.seed(32)

training_images, _ = emnist.extract_training_samples("balanced")

random_indices = np.random.choice(len(training_images), 25, replace=False)

fig, axes = plt.subplots(5, 5, figsize=(10, 10))

for i, ax in enumerate(axes.flat):
    image_index = random_indices[i]
    image = training_images[image_index]
    ax.imshow(image, cmap=cm.gray_r)  # Invert colors using gray_r colormap
    ax.axis('off')

plt.tight_layout()
plt.savefig('plots/emnist_examples.png')
