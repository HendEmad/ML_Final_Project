import numpy as np
import matplotlib.pyplot as plt


def visualize_images(images, labels, class_names, number_of_images=12):
    random_indices = np.random.choice(len(images), size=number_of_images)
    columns = 4
    rows = int(np.ceil(number_of_images / columns))
    plt.figure(figsize=(12, 3 * rows))
    for pos, idx in enumerate(random_indices):
        plt.subplot(rows, columns, pos + 1)
        plt.imshow(images[idx])
        plt.title(class_names[labels[idx]])
        plt.axis("off")
    plt.tight_layout()
    plt.show()

    