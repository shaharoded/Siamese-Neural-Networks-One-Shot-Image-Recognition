import matplotlib.pyplot as plt
import numpy as np


def imshow(img, labels=None):
    """
    Display an image grid with optional labels.
    
    Args:
    - img (Tensor): Grid of images.
    - labels (List[str], optional): List of labels corresponding to each pair.
    """
    npimg = img.numpy()
    plt.figure(figsize=(12, 6))
    plt.axis("off")
    
    # Display images
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    
    # Add labels above images
    if labels:
        num_pairs = len(labels)
        step = img.size(2) // num_pairs  # Calculate the width of each pair
        for i, label in enumerate(labels):
            plt.text(step * (i + 0.5), -10, label, ha='center', va='bottom',
                     fontsize=12, fontweight='bold', color="black",
                     bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 3})
    
    plt.show()