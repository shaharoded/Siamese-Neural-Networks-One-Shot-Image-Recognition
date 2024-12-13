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
    

def calculate_output_size(input_size, kernel_size, stride, padding, pool_kernel, pool_stride, pool_padding):
    """
    Calculate the output size of a convolutional layer followed by a max-pooling layer.
    
    Args:
        input_size (tuple): (Height, Width) of the input image.
        kernel_size (int): Size of the convolutional kernel (assumes square kernel).
        stride (int): Stride of the convolution.
        padding (int): Padding added to all sides of the input image.
        pool_kernel (int): Size of the max-pooling kernel.
        pool_stride (int): Stride of the max-pooling operation.
        pool_padding (int): Padding for the max-pooling operation.
        
    Returns:
        tuple: (Height, Width) of the final output image.
    """
    # Convolutional layer output
    h_in, w_in = input_size
    h_out = (h_in + 2 * padding - kernel_size) // stride + 1
    w_out = (w_in + 2 * padding - kernel_size) // stride + 1
    
    # Max-pooling layer output
    if pool_kernel > 0 and pool_stride > 0:
        h_out = (h_out + 2 * pool_padding - pool_kernel) // pool_stride + 1
        w_out = (w_out + 2 * pool_padding - pool_kernel) // pool_stride + 1
    
    return h_out, w_out


def calculate_cnn_output_size(cnn_blocks, input_size):
    """
    Calculate the output size of the CNN blocks to determine the input size for the FC layers.
    
    Args:
        cnn_blocks (list): List of dictionaries defining CNN block configurations.
        input_size (tuple): (Height, Width) of the input image.

    Returns:
        tuple: (Height, Width, Channels) of the final output to be used for the FC layer.
    """
    h, w = input_size  # Initial input size
    channels = 3  # Default input channels for an RGB image, adjust if different (e.g., grayscale = 1)

    for block in cnn_blocks:
        kernel_size = block.get("kernel_size", 1)
        stride = block.get("stride", 1)
        padding = block.get("padding", 0)
        pool_kernel = 2 if block.get("use_pooling", False) else 0
        pool_stride = 2 if block.get("use_pooling", False) else 0
        pool_padding = 0

        # Calculate convolution output size
        h = (h + 2 * padding - kernel_size) // stride + 1
        w = (w + 2 * padding - kernel_size) // stride + 1

        # Calculate max-pooling output size (if pooling is used)
        if pool_kernel > 0 and pool_stride > 0:
            h = (h + 2 * pool_padding - pool_kernel) // pool_stride + 1
            w = (w + 2 * pool_padding - pool_kernel) // pool_stride + 1

        # Update the number of channels after this layer
        channels = block.get("out_channels", channels)

    return h, w, channels