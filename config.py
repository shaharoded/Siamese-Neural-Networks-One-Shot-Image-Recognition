import os
import torch

# Paths
DATA_ROOT = "lfwa/Files"
TRAIN_FILE = "lfwa/train.txt"
TEST_FILE = "lfwa/test.txt"
MODEL_DIR = "trained_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Save Path
MODEL_NAME = "model1_best_checkpoint.pth"        # Path to save the best model
SAVE_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# DataLoader Parameters
BATCH_SIZE = 16      # Batch size for training and validation
NUM_WORKERS = 2     # Number of worker threads for DataLoader

# Image Transformation Parameters
IMAGE_SIZE = (105, 105)  # Matches the input size of the dataset

# Model Parameters:
DROPOUT = 0.0
BATCHNORM = False

# CNN Block Configurations
CNN_BLOCKS = [    
    # Original architecture
    {"out_channels": 64, "kernel_size": 10, "stride": 1, "padding": 0, "use_pooling": True, "use_batchnorm": BATCHNORM, "dropout_prob": DROPOUT},   # 105x105 -> 48x48
    {"out_channels": 128, "kernel_size": 7, "stride": 1, "padding": 0, "use_pooling": True, "use_batchnorm": BATCHNORM, "dropout_prob": DROPOUT},   # 48x48 -> 21x21
    {"out_channels": 128, "kernel_size": 4, "stride": 1, "padding": 0, "use_pooling": True, "use_batchnorm": BATCHNORM, "dropout_prob": DROPOUT},   # 21x21 -> 9x9
    {"out_channels": 256, "kernel_size": 4, "stride": 1, "padding": 0, "use_pooling": False, "use_batchnorm": BATCHNORM, "dropout_prob": DROPOUT},  # 9x9 -> 6x6
]

FC_LAYERS = [
    {"in_features": 256 * 6 * 6, "out_features": 4096, "use_batchnorm": BATCHNORM, "dropout_prob": DROPOUT}, # Fully Connected Layer, # 6x6 -> 1X4096
    {"in_features": 4096, "out_features": 1, "use_batchnorm": False, "dropout_prob": 0.0},  # Final similarity score layer (1X4096 -> 1X1)
]

# Training Parameters
VAL_SPLIT = 0.2                   # Fraction of data for validation (out of train)
AUGMENT_RATIO = 3.0               # Augment the data X3
MIN_EPOCHS = 20                   # Min number of training epochs
MAX_EPOCHS = 200                      # Max number of training epochs
LEARNING_RATE = 1e-4              # Learning rate for the optimizer
L2_REG = 1e-4                     # L2 regularization strength
EARLY_STOP_PATIENCE = 10          # Number of epochs for early stopping

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Device to use ('cpu' or 'cuda')