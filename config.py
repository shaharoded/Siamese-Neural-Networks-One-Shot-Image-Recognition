import os
import torch

# Paths
DATA_ROOT = "lfwa/Files"
TRAIN_FILE = "lfwa/train.txt"
TEST_FILE = "lfwa/test.txt"
MODEL_DIR = "trained_models"

# DataLoader Parameters
BATCH_SIZE = 4      # Batch size for training and validation
SHUFFLE = True      # Shuffle data in DataLoader
NUM_WORKERS = 2     # Number of worker threads for DataLoader

# Image Transformation Parameters
IMAGE_SIZE = (105, 105)  # Matches the input size in the paper

# Model Parameters:
# CNN Block Configurations
CNN_BLOCKS = [
    {"out_channels": 64, "kernel_size": 10, "stride": 1, "padding": 0, "use_pooling": True, "use_batchnorm": False, "dropout_prob": 0.0},
    {"out_channels": 128, "kernel_size": 7, "stride": 1, "padding": 0, "use_pooling": True, "use_batchnorm": False, "dropout_prob": 0.0},
    {"out_channels": 128, "kernel_size": 4, "stride": 1, "padding": 0, "use_pooling": True, "use_batchnorm": False, "dropout_prob": 0.0},
    {"out_channels": 256, "kernel_size": 4, "stride": 1, "padding": 0, "use_pooling": False, "use_batchnorm": False, "dropout_prob": 0.0},
]

# Fully Connected Layer Configurations
FC_LAYERS = [
    {"in_features": 256 * 6 * 6, "out_features": 4096},  # Fully Connected Layer
]

# Training Parameters
VAL_SPLIT = 0.2                   # Fraction of data for validation (out of train)
MIN_EPOCHS = 50                   # Min number of training epochs
MAX_EPOCHS = 500                      # Max number of training epochs
LEARNING_RATE = 1e-4              # Learning rate for the optimizer
L2_REG = 1e-5                     # L2 regularization strength
EARLY_STOP_PATIENCE = 15          # Number of epochs for early stopping

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Device to use ('cpu' or 'cuda')

# Save Path
MODEL_NAME = "model1_best_checkpoint.pth"        # Path to save the best model
SAVE_PATH = os.path.join(MODEL_DIR, MODEL_NAME)