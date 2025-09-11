# src/config.py
import os

# --- Training Hyperparameters ---
BATCH_SIZE = 8
LEARNING_RATE = 0.001
EPOCHS = 5

# --- Data & Model Specifications ---
IMG_SIZE = 128
VOXEL_SIZE = 32

# --- Paths (Configured for Google Colab) ---
# Note: Ensure your Google Drive is mounted at /content/drive
BASE_DRIVE_PATH = "/content/"
DATA_PATH = os.path.join(BASE_DRIVE_PATH, "ModelNet10/")
SAVE_PATH = os.path.join(BASE_DRIVE_PATH, "pix2vox_outputs/")

# --- Debugging ---
# If True, runs training on a small subset of data for quick checks.
DEBUG = True
DEBUG_BATCHES = 2
