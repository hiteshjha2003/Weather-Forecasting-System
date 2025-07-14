# models_trainings/config.py

import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
PATCH_SIZE = 24
SEQ_LEN = 96
PRED_LEN = 24
LR = 1e-4
EPOCHS = 100
TARGET_COLUMNS = ["Temperature (C)", "Humidity"]
EMBED_DIM = 128
NHEAD = 8
NUM_LAYERS = 4
PATIENCE = 10
