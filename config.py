import torch

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Video / Vision
IMG_SIZE = 224
PATCH_SIZE = 16
NUM_FRAMES = 8

# Model
EMBED_DIM = 512
NUM_HEADS = 8
NUM_LAYERS = 6
TOP_K = 2

# Training
BATCH_SIZE = 2
LR = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 20

# Tokenizer
MAX_LEN = 20
VOCAB_SIZE = 5000


RESUME_CHECKPOINT_PATH = None 
CHECKPOINT_SAVE_DIR = "/kaggle/working/checkpoints/"