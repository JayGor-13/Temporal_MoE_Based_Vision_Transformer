from torch.utils.data import DataLoader

from .config import *
from .data.tokenizer import SimpleTokenizer
from .data.msvd_dataset import MSVDVideoCaptionDataset
from .models.video_captioning_moe import VideoCaptioningMoE
from .train.train_loop import train_model

tokenizer = SimpleTokenizer(vocab_size=VOCAB_SIZE)

dataset = MSVDVideoCaptionDataset(...)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = VideoCaptioningMoE(
    vocab_size=tokenizer.vocab_size_property,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    top_k=TOP_K,
    num_layers=NUM_LAYERS
).to(DEVICE)

train_model(model, loader, None, tokenizer, device=DEVICE, epochs=2)
