from torch.utils.data import DataLoader

from config import *
from data.tokenizer import SimpleTokenizer
from data.msvd_dataset import MSVDVideoCaptionDataset
from models.video_captioning_moe import VideoCaptioningMoE
from train.train_loop import train_model

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = SimpleTokenizer(vocab_size=VOCAB_SIZE)

    dataset = MSVDVideoCaptionDataset(
        video_dir="./dataset/video_files/video_files",
        csv_path="./dataset/video_corpus.csv",
        txt_path="./dataset/annotations.txt",
        tokenizer=tokenizer,
        frames=NUM_FRAMES
    )

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = VideoCaptioningMoE(
        vocab_size=len(tokenizer.word2idx),
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        top_k=TOP_K,
        num_layers=NUM_LAYERS,
        decoder_layers=6
    ).to(device)

    train_model(model, loader, None, tokenizer, device=device, epochs=2)

if __name__ == "__main__":
    main()

