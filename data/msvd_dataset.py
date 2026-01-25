import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from ..config import NUM_SAMPLES

class MSVDVideoCaptionDataset(Dataset):
    def __init__(self, video_dir, csv_path, txt_path, tokenizer, max_len=20, limit=NUM_SAMPLES, frames=8):
        self.video_dir = video_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.frames = frames
        
        # Load metadata
        print(f"Loading metadata from {csv_path}...")
        self.data = pd.read_csv(csv_path)
        self.data = self.data[self.data["Language"] == "English"].head(limit)
        
        # Load captions
        print(f"Loading captions from {txt_path}...")
        with open(txt_path, "r", encoding='utf-8') as f:
            self.captions = [line.strip() for line in f if line.strip()]
        
        print(f"✓ Loaded {len(self.data)} English video samples.")
        print(f"✓ Loaded {len(self.captions)} captions.")
        
        # Build vocabulary if tokenizer needs it
        if hasattr(tokenizer, 'build_vocab') and not tokenizer.vocab_built:
            tokenizer.build_vocab(self.captions)
    
    def __len__(self):
        return len(self.data)
    
    def _load_video(self, path):
        """Load and preprocess video frames"""
        if not os.path.exists(path):
            return torch.zeros(self.frames, 3, 224, 224)
        
        cap = cv2.VideoCapture(path)
        frames = []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total == 0:
            cap.release()
            return torch.zeros(self.frames, 3, 224, 224)
        
        # Sample frames uniformly
        idxs = np.linspace(0, total - 1, self.frames, dtype=int)
        
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            
            # Preprocess frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frames.append(frame)
        
        cap.release()
        
        # Padding if not enough frames
        while len(frames) < self.frames:
            frames.append(torch.zeros(3, 224, 224))
        
        return torch.stack(frames)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        vid_name = f"{row['VideoID']}_{int(row['Start'])}_{int(row['End'])}.avi"
        vid_path = os.path.join(self.video_dir, vid_name)
        
        # Load video frames
        video = self._load_video(vid_path)
        
        # Get corresponding caption
        caption = self.captions[idx % len(self.captions)]
        
        # Tokenize caption
        tokens = self.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=self.max_len
        )
        
        return {
            "video": video,  # Shape: (num_frames, 3, 224, 224)
            "input_ids": tokens['input_ids'].squeeze(0),  # Shape: (max_len,)
            "attention_mask": tokens['attention_mask'].squeeze(0),  # Shape: (max_len,)
            "caption": caption  # Original caption text
        }