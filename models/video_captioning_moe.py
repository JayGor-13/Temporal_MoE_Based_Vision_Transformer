import torch
import torch.nn as nn

from ..models.vit_encoder import TemporalMoEViT_Encoder
from ..models.decoder import TransformerDecoder
from ..config import NUM_FRAMES, PATCH_SIZE
# =========================================================================
# --- STEP 4: UPGRADE THE MASTER VideoCaptioningMoE MODEL ---
# This model now correctly handles the auxiliary losses from the new encoder.
# =========================================================================

class VideoCaptioningMoE(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_heads=8,
                 top_k=2, num_layers=6, decoder_layers=6):
        super().__init__()
        
        # Create an ENCODER that is just the TemporalMoEViT part

        self.encoder = TemporalMoEViT_Encoder(embed_dim, num_heads, top_k, num_layers)
        
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=embed_dim,
            nhead=num_heads,
            num_layers=decoder_layers
        )

    def forward(self, video_frames, tgt_ids, expert_kwargs=None):
        # We need to know whether to compute the aux losses
        compute_losses = self.training 

        # Encode video into memory and get auxiliary losses
        memory, diagnostics = self.encoder(video_frames, expert_kwargs, compute_router_losses=compute_losses)
        
        # Decode captions conditioned on the video memory
        logits = self.decoder(tgt_ids, memory)
        
        return logits, diagnostics