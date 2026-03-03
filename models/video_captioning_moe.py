import torch
from torch import nn

from config import CFG
from models.decoder import TransformerDecoder
from models.vit_encoder import TemporalMoEViTEncoder


class VideoCaptioningMoE(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dense_only: bool = False,
        no_caption_conditioning: bool = False,
        no_temporal_bias: bool = False,
        no_cross_modal_gating: bool = False,
    ):
        super().__init__()
        self.no_cross_modal_gating = no_cross_modal_gating
        self.encoder = TemporalMoEViTEncoder(
            dense_only=dense_only,
            no_temporal_bias=no_temporal_bias,
            no_caption_conditioning=no_caption_conditioning,
        )
        self.gate = nn.Linear(CFG.embed_dim, CFG.embed_dim)  # alpha = sigmoid(Wg h_video)
        self.decoder = TransformerDecoder(vocab_size, CFG.embed_dim, CFG.num_heads, CFG.num_layers, CFG.dropout)

    def forward(self, video_frames, tgt_ids):
        text_state = torch.zeros(video_frames.size(0), CFG.embed_dim, device=video_frames.device)
        if tgt_ids.numel() > 0:
            text_state = self.decoder.embedding(tgt_ids).mean(dim=1)

        memory, diagnostics = self.encoder(video_frames, text_state)
        # CLS token acts as the global video representation for auxiliary alignment losses.
        diagnostics["video_emb"] = memory[:, 0, :]

        if self.no_cross_modal_gating:
            alpha = torch.ones_like(memory)
        else:
            alpha = torch.sigmoid(self.gate(memory))
            memory = alpha * memory
        diagnostics["cross_modal_alpha_mean"] = alpha.mean()

        logits = self.decoder(tgt_ids, memory)
        return logits, diagnostics
