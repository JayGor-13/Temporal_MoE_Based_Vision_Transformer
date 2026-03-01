import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, nhead: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout)
        self.dec = nn.TransformerDecoder(layer, num_layers)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt_ids, memory):
        x = self.embedding(tgt_ids) * math.sqrt(self.d_model)
        x = self.pos(x)
        m = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)
        y = self.dec(x, memory, tgt_mask=m)
        return self.out(y)
