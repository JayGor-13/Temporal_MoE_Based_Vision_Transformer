import torch
import torch.nn.functional as F


def captioning_loss(logits, targets, pad_id=0, label_smoothing=0.0):
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=pad_id, label_smoothing=label_smoothing)


def clip_style_contrastive(video_emb, text_emb, tau=0.07):
    video_emb = F.normalize(video_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)
    logits = video_emb @ text_emb.t() / tau
    labels = torch.arange(logits.size(0), device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
