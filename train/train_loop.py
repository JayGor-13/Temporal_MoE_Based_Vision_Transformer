from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader

from config import CFG
from train.losses import captioning_loss, clip_style_contrastive
from train.eval import greedy_decode, beam_search_decode, coco_like_metrics
from utils.misc import approximate_flops
from utils.param_count import count_parameters


def _masked_mean(x, mask):
    mask = mask.unsqueeze(-1).type_as(x)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return (x * mask).sum(dim=1) / denom


def run_epoch(model, loader, optimizer, device, train=True):
    model.train(train)
    losses = []
    for batch in loader:
        video = batch["video"].to(device)
        ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)

        logits, diagnostics = model(video, ids[:, :-1])
        cap_loss = captioning_loss(logits, ids[:, 1:], label_smoothing=CFG.label_smoothing)
        aux = sum(v for k, v in diagnostics.items() if "loss" in k and torch.is_tensor(v)) if diagnostics else 0.0

        # CLIP-style pretraining component: keep both embeddings in model embed space.
        v_emb = diagnostics.get("video_emb")
        if v_emb is None:
            probs = torch.softmax(logits, dim=-1)
            v_emb = (probs @ model.decoder.embedding.weight).mean(dim=1)
        t_emb = _masked_mean(model.decoder.embedding(ids), attn)
        c_loss = clip_style_contrastive(v_emb, t_emb, tau=CFG.tau)

        loss = cap_loss + CFG.aux_loss_weight * aux + 0.1 * c_loss
        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
            optimizer.step()
        losses.append(loss.item())
    return sum(losses) / max(1, len(losses))


def evaluate(model, loader, device):
    model.eval()
    preds, refs = [], []
    with torch.no_grad():
        for batch in loader:
            v = batch["video"].to(device)
            g = greedy_decode(model, v, max_len=CFG.max_len)
            preds.extend([" ".join(map(str, x.tolist())) for x in g])
            refs.extend(batch["caption"])
    return coco_like_metrics(preds, refs)


def run_training(model, train_ds, val_ds, seed, run_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_bs = min(CFG.finetune_batch_size, len(train_ds))
    if device.type == "cpu":
        train_bs = min(train_bs, 1)
    train_loader = DataLoader(train_ds, batch_size=train_bs)
    val_loader = DataLoader(val_ds, batch_size=1)

    opt = torch.optim.AdamW(model.parameters(), lr=CFG.finetune_lr, weight_decay=CFG.weight_decay)
    train_loss = run_epoch(model, train_loader, opt, device, train=True)
    metrics = evaluate(model, val_loader, device)

    batch = next(iter(val_loader))
    beam = beam_search_decode(model, batch["video"].to(device), max_len=CFG.max_len, beam_size=5, length_penalty=CFG.length_penalty)
    metrics["beam_example_len"] = int(beam.size(1))
    metrics["train_loss"] = train_loss
    metrics["seed"] = seed
    metrics.update(count_parameters(model))
    metrics["flops_approx"] = approximate_flops(CFG.num_frames * (224 // 16) ** 2, CFG.embed_dim, CFG.num_layers)

    out_dir = Path(CFG.results_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
