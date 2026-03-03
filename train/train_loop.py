from copy import deepcopy
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import CFG
from train.losses import captioning_loss, clip_style_contrastive
from train.eval import greedy_decode, beam_search_decode, coco_caption_metrics
from utils.misc import approximate_flops
from utils.param_count import count_parameters


def _masked_mean(x, mask):
    mask = mask.unsqueeze(-1).type_as(x)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return (x * mask).sum(dim=1) / denom


def _build_scheduler(optimizer, total_steps: int):
    if total_steps <= 0:
        return None

    warmup_steps = max(1, int(total_steps * CFG.warmup_ratio))

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _to_float(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def run_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    device,
    train=True,
    caption_weight=1.0,
    aux_weight=0.0,
    contrastive_weight=0.0,
):
    model.train(train)
    total_losses = []
    cap_losses = []
    aux_losses = []
    contrastive_losses = []

    for batch in loader:
        video = batch["video"].to(device)
        ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)

        logits, diagnostics = model(video, ids[:, :-1])

        cap_loss = captioning_loss(logits, ids[:, 1:], label_smoothing=CFG.label_smoothing)

        aux = sum(v for k, v in diagnostics.items() if "loss" in k and torch.is_tensor(v)) if diagnostics else 0.0
        if not torch.is_tensor(aux):
            aux = torch.tensor(float(aux), device=device)

        v_emb = diagnostics.get("video_emb")
        if v_emb is None:
            probs = torch.softmax(logits, dim=-1)
            v_emb = (probs @ model.decoder.embedding.weight).mean(dim=1)
        t_emb = _masked_mean(model.decoder.embedding(ids), attn)
        c_loss = clip_style_contrastive(v_emb, t_emb, tau=CFG.tau)

        loss = caption_weight * cap_loss + aux_weight * aux + contrastive_weight * c_loss

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        total_losses.append(_to_float(loss))
        cap_losses.append(_to_float(cap_loss))
        aux_losses.append(_to_float(aux))
        contrastive_losses.append(_to_float(c_loss))

    denom = max(1, len(total_losses))
    return {
        "loss": sum(total_losses) / denom,
        "caption_loss": sum(cap_losses) / denom,
        "aux_loss": sum(aux_losses) / denom,
        "contrastive_loss": sum(contrastive_losses) / denom,
    }


def evaluate(model, loader, device, tokenizer):
    model.eval()
    preds, refs = [], []
    with torch.no_grad():
        for batch in loader:
            v = batch["video"].to(device)
            decoded_ids = greedy_decode(model, v, max_len=CFG.max_len)
            preds.extend(tokenizer.batch_decode(decoded_ids))
            refs.extend([str(x).strip().lower() for x in batch["caption"]])

    metrics = coco_caption_metrics(preds, refs)
    return metrics, preds, refs


def run_training(
    model,
    train_ds,
    val_ds,
    test_ds,
    seed,
    run_name,
    tokenizer,
    no_aux_loss: bool = False,
    no_contrastive_pretrain: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    pin_memory = CFG.pin_memory and device.type == "cuda"
    num_workers = max(0, int(CFG.num_workers))
    persistent_workers = num_workers > 0

    pretrain_bs = min(max(1, CFG.pretrain_batch_size), len(train_ds))
    finetune_bs = min(max(1, CFG.finetune_batch_size), len(train_ds))

    pretrain_loader = DataLoader(
        train_ds,
        batch_size=pretrain_bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=finetune_bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    history = []

    # Stage 1: contrastive pretraining
    if CFG.pretrain_epochs > 0 and not no_contrastive_pretrain:
        pre_opt = torch.optim.AdamW(model.parameters(), lr=CFG.pretrain_lr, weight_decay=CFG.weight_decay)
        pre_sched = _build_scheduler(pre_opt, total_steps=CFG.pretrain_epochs * len(pretrain_loader))

        for epoch in range(1, CFG.pretrain_epochs + 1):
            train_stats = run_epoch(
                model,
                pretrain_loader,
                pre_opt,
                pre_sched,
                device,
                train=True,
                caption_weight=0.0,
                aux_weight=0.0,
                contrastive_weight=CFG.contrastive_loss_weight,
            )
            history.append({"phase": "pretrain", "epoch": epoch, **train_stats})

    # Stage 2: captioning fine-tuning
    finetune_opt = torch.optim.AdamW(model.parameters(), lr=CFG.finetune_lr, weight_decay=CFG.weight_decay)
    finetune_sched = _build_scheduler(finetune_opt, total_steps=CFG.finetune_epochs * len(train_loader))

    best_state = deepcopy(model.state_dict())
    best_val = {}
    best_epoch = 0
    best_cider = float("-inf")

    aux_weight = 0.0 if no_aux_loss else CFG.aux_loss_weight
    contrastive_weight = 0.0 if no_contrastive_pretrain else CFG.contrastive_loss_weight

    for epoch in range(1, CFG.finetune_epochs + 1):
        train_stats = run_epoch(
            model,
            train_loader,
            finetune_opt,
            finetune_sched,
            device,
            train=True,
            caption_weight=1.0,
            aux_weight=aux_weight,
            contrastive_weight=contrastive_weight,
        )
        val_metrics, _, _ = evaluate(model, val_loader, device, tokenizer)
        val_cider = float(val_metrics.get("CIDEr", 0.0))

        history.append({"phase": "finetune", "epoch": epoch, **train_stats, **{f"val_{k}": v for k, v in val_metrics.items()}})

        if val_cider > best_cider:
            best_cider = val_cider
            best_epoch = epoch
            best_val = val_metrics
            best_state = deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    test_metrics, preds, refs = evaluate(model, test_loader, device, tokenizer)

    batch = next(iter(test_loader))
    beam = beam_search_decode(
        model,
        batch["video"].to(device),
        max_len=CFG.max_len,
        beam_size=5,
        length_penalty=CFG.length_penalty,
    )

    metrics = dict(test_metrics)
    for k, v in best_val.items():
        metrics[f"val_{k}"] = v
    metrics["best_epoch"] = int(best_epoch)
    metrics["beam_example_len"] = int(beam.size(1))
    metrics["train_loss"] = float(history[-1]["loss"]) if history else 0.0
    metrics["seed"] = seed
    metrics.update(count_parameters(model))
    metrics["flops_approx"] = approximate_flops(CFG.num_frames * (224 // 16) ** 2, CFG.embed_dim, CFG.num_layers)

    out_dir = Path(CFG.results_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "seed": seed,
            "best_epoch": best_epoch,
            "metrics": metrics,
        },
        out_dir / "best_model.pt",
    )

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "predictions.json").write_text(json.dumps([{"pred": p, "ref": r} for p, r in zip(preds, refs)], indent=2))

    return metrics
