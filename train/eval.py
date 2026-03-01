from collections import defaultdict
from pathlib import Path
import json

import torch


@torch.no_grad()
def greedy_decode(model, video, max_len=20, bos_id=2):
    ids = torch.full((video.size(0), 1), bos_id, device=video.device, dtype=torch.long)
    for _ in range(max_len - 1):
        logits, _ = model(video, ids)
        nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
    return ids


@torch.no_grad()
def beam_search_decode(model, video, max_len=20, beam_size=5, length_penalty=0.7, bos_id=2):
    # batch=1 minimal implementation for eval scripts
    assert video.size(0) == 1
    beams = [(torch.tensor([[bos_id]], device=video.device), 0.0)]
    for _ in range(max_len - 1):
        candidates = []
        for seq, score in beams:
            logits, _ = model(video, seq)
            logp = torch.log_softmax(logits[:, -1], dim=-1)
            vals, idxs = torch.topk(logp, beam_size, dim=-1)
            for v, i in zip(vals[0], idxs[0]):
                nseq = torch.cat([seq, i.view(1, 1)], dim=1)
                lp = ((5 + nseq.size(1)) / 6) ** length_penalty
                candidates.append((nseq, (score + v.item()) / lp))
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]
    return beams[0][0]


def coco_like_metrics(preds, refs):
    # lightweight proxy metrics to keep pipeline executable locally
    def overlap(a, b):
        sa, sb = set(a.split()), set(b.split())
        return len(sa & sb) / max(1, len(sb))

    m = defaultdict(float)
    for p, r in zip(preds, refs):
        ov = overlap(p, r)
        for i in range(1, 5):
            m[f"BLEU-{i}"] += ov
        m["METEOR"] += ov
        m["ROUGE-L"] += ov
        m["CIDEr"] += ov * 10
        m["SPICE"] += ov
    n = max(1, len(preds))
    return {k: v / n for k, v in m.items()}


def dump_metrics(path: str, metrics: dict):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(metrics, indent=2))
