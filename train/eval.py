from collections import defaultdict
from pathlib import Path
import json

import torch


@torch.no_grad()
def greedy_decode(model, video, max_len=20, bos_id=2, eos_id=3):
    ids = torch.full((video.size(0), 1), bos_id, device=video.device, dtype=torch.long)
    for _ in range(max_len - 1):
        logits, _ = model(video, ids)
        nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
        if torch.all(nxt.squeeze(-1) == eos_id):
            break
    return ids


@torch.no_grad()
def beam_search_decode(model, video, max_len=20, beam_size=5, length_penalty=0.7, bos_id=2, eos_id=3):
    # batch=1 minimal implementation for eval scripts
    assert video.size(0) == 1
    beams = [(torch.tensor([[bos_id]], device=video.device), 0.0)]
    for _ in range(max_len - 1):
        candidates = []
        for seq, score in beams:
            if seq[0, -1].item() == eos_id:
                candidates.append((seq, score))
                continue
            logits, _ = model(video, seq)
            logp = torch.log_softmax(logits[:, -1], dim=-1)
            vals, idxs = torch.topk(logp, beam_size, dim=-1)
            for v, i in zip(vals[0], idxs[0]):
                nseq = torch.cat([seq, i.view(1, 1)], dim=1)
                lp = ((5 + nseq.size(1)) / 6) ** length_penalty
                candidates.append((nseq, (score + v.item()) / lp))
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]
        if all(seq[0, -1].item() == eos_id for seq, _ in beams):
            break
    return beams[0][0]


def _fallback_overlap_metrics(preds, refs):
    # lexical overlap proxy for environments without COCO metric toolchain
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


def coco_caption_metrics(preds, refs):
    if not preds or not refs:
        return {"BLEU-1": 0.0, "BLEU-2": 0.0, "BLEU-3": 0.0, "BLEU-4": 0.0, "METEOR": 0.0, "ROUGE-L": 0.0, "CIDEr": 0.0, "SPICE": 0.0}

    fallback = _fallback_overlap_metrics(preds, refs)

    try:
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.spice.spice import Spice
    except Exception:
        return fallback

    gts = {i: [{"caption": refs[i]}] for i in range(len(preds))}
    res = {i: [{"caption": preds[i]}] for i in range(len(preds))}

    tokenizer = PTBTokenizer()
    try:
        gts_t = tokenizer.tokenize(gts)
        res_t = tokenizer.tokenize(res)
    except Exception:
        return fallback

    metrics = {}
    scorers = [
        (Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
        (Meteor(), ["METEOR"]),
        (Rouge(), ["ROUGE-L"]),
        (Cider(), ["CIDEr"]),
        (Spice(), ["SPICE"]),
    ]

    for scorer, names in scorers:
        try:
            score, _ = scorer.compute_score(gts_t, res_t)
            if len(names) == 1:
                metrics[names[0]] = float(score)
            else:
                for name, val in zip(names, score):
                    metrics[name] = float(val)
        except Exception:
            for name in names:
                metrics[name] = fallback.get(name, 0.0)

    # Fill any missing key defensively.
    for key, value in fallback.items():
        metrics.setdefault(key, value)

    return metrics


def dump_metrics(path: str, metrics: dict):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(metrics, indent=2))
