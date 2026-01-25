import torch.nn as nn

def captioning_loss(logits, targets, pad_id):
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    return criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
