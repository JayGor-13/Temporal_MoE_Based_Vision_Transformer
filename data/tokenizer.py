import re
from collections import Counter

import torch


class SimpleTokenizer:
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.word2idx = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_built = False

    def _tokenize_text(self, text):
        text = re.sub(r"[^a-z0-9\s]", "", text.lower())
        return text.split()

    def build_vocab(self, texts):
        freq = Counter()
        for t in texts:
            freq.update(self._tokenize_text(t))
        for i, (w, _) in enumerate(freq.most_common(self.vocab_size - 4), start=4):
            self.word2idx[w] = i
            self.idx2word[i] = w
        self.vocab_built = True

    def encode(self, text, max_length=20):
        if not self.vocab_built:
            self.build_vocab([text])
        ids = [self.word2idx["[CLS]"]] + [self.word2idx.get(t, 1) for t in self._tokenize_text(text)][: max_length - 2] + [self.word2idx["[SEP]"]]
        attn = [1] * len(ids)
        while len(ids) < max_length:
            ids.append(0)
            attn.append(0)
        return {"input_ids": torch.tensor(ids, dtype=torch.long), "attention_mask": torch.tensor(attn, dtype=torch.long)}

    def __call__(self, text, max_length=20, **kwargs):
        return self.encode(text, max_length=max_length)

    @property
    def pad_token_id(self):
        return 0
