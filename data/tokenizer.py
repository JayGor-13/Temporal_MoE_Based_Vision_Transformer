import re
import torch
from collections import Counter

class SimpleTokenizer:
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.word2idx = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
        self.idx2word = {0: "[PAD]", 1: "[UNK]", 2: "[CLS]", 3: "[SEP]"}
        self.vocab_built = False
    
    def build_vocab(self, texts):
        """Build vocabulary from list of texts"""
        print("Building vocabulary...")
        words = []
        for text in texts:
            words.extend(self._tokenize_text(text))
        
        # Get most common words
        word_freq = Counter(words)
        most_common = word_freq.most_common(self.vocab_size - 4)
        
        # Add to vocab
        for idx, (word, _) in enumerate(most_common, start=4):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        self.vocab_built = True
        print(f"✓ Vocabulary built with {len(self.word2idx)} tokens")
    
    def _tokenize_text(self, text):
        """Simple tokenization: lowercase and split"""
        text = text.lower()
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.split()
    
    def encode(self, text, max_length=20, padding='max_length', truncation=True):
        """Encode text to token IDs"""
        if not self.vocab_built:
            raise ValueError("Vocabulary not built! Call build_vocab first.")
        
        tokens = self._tokenize_text(text)
        
        # Convert to IDs
        ids = [self.word2idx.get(token, self.word2idx["[UNK]"]) for token in tokens]
        
        # Truncate
        if truncation and len(ids) > max_length - 2:
            ids = ids[:max_length - 2]
        
        # Add CLS and SEP
        ids = [self.word2idx["[CLS]"]] + ids + [self.word2idx["[SEP]"]]
        
        # Create attention mask
        attention_mask = [1] * len(ids)
        
        # Pad
        if padding == 'max_length':
            pad_length = max_length - len(ids)
            ids += [self.word2idx["[PAD]"]] * pad_length
            attention_mask += [0] * pad_length
        
        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }
    
    def __call__(self, text, **kwargs):
        """Make tokenizer callable like HuggingFace tokenizers"""
        result = self.encode(text, **kwargs)
        # Return with batch dimension for compatibility
        return {
            'input_ids': result['input_ids'].unsqueeze(0),
            'attention_mask': result['attention_mask'].unsqueeze(0)
        }
    
    def convert_ids_to_tokens(self, ids):
        """Convert token IDs to tokens (list of strings)"""
        if torch.is_tensor(ids):
            ids = ids.tolist()
        # Handle single integer
        if isinstance(ids, int):
            return [self.idx2word.get(ids, "[UNK]")]
        # Handle list of integers
        return [self.idx2word.get(idx, "[UNK]") for idx in ids]
    
    def convert_tokens_to_ids(self, tokens):
        """Convert tokens to token IDs"""
        if isinstance(tokens, str):
            tokens = [tokens]
        return [self.word2idx.get(token, self.word2idx["[UNK]"]) for token in tokens]
    
    def decode(self, ids, skip_special_tokens=True):
        """Decode token IDs back to text"""
        if torch.is_tensor(ids):
            ids = ids.tolist()
        tokens = [self.idx2word.get(idx, "[UNK]") for idx in ids]
        
        # Remove special tokens if requested
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in ["[PAD]", "[CLS]", "[SEP]", "[UNK]"]]
        
        return " ".join(tokens)
    
    @property
    def vocab_size_property(self):
        """Return vocabulary size (for model compatibility)"""
        return len(self.word2idx)
    
    @property
    def pad_token_id(self):
        return self.word2idx["[PAD]"]
    
    @property
    def unk_token_id(self):
        return self.word2idx["[UNK]"]
    
    @property
    def cls_token_id(self):
        return self.word2idx["[CLS]"]
    
    @property
    def sep_token_id(self):
        return self.word2idx["[SEP]"]