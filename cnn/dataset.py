# dataset.py
import torch
from torch.utils.data import Dataset
import re


class ToxicCommentsDataset(Dataset):
    def __init__(self, texts, labels, word2idx, max_length=200):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.max_length = max_length
        self.labels = torch.LongTensor(labels.values if hasattr(labels, 'values') else labels)
    
    def tokenize(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text.split()
    
    def text_to_sequence(self, text):
        tokens = self.tokenize(text)
        sequence = [self.word2idx.get(token, 1) for token in tokens]
        
        if len(sequence) < self.max_length:
            sequence = sequence + [0] * (self.max_length - len(sequence))
        else:
            sequence = sequence[:self.max_length]
        
        return sequence
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        sequence = self.text_to_sequence(text)
        
        return {
            'sequence': torch.LongTensor(sequence),
            'label': self.labels[idx]
        }