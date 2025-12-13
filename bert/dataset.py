import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer


class ToxicCommentsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer=None, max_length=128):
        self.texts = texts
        self.labels = labels
        
        if tokenizer is None:
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        else:
            self.tokenizer = tokenizer
        
        self.max_length = max_length
        self.labels = torch.LongTensor(labels.values if hasattr(labels, 'values') else labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': self.labels[idx]
        }