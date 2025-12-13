import pandas as pd
import numpy as np
from sklearn.utils import resample


class KaggleDataLoader:
    def __init__(self, train_path='./data/train.csv', test_path='./data/test.csv'):
        self.train_path = train_path
        self.test_path = test_path
        self.toxicity_columns = ['toxic', 'severe_toxic', 'obscene', 
                                 'threat', 'insult', 'identity_hate']
    
    def load_data(self, use_full_dataset=True):
        import os
        if not os.path.exists(self.train_path):
            raise FileNotFoundError(f"Dataset not found at: {self.train_path}")
        
        df = pd.read_csv(self.train_path)
        print(f"Loaded {len(df):,} comments")
        
        df['label'] = df[self.toxicity_columns].max(axis=1)
        df = df[['comment_text', 'label']]
        df.columns = ['text', 'label']
        df = df.dropna()
        
        toxic_count = df['label'].sum()
        non_toxic_count = len(df) - toxic_count
        toxic_pct = (toxic_count / len(df)) * 100
        
        print(f"Toxic: {toxic_count:,} ({toxic_pct:.2f}%), Non-toxic: {non_toxic_count:,} ({100-toxic_pct:.2f}%)")
        
        if not use_full_dataset:
            sample_size = 50000
            print(f"Sampling {sample_size:,} comments...")
            df = df.sample(n=min(sample_size, len(df)), random_state=42, replace=False)
        
        df_balanced = self._balance_classes(df)
        return df_balanced
    
    def _balance_classes(self, df):
        df_toxic = df[df['label'] == 1]
        df_non_toxic = df[df['label'] == 0]
        
        min_count = min(len(df_toxic), len(df_non_toxic))
        
        df_toxic_balanced = resample(df_toxic, n_samples=min_count, random_state=42, replace=False)
        df_non_toxic_balanced = resample(df_non_toxic, n_samples=min_count, random_state=42, replace=False)
        
        df_balanced = pd.concat([df_toxic_balanced, df_non_toxic_balanced])
        df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"Balanced dataset: {len(df_balanced):,} comments (50/50 split)")
        return df_balanced
