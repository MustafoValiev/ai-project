import pandas as pd
import numpy as np
import re
import os
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def setup_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("*"*60)
        print("GPU DETECTED AND ACTIVATED")
        print("*"*60)
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print("*"*60)
    else:
        device = torch.device('cpu')
        print("*"*60)
        print("GPU NOT AVAILABLE - Using CPU")
        print("*"*60)
    return device


class KaggleDataLoader:
    def __init__(self, train_path='./data/train.csv', test_path='./data/test.csv'):
        self.train_path = train_path
        self.test_path = test_path
        self.toxicity_columns = ['toxic', 'severe_toxic', 'obscene', 
                                 'threat', 'insult', 'identity_hate']
    
    def load_data(self, use_full_dataset=True):
        print("\n" + ("*"*60))
        print("LOADING KAGGLE DATASET")
        print("*"*60)
        
        if not os.path.exists(self.train_path):
            raise FileNotFoundError(f"Dataset not found at: {self.train_path}")
        
        df = pd.read_csv(self.train_path)
        print(f"Loaded {len(df):,} comments from training set")
        
        df['label'] = df[self.toxicity_columns].max(axis=1)
        df = df[['comment_text', 'label']]
        df.columns = ['text', 'label']
        df = df.dropna()
        
        toxic_count = df['label'].sum()
        non_toxic_count = len(df) - toxic_count
        toxic_pct = (toxic_count / len(df)) * 100
        
        print(f"\nOriginal Distribution:")
        print(f"  Toxic:     {toxic_count:>7,} ({toxic_pct:>5.2f}%)")
        print(f"  Non-Toxic: {non_toxic_count:>7,} ({100-toxic_pct:>5.2f}%)")
        
        if not use_full_dataset:
            sample_size = 50000
            print(f"\nSampling {sample_size:,} comments for faster training...")
            df = df.sample(n=min(sample_size, len(df)), random_state=42, replace=False)
        
        df_balanced = self._balance_classes(df)
        
        return df_balanced
    
    def _balance_classes(self, df):
        from sklearn.utils import resample
        
        print(f"\nBalancing dataset...")
        
        df_toxic = df[df['label'] == 1]
        df_non_toxic = df[df['label'] == 0]
        
        min_count = min(len(df_toxic), len(df_non_toxic))
        
        df_toxic_balanced = resample(df_toxic, n_samples=min_count, random_state=42, replace=False)
        df_non_toxic_balanced = resample(df_non_toxic, n_samples=min_count, random_state=42, replace=False)
        
        df_balanced = pd.concat([df_toxic_balanced, df_non_toxic_balanced])
        df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"Balanced to {len(df_balanced):,} comments")
        print(f"  Toxic:     {df_balanced['label'].sum():>7,} (50.0%)")
        print(f"  Non-Toxic: {len(df_balanced) - df_balanced['label'].sum():>7,} (50.0%)")
        
        return df_balanced


class TextPreprocessor:
    def __init__(self):
        self.patterns = {
            'url': r'http\S+|www\.\S+',
            'ip': r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',
            'username': r'@\w+',
            'newline': r'\n',
        }
    
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = re.sub(self.patterns['ip'], ' ', text)
        text = re.sub(self.patterns['url'], ' ', text)
        text = re.sub(self.patterns['username'], ' ', text)
        text = re.sub(self.patterns['newline'], ' ', text)
        text = re.sub(r'[^a-zA-Z\s\.\!\?]', ' ', text)
        text = ' '.join(text.split())
        
        return text
    
    def preprocess_dataset(self, texts, show_progress=True):
        if show_progress and len(texts) > 1000:
            return [self.clean_text(text) for text in tqdm(texts, desc="Preprocessing")]
        else:
            return [self.clean_text(text) for text in texts]


class ToxicCommentsDataset(Dataset):
    def __init__(self, texts, labels, vectorizer=None, max_features=10000):
        self.texts = texts
        self.labels = labels
        
        if vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 3),
                min_df=3,
                max_df=0.9,
                strip_accents='unicode',
                sublinear_tf=True
            )
            self.features = self.vectorizer.fit_transform(texts).toarray()
        else:
            self.vectorizer = vectorizer
            self.features = vectorizer.transform(texts).toarray()
        
        self.features = torch.FloatTensor(self.features)
        self.labels = torch.LongTensor(labels.values if hasattr(labels, 'values') else labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class ToxicCommentNN(nn.Module):
    def __init__(self, input_size, hidden_sizes=[1024, 512, 256], dropout=0.4):
        super(ToxicCommentNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 2))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + ("*"*60))
    print("MODEL ARCHITECTURE ANALYSIS")
    print("*"*60)
    print(f"\nTotal Parameters:      {total_params:>12,}")
    print(f"Trainable Parameters:  {trainable_params:>12,}")
    print(f"Model Size:            {total_params * 4 / 1e6:>12.2f} MB")
    print("*"*60)
    
    return trainable_params


class ToxicClassifier:
    def __init__(self, device, input_size=10000, hidden_sizes=[1024, 512, 256]):
        self.device = device
        self.model = ToxicCommentNN(input_size, hidden_sizes).to(device)
        self.preprocessor = TextPreprocessor()
        self.vectorizer = None
        self.training_history = {'train_loss': [], 'val_acc': [], 'val_f1': []}
        
    def train_model(self, train_loader, val_loader, epochs=15, lr=0.0005):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=lr*10, 
            steps_per_epoch=len(train_loader), 
            epochs=epochs,
            pct_start=0.1
        )
        
        print("\n" + ("*"*60))
        print("TRAINING NEURAL NETWORK")
        print("*"*60)
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"Initial Learning Rate: {lr}")
        print(f"Optimizer: AdamW with OneCycleLR")
        print("*"*60)
        
        best_val_f1 = 0
        patience_counter = 0
        patience = 5
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            for features, labels in pbar:
                features, labels = features.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_train_loss = train_loss / len(train_loader)
            train_acc = 100 * train_correct / train_total
            
            self.model.eval()
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for features, labels in val_loader:
                    features = features.to(self.device)
                    outputs = self.model(features)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    val_preds.extend(predicted.cpu().numpy())
                    val_labels.extend(labels.numpy())
            
            val_preds = np.array(val_preds)
            val_labels = np.array(val_labels)
            
            val_acc = accuracy_score(val_labels, val_preds) * 100
            _, _, val_f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='binary')
            
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['val_f1'].append(val_f1)
            
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Loss: {avg_train_loss:.4f}, "
                  f"Train Acc: {train_acc:.2f}%, "
                  f"Val Acc: {val_acc:.2f}%, "
                  f"Val F1: {val_f1:.4f}")
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                print(f"  New best F1: {best_val_f1:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
        
        print("*"*60)
        print(f"Training Complete! Best Val F1: {best_val_f1:.4f}")
        print("*"*60)
        
        return self.training_history
    
    def evaluate(self, test_loader):
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for features, labels in tqdm(test_loader, desc="Evaluating"):
                features = features.to(self.device)
                outputs = self.model(features)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        return np.array(all_labels), np.array(all_preds), np.array(all_probs)
    
    def predict_toxicity(self, text):
        self.model.eval()
        
        cleaned = self.preprocessor.clean_text(text)
        features = self.vectorizer.transform([cleaned]).toarray()
        features = torch.FloatTensor(features).to(self.device)
        
        with torch.no_grad():
            output = self.model(features)
            probs = torch.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
        
        return {
            'is_toxic': bool(predicted.item()),
            'toxicity_score': float(probs[0, 1].item()),
            'confidence': float(torch.max(probs).item()),
            'label': 'TOXIC' if predicted.item() == 1 else 'SAFE'
        }
    
    def save_model(self, filepath='models/toxic_detector.pth'):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'vectorizer': self.vectorizer,
            'preprocessor': self.preprocessor,
            'training_history': self.training_history
        }, filepath)
        
        file_size = os.path.getsize(filepath) / (1024 * 1024)
        print(f"\nModel saved to: {filepath} ({file_size:.2f} MB)")


def generate_visualizations(y_test, y_pred, y_probs, training_history, save_path='results/'):
    os.makedirs(save_path, exist_ok=True)
    
    fig = plt.figure(figsize=(18, 10))
    
    ax1 = plt.subplot(2, 3, 1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
               xticklabels=['Non-Toxic', 'Toxic'],
               yticklabels=['Non-Toxic', 'Toxic'])
    ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Actual')
    ax1.set_xlabel('Predicted')
    
    ax2 = plt.subplot(2, 3, 2)
    if training_history.get('train_loss'):
        ax2.plot(training_history['train_loss'], marker='o', label='Training Loss')
        ax2.set_title('Training Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(2, 3, 3)
    if training_history.get('val_f1'):
        ax3.plot(training_history['val_f1'], marker='s', color='green', label='Validation F1')
        ax3.set_title('Validation F1-Score', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('F1-Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    ax4 = plt.subplot(2, 3, 4)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    roc_auc = roc_auc_score(y_test, y_probs)
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    values = [accuracy, precision, recall, f1, roc_auc]
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum']
    
    bars = ax4.barh(metrics, values, color=colors)
    ax4.set_xlim(0, 1.0)
    ax4.set_title('Performance Metrics', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Score')
    
    for bar, value in zip(bars, values):
        ax4.text(value + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{value:.4f}', va='center', fontweight='bold')
    
    ax5 = plt.subplot(2, 3, 5)
    tn, fp, fn, tp = cm.ravel()
    categories = ['True\nNegative', 'False\nPositive', 'False\nNegative', 'True\nPositive']
    counts = [tn, fp, fn, tp]
    colors_dist = ['lightgreen', 'salmon', 'orange', 'lightblue']
    
    bars = ax5.bar(categories, counts, color=colors_dist)
    ax5.set_title('Prediction Distribution', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Count')
    
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = f"""
    PERFORMANCE SUMMARY
    {'='*40}
    
    Total Test Samples: {len(y_test):,}
    
    Accuracy:   {accuracy:.4f} ({accuracy*100:.2f}%)
    Precision:  {precision:.4f}
    Recall:     {recall:.4f}
    F1-Score:   {f1:.4f}
    ROC-AUC:    {roc_auc:.4f}
    
    Confusion Matrix:
      TP: {tp:,}  TN: {tn:,}
      FP: {fp:,}  FN: {fn:,}
    
    Insights:
    • Caught {recall*100:.1f}% of toxic comments
    • {precision*100:.1f}% of flags were correct
    • False alarm: {fp/(fp+tn)*100:.1f}%
    """
    
    ax6.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}model_evaluation.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualizations saved to: {save_path}model_evaluation.png")
    
    return fig


def print_detailed_report(y_test, y_pred, y_probs):
    print("\n" + ("*"*60))
    print("DETAILED CLASSIFICATION REPORT")
    print("*"*60)
    
    print("\n" + classification_report(
        y_test, y_pred,
        target_names=['Non-Toxic', 'Toxic'],
        digits=4
    ))
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print("Confusion Matrix Breakdown:")
    print(f"  True Negatives:  {tn:>6,}")
    print(f"  False Positives: {fp:>6,}")
    print(f"  False Negatives: {fn:>6,}")
    print(f"  True Positives:  {tp:>6,}")
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    print(f"\nKey Metrics:")
    print(f"  Recall:      {recall*100:>6.2f}%")
    print(f"  Precision:   {precision*100:>6.2f}%")
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_probs)
    
    print(f"\nOverall:")
    print(f"  Accuracy:    {accuracy*100:>6.2f}%")
    print(f"  ROC-AUC:     {roc_auc:>7.4f}")
    print("*"*60)


def interactive_testing(classifier):
    print("\n" + ("*"*60))
    print("INTERACTIVE TESTING MODE")
    print("*"*60)
    
    test_cases = [
        "You are amazing and very helpful, thank you!",
        "I hate you so much, you worthless piece of garbage",
        "This is a great discussion, I appreciate your input",
        "Stupid idiot, nobody wants you here",
        "Have a wonderful day!",
        "You should die, loser",
        "I respectfully disagree",
        "You're disgusting trash"
    ]
    
    print("\nTesting sample comments:\n")
    
    for i, comment in enumerate(test_cases, 1):
        result = classifier.predict_toxicity(comment)
        status = "TOXIC" if result['is_toxic'] else "SAFE"
        print(f"{i}. {comment[:60]}")
        print(f"   {status} | Score: {result['toxicity_score']:.3f}\n")
    
    print("\n" + ("*"*60))
    print("Try your own comments (type 'quit' to exit)")
    print("*"*60)
    
    while True:
        try:
            user_input = input("\nEnter comment: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q', '']:
                print("\nThank you!")
                break
            
            result = classifier.predict_toxicity(user_input)
            
            print(f"\nResult: {result['label']}")
            print(f"Toxicity: {result['toxicity_score']:.3f}")
            print(f"Confidence: {result['confidence']:.3f}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    print("*"*60)
    print("*"*60)
    
    TRAIN_VAL_SPLIT = 0.85
    VAL_SIZE = 0.15
    BATCH_SIZE = 256
    EPOCHS = 15
    LEARNING_RATE = 0.0005
    INPUT_SIZE = 10000
    HIDDEN_SIZES = [1024, 512, 256]
    USE_FULL_DATASET = True
    
    try:
        print("\n" + ("*"*60))
        print("STEP 1: DEVICE SETUP")
        print("*"*60)
        device = setup_device()
        
        print("\n" + ("*"*60))
        print("STEP 2: LOADING DATASET")
        print("*"*60)
        
        loader = KaggleDataLoader(train_path='./data/train.csv')
        df = loader.load_data(use_full_dataset=USE_FULL_DATASET)
        
        print("\n" + ("*"*60))
        print("STEP 3: TEXT PREPROCESSING")
        print("*"*60)
        
        preprocessor = TextPreprocessor()
        print("Cleaning text...")
        df['cleaned'] = preprocessor.preprocess_dataset(df['text'].values)
        print("Preprocessing complete")
        
        print("\n" + ("*"*60))
        print("STEP 4: DATA SPLITTING")
        print("*"*60)
        
        train_val_df, test_df = train_test_split(
            df, test_size=(1-TRAIN_VAL_SPLIT), random_state=42, stratify=df['label']
        )
        
        train_df, val_df = train_test_split(
            train_val_df, test_size=VAL_SIZE, random_state=42, stratify=train_val_df['label']
        )
        
        print(f"Training:   {len(train_df):>6,} ({len(train_df)/len(df)*100:.1f}%)")
        print(f"Validation: {len(val_df):>6,} ({len(val_df)/len(df)*100:.1f}%)")
        print(f"Test:       {len(test_df):>6,} ({len(test_df)/len(df)*100:.1f}%)")
        
        print("\n" + ("*"*60))
        print("STEP 5: CREATING DATASETS")
        print("*"*60)
        
        print("Creating TF-IDF features...")
        train_dataset = ToxicCommentsDataset(
            train_df['cleaned'].values,
            train_df['label'].values,
            max_features=INPUT_SIZE
        )
        vectorizer = train_dataset.vectorizer
        
        val_dataset = ToxicCommentsDataset(
            val_df['cleaned'].values,
            val_df['label'].values,
            vectorizer=vectorizer
        )
        
        test_dataset = ToxicCommentsDataset(
            test_df['cleaned'].values,
            test_df['label'].values,
            vectorizer=vectorizer
        )
        
        print(f"Vocabulary: {len(vectorizer.vocabulary_):,} words")
        print(f"Features: {INPUT_SIZE:,}")
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
        
        print(f"Batch size: {BATCH_SIZE}")
        
        print("\n" + ("*"*60))
        print("STEP 6: MODEL ARCHITECTURE")
        print("*"*60)
        
        classifier = ToxicClassifier(device, INPUT_SIZE, HIDDEN_SIZES)
        classifier.vectorizer = vectorizer
        
        print(f"\nModel: {classifier.model}")
        total_params = count_parameters(classifier.model)
        
        print("\n" + ("*"*60))
        print("STEP 7: TRAINING")
        print("*"*60)
        
        start_time = time.time()
        training_history = classifier.train_model(
            train_loader, val_loader,
            epochs=EPOCHS,
            lr=LEARNING_RATE
        )
        training_time = time.time() - start_time
        
        print(f"\nTraining Time: {training_time:.2f}s ({training_time/60:.2f}min)")
        
        print("\n" + ("*"*60))
        print("STEP 8: EVALUATION")
        print("*"*60)
        
        print("Evaluating...")
        y_test, y_pred, y_probs = classifier.evaluate(test_loader)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        roc_auc = roc_auc_score(y_test, y_probs)
        
        print("\n" + ("*"*60))
        print("TEST SET RESULTS")
        print("*"*60)
        print(f"\nAccuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        
        print_detailed_report(y_test, y_pred, y_probs)
        
        print("\n" + ("*"*60))
        print("STEP 9: VISUALIZATIONS")
        print("*"*60)
        
        generate_visualizations(y_test, y_pred, y_probs, training_history)
        
        print("\n" + ("*"*60))
        print("STEP 10: SAVING MODEL")
        print("*"*60)
        
        classifier.save_model('models/toxic_detector.pth')
        
        print("\n" + ("*"*60))
        print("STEP 11: INTERACTIVE TESTING")
        print("*"*60)
        
        interactive_testing(classifier)
        
        print("\n" + ("*"*60))
        print("PROJECT COMPLETED")
        print(f"\nSummary:")
        print(f"  Dataset:       {len(df):,} comments")
        print(f"  Parameters:    {total_params:,}")
        print(f"  Training Time: {training_time:.2f}s")
        print(f"  Accuracy:      {accuracy*100:.2f}%")
        print(f"  F1-Score:      {f1:.4f}")
        print(f"  Device:        {device}")
        
        print(f"\nFiles:")
        print(f"  models/toxic_detector_gpu.pth")
        print(f"  results/model_evaluation.png")
        
        print("*"*60)
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nEnsure dataset is at: ./data/train.csv")

    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()

        
if __name__ == "__main__":
    main()