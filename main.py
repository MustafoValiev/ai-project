import os
import time
import torch
import platform
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from transformers import DistilBertTokenizer

from data_loader import KaggleDataLoader
from dataset import ToxicCommentsDataset
from model import ToxicClassifier
from utils import count_parameters, generate_visualizations, print_detailed_report, interactive_testing


def setup_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def main():
    TRAIN_VAL_SPLIT = 0.85
    VAL_SIZE = 0.15
    BATCH_SIZE = 16
    EPOCHS = 5
    LEARNING_RATE = 2e-5
    HIDDEN_SIZES = [512, 256]
    USE_FULL_DATASET = True
    MAX_LENGTH = 128
    
    try:
        device = setup_device()
        
        loader = KaggleDataLoader(train_path='./data/train.csv')
        df = loader.load_data(use_full_dataset=USE_FULL_DATASET)
        
        train_val_df, test_df = train_test_split(
            df, test_size=(1-TRAIN_VAL_SPLIT), random_state=42, stratify=df['label']
        )
        
        train_df, val_df = train_test_split(
            train_val_df, test_size=VAL_SIZE, random_state=42, stratify=train_val_df['label']
        )
        
        print(f"Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")
        
        print("Creating BERT datasets...")
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        train_dataset = ToxicCommentsDataset(
            train_df['text'].values,
            train_df['label'].values,
            tokenizer=tokenizer,
            max_length=MAX_LENGTH
        )
        
        val_dataset = ToxicCommentsDataset(
            val_df['text'].values,
            val_df['label'].values,
            tokenizer=tokenizer,
            max_length=MAX_LENGTH
        )
        
        test_dataset = ToxicCommentsDataset(
            test_df['text'].values,
            test_df['label'].values,
            tokenizer=tokenizer,
            max_length=MAX_LENGTH
        )
        
        num_workers = 0 if platform.system() == 'Windows' else 2
        pin_memory = torch.cuda.is_available() and platform.system() != 'Windows'
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        
        classifier = ToxicClassifier(device, HIDDEN_SIZES)
        
        print(f"\nModel architecture:\n{classifier.model}")
        total_params = count_parameters(classifier.model)
        
        start_time = time.time()
        training_history = classifier.train_model(
            train_loader, val_loader,
            epochs=EPOCHS,
            lr=LEARNING_RATE
        )
        training_time = time.time() - start_time
        
        print(f"Training time: {training_time:.2f}s ({training_time/60:.2f}min)")
        
        print("\nEvaluating on test set...")
        y_test, y_pred, y_probs = classifier.evaluate(test_loader)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        roc_auc = roc_auc_score(y_test, y_probs)
        
        print(f"\nTest Results:")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        
        print_detailed_report(y_test, y_pred, y_probs)
        
        generate_visualizations(y_test, y_pred, y_probs, training_history)
        
        classifier.save_model('models/toxic_detector.pth')
        
        interactive_testing(classifier)
        
        print(f"\nDone! Dataset: {len(df):,} comments, Parameters: {total_params:,}, Accuracy: {accuracy*100:.2f}%")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure data/train.csv exists")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

        
if __name__ == "__main__":
    main()