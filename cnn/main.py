import os
import time
import torch
import platform
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from data_loader import KaggleDataLoader
from dataset import ToxicCommentsDataset
from model import ToxicClassifier
from utils import count_parameters, generate_visualizations, print_detailed_report


def setup_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    TRAIN_VAL_SPLIT = 0.85
    VAL_SIZE = 0.15
    BATCH_SIZE = 64
    EPOCHS = 10
    LEARNING_RATE = 0.001
    USE_FULL_DATASET = True
    MAX_LENGTH = 200
    VOCAB_SIZE = 30000
    EMBEDDING_DIM = 300
    NUM_FILTERS = 128
    FILTER_SIZES = [3, 4, 5]

    device = setup_device()

    loader = KaggleDataLoader(train_path='../data/train.csv')
    df = loader.load_data(use_full_dataset=USE_FULL_DATASET)

    train_val_df, test_df = train_test_split(
        df,
        test_size=(1 - TRAIN_VAL_SPLIT),
        random_state=42,
        stratify=df['label']
    )

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=VAL_SIZE,
        random_state=42,
        stratify=train_val_df['label']
    )

    classifier = ToxicClassifier(
        device=device,
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        num_filters=NUM_FILTERS,
        filter_sizes=FILTER_SIZES
    )

    classifier.build_vocabulary(
        train_df['text'].values,
        max_vocab_size=VOCAB_SIZE
    )

    train_dataset = ToxicCommentsDataset(
        train_df['text'].values,
        train_df['label'].values,
        word2idx=classifier.word2idx,
        max_length=MAX_LENGTH
    )

    val_dataset = ToxicCommentsDataset(
        val_df['text'].values,
        val_df['label'].values,
        word2idx=classifier.word2idx,
        max_length=MAX_LENGTH
    )

    test_dataset = ToxicCommentsDataset(
        test_df['text'].values,
        test_df['label'].values,
        word2idx=classifier.word2idx,
        max_length=MAX_LENGTH
    )

    num_workers = 0 if platform.system() == 'Windows' else 2
    pin_memory = torch.cuda.is_available() and platform.system() != 'Windows'

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    total_params = count_parameters(classifier.model)

    start_time = time.time()
    training_history = classifier.train_model(
        train_loader,
        val_loader,
        epochs=EPOCHS,
        lr=LEARNING_RATE
    )
    training_time = time.time() - start_time

    y_test, y_pred, y_probs = classifier.evaluate(test_loader)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average='binary'
    )
    roc_auc = roc_auc_score(y_test, y_probs)

    print_detailed_report(y_test, y_pred, y_probs)

    os.makedirs('results', exist_ok=True)
    generate_visualizations(
        y_test,
        y_pred,
        y_probs,
        training_history,
        save_path='results/'
    )

    classifier.save_model('models/toxic_detector_cnn.pth')


if __name__ == "__main__":
    main()
