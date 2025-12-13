# model.py
import torch
import torch.nn as nn
import numpy as np


class ToxicCommentCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, num_filters=128, 
                 filter_sizes=[3, 4, 5], dropout=0.5, pretrained_embeddings=None):
        super(ToxicCommentCNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            self.embedding.weight.requires_grad = True
        
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, 
                     out_channels=num_filters, 
                     kernel_size=fs)
            for fs in filter_sizes
        ])
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(num_filters)
            for _ in filter_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(len(filter_sizes) * num_filters, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 2)
        
    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.permute(0, 2, 1)
        
        conved = []
        for conv, bn in zip(self.convs, self.batch_norms):
            c = conv(embedded)
            c = bn(c)
            c = torch.relu(c)
            c = torch.max_pool1d(c, c.shape[2]).squeeze(2)
            conved.append(c)
        
        cat = torch.cat(conved, dim=1)
        cat = self.dropout(cat)
        
        x = torch.relu(self.bn1(self.fc1(cat)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class ToxicClassifier:
    def __init__(self, device, vocab_size=30000, embedding_dim=300, 
                 num_filters=128, filter_sizes=[3, 4, 5]):
        self.device = device
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        
        self.model = ToxicCommentCNN(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_filters=num_filters,
            filter_sizes=filter_sizes
        ).to(device)
        
        self.training_history = {'train_loss': [], 'val_acc': [], 'val_f1': []}
        
    def build_vocabulary(self, texts, max_vocab_size=30000):
        from collections import Counter
        
        word_counts = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            word_counts.update(tokens)
        
        most_common = word_counts.most_common(max_vocab_size - 2)
        
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        print(f"Vocabulary size: {len(self.word2idx):,}")
        return self.word2idx
    
    def load_pretrained_embeddings(self, glove_path='glove.6B.300d.txt'):
        import os
        
        if not os.path.exists(glove_path):
            print("GloVe embeddings not found, using random initialization")
            return None
        
        print("Loading GloVe embeddings...")
        embeddings_index = {}
        
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        
        embedding_matrix = np.random.randn(len(self.word2idx), self.embedding_dim).astype('float32') * 0.01
        
        found = 0
        for word, idx in self.word2idx.items():
            if word in embeddings_index:
                embedding_matrix[idx] = embeddings_index[word]
                found += 1
        
        print(f"Found {found}/{len(self.word2idx)} words in GloVe")
        return embedding_matrix
    
    def tokenize(self, text):
        import re
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text.split()
    
    def text_to_sequence(self, text, max_length=200):
        tokens = self.tokenize(text)
        sequence = [self.word2idx.get(token, 1) for token in tokens]
        
        if len(sequence) < max_length:
            sequence = sequence + [0] * (max_length - len(sequence))
        else:
            sequence = sequence[:max_length]
        
        return sequence
        
    def train_model(self, train_loader, val_loader, epochs=10, lr=0.001):
        import torch.optim as optim
        import numpy as np
        from tqdm import tqdm
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                         factor=0.5, patience=2, verbose=True)
        
        print(f"\nTraining CNN for {epochs} epochs on {self.device}")
        print(f"Learning rate: {lr}")
        
        best_val_f1 = 0
        patience_counter = 0
        patience = 5
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            for batch in pbar:
                sequences = batch['sequence'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(sequences)
                loss = criterion(outputs, labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
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
                for batch in val_loader:
                    sequences = batch['sequence'].to(self.device)
                    labels = batch['label']
                    
                    outputs = self.model(sequences)
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
            
            scheduler.step(val_f1)
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}")
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                print(f"  New best F1: {best_val_f1:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        print(f"Training complete. Best F1: {best_val_f1:.4f}")
        return self.training_history
    
    def evaluate(self, test_loader):
        import numpy as np
        from tqdm import tqdm
        
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                sequences = batch['sequence'].to(self.device)
                labels = batch['label']
                
                outputs = self.model(sequences)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        return np.array(all_labels), np.array(all_preds), np.array(all_probs)
    
    def predict_toxicity(self, text, threshold=0.5):
        self.model.eval()
        
        sequence = self.text_to_sequence(text)
        sequence_tensor = torch.LongTensor([sequence]).to(self.device)
        
        with torch.no_grad():
            output = self.model(sequence_tensor)
            probs = torch.softmax(output, dim=1)
            toxicity_prob = float(probs[0, 1].item())
        
        is_toxic = toxicity_prob >= threshold
        
        return {
            'is_toxic': is_toxic,
            'toxicity_score': toxicity_prob,
            'confidence': max(toxicity_prob, 1 - toxicity_prob),
            'label': 'TOXIC' if is_toxic else 'SAFE'
        }
    
    def save_model(self, filepath='models/toxic_detector_cnn.pth'):
        import os
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'training_history': self.training_history
        }, filepath)
        
        file_size = os.path.getsize(filepath) / (1024 * 1024)
        print(f"Model saved to {filepath} ({file_size:.2f} MB)")
    
    def load_model(self, filepath='models/toxic_detector_cnn.pth'):
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.word2idx = checkpoint['word2idx']
        self.idx2word = checkpoint['idx2word']
        self.vocab_size = checkpoint['vocab_size']
        self.embedding_dim = checkpoint['embedding_dim']
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.training_history = checkpoint.get('training_history', {})
        
        print(f"Model loaded from {filepath}")