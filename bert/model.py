#model.py
import torch
import torch.nn as nn
from transformers import DistilBertModel


class ToxicCommentNN(nn.Module):
    def __init__(self, bert_model, hidden_sizes=[512, 256], dropout=0.3):
        super(ToxicCommentNN, self).__init__()
        
        self.bert = bert_model
        for param in self.bert.parameters():
            param.requires_grad = False
        
        layers = []
        prev_size = 768
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 2))
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        return self.classifier(pooled_output)


class ToxicClassifier:
    def __init__(self, device, hidden_sizes=[512, 256]):
        self.device = device
        from transformers import DistilBertTokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.model = ToxicCommentNN(bert_model, hidden_sizes).to(device)
        self.training_history = {'train_loss': [], 'val_acc': [], 'val_f1': []}
        
    def train_model(self, train_loader, val_loader, epochs=5, lr=2e-5):
        import torch.nn as nn
        import torch.optim as optim
        import numpy as np
        from tqdm import tqdm
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        
        print(f"\nTraining for {epochs} epochs on {self.device}")
        print(f"Learning rate: {lr}")
        
        best_val_f1 = 0
        patience_counter = 0
        patience = 3
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
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
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label']
                    
                    outputs = self.model(input_ids, attention_mask)
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
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label']
                
                outputs = self.model(input_ids, attention_mask)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        return np.array(all_labels), np.array(all_preds), np.array(all_probs)
    
    def predict_toxicity(self, text, threshold=0.5):
        self.model.eval()
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            output = self.model(input_ids, attention_mask)
            probs = torch.softmax(output, dim=1)
            toxicity_prob = float(probs[0, 1].item())
        
        is_toxic = toxicity_prob >= threshold
        
        return {
            'is_toxic': is_toxic,
            'toxicity_score': toxicity_prob,
            'confidence': max(toxicity_prob, 1 - toxicity_prob),
            'label': 'TOXIC' if is_toxic else 'SAFE'
        }
    
    def save_model(self, filepath='models/toxic_detector.pth'):
        import os
        import torch
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'tokenizer_name': 'distilbert-base-uncased',
            'training_history': self.training_history
        }, filepath)
        
        file_size = os.path.getsize(filepath) / (1024 * 1024)
        print(f"Model saved to {filepath} ({file_size:.2f} MB)")
