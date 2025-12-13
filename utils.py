import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1e6:.2f} MB")
    
    return trainable_params


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
    
    Caught {recall*100:.1f}% of toxic comments
    {precision*100:.1f}% of flags were correct
    False alarm rate: {fp/(fp+tn)*100:.1f}%
    """
    
    ax6.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}model_evaluation.png', dpi=300, bbox_inches='tight')
    print(f"Visualizations saved to {save_path}model_evaluation.png")
    
    return fig


def print_detailed_report(y_test, y_pred, y_probs):
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['Non-Toxic', 'Toxic'],
        digits=4
    ))
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print("Confusion Matrix:")
    print(f"  True Negatives:  {tn:,}")
    print(f"  False Positives: {fp:,}")
    print(f"  False Negatives: {fn:,}")
    print(f"  True Positives:  {tp:,}")
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_probs)
    
    print(f"\nRecall: {recall*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"ROC-AUC: {roc_auc:.4f}")


def interactive_testing(classifier):
    test_cases = [
        "You are amazing and very helpful, thank you!",
        "I hate you so much, you worthless piece of garbage",
        "This is a great discussion, I appreciate your input",
        "Stupid idiot, nobody wants you here",
        "Have a wonderful day!",
        "You should die, loser",
        "I respectfully disagree",
        "You're disgusting trash",
        "i don't think you are bad person"
    ]
    
    print("\nTesting sample comments:")
    for i, comment in enumerate(test_cases, 1):
        result = classifier.predict_toxicity(comment)
        status = "TOXIC" if result['is_toxic'] else "SAFE"
        print(f"{i}. {comment[:60]}")
        print(f"   {status} | Score: {result['toxicity_score']:.3f}")
    
    print("\nEnter your own comments (type 'quit' to exit)")
    
    while True:
        try:
            user_input = input("\nEnter comment: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q', '']:
                break
            
            result = classifier.predict_toxicity(user_input)
            print(f"Result: {result['label']}")
            print(f"Toxicity: {result['toxicity_score']:.3f}")
            print(f"Confidence: {result['confidence']:.3f}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")