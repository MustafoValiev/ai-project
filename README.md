# Toxic Comment Detector

BERT-based toxic comment classification using PyTorch and Transformers.

## Project Structure

```
ai-project/
├── main.py              # Main training script
├── data_loader.py        # Data loading and preprocessing
├── dataset.py           # PyTorch dataset class
├── model.py             # Model architecture and classifier
├── utils.py             # Visualization and utility functions
├── requirements.txt     # Dependencies
├── data/
│   ├── train.csv        # Training data
│   └── test.csv         # Test data
├── models/              # Saved models
└── results/             # Evaluation visualizations
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

pip install -r requirements.txt
```

For GPU support, install PyTorch with CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Usage

Train the model:
```bash
python main.py
```

The script will:
- Load and preprocess the dataset
- Train a BERT-based classifier
- Evaluate on test set
- Save model to `models/toxic_detector.pth`
- Generate visualizations in `results/`
- Run interactive testing

## Model Architecture

- **Base Model**: DistilBERT (distilbert-base-uncased)
- **Classifier**: 2-layer neural network (512 → 256 → 2)
- **Features**: BERT embeddings (768 dimensions)
- **Context Understanding**: Full sentence context with attention mechanism

Run the demo:
```bash
python app.py
```