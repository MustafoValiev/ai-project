from flask import Flask, render_template, request, jsonify
import torch
import os
from pathlib import Path

app = Flask(__name__)

model = None
device = None

def load_model():
    global model, device
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        from model import ToxicClassifier
        model = ToxicClassifier(device=device)

        model_path = 'models/toxic_detector.pth'
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return False

        checkpoint = torch.load(model_path, map_location=device)
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.training_history = checkpoint.get('training_history', {})

        file_size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"Model loaded ({file_size:.2f} MB)")
        return True

    except Exception as e:
        print(f"Model load error: {e}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()

        if not text:
            return jsonify({'error': 'Empty input'}), 400
        if len(text) > 5000:
            return jsonify({'error': 'Text too long'}), 400
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        result = model.predict_toxicity(text)
        result['text_length'] = len(text)
        result['word_count'] = len(text.split())
        result['char_count'] = len(text)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        data = request.get_json()
        texts = data.get('texts', [])

        if not texts:
            return jsonify({'error': 'No input'}), 400
        if len(texts) > 100:
            return jsonify({'error': 'Too many inputs'}), 400
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        results = []
        for i, text in enumerate(texts):
            if text and text.strip():
                result = model.predict_toxicity(text.strip())
                result['index'] = i
                result['text_preview'] = text[:100]
                results.append(result)

        return jsonify({'results': results, 'total': len(results)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'device': str(device) if device else 'unknown'
    })

@app.route('/stats')
def stats():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)

        return jsonify({
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(device),
            'training_history': model.training_history
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    Path('models').mkdir(exist_ok=True)
    Path('templates').mkdir(exist_ok=True)
    Path('static').mkdir(exist_ok=True)

    if not load_model():
        raise RuntimeError("Model failed to load")

    app.run(debug=True, host='127.0.0.1', port=5001)
