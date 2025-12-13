from flask import Flask, render_template, request, jsonify
import torch
import os

app = Flask(__name__)

model = None
device = None

def load_model():
    global model, device

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from model import ToxicClassifier
    model = ToxicClassifier(device=device)

    model_path = 'models/toxic_detector_cnn.pth'
    if not os.path.exists(model_path):
        return False

    model.load_model(model_path)
    return True

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

        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        result = model.predict_toxicity(text)
        result['text_length'] = len(text)
        result['word_count'] = len(text.split())

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

        results = []
        for text in texts:
            if text and text.strip():
                result = model.predict_toxicity(text.strip())
                result['text'] = text[:100]
                results.append(result)

        return jsonify({'results': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'device': str(device)
    })

if __name__ == '__main__':
    if not load_model():
        raise RuntimeError('Model failed to load')

    app.run(debug=True, host='127.0.0.1', port=5000)
