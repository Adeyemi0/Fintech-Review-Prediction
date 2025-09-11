import os
import json
import joblib
import torch
import numpy as np
from flask import Flask, request, render_template_string, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

# -----------------------
# Load artifacts
# -----------------------
SAVE_DIR = "./model"

try:
    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR)
    model.eval()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(DEVICE)
    
    # Load MultiLabelBinarizer and labels
    mlb = joblib.load(os.path.join(SAVE_DIR, "mlb.joblib"))
    with open(os.path.join(SAVE_DIR, "labels.json"), "r", encoding="utf-8") as f:
        labels = json.load(f)
    
    MODEL_LOADED = True
    print(f"Model loaded successfully on device: {DEVICE}")
    print(f"Available labels: {labels}")
    
except Exception as e:
    MODEL_LOADED = False
    print(f"Error loading model: {e}")
    tokenizer = None
    model = None
    mlb = None
    labels = []

# Sigmoid for probabilities
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# -----------------------
# Prediction function (single text only)
# -----------------------
def predict_single(text, threshold=0.5):
    """Predict categories for a single text."""
    if not MODEL_LOADED:
        return [], []
    
    # Tokenize
    encodings = tokenizer(
        [text],  # Wrap in list since model expects batch
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    ).to(DEVICE)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits.cpu().numpy()
    
    # Convert to probabilities
    probs = sigmoid(logits)
    
    # Apply fixed threshold (0.5)
    pred_bin = (probs >= threshold).astype(int)
    
    # Decode to label names
    row_2d = np.array([pred_bin[0]])
    categories = mlb.inverse_transform(row_2d)[0]
    
    return list(categories), probs[0]

# HTML Template with embedded CSS + LinkedIn Footer
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fintech Review Category Classifier</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            display: flex;
            flex-direction: column;
        }

        .container {
            max-width: 1000px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
            overflow: hidden;
            flex: 1;
        }

        .header {
            background: linear-gradient(45deg, #2c3e50, #4a6741);
            color: white;
            padding: 30px;
            text-align: center;
        }

        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }

        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }

        .main-content {
            padding: 40px;
        }

        .input-section {
            margin-bottom: 30px;
        }

        .form-group {
            margin-bottom: 20px;
        }

        label {
            display: block;
            margin-bottom: 10px;
            font-weight: 600;
            color: #333;
            font-size: 1.1em;
        }

        textarea {
            width: 100%;
            min-height: 120px;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 16px;
            font-family: inherit;
            resize: vertical;
            transition: all 0.3s ease;
        }

        textarea:focus {
            border-color: #667eea;
            outline: none;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }

        .controls {
            display: flex;
            gap: 20px;
            align-items: center;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }

        .btn {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 15px 30px;
            font-size: 16px;
            font-weight: 600;
            border-radius: 25px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }

        .btn:active {
            transform: translateY(0);
        }

        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .results-section {
            margin-top: 30px;
        }

        .result-card {
            background: #f8f9ff;
            border: 1px solid #e0e8ff;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
        }

        .original-text {
            background: #fff;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            margin-bottom: 20px;
            font-style: italic;
            color: #555;
        }

        .categories {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 15px;
        }

        .category-tag {
            background: linear-gradient(45deg, #48bb78, #38a169);
            color: white;
            padding: 8px 15px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 500;
            box-shadow: 0 2px 5px rgba(72, 187, 120, 0.3);
        }

        .no-categories {
            color: #666;
            font-style: italic;
            padding: 10px;
            background: #f0f0f0;
            border-radius: 8px;
        }

        .loading {
            display: none;
            text-align: center;
            padding: 20px;
            color: #667eea;
            font-size: 18px;
        }

        .loading.show {
            display: block;
        }

        .error {
            background: #fed7d7;
            color: #c53030;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 4px solid #c53030;
        }

        .model-status {
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-weight: 500;
        }

        .model-status.loaded {
            background: #c6f6d5;
            color: #22543d;
            border-left: 4px solid #38a169;
        }

        .model-status.error {
            background: #fed7d7;
            color: #c53030;
            border-left: 4px solid #c53030;
        }

        footer {
            text-align: center;
            padding: 20px;
            background: #2c3e50;
            color: #ecf0f1;
            font-size: 14px;
            margin-top: auto;
        }

        footer a {
            color: #667eea;
            text-decoration: none;
            font-weight: 600;
        }

        footer a:hover {
            text-decoration: underline;
        }

        @media (max-width: 768px) {
            .header h1 {
                font-size: 2em;
            }
            
            .main-content {
                padding: 20px;
            }
            
            .controls {
                flex-direction: column;
                align-items: stretch;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏦 Fintech Review Classifier</h1>
            <p>Classify your customer review into relevant categories</p>
        </div>
        
        <div class="main-content">
            {% if model_loaded %}
                <div class="model-status loaded">
                    ✅ Model loaded successfully! Available categories: {{ labels|length }}
                </div>
            {% else %}
                <div class="model-status error">
                    ❌ Model could not be loaded. Please check if the model files exist in './model' directory.
                </div>
            {% endif %}
            
            <form id="classifyForm" {% if not model_loaded %}style="opacity: 0.5; pointer-events: none;"{% endif %}>
                <div class="input-section">
                    <div class="form-group">
                        <label for="review_text">Enter Customer Review:</label>
                        <textarea id="review_text" name="review_text" placeholder="Type a single customer review here..." required>{{ sample_text if sample_text else 'The app crashes every time I try to open it.' }}</textarea>
                    </div>
                    
                    <div class="controls">
                        <button type="submit" class="btn" {% if not model_loaded %}disabled{% endif %}>
                            🔍 Classify Review
                        </button>
                    </div>
                </div>
            </form>
            
            <div class="loading" id="loading">
                <div>🤖 Analyzing review...</div>
            </div>
            
            <div class="results-section" id="results" style="display: none;">
                <!-- Results will be inserted here -->
            </div>
        </div>
    </div>

    <footer>
        Made with ❤️ by Adediran Adeyemi — <a href="https://www.linkedin.com/in/adediran-adeyemi-17103b114/" target="_blank">Connect with me on LinkedIn</a>
    </footer>

    <script>
        // Handle form submission
        document.getElementById('classifyForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const loading = document.getElementById('loading');
            const results = document.getElementById('results');
            
            // Show loading, hide results
            loading.classList.add('show');
            results.style.display = 'none';
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.error) {
                    throw new Error(data.error);
                }
                
                displayResults(data);
                
            } catch (error) {
                results.innerHTML = '<div class="error">❌ Error: ' + error.message + '</div>';
                results.style.display = 'block';
            } finally {
                loading.classList.remove('show');
            }
        });

        function displayResults(data) {
            const results = document.getElementById('results');
            
            // Clear any existing content completely
            results.innerHTML = '';
            
            // Create results header
            const header = document.createElement('h2');
            header.textContent = '🎯 Classification Result';
            results.appendChild(header);
            
            // Only one result expected
            const result = data.results[0];
            const card = document.createElement('div');
            card.className = 'result-card';
            
            // Original text section
            const textDiv = document.createElement('div');
            textDiv.className = 'original-text';
            textDiv.innerHTML = `<strong>Review:</strong> "${result.text}"`;
            card.appendChild(textDiv);
            
            // Categories section
            const categoriesDiv = document.createElement('div');
            categoriesDiv.className = 'categories';
            
            if (result.categories.length > 0) {
                result.categories.forEach(cat => {
                    const tag = document.createElement('span');
                    tag.className = 'category-tag';
                    tag.textContent = cat;
                    categoriesDiv.appendChild(tag);
                });
            } else {
                const noCategories = document.createElement('div');
                noCategories.className = 'no-categories';
                noCategories.textContent = 'No categories above threshold';
                categoriesDiv.appendChild(noCategories);
            }
            card.appendChild(categoriesDiv);
            
            results.appendChild(card);
            results.style.display = 'block';
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(
        HTML_TEMPLATE,
        model_loaded=MODEL_LOADED,
        labels=labels,
        sample_text=""
    )

@app.route('/predict', methods=['POST'])
def predict_route():
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded. Please check model files.'}), 500
    
    try:
        review_text = request.form.get('review_text', '').strip()
        
        if not review_text:
            return jsonify({'error': 'Please enter a review.'}), 400
        
        # Predict for SINGLE review only
        categories, _ = predict_single(review_text, threshold=0.5)
        
        # Format result (only one result object)
        result = {
            'text': review_text,
            'categories': categories
        }
        
        return jsonify({
            'success': True,
            'results': [result],  # Still wrapped in list for frontend compatibility
            'threshold': 0.5
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL_LOADED,
        'device': DEVICE if MODEL_LOADED else 'N/A',
        'labels_count': len(labels) if labels else 0
    })

if __name__ == '__main__':
    print("="*50)
    print("🚀 Starting Fintech Review Classification App")
    print("="*50)
    if MODEL_LOADED:
        print(f"✅ Model loaded successfully on {DEVICE}")
        print(f"📋 Available categories: {len(labels)}")
        print(f"🏷️  Categories: {', '.join(labels[:5])}{'...' if len(labels) > 5 else ''}")
    else:
        print("❌ Model failed to load - app will run in demo mode")
    print("🌐 Open your browser to: http://localhost:5000")
    print("="*50)
    
    app.run(host='0.0.0.0', port=5000)
