from flask import Flask, request, jsonify, render_template_string
import joblib
import json

app = Flask(__name__)

# Load models and vectorizer once when the app starts
vectorizer_path = r"C:model\vectorizer.pkl"
model_path = r"model\models.pkl"
metadata_path = r"model\metadata.json"

vectorizer = joblib.load(vectorizer_path)
loaded_models = joblib.load(model_path)
model = loaded_models['enhanced_rf']

with open(metadata_path, 'r') as f:
    metadata = json.load(f)

label_mapping = {i: cat for i, cat in enumerate(metadata['classes'])}

# HTML template with embedded CSS and footer credit added
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Text Category Classifier</title>
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
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            padding: 40px;
            max-width: 600px;
            width: 100%;
            animation: slideUp 0.6s ease-out;
        }
        
        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .header h1 {
            color: #333;
            font-size: 2.2em;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .header p {
            color: #666;
            font-size: 1.1em;
        }
        
        .form-group {
            margin-bottom: 25px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }
        
        textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #e1e5e9;
            border-radius: 12px;
            font-size: 16px;
            font-family: inherit;
            resize: vertical;
            transition: border-color 0.3s ease, box-shadow 0.3s ease;
        }
        
        textarea:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            font-size: 16px;
            font-weight: 600;
            border-radius: 12px;
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            width: 100%;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        }
        
        .btn:active {
            transform: translateY(0);
        }
        
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .result {
            margin-top: 25px;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            font-size: 18px;
            font-weight: 600;
            animation: fadeIn 0.5s ease-out;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        .result.success {
            background: linear-gradient(135deg, #4CAF50, #45a049);
            color: white;
            box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
        }
        
        .result.error {
            background: linear-gradient(135deg, #f44336, #d32f2f);
            color: white;
            box-shadow: 0 4px 15px rgba(244, 67, 54, 0.3);
        }
        
        .loading {
            display: none;
            text-align: center;
            margin-top: 20px;
        }
        
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .example-text {
            font-size: 14px;
            color: #888;
            font-style: italic;
            margin-top: 5px;
        }
        
        .footer {
            margin-top: 30px;
            text-align: center;
            font-size: 14px;
            color: #eee;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .footer a {
            color: #a3bffa;
            text-decoration: none;
            font-weight: 600;
        }
        .footer a:hover {
            text-decoration: underline;
        }
        
        @media (max-width: 480px) {
            .container {
                padding: 25px;
                margin: 10px;
            }
            
            .header h1 {
                font-size: 1.8em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Review Classifier</h1>
            <p>Enter your text below and discover its category</p>
        </div>
        
        <form id="classifyForm">
            <div class="form-group">
                <label for="textInput">Enter your text or review:</label>
                <textarea 
                    id="textInput" 
                    name="text" 
                    rows="6" 
                    placeholder="Paste or type your text here..."
                    required
                ></textarea>
                <div class="example-text">
                    Example: "Good app, does what it says it does, but recently, there's been a very disturbing downtime. It could take days to log in to access your funds."
                </div>
            </div>
            
            <button type="submit" class="btn" id="submitBtn">
                Classify Text
            </button>
        </form>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Analyzing text...</p>
        </div>
        
        <div id="result"></div>
    </div>

    <div class="footer">
        Created by Adediran Adeyemi. Connect with me on 
        <a href="https://www.linkedin.com/in/adediran-adeyemi-17103b114/" target="_blank" rel="noopener noreferrer">LinkedIn</a>.
    </div>

    <script>
        document.getElementById('classifyForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const textInput = document.getElementById('textInput');
            const submitBtn = document.getElementById('submitBtn');
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');
            
            const text = textInput.value.trim();
            
            if (!text) {
                showResult('Please enter some text to classify.', 'error');
                return;
            }
            
            // Show loading state
            submitBtn.disabled = true;
            submitBtn.textContent = 'Analyzing...';
            loading.style.display = 'block';
            result.innerHTML = '';
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text: text })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    showResult(`Category: ${data.category}`, 'success');
                } else {
                    showResult(`Error: ${data.error}`, 'error');
                }
                
            } catch (error) {
                showResult('Network error. Please try again.', 'error');
            } finally {
                // Reset loading state
                submitBtn.disabled = false;
                submitBtn.textContent = 'Classify Text';
                loading.style.display = 'none';
            }
        });
        
        function showResult(message, type) {
            const result = document.getElementById('result');
            result.innerHTML = message;
            result.className = `result ${type}`;
        }
        
        // Auto-resize textarea
        document.getElementById('textInput').addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 200) + 'px';
        });
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        user_text = data['text'].strip()
        if not user_text:
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        # Vectorize input text
        X = vectorizer.transform([user_text])
        
        # Predict label
        pred_label = model.predict(X)[0]
        
        # Map label to category
        category = label_mapping.get(pred_label, "Unknown")
        
        return jsonify({'category': category})
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
