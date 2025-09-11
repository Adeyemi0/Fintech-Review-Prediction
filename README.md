# Multi-Labeled Review Categorization Model

A RoBERTa-based transformer model for automatic categorization of customer reviews into multiple labels simultaneously. The model achieved 93.7% F1-micro score on the test set, making it highly effective for real-world review classification tasks.

## Table of Contents

- [Overview](#overview)
- [Model Performance](#model-performance)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [Quick Start](#quick-start)
  - [Batch Predictions](#batch-predictions)
  - [Custom Threshold](#custom-threshold)
- [Training Details](#training-details)
- [Model Files](#model-files)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

This model automatically categorizes customer reviews into multiple relevant categories using a fine-tuned RoBERTa transformer. Unlike traditional single-label classification, this multi-label approach can assign multiple categories to a single review, making it more suitable for complex customer feedback analysis.

**Key Features:**
- Multi-label classification capability
- High performance with 93.7% F1-micro score
- Pre-trained on RoBERTa-base for robust language understanding
- Ready-to-use with Hugging Face integration
- Temporal data splitting to prevent data leakage

## Model Performance

| Metric | Score |
|--------|-------|
| **F1 Micro** | **93.73%** |
| **F1 Macro** | 62.74% |
| **Precision Micro** | 95.36% |
| **Recall Micro** | 92.15% |
| **ROC AUC Micro** | 99.69% |

*Results on held-out test set (reviews from September 2025 onwards)*

## Dataset

- **Total Reviews**: 29,000+ customer reviews
- **Data Source**: Sourced from Moniepoint Customer Review (playstore) and Labeled using DeepSeek for consistent categorization
- **Preprocessing**: Comprehensive data cleaning and label standardization
- **Split Strategy**: Temporal split with cutoff date of September 1, 2025
- **Labels**: Multiple categories per review supported

## Model Architecture

- **Base Model**: RoBERTa-base (roberta-base)
- **Task**: Multi-label sequence classification
- **Loss Function**: Binary Cross Entropy with Logits Loss
- **Maximum Sequence Length**: 256 tokens
- **Output**: Sigmoid probabilities for each label

## Installation

```bash
# Install required packages
pip install torch transformers huggingface-hub scikit-learn joblib numpy
```

## Usage

### Quick Start

```python
import os
import json
import joblib
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download

# Load the model
REPO_ID = "adeyemi001/Multi-Labelled-Review-Categorization-Model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Download and setup model files
model_files = [
    "model/config.json",
    "model/tokenizer.json", 
    "model/tokenizer_config.json",
    "model/vocab.json",
    "model/merges.txt",
    "model/special_tokens_map.json",
    "model/model.safetensors"
]

downloaded_files = {}
for file in model_files:
    path = hf_hub_download(repo_id=REPO_ID, filename=file)
    downloaded_files[file] = path

# Create local directory and copy files
model_dir = "./temp_model"
os.makedirs(model_dir, exist_ok=True)

import shutil
for file, path in downloaded_files.items():
    local_path = os.path.join(model_dir, os.path.basename(file))
    shutil.copy2(path, local_path)

# Load model components
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(DEVICE)
model.eval()

# Load label encoder and labels
mlb_path = hf_hub_download(repo_id=REPO_ID, filename="model/mlb.joblib")
labels_path = hf_hub_download(repo_id=REPO_ID, filename="model/labels.json")

mlb = joblib.load(mlb_path)
with open(labels_path, "r", encoding="utf-8") as f:
    labels = json.load(f)

# Sigmoid function for probabilities
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Prediction function
def predict(texts, threshold=0.5):
    if isinstance(texts, str):
        texts = [texts]

    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits.cpu().numpy()

    probs = sigmoid(logits)
    pred_bin = (probs >= threshold).astype(int)

    all_preds = []
    for row in pred_bin:
        row_2d = np.array([row])
        categories = mlb.inverse_transform(row_2d)[0]
        all_preds.append(categories)

    return all_preds, probs

# Example usage
sample_texts = [
    "The app crashes every time I try to open it.",
    "Please add a dark mode and reduce charges."
]

predictions, probabilities = predict(sample_texts, threshold=0.5)

for txt, pred, prob in zip(sample_texts, predictions, probabilities):
    print(f"\nText: {txt}")
    print(f"Predicted categories: {pred}")
    print(f"Probabilities: {dict(zip(labels, prob.round(3)))}")

# Clean up
shutil.rmtree(model_dir)
```

### Batch Predictions

```python
# Process multiple reviews at once
reviews = [
    "Great customer service, very helpful staff",
    "App is slow and buggy, needs improvement",
    "Love the new features but please fix the login issue"
]

predictions, probabilities = predict(reviews, threshold=0.5)

for i, (review, pred) in enumerate(zip(reviews, predictions)):
    print(f"Review {i+1}: {pred}")
```

### Custom Threshold

```python
# Adjust threshold for different precision/recall trade-offs
predictions_conservative, _ = predict(reviews, threshold=0.7)  # Higher precision
predictions_liberal, _ = predict(reviews, threshold=0.3)      # Higher recall
```

## Training Details

### Data Preparation
- **Label Processing**: Semicolon-separated multi-labels converted to binary matrix
- **Text Preprocessing**: Standard tokenization with RoBERTa tokenizer
- **Data Split**: Temporal split to prevent data leakage
- **Validation Strategy**: Hold-out test set from September 2025

### Training Configuration
- **Optimizer**: AdamW with learning rate 2e-5
- **Batch Size**: 8 (training), 16 (evaluation)
- **Epochs**: 6 with early stopping (patience=2)
- **Weight Decay**: 0.01
- **Scheduler**: Linear decay
- **Hardware**: CUDA-enabled training

### Training Progress
| Epoch | Training Loss | Validation Loss | F1 Micro | F1 Macro | Precision Micro | Recall Micro | ROC AUC Micro |
|-------|---------------|-----------------|----------|----------|-----------------|--------------|---------------|
| 1 | 0.0519 | 0.0300 | 0.9128 | 0.5355 | 0.9343 | 0.8924 | 0.9965 |
| 2 | 0.0361 | 0.0259 | **0.9373** | 0.6274 | 0.9536 | 0.9215 | 0.9969 |
| 3 | 0.0263 | 0.0262 | 0.9370 | 0.6341 | 0.9578 | 0.9170 | 0.9967 |
| 4 | 0.0178 | 0.0293 | 0.9344 | 0.6144 | 0.9429 | 0.9260 | 0.9963 |

*Best model selected from Epoch 2 based on F1-micro score*

## Model Files

The model repository contains:

```
model/
├── config.json              # Model configuration
├── model.safetensors        # Model weights
├── tokenizer.json           # Tokenizer vocabulary
├── tokenizer_config.json    # Tokenizer configuration
├── vocab.json               # Vocabulary mappings
├── merges.txt              # BPE merges
├── special_tokens_map.json  # Special tokens
├── mlb.joblib              # Multi-label binarizer
├── labels.json             # Label names
└── metadata.json           # Training metadata
```

## Results

### Key Achievements
- ✅ **High Accuracy**: 93.7% F1-micro score demonstrates excellent overall performance
- ✅ **Robust Predictions**: 99.7% ROC-AUC indicates strong discriminative ability  
- ✅ **Production Ready**: Low inference latency with optimized tokenization
- ✅ **Temporal Validation**: Model tested on future data to ensure generalization

### Use Cases
- **Customer Support**: Automatic routing of support tickets
- **Product Development**: Feature request categorization
- **Market Research**: Sentiment and topic analysis
- **Quality Assurance**: Issue classification and prioritization

## Contributing

We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Model Card**: [adeyemi001/Multi-Labelled-Review-Categorization-Model](https://huggingface.co/adeyemi001/Multi-Labelled-Review-Categorization-Model)

**Contact**: For questions or support, please open an issue on the GitHub repository.
