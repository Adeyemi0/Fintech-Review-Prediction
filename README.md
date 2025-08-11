# Fintech Review Categorization with Enhanced Imbalanced Learning

A comprehensive machine learning pipeline for automatically categorizing fintech customer reviews using advanced imbalanced learning techniques. This project addresses the challenge of predicting review categories in financial technology applications where some categories are significantly underrepresented.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Review Categories](#review-categories)
4. [Model Architecture](#model-architecture)
5. [Performance Results](#performance-results)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Data Preprocessing](#data-preprocessing)
9. [Model Training](#model-training)
10. [Model Evaluation](#model-evaluation)
11. [Deployment](#deployment)
12. [File Structure](#file-structure)
13. [Dependencies](#dependencies)
14. [Contributing](#contributing)
15. [License](#license)

## Overview

This project implements an advanced text classification system specifically designed for fintech customer reviews. The system automatically categorizes customer feedback into predefined categories, enabling financial technology companies to better understand customer pain points and improve their services.

The main challenge addressed is **class imbalance** - some review categories (like General Feedback) are much more common than others (like Feature Requests or Customer Service complaints). The solution implements multiple advanced techniques including:

- **Focal Loss Neural Networks** for handling imbalanced data
- **Cost-Sensitive Learning** with class weights
- **Threshold Optimization** for better minority class detection
- **Rule-Based Text Augmentation** for minority classes
- **Ensemble Modeling** with automatic best model selection

## Features

- ✨ **Advanced Imbalance Handling**: Multiple techniques to address class imbalance
- 🧠 **Multiple Model Architecture**: Focal Loss NN, Cost-Sensitive LR, Threshold-Optimized LR, Enhanced RF
- 📈 **Automatic Model Selection**: Best model chosen based on minority class performance
- 🔄 **Data Augmentation**: Rule-based text augmentation for underrepresented classes
- 🎯 **Threshold Optimization**: Custom thresholds for each class to maximize F1-score
- 💾 **Model Persistence**: Complete save/load functionality for deployment
- 📊 **Comprehensive Evaluation**: Detailed performance metrics focusing on minority classes

## Review Categories

The system categorizes fintech reviews into 9 distinct categories:

| Category | Description | Example Issues |
|----------|-------------|----------------|
| **Transaction Fees/Charges** | Complaints about transaction costs and pricing | High fees, unexpected charges, pricing transparency |
| **Failed Transactions** | Issues with incomplete or unsuccessful transactions | Payment failures, transfer errors, processing issues |
| **App Performance Issues** | Technical problems, slow loading, crashes | App crashes, slow performance, UI bugs |
| **Account Setup & Verification** | Registration, KYC, and identity verification issues | Account creation problems, document verification |
| **Login & Authentication** | Access problems, password/PIN issues, biometric failures | Login errors, password reset issues, 2FA problems |
| **App Installation & Updates** | Download, installation, and update-related problems | Installation failures, update issues |
| **Feature Requests** | Suggestions for new functionality and improvements | New feature suggestions, UI improvements |
| **Customer Service** | Support experience and response quality | Support responsiveness, service quality |
| **General Feedback** | Overall satisfaction, praise, and general comments | App ratings, general satisfaction, compliments |

## Model Architecture

### Core Components

1. **Text Preprocessing Pipeline**
   - Comprehensive text cleaning and normalization
   - TF-IDF vectorization with n-gram features (1-2)
   - Stop word removal and feature limitation (10K features)

2. **Focal Loss Neural Network**
   ```python
   class FocalLossNet(nn.Module):
       - Input Layer → Hidden(128) → ReLU → Dropout(0.3)
       - Hidden(128) → Hidden(64) → ReLU → Dropout(0.3)  
       - Hidden(64) → Output(num_classes)
   ```

3. **Cost-Sensitive Models**
   - Logistic Regression with computed class weights
   - Random Forest with enhanced parameters
   - Class weights scaled by 1.5x for stronger minority emphasis

4. **Threshold Optimization**
   - Per-class threshold optimization using precision-recall curves
   - F1-score maximization for each category
   - Custom prediction logic with optimized thresholds

### Data Augmentation Strategy

Rule-based text augmentation for minority classes:
- **Clause Swapping**: Reorder clauses connected by conjunctions
- **Lexical Substitution**: Replace common words with synonyms
- **Sentence Restructuring**: Reorder sentences within reviews
- **Targeted Augmentation**: Focus on classes with <800 samples

## Performance Results

### Model Comparison

| Model | Weighted F1 | Macro F1 | Minority Avg F1 |
|-------|-------------|----------|-----------------|
| **Cost-Sensitive LR** ⭐ | **0.8814** | **0.6866** | **0.6068** |
| Threshold-Optimized LR | 0.8844 | 0.6881 | 0.6037 |
| Focal Loss NN | 0.8694 | 0.6148 | 0.4838 |
| Enhanced RF | 0.8432 | 0.6275 | 0.5596 |

**Best Model**: Cost-Sensitive Logistic Regression (Minority Avg F1: 0.6068)

### Detailed Performance by Category

| Category | Precision | Recall | F1-Score | Support |
|----------|-----------|---------|----------|---------|
| Account Setup & Verification | 0.58 | 0.73 | 0.65 | 179 |
| App Installation & Updates | 0.59 | 0.75 | 0.66 | 110 |
| App Performance Issues | 0.50 | 0.74 | 0.60 | 207 |
| Customer Service | 0.43 | 0.74 | 0.55 | 62 |
| **Failed Transactions** | **0.87** | **0.88** | **0.88** | 496 |
| Feature Requests | 0.41 | 0.60 | 0.49 | 84 |
| **General Feedback** | **0.99** | **0.90** | **0.94** | 3310 |
| Login & Authentication | 0.70 | 0.77 | 0.73 | 145 |
| | | | | |
| **Overall Accuracy** | | | **0.87** | 4593 |
| **Macro Average** | 0.63 | 0.76 | 0.69 | 4593 |
| **Weighted Average** | 0.90 | 0.87 | 0.88 | 4593 |

## Installation

### Prerequisites
- Python 3.7+
- PyTorch
- Scikit-learn
- Pandas, NumPy
- CUDA (optional, for GPU acceleration)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/fintech-review-categorization.git
   cd fintech-review-categorization
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create required directories**
   ```bash
   mkdir -p model/saved_models
   ```

## Usage

### Training a New Model

```python
import pandas as pd
from enhanced_imbalanced_classifier import EnhancedImbalancedTextClassifier

# Load your data
df = pd.read_csv('your_review_data.csv')

# Initialize classifier
classifier = EnhancedImbalancedTextClassifier(
    base_path="./model/saved_models"
)

# Train the model
classifier.fit(
    df=df,
    text_col='content',          # Column with review text
    label_col='category_clean'   # Column with category labels
)

# Save the model
classifier.save_model()
```

### Making Predictions

```python
# Load saved model
classifier = EnhancedImbalancedTextClassifier.load_model("./model/saved_models")

# Predict categories
reviews = [
    "The app keeps crashing when I try to make a payment",
    "Love the new UI design, very intuitive!",
    "Can't login with my fingerprint anymore"
]

predictions = classifier.predict(reviews)
print(predictions)
# Output: ['App Performance Issues', 'General Feedback', 'Login & Authentication']
```

### Complete Training Pipeline

```python
# Use the provided training function
from enhanced_imbalanced_classifier import train_and_save_model

# Train and save model in one step
classifier = train_and_save_model(df)
```

## Data Preprocessing

The preprocessing pipeline includes:

1. **Text Cleaning**
   - Convert to lowercase
   - Remove special characters (keep alphanumeric and basic punctuation)
   - Normalize whitespace
   - Remove empty texts

2. **Feature Engineering**
   - TF-IDF vectorization with 1-2 grams
   - Maximum 10,000 features
   - English stop word removal

3. **Data Splitting**
   - Stratified train-test split (80-20)
   - Ensures all categories represented in both sets
   - Special handling for category-specific texts

## Model Training

### Training Process

1. **Data Preparation**
   - Text cleaning and preprocessing
   - Stratified splitting to maintain class distribution
   - Label encoding for numerical processing

2. **Augmentation**
   - Identify minority classes (below median count)
   - Apply rule-based augmentation
   - Generate 3 variations per minority sample

3. **Model Training**
   - Train 4 different model architectures
   - Compute class weights for imbalance handling
   - Optimize thresholds for threshold-based models

4. **Model Selection**
   - Evaluate all models on test set
   - Select best model based on minority class performance
   - Generate comprehensive performance reports

### Hyperparameters

```python
# TF-IDF Vectorizer
max_features = 10000
ngram_range = (1, 2)

# Focal Loss Network
learning_rate = 0.01
epochs = 50
batch_size = 32
gamma = 2  # Focal loss parameter

# Random Forest
n_estimators = 200
max_depth = 10
min_samples_split = 5

# Class Weight Scaling
weight_multiplier = 1.5
```

## Model Evaluation

### Metrics Used

- **Weighted F1-Score**: Accounts for class imbalance in overall performance
- **Macro F1-Score**: Treats all classes equally, better for imbalanced data
- **Minority Average F1**: Custom metric focusing on underrepresented classes
- **Per-Class Performance**: Individual precision, recall, and F1 for each category

### Evaluation Focus

The model selection prioritizes **minority class performance** rather than overall accuracy, ensuring that underrepresented categories (Customer Service, Feature Requests, etc.) are properly detected.

## Deployment

### Model Persistence

The system saves all necessary components:

```
model/saved_models/
├── vectorizer.pkl              # TF-IDF vectorizer
├── models.pkl                  # All sklearn models
├── thresholds.json            # Optimized thresholds
├── metadata.json              # Model metadata
├── full_classifier.pkl        # Complete classifier
├── focal_loss_model.pth       # PyTorch model weights
└── focal_loss_params.json     # PyTorch model parameters
```

### Loading for Inference

```python
# Simple loading
classifier = EnhancedImbalancedTextClassifier.load_model("./model/saved_models")

# Make predictions
predictions = classifier.predict(["Your review text here"])
```

## File Structure

```
fintech-review-categorization/
├── enhanced_imbalanced_classifier.py    # Main classifier implementation
├── requirements.txt                     # Python dependencies
├── README.md                           # This file
├── examples/
│   ├── training_example.py            # Training script example
│   ├── inference_example.py           # Inference script example
│   └── evaluation_example.py          # Model evaluation example
├── model/
│   └── saved_models/                  # Saved model artifacts
├── data/
│   ├── sample_data.csv               # Sample training data
│   └── data_description.md           # Data format description
├── notebooks/
│   ├── exploratory_analysis.ipynb    # Data exploration
│   ├── model_comparison.ipynb        # Model comparison analysis
│   └── performance_analysis.ipynb    # Performance deep dive
└── tests/
    ├── test_classifier.py            # Unit tests
    ├── test_preprocessing.py         # Preprocessing tests
    └── test_models.py                # Model tests
```

## Dependencies

### Core Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
torch>=1.9.0
joblib>=1.0.0
```

### Full Requirements

```
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.2.2
torch==2.0.0
torchvision==0.15.0
joblib==1.2.0
matplotlib==3.7.1
seaborn==0.12.2
jupyter==1.0.0
pytest==7.3.1
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linting
flake8 enhanced_imbalanced_classifier.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Built with scikit-learn, PyTorch, and pandas
- Inspired by the challenges of real-world fintech customer feedback analysis
- Special focus on minority class detection in imbalanced datasets

## Contact

- **Author**: Adediran Adeyemi
- **Email**: adediran.yemite@yahoo.com
- **LinkedIn**: https://www.linkedin.com/in/adediran-adeyemi-17103b114/
- **GitHub**: https://github.com/Adeyemi0

---

*For questions, issues, or contributions, please open an issue on GitHub or contact the maintainer directly.*
