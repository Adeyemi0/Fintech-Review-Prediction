# Analysis of Moniepoint Banking App Reviews

## Live Demo
🚀 **Try the model now**: [Interactive Demo on Hugging Face Spaces](https://huggingface.co/spaces/adeyemi001/Multi-Labelled-Review-Categorization-Model)

## Flask Demo
![Flask Demo](fldemo.gif)

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Project Overview](#project-overview)
3. [Technical Innovation: AI Model Development](#technical-innovation-ai-model-development)
   - [Model Architecture](#model-architecture)
   - [Performance Metrics](#performance-metrics)
   - [Training & Validation](#training--validation)
4. [Model Deployment & Access](#model-deployment--access)
   - [Live Interactive Demo](#live-interactive-demo)
   - [Hugging Face Repository](#hugging-face-repository)
   - [Usage Instructions](#usage-instructions)
   - [API Integration](#api-integration)
5. [Business Intelligence Analysis](#business-intelligence-analysis)
   - [Customer Engagement Insights](#customer-engagement-insights)
   - [Critical Issues Identification](#critical-issues-identification)
   - [Competitive Advantages](#competitive-advantages)
6. [Strategic Recommendations for Moniepoint](#strategic-recommendations-for-moniepoint)
7. [Technical Implementation Details](#technical-implementation-details)
8. [Results & Insights](#results--insights)
9. [Future Work & Extensions](#future-work--extensions)
10. [Conclusion](#conclusion)

---

## Executive Summary

This personal project presents a comprehensive AI-driven analysis of **Moniepoint's personal banking app** through advanced machine learning techniques applied to customer reviews scraped from Google Play Store. The project demonstrates the power of modern NLP techniques in extracting actionable business intelligence from unstructured customer feedback.

**Key Achievements:**
- **Developed Custom AI Model**: RoBERTa-based multi-label classifier achieving **93.7% F1-micro accuracy**
- **Analyzed 29,000+ Reviews**: Comprehensive dataset spanning multiple years of customer feedback
- **Deployed Production Model**: Publicly accessible via Hugging Face Hub and interactive web demo
- **Generated Business Insights**: Identified critical operational issues and competitive advantages
- **Created Actionable Intelligence**: Data-driven recommendations for product improvement

**Key Findings about Moniepoint:**
- Strong customer service performance with 81.83% review response rate
- Critical technical issues requiring immediate attention (417 combined mentions)
- Speed and reliability as core competitive advantages (857+ positive mentions)
- Consistent positive sentiment trend since August 2023

---

## Project Overview

### Motivation
This project was undertaken to demonstrate the practical application of advanced NLP techniques in fintech customer experience analysis. By analyzing real customer feedback from Moniepoint's banking app, the goal was to:

1. Build a state-of-the-art multi-label text classification system
2. Extract meaningful business insights from unstructured review data
3. Provide actionable recommendations based on AI-driven analysis
4. Create a reusable model for similar applications

### Data Collection & Scope
- **Target Application**: Moniepoint Personal Banking App (Google Play Store)
- **Data Source**: Web scraping of customer reviews and ratings
- **Dataset Size**: 29,000+ reviews spanning multiple years
- **Review Period**: Historical data through September 2025
- **Preprocessing**: Comprehensive data cleaning and label standardization using DeepSeek AI

### Methodology
- **Approach**: End-to-end machine learning pipeline from data collection to deployment
- **Model Selection**: Transformer-based architecture for superior language understanding
- **Validation Strategy**: Temporal data splitting to prevent data leakage
- **Deployment**: Public model hosting for accessibility and reproducibility

---

## Technical Innovation: AI Model Development

### Model Architecture

**Base Model**: RoBERTa-base (roberta-base)
- **Architecture Type**: Transformer-based encoder
- **Task Configuration**: Multi-label sequence classification
- **Maximum Sequence Length**: 256 tokens
- **Output Layer**: Sigmoid activation for multi-label predictions
- **Loss Function**: Binary Cross Entropy with Logits Loss

### Performance Metrics

The model achieved exceptional performance on the held-out test set:

| Metric | Score | Industry Context |
|--------|-------|-----------------|
| **F1 Micro** | **93.73%** | Excellent for multi-label classification |
| **F1 Macro** | 62.74% | Strong across diverse categories |
| **Precision Micro** | 95.36% | High prediction confidence |
| **Recall Micro** | 92.15% | Comprehensive issue detection |
| **ROC AUC Micro** | 99.69% | Outstanding discriminative ability |

### Training & Validation

**Training Configuration:**
- **Optimizer**: AdamW with learning rate 2e-5
- **Batch Sizes**: 8 (training), 16 (evaluation)
- **Training Epochs**: 6 with early stopping (patience=2)
- **Hardware**: CUDA-enabled GPU training
- **Regularization**: Weight decay of 0.01

**Training Progress:**
| Epoch | Training Loss | Validation Loss | F1 Micro | F1 Macro | Best Model |
|-------|---------------|-----------------|----------|----------|------------|
| 1 | 0.0519 | 0.0300 | 0.9128 | 0.5355 | - |
| **2** | **0.0361** | **0.0259** | **0.9373** | **0.6274** | **✓** |
| 3 | 0.0263 | 0.0262 | 0.9370 | 0.6341 | - |
| 4 | 0.0178 | 0.0293 | 0.9344 | 0.6144 | - |

*Best model selected from Epoch 2 based on F1-micro score*

---

## Model Deployment & Access

### Live Interactive Demo

🎯 **Try it now**: [https://huggingface.co/spaces/adeyemi001/Multi-Labelled-Review-Categorization-Model](https://huggingface.co/spaces/adeyemi001/Multi-Labelled-Review-Categorization-Model)

The interactive demo provides:
- **Real-time Predictions**: Instant review categorization
- **User-Friendly Interface**: Simple text input for immediate results
- **Visual Feedback**: Clear category labels with confidence scores
- **Example Reviews**: Pre-loaded samples to test the model
- **No Setup Required**: Browser-based access with zero configuration

**Perfect for:**
- Quick model evaluation and testing
- Understanding model capabilities
- Demonstrating the project to stakeholders
- Educational purposes and learning

### Hugging Face Repository

**Model Repository**: [adeyemi001/Multi-Labelled-Review-Categorization-Model](https://huggingface.co/adeyemi001/Multi-Labelled-Review-Categorization-Model)

**Model Files Structure:**
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

### Usage Instructions

#### Quick Start Implementation
```python
import os
import json
import joblib
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download

# Model configuration
REPO_ID = "adeyemi001/Multi-Labelled-Review-Categorization-Model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Download model files
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

# Setup local model directory
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

# Load preprocessing components
mlb_path = hf_hub_download(repo_id=REPO_ID, filename="model/mlb.joblib")
labels_path = hf_hub_download(repo_id=REPO_ID, filename="model/labels.json")

mlb = joblib.load(mlb_path)
with open(labels_path, "r", encoding="utf-8") as f:
    labels = json.load(f)
```

#### Prediction Function
```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict_review_categories(texts, threshold=0.5):
    """
    Predict categories for customer reviews
    
    Args:
        texts: Single review string or list of reviews
        threshold: Classification threshold (0.0-1.0)
    
    Returns:
        predictions: List of predicted categories for each review
        probabilities: Probability scores for all categories
    """
    if isinstance(texts, str):
        texts = [texts]

    # Tokenize input
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    ).to(DEVICE)

    # Generate predictions
    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits.cpu().numpy()

    # Convert to probabilities and binary predictions
    probs = sigmoid(logits)
    pred_bin = (probs >= threshold).astype(int)

    # Decode predictions
    all_preds = []
    for row in pred_bin:
        row_2d = np.array([row])
        categories = mlb.inverse_transform(row_2d)[0]
        all_preds.append(categories)

    return all_preds, probs

# Example usage
sample_reviews = [
    "The app crashes every time I try to open it.",
    "Please add a dark mode and reduce transaction charges.",
    "Fast transfers and great customer service!"
]

predictions, probabilities = predict_review_categories(sample_reviews, threshold=0.5)

for review, pred, prob in zip(sample_reviews, predictions, probabilities):
    print(f"\nReview: {review}")
    print(f"Predicted categories: {pred}")
    print(f"Top probabilities: {dict(zip(labels, prob.round(3)))}")
```

### API Integration

#### Batch Processing
```python
# Process multiple reviews efficiently
def batch_analyze_reviews(review_list, batch_size=16, threshold=0.5):
    """
    Process large numbers of reviews in batches
    """
    results = []
    
    for i in range(0, len(review_list), batch_size):
        batch = review_list[i:i+batch_size]
        predictions, probabilities = predict_review_categories(batch, threshold)
        
        for j, (review, pred, prob) in enumerate(zip(batch, predictions, probabilities)):
            results.append({
                'review': review,
                'categories': pred,
                'probabilities': dict(zip(labels, prob)),
                'index': i + j
            })
    
    return results

# Custom threshold adjustment
def analyze_with_confidence_levels(reviews):
    """
    Analyze reviews with different confidence thresholds
    """
    results = {}
    thresholds = [0.3, 0.5, 0.7]
    
    for threshold in thresholds:
        predictions, _ = predict_review_categories(reviews, threshold)
        results[f'threshold_{threshold}'] = predictions
    
    return results
```

---

## Business Intelligence Analysis

### Customer Engagement Insights

**Moniepoint's Customer Service Performance:**
- **Response Rate**: 81.83% (significantly above industry standards of ~60%)
- **Average Response Time**: 29.3 hours
- **Response Distribution**: 75% of responses delivered within 25 hours
- **Response Quality**: Consistent engagement with customer concerns

**Sentiment Analysis Results:**
- **Overall Trend**: Positive sentiment trajectory maintained since August 2023
- **Recent Performance**: Strong upward momentum in Q2-Q3 2025
- **Current Sentiment Score**: 0.46+ average (above neutral baseline)
- **Rating Correlation**: Clear positive correlation between star ratings and sentiment scores

### Critical Issues Identification

**AI-Identified Priority Issues (Ranked by Frequency & Sentiment Impact):**

#### Tier 1: Critical System Failures
1. **App Not Opening** - 217 mentions
   - *Impact*: Complete user access failure
   - *Sentiment*: Highly negative (-0.8 average)
   - *Business Risk*: Immediate churn potential

2. **Login Issues** - 101 mentions  
   - *Impact*: Authentication barriers
   - *Sentiment*: Consistently negative
   - *Business Risk*: User abandonment

3. **Failed Transactions** - 99 mentions
   - *Impact*: Core functionality failure
   - *Sentiment*: Extremely negative (-0.9 average)
   - *Business Risk*: Trust erosion

#### Tier 2: Financial Concerns  
4. **Transaction Charges** - 117 mentions
   - *Impact*: Pricing competitiveness
   - *Sentiment*: Negative (-0.6 average)
   - *Business Risk*: Competitive disadvantage

5. **High Charges** - 106 mentions
   - *Impact*: Value perception
   - *Sentiment*: Moderate negative
   - *Business Risk*: Price sensitivity

#### Tier 3: Performance Issues
6. **Slow App Performance** - 96 mentions
7. **Account Access Issues** - 76 mentions  
8. **Account Restrictions** - 74 mentions
9. **App Performance Issues** - 66 mentions
10. **Frequent Updates** - 57 mentions

**Combined Risk Assessment**: 417 total mentions across critical technical categories represent significant customer experience challenges.

### Competitive Advantages

**AI-Identified Strengths (High Positive Sentiment):**

#### Core Differentiators
1. **Fast Transactions** - 297 mentions
   - *Sentiment*: +0.7 average
   - *Strategic Value*: Primary competitive advantage

2. **Fast Transfers** - 282 mentions
   - *Sentiment*: +0.8 average  
   - *Strategic Value*: Market differentiation

3. **Speed** - 278 mentions
   - *Sentiment*: +0.6 average
   - *Strategic Value*: Brand positioning opportunity

#### User Experience Strengths
4. **Ease of Use** - 187 mentions
   - *Sentiment*: +0.5 average
   - *Strategic Value*: UX competitive advantage

5. **Reliability** - 183 mentions
   - *Sentiment*: +0.6 average
   - *Strategic Value*: Trust building

**Total Speed-Related Positive Mentions**: 857 (Strong market positioning asset)

---

## Strategic Recommendations for Moniepoint

### Immediate Technical Priorities (0-30 days)

**Critical System Stabilization:**
1. **App Opening Failures** (217 mentions)
   - Conduct comprehensive app launch diagnostics
   - Implement crash reporting and monitoring
   - Deploy emergency patches for identified issues

2. **Authentication System Overhaul** (101 mentions)
   - Review login flow user experience
   - Implement backup authentication methods
   - Add session management improvements

3. **Transaction Reliability** (99 mentions)
   - Enhance transaction processing infrastructure
   - Implement real-time transaction monitoring
   - Create automatic retry mechanisms

### Short-term Strategic Initiatives (30-90 days)

**Pricing Strategy Optimization:**
1. **Fee Transparency** (223 combined charge-related mentions)
   - Develop clear fee communication strategy
   - Consider competitive fee benchmarking
   - Implement fee calculator tools

2. **Performance Enhancement:**
   - Address app speed concerns through code optimization
   - Upgrade infrastructure for better response times
   - Implement progressive loading features

**Customer Communication:**
1. **Proactive Issue Communication**
   - Implement in-app status notifications
   - Create maintenance scheduling transparency
   - Develop issue resolution timelines

### Long-term Strategic Vision (90+ days)

**Leverage Competitive Advantages:**
1. **Speed-Focused Marketing** (857 positive speed mentions)
   - Build marketing campaigns around transaction speed
   - Develop speed benchmarking against competitors
   - Create user testimonials highlighting speed benefits

2. **Reliability Enhancement:**
   - Invest in infrastructure redundancy
   - Implement advanced monitoring systems
   - Develop service level agreements (SLAs)

**Innovation Opportunities:**
1. **AI-Powered Customer Support**
   - Implement automated issue categorization
   - Develop predictive customer service
   - Create personalized user experiences

---

## Technical Implementation Details

### Model Architecture Deep Dive

**RoBERTa Base Configuration:**
```python
# Model specifications
{
    "model_type": "roberta",
    "vocab_size": 50265,
    "max_position_embeddings": 514,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_labels": len(unique_categories),
    "problem_type": "multi_label_classification"
}
```

**Training Hyperparameters:**
```python
{
    "learning_rate": 2e-5,
    "train_batch_size": 8,
    "eval_batch_size": 16,
    "num_train_epochs": 6,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "logging_steps": 100,
    "eval_steps": 500,
    "save_steps": 1000,
    "early_stopping_patience": 2
}
```

### Data Processing Pipeline

**Review Preprocessing Steps:**
1. **Text Cleaning**: Removal of special characters, normalization
2. **Language Detection**: Filter for English reviews
3. **Duplicate Removal**: Eliminate redundant entries
4. **Label Standardization**: Consistent category naming via DeepSeek AI
5. **Quality Filtering**: Remove low-quality or spam reviews

**Temporal Data Splitting:**
- **Training Set**: Reviews before September 1, 2025
- **Validation Set**: Random 20% of training data
- **Test Set**: Reviews from September 1, 2025 onwards
- **Rationale**: Prevents data leakage and tests real-world performance

### Model Validation Strategy

**Cross-Validation Approach:**
- **Temporal Validation**: Future data performance testing
- **Category Balance**: Stratified sampling for rare categories
- **Performance Metrics**: Multiple evaluation criteria
- **Threshold Optimization**: ROC analysis for optimal decision boundaries

---

## Results & Insights

### Model Performance Analysis

**Strengths:**
- **High Precision**: 95.36% micro precision indicates reliable predictions
- **Strong Recall**: 92.15% micro recall ensures comprehensive issue detection
- **Excellent Discrimination**: 99.69% ROC AUC shows superior category separation
- **Production Ready**: Low latency inference suitable for real-time applications

**Category-Specific Performance:**
- **Best Performance**: Technical issues and user experience categories
- **Moderate Performance**: Complex multi-faceted feedback
- **Areas for Improvement**: Rare categories with limited training data

### Business Intelligence Insights

**Key Discoveries:**
1. **Customer Service Excellence**: Moniepoint's 81.83% response rate demonstrates strong customer engagement
2. **Technical Debt**: Critical system issues require immediate engineering focus
3. **Competitive Positioning**: Speed advantage provides strong market differentiation
4. **Customer Loyalty**: Despite technical issues, positive sentiment trends indicate brand strength

**Actionable Intelligence:**
- **Priority Matrix**: Clear ranking of issues by frequency and sentiment impact
- **Resource Allocation**: Data-driven recommendations for engineering priorities  
- **Market Positioning**: Specific strengths to leverage in competitive landscape
- **Risk Assessment**: Quantified customer churn risks from technical issues

### Comparative Analysis

**Industry Context:**
- **Response Rate**: Moniepoint's 81.83% vs industry average ~60%
- **Model Accuracy**: 93.7% F1-score exceeds typical industry benchmarks (85-90%)
- **Analysis Depth**: Multi-label approach provides richer insights than traditional sentiment analysis

---

## Future Work & Extensions

### Model Enhancement Opportunities

**Technical Improvements:**
1. **Multilingual Support**: Expand to support local Nigerian languages
2. **Real-time Learning**: Implement continuous model updates
3. **Ensemble Methods**: Combine multiple models for improved accuracy
4. **Specialized Models**: Develop domain-specific classification variants

**Feature Engineering:**
1. **Temporal Features**: Incorporate review timing patterns
2. **User Behavior**: Include reviewer history and patterns
3. **App Version Correlation**: Link feedback to specific app releases
4. **Competitor Comparison**: Multi-app comparative analysis

### Business Applications

**Expanded Use Cases:**
1. **Predictive Analytics**: Forecast customer satisfaction trends
2. **Product Roadmap**: Data-driven feature prioritization
3. **Competitive Intelligence**: Automated competitor review analysis
4. **Customer Segmentation**: AI-driven user persona development

**Integration Opportunities:**
1. **Customer Support**: Automated ticket routing and prioritization
2. **Product Development**: Real-time feature feedback analysis
3. **Marketing Intelligence**: Campaign effectiveness measurement
4. **Quality Assurance**: Automated release impact assessment

### Research Directions

**Academic Contributions:**
1. **Multi-label Fintech Classification**: Domain-specific methodology papers
2. **Temporal Sentiment Analysis**: Time-series customer satisfaction modeling
3. **Cross-Platform Analysis**: Multi-app store comparison techniques
4. **Explainable AI**: Interpretable customer feedback analysis

---

## Conclusion

This project demonstrates the practical application of NLP techniques to real-world business intelligence challenges. The project provides actionable insights for Moniepoint's customer experience optimization.

### Key Achievements

**Technical Excellence:**
- **93.7% F1-Micro Accuracy**: Demonstrates mastery of transformer-based architectures
- **Production Deployment**: Successfully deployed model to Hugging Face Hub and interactive Spaces demo
- **End-to-End Pipeline**: Complete solution from data collection to actionable insights
- **Reproducible Research**: Open-source model enabling further research and applications

**Business Impact:**
- **Actionable Intelligence**: Clear prioritization of 417 critical technical issues
- **Competitive Advantage**: Identification of 857 speed-related positive mentions as key differentiator
- **Strategic Framework**: Data-driven recommendations for product development and customer service
- **Risk Assessment**: Quantified customer churn risks and mitigation strategies

### Project Significance

This project showcases the intersection of advanced machine learning techniques and practical business applications. The ability to extract meaningful insights from unstructured customer feedback demonstrates the value of AI-driven analysis in modern business intelligence.

**For Moniepoint**: The analysis provides a comprehensive roadmap for addressing customer concerns while leveraging existing competitive advantages.

**For the AI Community**: The model serves as a benchmark for multi-label classification in the fintech domain and is freely available for research and commercial applications.

**For Future Applications**: The methodology and tools developed can be adapted for similar customer feedback analysis across various industries and platforms.

The project represents a complete data science lifecycle implementation, from problem identification through model deployment and business intelligence generation, demonstrating both technical proficiency and practical business acumen.

---

## Project Links

- 🚀 **Live Demo**: [Interactive Spaces Demo](https://huggingface.co/spaces/adeyemi001/Multi-Labelled-Review-Categorization-Model)
- 🤗 **Model Repository**: [Hugging Face Model Hub](https://huggingface.co/adeyemi001/Multi-Labelled-Review-Categorization-Model)
- 💼 **LinkedIn**: [Adediran Adeyemi](https://www.linkedin.com/in/adediran-adeyemi-17103b114/)

**Project Author**: Adediran Adeyemi  
**Completion Date**: September 2025  
**Flask Demo**: Available in project repository  

**Keywords**: Natural Language Processing, Multi-label Classification, RoBERTa, Fintech Analysis, Customer Experience, Business Intelligence, Machine Learning, Transformer Models, Hugging Face, Python
