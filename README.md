## Team Members

- **Sheng Zhang** - Student ID: 101311288
- **Izuchukwu Amadi** - Student ID: 101322414
- **Osasuyi Uhunmwagho** - Student ID: 101301977

---
# Skin Cancer Detection Using Artificial Intelligence

An intelligent agent system that analyzes skin lesion images and predicts whether they are benign or malignant using deep learning and various AI principles.

---

## Background

Skin cancer is one of the most common cancers worldwide, but early detection has been proven to greatly increase survival rates. Dermatologists are usually burdened with the stress of visual examination and biopsy confirmation, which can be both time-consuming and subjective. However, with the utilization of artificial intelligence, we can expedite this process by designing an agent capable of classifying skin lesion images and identifying cases that are potentially cancerous.

This project demonstrates how multiple artificial intelligence concepts—including agents, Bayes theorem, neural networks, and rule-based reasoning—can work together cohesively to solve a real-world medical problem.

---

## Project Description

This project implements an AI-powered skin cancer detection system that:

- **Analyzes skin lesion images** using deep learning models (ResNet, CNN, Transformers)
- **Classifies lesions** as benign or malignant with high accuracy
- **Uses transfer learning** from ImageNet-pretrained models for better performance
- **Implements data augmentation** to improve model generalization
- **Provides comprehensive evaluation** with metrics, visualizations, and cross-validation
- **Supports multiple datasets** (ISIC Archive, HAM10000) through flexible CSV-based loading

### Technologies Used

- **PyTorch**: Deep learning framework for neural network implementation
- **torchvision**: Image transformations and pretrained models (ResNet)
- **scikit-learn**: Machine learning utilities and metrics
- **pandas**: Data manipulation for CSV-based dataset management
- **matplotlib/seaborn**: Visualization of results and model performance

### Key Features
- **Flexible Data Loading**: CSV-based system that can aggregate multiple medical imaging datasets
- **Data Augmentation**: Random flips, rotations, color jitter, and affine transformations
- **Multiple Model Architectures**: SimpleCNN, ResNet (18/34/50), and Vision Transformer
- **Comprehensive Evaluation**: Metrics, confusion matrices, ROC curves, learning curves
- **Cross-Validation**: K-fold, stratified k-fold, and leave-one-out validation
- **Model Explainability**: LIME integration for understanding model decisions

---

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Dataset Preparation](#dataset-preparation)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Project Structure Details](#project-structure-details)
- [Team Members](#team-members)
- [References](#references)

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- CUDA-capable GPU (optional, for faster training)

### Step-by-Step Installation

1. **Clone the repository** (or download the project files)

2. **Navigate to the project directory**
   ```bash
   cd 3106-Final
   ```

3. **Install required dependencies**
   ```bash
   pip install -r requirments.txt
   ```
   
   Note: If you encounter issues, you may need to install PyTorch separately:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Verify installation**
   ```bash
   python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
   ```

---
## Usage

### Quick Start

1. **Prepare your dataset** (see [Dataset Preparation](#dataset-preparation) below)

2. **Run training**
   ```bash
   python codebase/main.py
   ```

3. **Prediction**
   python -m codebase.predict <path_to_image_file>

4. **View results**
   - Training progress will be displayed in the console
   - Best model will be saved to `checkpoints/best_model.pth`
   - Evaluation metrics will be printed at the end

### Basic Usage Example

```python
from codebase.config import TrainingConfig
from codebase.training.trainer import Trainer
from codebase.models.neural_networks import ResNetModel
from codebase.data import get_dataloaders

# Configure training
config = TrainingConfig(
    num_epochs=20,
    learning_rate=1e-4,
    batch_size=32
)

# Load data
train_loader, val_loader, test_loader = get_dataloaders(config)

# Initialize model
model = ResNetModel({
    "num_classes": 2,
    "pretrained": True,
    "resnet_version": "resnet18"
})

# Train
trainer = Trainer(model, train_loader, val_loader, config)
history = trainer.fit()

# Evaluate
metrics = trainer.evaluate(test_loader)
print(metrics)
```

---

## Dataset Preparation

### CSV Format

Create a CSV file (`data/labels.csv`) with the following format:

```csv
filepath,label
data/images/img_001.jpg,0
data/images/img_002.jpg,1
data/images/img_003.jpg,0
```

- **filepath**: Path to the image file (relative to project root)
- **label**: Class label
  - `0` or `benign` = Benign (non-cancerous)
  - `1` or `malignant` = Malignant (cancerous)

### Some supported Datasets

- **ISIC Archive**: Large publicly available database of skin lesion images
- **HAM10000**: Collection of dermatoscopic images with confirmed diagnoses

### Data Organization

```
codebase/
  data/
    images/          # Your image files
      img_001.jpg
      img_002.jpg
      ...
    labels.csv       # CSV file mapping images to labels
```

---

## Model Training

### Configuration

Training parameters can be adjusted in `codebase/main.py`:

```python
config = TrainingConfig(
    num_epochs=20,              # Number of training epochs
    learning_rate=1e-4,         # Learning rate
    batch_size=32,              # Batch size
    early_stopping_patience=5   # Stop if no improvement for 5 epochs
)
```

### Available Models

1. **SimpleCNN**: Custom convolutional neural network
2. **ResNet**: Pretrained ResNet (18, 34, or 50 layers)
3. **TransformerModel**: Vision Transformer architecture

### Training Process

The training pipeline:
1. Loads and splits data (70% train, 15% validation, 15% test)
2. Applies data augmentation to training set
3. Trains model with early stopping
4. Saves best model checkpoint
5. Evaluates on test set

---

## Evaluation

### Metrics

The system calculates:
- **Accuracy**: Overall correctness
- **Precision**: Of predicted positives, how many are correct
- **Recall**: Of actual positives, how many were found
- **F1 Score**: Harmonic mean of precision and recall

### Visualizations

Generate plots using the evaluation module:

```python
from codebase.evaluation import (
    plot_learning_curves,
    plot_confusion_matrix,
    plot_roc_curve
)

# Plot training history
plot_learning_curves(history, save_path='results/learning_curves.png')

# Plot confusion matrix
plot_confusion_matrix(y_true, y_pred, save_path='results/confusion_matrix.png')

# Plot ROC curve
plot_roc_curve(y_true, y_pred_proba, save_path='results/roc_curve.png')
```

### Cross-Validation

```python
from codebase.evaluation import stratified_k_fold_cross_validation

results = stratified_k_fold_cross_validation(model, X, y, k=5)
print(f"Mean Accuracy: {results['mean_accuracy']:.3f}")
```

---

## Project Structure Details

```
3106-Final/
├── codebase/
│   ├── data/                    # Data handling module
│   │   ├── dataset_loader.py    # CSV-based data loading
│   │   ├── augmentation.py       # Data augmentation transforms
│   │   ├── preprocessor.py      # Image preprocessing
│   │   └── labels.csv           # Dataset labels
│   │
│   ├── models/                  # Model architectures
│   │   ├── neural_networks.py   # CNN, ResNet, Transformer
│   │   ├── interpretable_models.py  # Logistic Regression, Random Forest
│   │   ├── ensemble.py          # Ensemble methods
│   │   └── explainability.py    # Model interpretation (LIME)
│   │
│   ├── training/                # Training utilities
│   │   ├── trainer.py           # Main training loop
│   │   └── callbacks.py         # Early stopping, checkpoints
│   │
│   ├── evaluation/              # Evaluation and visualization
│   │   ├── metrics.py           # Metric calculations
│   │   ├── validation.py        # Cross-validation strategies
│   │   └── visualization.py     # Plotting functions
│   │
│   ├── config.py                # Training configuration
│   └── main.py                  # Entry point
│
├── documents/                   # Project documentation
├── requirments.txt              # Python dependencies
└── README.md                    # This file
```

### Module Responsibilities

- **Data Module**: Handles dataset loading, preprocessing, and augmentation
- **Models Module**: Implements various neural network architectures
- **Training Module**: Manages training loop, optimization, and callbacks
- **Evaluation Module**: Provides metrics, validation, and visualization tools

---



## References

### Datasets

- **ISIC Archive**: [https://www.isic-archive.com/](https://www.isic-archive.com/)
- **HAM10000**: [https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

### Technologies and Libraries

- **PyTorch**: [https://pytorch.org/](https://pytorch.org/)
- **scikit-learn**: [https://scikit-learn.org/](https://scikit-learn.org/)
- **LIME**: [https://github.com/marcotcr/lime](https://github.com/marcotcr/lime)

### AI Concepts Implemented

- **Agents and Agent-Based Systems**: The model acts as an intelligent learning agent
- **Neural Networks**: Core learning component using deep learning
- **Bayes Theorem**: Used for probability interpretation of model outputs
- **Rule-Based Systems**: Confidence thresholds for uncertain predictions
- **Transfer Learning**: Using ImageNet-pretrained models

---

## Future Improvements

- [ ] Support for multi-class classification (different cancer types)
- [ ] Real-time inference API
- [ ] Web interface for dermatologists
- [ ] Integration with medical imaging systems
- [ ] Ensemble of multiple models for improved accuracy
