# Scientific Abstract Classification

Smart topic classification for scientific abstracts using machine learning.

## 📋 Overview

This project implements a multi-class classification system for arXiv scientific abstracts. It supports various text vectorization methods (Bag-of-Words, TF-IDF, LSA, Sentence Embeddings) and multiple classification algorithms (KNN, Decision Tree, Random Forest, Naive Bayes, etc.).

## 🏗️ Project Structure

```
.
├── app/                    # Streamlit web application
│   ├── pages/             # App pages (home, data exploration, experiments, etc.)
│   ├── services/          # Business logic services
│   ├── states/            # App state management
│   └── styles/            # CSS styles
├── configs/               # Configuration files (YAML)
├── data/                  # Data directory
│   ├── arxiv_train_augmented.csv
│   └── arxiv_test_untouched.csv
├── src/                   # Source code
│   ├── config/           # Configuration management
│   ├── models/           # Classifier implementations
│   ├── preprocessing/    # Data loading and preprocessing
│   ├── training/         # Training logic
│   ├── utils/            # Utility functions
│   └── vectorizers/      # Text vectorization methods
├── scripts/              # Utility scripts
├── notebooks/            # Jupyter notebooks for experiments
├── main.py              # Application entry point
└── pyproject.toml       # Project dependencies and configuration
```

## 🚀 Getting Started

### Prerequisites

- Python >= 3.12
- uv (recommended) or pip

### Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd scientific-abstract-classification
```

2. Install dependencies:

```bash
uv sync
```

Or with pip:

```bash
pip install -e .
```

### Running the Application

Start the Streamlit web app:

```bash
streamlit run main.py
```

The app will be available at `http://localhost:8501`

## 📊 Features

- **Data Exploration**: Browse and analyze the arXiv dataset
- **Data Sampling**: Extract balanced/imbalanced subsets
- **Data Processing**: Apply preprocessing and transformations
- **Model Experiments**: Train and evaluate various classifiers
- **Live Prediction**: Make predictions on new abstracts

## 🧪 Supported Models

### Classifiers

- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Naive Bayes (Gaussian, Multinomial)
- Logistic Regression
- AdaBoost
- Gradient Boosting
- Stacking Ensemble
- XGBoost (optional)
- LightGBM (optional)
- CatBoost

### Vectorization Methods

- Bag-of-Words (BoW)
- TF-IDF
- Latent Semantic Analysis (LSA)
- Sentence Embeddings (E5)
- Fusion (TF-IDF + LSA)
- FAISS-indexed embeddings

## 🛠️ Configuration

Edit `configs/config.yaml` to customize:

- Data paths
- Random seed
- Train/test split ratio
- Default sampling parameters

## 📝 License

MIT License

## 👥 Contributors

MIX002 Team
