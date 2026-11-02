# EMNIST Handwritten Character Recognition
### Comparative Study of Classification Algorithms

A team-based ML study comparing classical and custom classifiers on the **EMNIST ByClass** dataset for handwritten digit and character recognition.

> Course project · Cosmic Coders · RKMVERI  
> Team members: Arnab Singha and Kanan Pandit

## Problem
Handwriting varies substantially and the dataset is class-imbalanced. EMNIST ByClass contains **62 classes**: 10 digits, 26 uppercase letters, and 26 lowercase letters.

## Dataset
- **EMNIST ByClass**
- **Classes:** 62
- **Original image size:** 28 × 28 grayscale
- **Source:** [EMNIST on Kaggle](https://www.kaggle.com/datasets/crawford/emnist)

## Models compared
- Logistic Regression
- Softmax Regression
- Decision Tree
- Random Forest
- Two-layer hierarchical classifier:
  - Stage 1: Digit / Uppercase / Lowercase
  - Stage 2: character classification within the predicted group

## Workflow
`Dataset → preprocessing & normalization → model training → evaluation → comparative analysis`

## Evaluation
Accuracy, precision, recall, and F1-score. Consult the notebook and report for exact model-level values.

## Tech stack
Python · NumPy · pandas · scikit-learn · Matplotlib · Google Colab

## Repository contents
- `COSMIC_CODERS_.ipynb` — main notebook
- `ML Project Final Report(cosmic coders).pdf` — project report

## Run
Open the notebook in Google Colab or Jupyter and execute the cells. Dataset download/environment setup may be needed.

## Repository
[View source code](https://github.com/kananpandit02/emnist_classification)

**Kanan Pandit** · [Portfolio](https://kananpanditportfolio.netlify.app/) · [LinkedIn](https://linkedin.com/in/kananpandit02)
