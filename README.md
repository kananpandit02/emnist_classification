# ✨ EMNIST Handwritten Character Recognition
_A Comparative Study of Classification Algorithms_  
**By Cosmic Coders** 🚀  
![License: None](https://img.shields.io/badge/License-None-lightgrey)  
![Python](https://img.shields.io/badge/Python-3.10-blue.svg)  
![Colab](https://img.shields.io/badge/Google%20Colab-%23000000.svg?logo=googlecolab&logoColor=white)  
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-Classification-orange)

---

## 📘 Overview

This project presents a comparative study of traditional machine learning algorithms applied to the **EMNIST ByClass** dataset for recognizing handwritten **digits, uppercase letters**, and **lowercase letters**.

> 📍 _Course Project | Ramakrishna Mission Vivekananda Educational and Research Institute, Belur Math_  
> 🧠 _Team_: **Cosmic Coders**  
> 👨‍💻 _Members_: Arnab Singha (B2430032), Kanan Pandit (B2430051)  
> 📅 _Date_: November 25, 2024

---

## 🎯 Objectives

- 🔍 Preprocess and normalize the **EMNIST ByClass** dataset.
- ⚙️ Train & evaluate 5 different machine learning models:
  - Logistic Regression
  - Softmax Regression
  - Decision Tree
  - Random Forest
  - Two-Layer Hierarchical Softmax Model
- 📊 Compare the performance using metrics like Accuracy, Precision, Recall, and F1-score.
- 💡 Analyze strengths, weaknesses, and real-world applicability of each method.

---

## 📁 Dataset

- **Source**: [Kaggle EMNIST Dataset](https://www.kaggle.com/datasets/crawford/emnist)
- **Split Used**: `ByClass` (62 classes: 10 digits + 26 uppercase + 26 lowercase letters)
- **Size**: 
  - 🧪 Train set: 697,932 samples  
  - 🧾 Test set: 116,323 samples  
- **Features**: 28×28 grayscale images (downscaled to 14×14 for efficiency)
- **Challenge**: Imbalanced class distribution and handwriting variability

---

## 🛠️ Tools & Libraries

| Tool          | Purpose                             |
|---------------|-------------------------------------|
| Python 3.10   | Programming language                |
| Pandas/NumPy  | Data handling and computation       |
| Scikit-learn  | Machine learning models             |
| Matplotlib    | Data visualization                  |
| Seaborn       | Statistical plots                   |
| Google Colab  | Cloud-based model training          |
| PIL           | Image handling                      |

---

## ⚙️ Methodology

### ✅ Models Implemented

1. **Logistic Regression**
2. **Softmax Regression**
3. **Decision Tree Classifier**
4. **Random Forest Classifier**
5. **Custom Two-Layer Hierarchical Classifier**
   - Layer 1: Classifies into `Digit`, `Uppercase`, `Lowercase`
   - Layer 2: Specialized classifier per group

---

## 📈 Results

| 🔢 Model                  | ✅ Test Accuracy | 🎯 Precision | 🔁 Recall | 📊 F1 Score |
|---------------------------|------------------|--------------|------------|-------------|
| Logistic Regression        | 72%              | 0.69         | 0.72       | 0.70        |
| Softmax Regression         | **73%**          | **0.71**     | **0.73**   | **0.71**    |
| Decision Tree              | 71%              | 0.69         | 0.71       | 0.70        |
| Random Forest              | 48%              | 0.38         | 0.48       | 0.36        |
| Two-Layer Hierarchical     | 58%              | 0.54         | 0.58       | 0.54        |

> 🏆 **Best Performing Model**: **Softmax Regression**

---

## 🚧 Challenges Faced

- ⚖️ Severe class imbalance across 62 character classes  
- 🧠 Overfitting in complex models like Decision Trees  
- 🧮 Random Forest underperformance due to poor hyperparameter tuning  
- 💻 Limited compute capacity restricted SVM & deep learning usage  
- 🔄 Error propagation in the Two-Layer Model

---

## 🔍 Key Insights

- Simpler models like Logistic & Softmax perform decently and are interpretable.
- Decision Trees can overfit, Random Forest needs proper tuning.
- Hierarchical approaches are promising but complex to optimize.
- CNNs (not covered here) could drastically improve performance.

---

## 🧪 Future Work

- 🧠 Integrate **Convolutional Neural Networks (CNNs)** for automatic feature extraction
- ⚖️ Use **class-weighted loss** or **oversampling** to handle imbalances
- 🔧 Perform **hyperparameter tuning** using grid/random search
- 🔄 Use **cross-validation** for robust generalization
- ⚙️ Explore **hybrid models** combining ML + DL
- 💻 Scale using **distributed learning** across GPUs

---

## 📚 Files in This Repository

| File | Description |
|------|-------------|
| [`COSMIC_CODERS_.ipynb`](./COSMIC_CODERS_.ipynb) | Jupyter notebook with full implementation |
| [`ML Project Final Report(cosmic coders).pdf`](./ML%20Project%20Final%20Report(cosmic%20coders).pdf) | Full project report with literature review, methodology, and results |
| [`README.md`](./README.md) | This file |

---

## 🙏 Acknowledgements

Special thanks to:

> **Br. Bhaswarachaitanya (Tamal Maharaj)**  
> Assistant Professor, Department of Data Science  
> RKMVERI, Belur Math, West Bengal  
> _for his guidance, mentorship, and continuous encouragement._

---

## 📄 License

This project was developed as part of an academic research paper and is **not licensed for public use or redistribution**.


---
## 🌐 Connect With Us

- **Arnab Singha** (B2430032)  
  [🌐 Portfolio](https://arnabsingha200228.github.io/)  
  ✉️ arnabsingha200228@gmail.com  

- **Kanan Pandit** (B2430051)  
  [🌐 Portfolio](https://kananpanditportfolio.netlify.app/)  
  ✉️ kananpandot02@gmail.com  

- **Ramakrishna Mission Vivekananda Educational and Research Institute**  
  📍 Belur Math, Howrah, West Bengal


---

> _“Digitizing the future, one pixel at a time.”_ ✨
