# 🕵️‍♂️ Financial Fraud Detection (Model Benchmarking)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![XGBoost](https://img.shields.io/badge/Library-XGBoost-red)
![LightGBM](https://img.shields.io/badge/Library-LightGBM-green)
![Status](https://img.shields.io/badge/Status-Active_Development-yellow)

## 📌 Project Overview
This project aims to build and benchmark the performance of various **Machine Learning** algorithms in detecting fraudulent financial transactions.

The primary challenge in this dataset is the **Imbalanced Dataset** problem, where fraudulent transactions are significantly rarer than normal transactions. Consequently, standard accuracy is not sufficient to evaluate model performance.

The main focus of this experiment is to benchmark simple linear models against complex ensemble tree-based models to identify the most effective approach for fraud detection.

## 📂 Dataset
The dataset used in this project is the **Synthetic Financial Datasets For Fraud Detection** sourced from Kaggle.

- **Source:** [Kaggle - Synthetic Financial Datasets For Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1)
- **Description:** A simulation of mobile money transactions based on a sample of real financial logs.
- **Target Variable:** `isFraud` (1 = Fraud, 0 = Normal).

## 🛠️ Methodology & Algorithms
This project benchmarks the following 4 algorithms:

1.  **Logistic Regression:** Serves as a simple and interpretable **baseline model**.
2.  **Random Forest:** Utilizes **bagging** methods to reduce variance and prevent overfitting.
3.  **XGBoost:** A highly efficient **gradient boosting** algorithm, popular in tabular data competitions.
4.  **LightGBM:** A faster, memory-efficient gradient boosting variant suitable for large-scale datasets.

### Workflow:
1.  **EDA (Exploratory Data Analysis):** Analyzing data distribution and feature correlations.
2.  **Preprocessing:**
    * Dropping irrelevant features (`nameOrig`, `nameDest`).
    * *One-Hot Encoding* for categorical features (`type`).
    * *Robust Scaler* to handle outliers in numerical features.
3.  **Modeling:** Training the 4 models mentioned above.
4.  **Evaluation:** Using Confusion Matrix, Precision, Recall, and F1-Score.

## 📊 Preliminary Results (Phase 1: Imbalanced)
Currently, the models are trained on the **original imbalanced data**. The goal is to observe the models' natural behavior toward the minority class before applying any sampling techniques.

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | *0.946* | *0.998* | *0.946* | *0.971* |
| **Random Forest** | *0.999* | *0.998* | *0.947* | *0.971* |
| **XGBoost** | *0.993* | *0.998* | *0.993* | *0.995* |
| **LightGBM** | *0.947* | *0.998* | *0.947* | *0.971* |

*(Full results will be updated upon completion of the experiments)*

## 🚀 Roadmap & Future Work
The project development is divided into two phases. Currently, the project is in **Phase 1**.

- [x] **Phase 1:** Implementation of Baseline Models (LogReg, RF, XGB, LGBM) on *Imbalanced Data*.
- [ ] **Phase 2:** Implementation of **SMOTE (Synthetic Minority Over-sampling Technique)** to handle class imbalance.
- [ ] **Comparative Analysis:** Comparing the improvement in *Recall* metrics between Phase 1 and Phase 2 models.

## 💻 How to Run
1. Clone this repository.
2. Install the required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm
