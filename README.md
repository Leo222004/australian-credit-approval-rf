# Australian Credit Approval Prediction

## Project Overview
This project implements a Machine Learning system to predict the approval or rejection of a credit card application, which is a binary classification problem. The model is trained and evaluated on the Statlog (Australian Credit Approval) dataset.

## The Dataset
* **Source:** UCI Machine Learning Repository.
* **Size:** 690 instances and 14 independent features. The features are anonymized for confidentiality reasons.
* **Class Distribution:** The dataset is well-balanced, consisting of 55.51% approved applications (Class 1) and 44.49% rejected applications (Class 2).

## Technologies Used
* **Language:** Python 
* **Data Manipulation:** Pandas, NumPy 
* **Machine Learning:** Scikit-Learn (RandomForestClassifier, GridSearchCV, train_test_split) 
* **Data Visualization:** Matplotlib, Seaborn 

## Methodology
1. **Exploratory Data Analysis (EDA):** Conducted statistical analysis on numerical and categorical features. The analysis confirmed the dataset was clean, with 0 missing values and 0 duplicate rows.
2. **Feature Engineering:** Applied *One-Hot Encoding* to nominal categorical features (A4, A5, A6, A12) to prevent the algorithm from incorrectly assuming a mathematical hierarchy between categories.
3. **Model Training & Optimization:** Chosen algorithm is Random Forest (an ensemble method). Used `GridSearchCV` to perform an exhaustive search for the optimal hyperparameters. The best configuration found was: `{max_features: 0.5, max_samples: 0.25, n_estimators: 10}`.

## Results & Insights
The model generalized very well on the unseen test set (which represented 25% of the data):
* **Accuracy:** **87.86%**.
* **F1-Score:** **0.90** for the approved class and **0.84** for the rejected class, indicating the model differentiates between the two effectively without heavy bias toward either.
* **Key Business Insight:** Through Pearson correlation and visual analysis, features **A10** and **A7** were identified as the strongest predictors. Boxplot distributions clearly showed that high values in either of these features strongly correlate with a rejected credit application.
