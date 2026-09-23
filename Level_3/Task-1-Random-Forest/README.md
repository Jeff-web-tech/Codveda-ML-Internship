# Level 3 — Task 1: Random Forest Classification

## Overview

This project implements a **Random Forest Classifier** to predict customer churn using a telecom customer dataset.

The model is trained on the provided training dataset and evaluated on a separate testing dataset. The project includes data preprocessing, categorical feature encoding, model training, evaluation, and feature importance analysis.

---

## Dataset

The dataset contains information about telecom customers and whether they churned.

Two datasets were used:

- `churn-bigml-80.csv` — Training dataset
- `churn-bigml-20.csv` — Testing dataset

### Dataset Sizes

| Dataset | Rows | Columns |
|---------|-----:|--------:|
| Training | 2,666 | 20 |
| Testing | 667 | 20 |

The target variable is:

```text
Churn

where:

False = Customer did not churn
True = Customer churned

Technologies Used

Python
Pandas
NumPy
Scikit-learn
Matplotlib

Data Preprocessing

The following preprocessing steps were performed:

Loaded the training and testing datasets using Pandas.
Separated the features from the target variable.
Identified categorical and numerical features.
Applied One-Hot Encoding to the categorical features.
Combined the encoded categorical features with the numerical features.
Categorical Features

The following columns were encoded:

State
International plan
Voice mail plan

There were 16 numerical features and 55 encoded categorical features, resulting in 71 features for model training.

Processed Data Shapes
Training features: (2666, 71)
Testing features:  (667, 71)
Random Forest Model

A Random Forest Classifier was used with the following configuration:

RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

The model was trained using the processed training data and then used to make predictions on the testing data.

Model Evaluation

The model was evaluated using:

Accuracy
Precision
Recall
F1-score
Confusion Matrix
Classification Report
Results
Metric	Score
Accuracy	94.45%
Precision	98.33%
Recall	62.11%
F1-score	76.13%
Confusion Matrix
[[571   1]
 [ 36  59]]

The matrix represents:

	Predicted False	Predicted True
Actual False	571	1
Actual True	36	59

The model correctly classified 571 non-churning customers and 59 churning customers. It incorrectly classified 1 non-churning customer as a churner and missed 36 actual churners.

Classification Report
              precision    recall  f1-score   support

       False       0.94      1.00      0.97       572
        True       0.98      0.62      0.76        95

    accuracy                           0.94       667
   macro avg       0.96      0.81      0.86       667
weighted avg       0.95      0.94      0.94       667
Feature Importance

Random Forest provides feature importance scores that indicate how much each feature contributed to the model's predictions.

Top 10 Features
Rank	Feature	Importance
1	Total day charge	0.120286
2	Total day minutes	0.110514
3	Customer service calls	0.102726
4	Total eve minutes	0.057124
5	Total eve charge	0.052720
6	International plan_No	0.047117
7	Total intl calls	0.046900
8	International plan_Yes	0.044502
9	Total intl minutes	0.043988
10	Total night minutes	0.041509

The results indicate that features such as total day charges, total day minutes, and customer service calls had relatively high feature-importance values in this trained Random Forest model.

Visualizations
Confusion Matrix

Feature Importance

Project Structure
Task-1-Random-Forest/
│
├── dataset/
│   ├── churn-bigml-20.csv
│   └── churn-bigml-80.csv
│
├── results/
│   ├── random_forest_confusion_matrix.png
│   └── random_forest_feature_importance.png
│
├── src/
│   ├── data_loading.py
│   └── Random_Forest.py
│
├── requirements.txt
└── README.md

How to Run
1. Navigate to the project directory
cd "Level_3\Task-1-Random-Forest"
2. Install the required dependencies
pip install -r requirements.txt
3. Navigate to the source directory
cd src
4. Run the Random Forest script
python Random_Forest.py

The script will train the Random Forest model, display the evaluation results, calculate feature importance, and generate the visualization files in the results directory.

Results

The Random Forest model achieved:

94.45% Accuracy
98.33% Precision
62.11% Recall
76.13% F1-score

The confusion matrix shows that the model correctly identified 59 out of 95 actual churn cases while producing only 1 false positive.

The feature importance analysis identified Total day charge, Total day minutes, and Customer service calls among the most important features in the trained model.

Conclusion

The Random Forest model achieved an accuracy of approximately 94.45% on the testing dataset.

The evaluation also shows that accuracy alone does not fully describe the model's performance on the churn class. The model achieved a churn precision of 98.33%, while its churn recall was 62.11%.

Feature importance analysis was also used to identify the features that contributed most to the model's predictions.

Overall, this project demonstrates the complete process of building, evaluating, and interpreting a Random Forest classification model for customer churn prediction.