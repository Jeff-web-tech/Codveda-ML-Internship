# Task 2 — Linear Regression

## Overview

This project implements a Linear Regression model to predict house prices using a Boston Housing-style dataset. The project covers data loading, data validation, feature scaling, model training, prediction, evaluation, and visualization.

## Dataset

The dataset contains **506 observations** and **14 columns**.

* **13 features** are used as input variables.
* **MEDV** is the target variable representing the median value of owner-occupied homes.

### Features

`CRIM`, `ZN`, `INDUS`, `CHAS`, `NOX`, `RM`, `AGE`, `DIS`, `RAD`, `TAX`, `PTRATIO`, `B`, `LSTAT`

### Target

`MEDV`

## Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the dataset using Pandas.
2. Assigned descriptive column names.
3. Separated the features from the target variable.
4. Checked for missing values.
5. Checked for duplicate rows.
6. Checked the data types.
7. Examined feature ranges.
8. Split the data into training and testing sets using an 80/20 split.
9. Standardized the features using `StandardScaler`.
10. The scaler was fitted only on the training data to prevent data leakage.

### Dataset Split

* Training data: **404 observations**
* Testing data: **102 observations**
* Features: **13**

## Model

A **Linear Regression** model from Scikit-learn was used.

The model was trained using the standardized training features and corresponding house-price target values.

## Model Evaluation

The model was evaluated using four regression metrics:

| Metric   |   Score |
| -------- | ------: |
| MAE      |  3.1891 |
| MSE      | 24.2911 |
| RMSE     |  4.9286 |
| R² Score |  0.6688 |

The R² score of **0.6688** indicates that the model explains approximately **66.9% of the variation** in the target house prices.

## Visualization

An actual-vs-predicted plot was created to compare the model's predictions with the actual house prices.

The visualization is available at:

`results/actual_vs_predicted.png`

## Project Structure

```text
Task-2-Linear-Regression/
│
├── dataset/
│   └── house Prediction Data Set 2.csv
│
├── results/
│   └── actual_vs_predicted.png
│
├── src/
│   └── linear_regression.py
│
├── README.md
└── requirements.txt
```

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib

## How to Run

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Linear Regression model:

```bash
python src/linear_regression.py
```

The program will load the dataset, preprocess the data, train the model, generate predictions, calculate evaluation metrics, and display the actual-vs-predicted visualization.
