# Logistic Regression for Binary Classification

## Overview

This project implements Logistic Regression for binary classification using a customer churn dataset. The objective is to build a machine learning model that predicts whether a customer is likely to churn (`True`) or remain with the service (`False`).

The project also investigates the effect of class imbalance and compares a standard Logistic Regression model with a balanced Logistic Regression model.

## Dataset

The dataset contains customer information from a telecommunications service.

Two datasets were used:

* `churn-bigml-80.csv` — Training dataset
* `churn-bigml-20.csv` — Testing dataset

### Dataset Size

| Dataset  |  Rows | Columns |
| -------- | ----: | ------: |
| Training | 2,666 |      20 |
| Testing  |   667 |      20 |

The target variable is:

* `Churn` — whether the customer churned (`True`) or not (`False`)

### Target Distribution

Training data:

* No Churn (`False`): 2,278
* Churn (`True`): 388

Testing data:

* No Churn (`False`): 572
* Churn (`True`): 95

This shows that the dataset is imbalanced because there are significantly more non-churning customers than churning customers.

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Matplotlib

## Project Structure

```text
Task-1-Logistic-Regression/
│
├── dataset/
│   ├── churn-bigml-80.csv
│   └── churn-bigml-20.csv
│
├── src/
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── logistic_regression.py
│   ├── logistic_regression_balanced.py
│   ├── model_comparison.py
│   ├── evaluation.py
│   └── final_model.py
│
├── results/
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   ├── y_test.csv
│   ├── model_evaluation.txt
│   ├── balanced_model_evaluation.txt
│   ├── model_comparison.csv
│   └── confusion_matrix.png
│
├── screenshots/
├── README.md
└── requirements.txt
```

## Data Preprocessing

The following preprocessing steps were performed:

### 1. Binary Encoding

The following categorical variables were converted into binary values:

* `International plan`
* `Voice mail plan`

The values were encoded as:

* `No` → 0
* `Yes` → 1

### 2. One-Hot Encoding

The `State` variable was converted into multiple binary columns using One-Hot Encoding.

`handle_unknown="ignore"` was used to safely handle categories that may appear in the test dataset but were not present during training.

### 3. Feature Scaling

Numerical features were standardized using `StandardScaler`.

The scaler was fitted only on the training data and then used to transform both the training and testing data. This prevents information from the test dataset from influencing the preprocessing process.

### 4. Feature and Target Separation

The dataset was separated into:

* `X_train` — training features
* `y_train` — training target
* `X_test` — testing features
* `y_test` — testing target

After preprocessing:

* `X_train`: 2,666 rows × 69 features
* `X_test`: 667 rows × 69 features

## Logistic Regression Model

A standard Logistic Regression model was first trained:

```python
LogisticRegression(max_iter=1000)
```

The model achieved an accuracy of approximately **85.46%**.

However, the recall for the churn class was only **22.11%**, meaning that many customers who actually churned were missed.

## Handling Class Imbalance

Because the target variable was imbalanced, a second model was trained using:

```python
LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)
```

The `class_weight="balanced"` option gives greater importance to the minority class.

## Model Comparison

| Model                        | Accuracy | Churn Precision | Churn Recall | Churn F1 |
| ---------------------------- | -------: | --------------: | -----------: | -------: |
| Original Logistic Regression |   85.46% |          47.73% |       22.11% |   30.22% |
| Balanced Logistic Regression |   77.66% |          36.50% |       76.84% |   49.49% |

## Confusion Matrix — Final Model

The final balanced model produced:

```text
[[445 127]
 [ 22  73]]
```

This means:

* 445 non-churning customers were correctly classified.
* 127 non-churning customers were incorrectly classified as churners.
* 22 churning customers were incorrectly classified as non-churners.
* 73 churning customers were correctly identified.

Therefore, the model correctly identified **73 out of 95 actual churners**.

## Final Model

The **Balanced Logistic Regression model** was selected as the final model.

Although its overall accuracy (**77.66%**) is lower than the original model's accuracy (**85.46%**), it has a significantly higher recall for the churn class:

**76.84% compared with 22.11%.**

For a customer churn problem, identifying customers who are actually likely to leave is important. Therefore, the balanced model provides a better trade-off for this task.

## Conclusion

This project demonstrated the use of Logistic Regression for binary classification.

The preprocessing stage included binary encoding, one-hot encoding, feature scaling, and separation of features and target variables.

Two Logistic Regression models were evaluated. The standard model achieved higher overall accuracy, but it performed poorly at detecting customers who churned. Introducing class balancing significantly improved churn recall from **22.11% to 76.84%**.

The balanced Logistic Regression model was therefore selected as the final model because it is substantially better at identifying the minority churn class.

## How to Run

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run data loading:

```bash
python src/data_loading.py
```

Run preprocessing:

```bash
python src/preprocessing.py
```

Run the original Logistic Regression model:

```bash
python src/logistic_regression.py
```

Run the balanced model:

```bash
python src/logistic_regression_balanced.py
```

Compare the models:

```bash
python src/model_comparison.py
```

Run the final model:

```bash
python src/final_model.py
```

Generate the confusion matrix:

```bash
python src/evaluation.py
```
