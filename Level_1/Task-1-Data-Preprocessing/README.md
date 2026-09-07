# Codveda ML Internship – Task 1: Data Preprocessing

## Project Overview

This project focuses on preprocessing a customer churn dataset as part of Task 1 of the Codveda Technologies Machine Learning Internship.

The goal is to prepare the raw dataset for machine learning by checking for missing values, encoding categorical variables, scaling numerical features, separating the features from the target variable, and saving the processed datasets for further analysis and model development.

The dataset contains information about telecommunications customers and whether they churned or remained with the service.

## Dataset

The dataset used for this project is a customer churn dataset provided as part of the internship task.

It is divided into two files:

- `churn-bigml-80.csv` — Training dataset
- `churn-bigml-20.csv` — Testing dataset

The training dataset contains 2,666 records, while the testing dataset contains 667 records. Each record contains customer information such as account length, call usage, international plan, voicemail plan, and customer service calls.

The target variable is `Churn`, which indicates whether a customer left the service (`True`) or remained with the service (`False`).

## Preprocessing Steps

The following preprocessing steps were performed on the dataset:

### 1. Data Loading

The training and testing datasets were loaded using Pandas. Since the datasets are tab-separated, the `sep="\t"` parameter was used when reading the CSV files.

### 2. Missing Value Check

The datasets were checked for missing values using Pandas' `isnull().sum()` method.

No missing values were found in either the training or testing dataset. Therefore, no imputation or row removal was necessary.

### 3. Binary Encoding

The categorical variables `International plan` and `Voice mail plan` were converted into numerical values.

- `No` → `0`
- `Yes` → `1`

This allows these categorical variables to be used by machine learning algorithms.

### 4. One-Hot Encoding

The `State` column contains 51 different categories. One-hot encoding was used to convert each state into a separate binary feature.

`OneHotEncoder` from Scikit-learn was used for this transformation.

The original `State` column was then removed from the dataset.

### 5. Feature Scaling

Numerical features were standardized using `StandardScaler` from Scikit-learn.

The scaler was fitted only on the training data and then used to transform both the training and testing data. This prevents data leakage from the testing dataset.

### 6. Feature and Target Separation

The `Churn` column was separated from the remaining columns.

- `X_train` and `X_test` contain the input features.
- `y_train` and `y_test` contain the target variable, `Churn`.

### 7. Saving Processed Data

The processed datasets were saved as CSV files in the `results` folder:

- `X_train.csv`
- `X_test.csv`
- `y_train.csv`
- `y_test.csv`

## Results

After preprocessing, the datasets were successfully transformed and saved.

### Dataset Shapes

| Dataset | Shape |
|---|---:|
| X_train | 2,666 × 69 |
| X_test | 667 × 69 |
| y_train | 2,666 × 1 |
| y_test | 667 × 1 |

The final feature sets contain 69 features after categorical encoding and removal of the target variable.

### Target Distribution

The training dataset contains:

- `False` (No churn): 2,278
- `True` (Churn): 388

The testing dataset contains:

- `False` (No churn): 572
- `True` (Churn): 95

The similar distribution of the target variable in both datasets indicates that the training and testing data have a comparable churn pattern.

## Technologies Used

The following tools and libraries were used in this project:

- **Python** — Programming language used for data preprocessing
- **Pandas** — Data loading, inspection, manipulation, and CSV handling
- **Scikit-learn** — Categorical encoding and feature scaling
- **VS Code** — Development environment
- **Git & GitHub** — Version control and project repository management

## Project Structure

```text
Task-1-Data-Preprocessing/
├── dataset/
│   ├── churn-bigml-80.csv
│   └── churn-bigml-20.csv
├── src/
│   └── preprocessing.py
├── screenshots/
├── results/
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   └── y_test.csv
├── README.md
└── requirements.txt
```
```

## Conclusion

The customer churn dataset was successfully preprocessed and prepared for machine learning.

The preprocessing included data inspection, missing value checking, binary encoding, one-hot encoding, numerical feature scaling, and separation of features from the target variable.

The processed training and testing datasets were saved in the `results` folder and verified to have the expected dimensions. The data is now ready for further analysis and machine learning model development.
