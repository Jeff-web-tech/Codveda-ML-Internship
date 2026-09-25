# Level 3 Task 2 — SVM Classification

## Overview

This project implements Support Vector Machine (SVM) classification to predict customer churn using the Telco churn dataset.

Two SVM models were developed and compared:

- Linear SVM
- RBF (Radial Basis Function) SVM

The models were evaluated using accuracy, precision, recall, and AUC.

---

## Dataset

The dataset contains customer information and a binary `Churn` target.

### Dataset split

- Training data: 2,666 records
- Testing data: 667 records
- Original features: 19 predictors
- Target variable: `Churn`

The categorical features are:

- `State`
- `International plan`
- `Voice mail plan`

The numerical features are standardized using `StandardScaler`.

Categorical features are converted to numerical representations using `OneHotEncoder`.

After preprocessing, the dataset contains 71 features.

---

## Project Structure

```text
Task-2-SVM-Classification/
│
├── dataset/
│   ├── churn-bigml-80.csv
│   └── churn-bigml-20.csv
│
├── results/
│   ├── linear_svm_decision_boundary.png
│   ├── rbf_svm_decision_boundary.png
│   └── svm_comparison.csv
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
│
├── README.md
└── requirements.txt


Methodology
1. Data Preprocessing

The dataset was divided into features and the target variable.

Categorical features were encoded using one-hot encoding, while numerical features were standardized.

The preprocessing transformer was fitted only on the training data and then applied to the test data to prevent data leakage.

2. Linear SVM

A Support Vector Classifier using a linear kernel was trained:

SVC(
    kernel="linear",
    probability=True,
    random_state=42
)

3. RBF SVM

A Support Vector Classifier using the Radial Basis Function kernel was trained:

SVC(
    kernel="rbf",
    probability=True,
    random_state=42
)

4. Evaluation

Both models were evaluated on the unseen test dataset using:

Accuracy
Precision
Recall
AUC

5. Decision Boundary Visualization

Because the preprocessed dataset contains 71 dimensions, Principal Component Analysis (PCA) was used to reduce the data to two dimensions for visualization.

Separate Linear and RBF SVM models were trained on the two-dimensional representation to visualize their decision boundaries.

Results
Model	Accuracy	Precision	Recall	AUC
Linear SVM	0.8576	0.0000	0.0000	0.7685
RBF SVM	0.9100	0.8431	0.4526	0.9296
Linear SVM

The Linear SVM achieved an accuracy of 85.76% and an AUC of 0.7685.

At the default classification threshold, the model predicted no customers as churners. Consequently, its precision and recall for the churn class were both 0.00.

RBF SVM

The RBF SVM achieved an accuracy of 91.00%, precision of 84.31%, recall of 45.26%, and AUC of 0.9296.

The RBF model identified a larger proportion of actual churn cases while maintaining relatively high precision.

Decision Boundary Visualizations

The project includes PCA-based decision boundary visualizations:

results/linear_svm_decision_boundary.png
results/rbf_svm_decision_boundary.png

These visualizations provide a two-dimensional representation of the models' classification boundaries.

How to Run

Install the required dependencies:

pip install -r requirements.txt

Navigate to the source directory:

cd src

Run preprocessing:

python preprocessing.py

Train the SVM models:

python train.py

Evaluate the models and generate the results:

python evaluate.py

Technologies Used

Python
pandas
NumPy
scikit-learn
Matplotlib


Conclusion

This project demonstrates the application of Support Vector Machines to a binary customer churn classification problem.

Both linear and non-linear SVM approaches were implemented, evaluated, and visualized. The results demonstrate how different SVM kernels can produce different classification behavior on the same dataset.