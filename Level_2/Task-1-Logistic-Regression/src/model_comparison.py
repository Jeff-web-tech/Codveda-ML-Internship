import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load processed data
X_train = pd.read_csv("results/X_train.csv")
X_test = pd.read_csv("results/X_test.csv")

y_train = pd.read_csv("results/y_train.csv").squeeze()
y_test = pd.read_csv("results/y_test.csv").squeeze()

# Original Logistic Regression
original_model = LogisticRegression(max_iter=1000)
original_model.fit(X_train, y_train)
original_pred = original_model.predict(X_test)

# Balanced Logistic Regression
balanced_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)
balanced_model.fit(X_train, y_train)
balanced_pred = balanced_model.predict(X_test)

# Calculate metrics for both models
results = {
    "Model": [
        "Original Logistic Regression",
        "Balanced Logistic Regression"
    ],
    "Accuracy": [
        accuracy_score(y_test, original_pred),
        accuracy_score(y_test, balanced_pred)
    ],
    "Churn Precision": [
        precision_score(y_test, original_pred, pos_label=True),
        precision_score(y_test, balanced_pred, pos_label=True)
    ],
    "Churn Recall": [
        recall_score(y_test, original_pred, pos_label=True),
        recall_score(y_test, balanced_pred, pos_label=True)
    ],
    "Churn F1": [
        f1_score(y_test, original_pred, pos_label=True),
        f1_score(y_test, balanced_pred, pos_label=True)
    ]
}

comparison_df = pd.DataFrame(results)

print("\nModel Comparison")
print("================")
print(comparison_df.to_string(index=False))

# Save comparison
comparison_df.to_csv(
    "results/model_comparison.csv",
    index=False
)