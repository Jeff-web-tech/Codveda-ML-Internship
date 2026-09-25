import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report
)

from preprocessing import load_data, preprocess_data
from train import train_models

import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate an SVM model using the required metrics."""

    # Predictions
    y_pred = model.predict(X_test)

    # Probability estimates for AUC
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n{model_name}")
    print("-" * 40)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"AUC:       {auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["No Churn", "Churn"],
        zero_division=0
    ))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "auc": auc
    }

def plot_decision_boundaries(X_train, X_test, y_train, y_test):
    """Visualize Linear and RBF SVM decision boundaries in 2D."""

    # Reduce the 71-dimensional data to 2 dimensions
    pca = PCA(n_components=2, random_state=42)

    X_train_2d = pca.fit_transform(X_train)
    X_test_2d = pca.transform(X_test)

    # Train 2D models for visualization
    linear_svm_2d = SVC(kernel="linear")
    rbf_svm_2d = SVC(kernel="rbf")

    linear_svm_2d.fit(X_train_2d, y_train)
    rbf_svm_2d.fit(X_train_2d, y_train)

    models = [
        ("Linear SVM", linear_svm_2d),
        ("RBF SVM", rbf_svm_2d)
    ]

    for model_name, model in models:

        fig, ax = plt.subplots(figsize=(8, 6))

        DecisionBoundaryDisplay.from_estimator(
            model,
            X_train_2d,
            response_method="predict",
            alpha=0.25,
            ax=ax
        )

        scatter = ax.scatter(
            X_test_2d[:, 0],
            X_test_2d[:, 1],
            c=y_test,
            edgecolors="k",
            alpha=0.8
        )

        ax.set_title(f"{model_name} Decision Boundary")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")

        ax.legend(
            scatter.legend_elements()[0],
            ["No Churn", "Churn"],
            title="Class"
        )

        plt.tight_layout()

        filename = model_name.lower().replace(" ", "_")
        plt.savefig(
            f"../results/{filename}_decision_boundary.png",
            dpi=300
        )
        plt.close()

if __name__ == "__main__":
    # Load data
    train_data, test_data = load_data()



    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(
        train_data,
        test_data
    )

    # Train models
    linear_svm, rbf_svm = train_models(
        X_train,
        y_train
    )

    # Evaluate Linear SVM
    linear_results = evaluate_model(
        linear_svm,
        X_test,
        y_test,
        "Linear SVM"
    )

    # Evaluate RBF SVM
    rbf_results = evaluate_model(
        rbf_svm,
        X_test,
        y_test,
        "RBF SVM"
    )

    # Save model comparison
    comparison = pd.DataFrame(
        [linear_results, rbf_results],
        index=["Linear SVM", "RBF SVM"]
    )

    comparison.to_csv("../results/svm_comparison.csv")

    print("\nModel Comparison")
    print("=" * 60)
    print(comparison)

    plot_decision_boundaries(
        X_train,
        X_test,
        y_train,
        y_test
    )