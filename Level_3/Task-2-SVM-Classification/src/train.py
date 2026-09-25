from sklearn.svm import SVC

from preprocessing import load_data, preprocess_data


def train_models(X_train, y_train):
    """Train Linear and RBF SVM models."""

    # Linear SVM
    linear_svm = SVC(
        kernel="linear",
        probability=True,
        random_state=42
    )

    # RBF SVM
    rbf_svm = SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    )

    # Train both models
    linear_svm.fit(X_train, y_train)
    rbf_svm.fit(X_train, y_train)

    return linear_svm, rbf_svm


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

    print("SVM models trained successfully.")
    print("Linear SVM:", linear_svm)
    print("RBF SVM:", rbf_svm)