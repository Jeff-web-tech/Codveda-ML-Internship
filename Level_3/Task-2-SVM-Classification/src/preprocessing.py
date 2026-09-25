import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data():
    """Load the training and testing churn datasets."""

    train_path = "../dataset/churn-bigml-80.csv"
    test_path = "../dataset/churn-bigml-20.csv"

    train_data = pd.read_csv(train_path, sep="\t")
    test_data = pd.read_csv(test_path, sep="\t")

    return train_data, test_data


def preprocess_data(train_data, test_data):
    """Prepare features and target variables for SVM."""

    # Separate features and target
    X_train = train_data.drop("Churn", axis=1)
    y_train = train_data["Churn"].astype(int)

    X_test = test_data.drop("Churn", axis=1)
    y_test = test_data["Churn"].astype(int)

    # Categorical columns
    categorical_features = [
        "State",
        "International plan",
        "Voice mail plan"
    ]

    # Numerical columns
    numerical_features = [
        column for column in X_train.columns
        if column not in categorical_features
    ]

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                categorical_features
            ),
            (
                "numerical",
                StandardScaler(),
                numerical_features
            )
        ]
    )

    # Fit on training data
    X_train_processed = preprocessor.fit_transform(X_train)

    # Transform test data using fitted preprocessor
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, y_train, y_test

if __name__ == "__main__":
    train_data, test_data = load_data()

    X_train, X_test, y_train, y_test = preprocess_data(
        train_data,
        test_data
    )

    print("Training Data Shape:", train_data.shape)
    print("Testing Data Shape:", test_data.shape)

    print("Processed Training Shape:", X_train.shape)
    print("Processed Testing Shape:", X_test.shape)

    print("Training Target Shape:", y_train.shape)
    print("Testing Target Shape:", y_test.shape)