import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Load the Iris dataset
df = pd.read_csv("dataset/iris.csv")

# Separate features and target
X = df.drop(columns=["species"])
y = df["species"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Test different tree depths
depths = [1, 2, 3, 4, 5]

print("Decision Tree Pruning Results")
print("-" * 50)

for depth in depths:

    # Create pruned Decision Tree
    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=depth,
        random_state=42
    )

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"Max Depth: {depth}")
    print(f"Testing Accuracy: {accuracy:.4f}")
    print(f"Weighted F1-score: {f1:.4f}")
    print("-" * 50)
