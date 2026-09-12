import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

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

# Create the Decision Tree model
model = DecisionTreeClassifier(
    criterion="gini",
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

print("Decision Tree model trained successfully.")

# Display training and testing accuracy
print("\nTraining Accuracy:", model.score(X_train, y_train))
print("Testing Accuracy:", model.score(X_test, y_test))

# Visualize the Decision Tree
plt.figure(figsize=(15, 10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=model.classes_,
    filled=True
)

plt.title("Decision Tree Structure")
plt.savefig("results/decision_tree.png")
plt.show()

print("\nDecision tree visualization saved to results/decision_tree.png")