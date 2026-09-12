import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

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

# Create the pruned Decision Tree
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the pruned model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

print("Pruned Decision Tree trained successfully.")
print("\nMax Depth:", model.max_depth)
print("Testing Accuracy:", accuracy)
print("Weighted F1-score:", f1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualize the pruned tree
plt.figure(figsize=(15, 10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=model.classes_,
    filled=True
)

plt.title("Pruned Decision Tree (Max Depth = 3)")
plt.savefig("results/pruned_decision_tree.png")
plt.show()

print("\nPruned decision tree saved to results/pruned_decision_tree.png")