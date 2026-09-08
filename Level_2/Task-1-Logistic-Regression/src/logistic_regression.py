import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load processed data
X_train = pd.read_csv("results/X_train.csv")
X_test = pd.read_csv("results/X_test.csv")

y_train = pd.read_csv("results/y_train.csv").squeeze()
y_test = pd.read_csv("results/y_test.csv").squeeze()

# Create Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Evaluation")
print("----------------")
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Model training completed!")
print("Number of predictions:", len(y_pred))
print("First 10 predictions:")
print(y_pred[:10])