import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt

train_df = pd.read_csv("dataset/churn-bigml-80.csv", sep="\t")
test_df = pd.read_csv("dataset/churn-bigml-20.csv", sep="\t")

# Separate features (X) and target (y)
X_train = train_df.drop("Churn", axis=1)
y_train = train_df["Churn"]

X_test = test_df.drop("Churn", axis=1)
y_test = test_df["Churn"]

# print("X_train shape:", X_train.shape)
# print("y_train shape:", y_train.shape)
# print("X_test shape:", X_test.shape)
# print("y_test shape:", y_test.shape)

# Identify categorical and numerical columns
categorical_columns = [
    "State",
    "International plan",
    "Voice mail plan"
]

numerical_columns = [
    col for col in X_train.columns
    if col not in categorical_columns
]

# print("Categorical columns:", categorical_columns)
# print("Number of numerical columns:", len(numerical_columns))   


# Encoding categorical columns
encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

# Fit on training data and transform both datasets
X_train_encoded = encoder.fit_transform(X_train[categorical_columns])
X_test_encoded = encoder.transform(X_test[categorical_columns])

# print("Encoded training shape:", X_train_encoded.shape)
# print("Encoded testing shape:", X_test_encoded.shape)


# Get numerical features
X_train_numerical = X_train[numerical_columns].values
X_test_numerical = X_test[numerical_columns].values

# Combine numerical and encoded categorical features
X_train_processed = np.hstack([
    X_train_numerical,
    X_train_encoded
])

X_test_processed = np.hstack([
    X_test_numerical,
    X_test_encoded
])

# print("Final training shape:", X_train_processed.shape)
# print("Final testing shape:", X_test_processed.shape)


# Create the Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
rf_model.fit(X_train_processed, y_train)

# print("Random Forest model trained successfully!")

# Make predictions on the test data
y_pred = rf_model.predict(X_test_processed)

# print("Predictions made successfully!")
# print("First 10 predictions:", y_pred[:10])


# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# print("\nRandom Forest Evaluation")
# print("------------------------")
# print("Accuracy:", accuracy)
# print("Precision:", precision)
# print("Recall:", recall)
# print("F1-score:", f1)

print("\nConfusion Matrix:")
print(cm)

# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# Get feature importance
feature_importance = rf_model.feature_importances_

# Get encoded categorical feature names
encoded_feature_names = encoder.get_feature_names_out(categorical_columns)

# Combine numerical and categorical feature names
all_feature_names = numerical_columns + list(encoded_feature_names)

# Create a DataFrame
importance_df = pd.DataFrame({
    "Feature": all_feature_names,
    "Importance": feature_importance
})

# Sort from most important to least important
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

# print("\nTop 10 Most Important Features:")
# print(importance_df.head(10))



# # Plot confusion matrix
# plt.figure(figsize=(6, 5))

# plt.imshow(cm)

# plt.title("Random Forest Confusion Matrix")
# plt.xlabel("Predicted Label")
# plt.ylabel("Actual Label")

# plt.xticks([0, 1], ["False", "True"])
# plt.yticks([0, 1], ["False", "True"])

# # Add numbers inside the matrix
# for i in range(2):
#     for j in range(2):
#         plt.text(j, i, cm[i, j], ha="center", va="center")

# plt.tight_layout()

# # Save the figure
# plt.savefig(
#     "random_forest_confusion_matrix.png",
#     dpi=300,
#     bbox_inches="tight"
# )

# plt.show()

# Get the top 10 features
# top_features = importance_df.head(10)

# # Plot feature importance
# plt.figure(figsize=(10, 6))

# plt.barh(
#     top_features["Feature"][::-1],
#     top_features["Importance"][::-1]
# )

# plt.title("Top 10 Random Forest Feature Importances")
# plt.xlabel("Importance")
# plt.ylabel("Feature")

# plt.tight_layout()

# # Save the figure
# plt.savefig(
#     "random_forest_feature_importance.png",
#     dpi=300,
#     bbox_inches="tight"
# )

# plt.show()

