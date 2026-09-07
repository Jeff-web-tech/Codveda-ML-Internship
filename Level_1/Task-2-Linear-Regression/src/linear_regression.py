import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(
    "dataset/house Prediction Data Set 2.csv",
    header=None,
    sep=r"\s+"
)

# Assign column names
df.columns = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM",
    "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B",
    "LSTAT", "MEDV"
]

# Separate features and target
X = df.drop("MEDV", axis=1)
y = df["MEDV"]

# Display dataset information
# print("Dataset loaded successfully!")
# print("Features shape:", X.shape)
# print("Target shape:", y.shape)
# print("Target column:", y.name)

# Check for missing values
# print("\nMissing values:")
# print(df.isnull().sum())

# Check for duplicate rows
# print("\nDuplicate rows:", df.duplicated().sum())

# Check data types
# print("\nData types:")
# print(df.dtypes)

# Check feature ranges
# print("\nFeature ranges:")
# print(X.describe().loc[["min", "max"]])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# print("\nTraining and testing sets:")
# print("X_train:", X_train.shape)
# print("X_test:", X_test.shape)
# print("y_train:", y_train.shape)
# print("y_test:", y_test.shape)

# Standardize the features
scaler = StandardScaler()

# Fit the scaler only on the training data
X_train = scaler.fit_transform(X_train)

# Use the same scaler to transform the test data
X_test = scaler.transform(X_test)

# print("\nFeature scaling completed!")
# print("Scaled X_train shape:", X_train.shape)
# print("Scaled X_test shape:", X_test.shape)

# Verify feature scaling
# print("\nScaled training data mean:")
# print(X_train.mean(axis=0))

# print("\nScaled training data standard deviation:")
# print(X_train.std(axis=0))

# Create the Linear Regression model
model = LinearRegression()

# Train the model using the scaled training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# print("\nFirst 10 predictions:")
# print(y_pred[:10])

# print("\nFirst 10 actual values:")
# print(y_test.values[:10])

# print("\nLinear Regression model trained successfully!")


# Model evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# print("\nModel Evaluation:")
# print("MAE:", mae)
# print("MSE:", mse)
# print("RMSE:", rmse)
# print("R² Score:", r2)


# Actual vs Predicted values
plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

# Perfect prediction line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs Predicted House Prices")

plt.tight_layout()

# Save the plot
plt.savefig("results/actual_vs_predicted.png")

plt.show()