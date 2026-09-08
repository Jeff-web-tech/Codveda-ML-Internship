import pandas as pd

# Load training and testing datasets
train_df = pd.read_csv("dataset/churn-bigml-80.csv", sep="\t")
test_df = pd.read_csv("dataset/churn-bigml-20.csv", sep="\t")

# Display dataset shapes
print("Training Data Shape:", train_df.shape)
print("Testing Data Shape:", test_df.shape)

# Display first 5 rows
print("\nTraining Data:")
print(train_df.head())

print("\nTesting Data:")
print(test_df.head())

# Display column information
print("\nTraining Data Information:")
print(train_df.info())

print("\nTesting Data Information:")
print(test_df.info())

# Check missing values
print("\nMissing Values in Training Data:")
print(train_df.isnull().sum())

print("\nMissing Values in Testing Data:")
print(test_df.isnull().sum())

# Check target distribution
print("\nTraining Churn Distribution:")
print(train_df["Churn"].value_counts())

print("\nTesting Churn Distribution:")
print(test_df["Churn"].value_counts())