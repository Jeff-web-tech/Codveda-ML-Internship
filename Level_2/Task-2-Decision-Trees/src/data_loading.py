import pandas as pd

# Load the Iris dataset
df = pd.read_csv("dataset/iris.csv")

# Display basic information
print("Dataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nClass Distribution:")
print(df.iloc[:, -1].value_counts())