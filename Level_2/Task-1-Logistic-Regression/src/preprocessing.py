import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

train_df = pd.read_csv("dataset/churn-bigml-80.csv", sep="\t")
test_df = pd.read_csv("dataset/churn-bigml-20.csv", sep="\t")

# print(train_df.shape)
# print(test_df.shape)

# print("Training Data:")
# # print(train_df.head())

# print("\nTesting data:")
# # print(test_df.head())

# print("\nMissing values in training data:")
# print(train_df.isnull().sum())

# print("\nMissing values in testing data:")
# print(test_df.isnull().sum())

# #Checking for categorical variables
# print("\nData Types:")
# # print(train_df.dtypes)

# print("\nUnique States:")
# print(train_df["State"].unique())

# print("\nInternational Plan:")
# print(train_df["International plan"].unique())

# print("\nVoice Mail Plan:")
# print(train_df["Voice mail plan"].unique())

#Binary Encoding on categorical variables
train_df["International plan"]  = train_df["International plan"].map({
    "No": 0,
    "Yes": 1
})
train_df["Voice mail plan"] = train_df["Voice mail plan"].map({
    "No": 0,
    "Yes": 1
})
# print("\nAfter binary encoding on traning dataset:")
# print(train_df[["International plan", "Voice mail plan"]].head())

test_df["International plan"] = test_df["International plan"].map({
    "No": 0,
    "Yes": 1
})
test_df["Voice mail plan"] = test_df["Voice mail plan"].map({
    "No": 0,
    "Yes": 1
})
# print("\nAfter binary encoding on testing dataset:")
# print(test_df[["International plan", "Voice mail plan"]].head())

#One-hot encoding on state column
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

state_train = encoder.fit_transform(train_df[["State"]])
state_test = encoder.transform(test_df[["State"]])

# print("\nEncoded State data:")
# print(state_train[:5])

# print("\nShape of encoded State data:")
# print(state_train.shape)

state_columns = encoder.get_feature_names_out(["State"])

# print("\nState column names:")
# print(state_columns)

state_train_df = pd.DataFrame(
    state_train,
    columns=state_columns,
    index=train_df.index
)

state_test_df = pd.DataFrame(
    state_test,
    columns=state_columns,
    index=test_df.index
)

train_df = train_df.drop(columns=["State"])
test_df = test_df.drop(columns=["State"])

train_df = pd.concat([train_df, state_train_df], axis=1)
test_df = pd.concat([test_df, state_test_df], axis=1)

# print("\nProcessed training data shape:")
# print(train_df.shape)

# print("\nProcessed testing data shape:")
# print(test_df.shape)

# print("\nProcessed training data:")
# print(train_df.head())

#Scaling
numerical_columns = [
    "Account length",
    "Area code",
    "Number vmail messages",
    "Total day minutes",
    "Total day calls",
    "Total day charge",
    "Total eve minutes",
    "Total eve calls",
    "Total eve charge",
    "Total night minutes",
    "Total night calls",
    "Total night charge",
    "Total intl minutes",
    "Total intl calls",
    "Total intl charge",
    "Customer service calls"
]

# print("\nNumerical columns to be scaled:")
# print(numerical_columns)

scaler = StandardScaler()

scaler.fit(train_df[numerical_columns])

train_df[numerical_columns] = scaler.transform(
    train_df[numerical_columns]
)

test_df[numerical_columns] = scaler.transform(
    test_df[numerical_columns]
)

# print("\nScaled numerical data:")
# print(train_df[numerical_columns].head())

# print("\nMean of scaled numerical columns:")
# print(train_df[numerical_columns].mean())


# Separate features and target
X_train = train_df.drop(columns=["Churn"])
y_train = train_df["Churn"]

X_test = test_df.drop(columns=["Churn"])
y_test = test_df["Churn"]

# print("\nFeature and target shapes:")
# print("X_train:", X_train.shape)
# print("y_train:", y_train.shape)
# print("X_test:", X_test.shape)
# print("y_test:", y_test.shape)

# print("\nTraining target distribution:")
# print(y_train.value_counts())

# print("\nTesting target distribution:")
# print(y_test.value_counts())

# Save processed datasets
X_train.to_csv("results/X_train.csv", index=False)
X_test.to_csv("results/X_test.csv", index=False)

y_train.to_csv("results/y_train.csv", index=False)
y_test.to_csv("results/y_test.csv", index=False)

# print("\nProcessed datasets saved successfully!")


print("\nVerification of saved files:")

print("X_train:", pd.read_csv("results/X_train.csv").shape)
print("X_test:", pd.read_csv("results/X_test.csv").shape)
print("y_train:", pd.read_csv("results/y_train.csv").shape)
print("y_test:", pd.read_csv("results/y_test.csv").shape)