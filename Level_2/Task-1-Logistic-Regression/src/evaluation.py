import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the test data
X_test = pd.read_csv("results/X_test.csv")
y_test = pd.read_csv("results/y_test.csv").squeeze()

# Load the trained model
from sklearn.linear_model import LogisticRegression

X_train = pd.read_csv("results/X_train.csv")
y_train = pd.read_csv("results/y_train.csv").squeeze()

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display confusion matrix
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Churn", "Churn"]
)

display.plot()
plt.title("Logistic Regression Confusion Matrix")
plt.tight_layout()

# Save the figure
plt.savefig("results/confusion_matrix.png", dpi=300)

plt.show()