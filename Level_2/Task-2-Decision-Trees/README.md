# Decision Tree Classification

## Overview

This project implements a Decision Tree classification model using the Iris dataset. The objective is to train a Decision Tree, visualize its structure, evaluate its performance using classification metrics, and prune the tree to reduce overfitting.

## Dataset

The Iris dataset contains 150 samples representing three species of iris flowers:

* Setosa
* Versicolor
* Virginica

The dataset contains four numerical features:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

The target variable is `species`.

The dataset was divided into:

* 80% training data
* 20% testing data

The split used `random_state=42` and stratification to maintain the class distribution.

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Matplotlib

## Project Requirements

### 1. Train a Decision Tree

A Decision Tree Classifier was trained using the Gini impurity criterion.

The initial model achieved:

* Training Accuracy: **100%**
* Testing Accuracy: **93.33%**

The difference between training and testing accuracy indicated slight overfitting.

### 2. Visualize the Decision Tree

The structure of the Decision Tree was visualized using Scikit-learn's `plot_tree()` function.

The visualization is available in:

```text
results/decision_tree.png
```

### 3. Prune the Decision Tree

To reduce overfitting, different values of `max_depth` were tested.

| Max Depth | Testing Accuracy | Weighted F1-score |
| --------: | ---------------: | ----------------: |
|         1 |           66.67% |            0.5556 |
|         2 |           93.33% |            0.9333 |
|     **3** |       **96.67%** |        **0.9666** |
|         4 |           93.33% |            0.9333 |
|         5 |           93.33% |            0.9333 |

The best result was obtained with:

```text
max_depth = 3
```

The pruned model achieved a testing accuracy of **96.67%**, improving on the original model's 93.33% testing accuracy.

The pruned tree visualization is available in:

```text
results/pruned_decision_tree.png
```

### 4. Model Evaluation

The final pruned Decision Tree achieved:

* **Testing Accuracy:** 96.67%
* **Weighted F1-score:** 0.9666

Classification report:

| Class      | Precision | Recall | F1-score |
| ---------- | --------: | -----: | -------: |
| Setosa     |      1.00 |   1.00 |     1.00 |
| Versicolor |      1.00 |   0.90 |     0.95 |
| Virginica  |      0.91 |   1.00 |     0.95 |

The final model correctly classified **29 out of 30 test samples**.

The confusion matrix is available in:

```text
results/confusion_matrix.png
```

## Conclusion

The Decision Tree model performed very well on the Iris dataset. The initial model achieved a testing accuracy of 93.33% but showed signs of slight overfitting because it achieved 100% training accuracy.

Pruning the tree by setting `max_depth=3` improved the testing accuracy to **96.67%** and produced a weighted F1-score of **0.9666**. The pruned model correctly classified 29 out of 30 test samples.

These results demonstrate that reducing the complexity of the Decision Tree helped improve its ability to generalize to unseen data.

## Project Structure

```text
Task-2-Decision-Tree/
│
├── dataset/
│   └── iris.csv
│
├── results/
│   ├── confusion_matrix.png
│   ├── decision_tree.png
│   └── pruned_decision_tree.png
│
├── src/
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── decision_tree.py
│   ├── evaluation.py
│   ├── pruning.py
│   └── pruned_decision_tree.py
│
├── README.md
└── requirements.txt
```
