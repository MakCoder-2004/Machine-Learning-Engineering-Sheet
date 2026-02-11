# Random Forest

Random Forest is a **supervised learning algorithm** used for both **classification** and **regression** tasks.
It is an **ensemble method** that builds multiple decision trees during training and combines their predictions (majority voting for classification or averaging for regression) to improve performance and reduce overfitting.

---

### How Random Forest Works

1. **Bootstrap Sampling**: From the original dataset, random subsets are created with replacement.
2. **Decision Trees**: A decision tree is trained on each subset, but only on a **random subset of features** at each split.
3. **Aggregation**:

   * For classification → majority voting across all trees.
   * For regression → average prediction across all trees.

---

### Why Use Random Forest?

* Handles both **classification and regression** problems.
* Reduces overfitting compared to a single decision tree.
* Works well with high-dimensional data.
* Provides **feature importance scores**.
* Robust to noise and outliers.

---

### When to Use Random Forest

* When you need **high accuracy** without much tuning.
* When dataset has a mix of **categorical and numerical features**.
* When interpretability is less important than performance.
* When you want to know **feature importance**.

---

## Required Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

---

## Loading Data

```python
path = "data.csv"
df = pd.read_csv(path)
print(df)
```

---

## Visualize Class Distribution

```python
# Example visualization if your dataset has two features x1 and x2
plt.scatter(df.x1, df.x2, c=df.y, cmap='bwr')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Scatter Plot of x1 vs x2')
plt.show()
```

---

## Algorithm Implementation From Scratch

```python
from collections import Counter
import random

# Gini Impurity
def gini(y):
    counts = Counter(y)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / len(y)
        impurity -= prob_of_lbl**2
    return impurity

# Split dataset
def split_dataset(X, y, feature, threshold):
    left_idx = X[:, feature] <= threshold
    right_idx = X[:, feature] > threshold
    return X[left_idx], y[left_idx], X[right_idx], y[right_idx]

# Build a Decision Tree (simplified)
class DecisionTree:
    def __init__(self, max_depth=5, min_samples=2):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.tree = None

    def fit(self, X, y, depth=0):
        if len(set(y)) == 1 or depth >= self.max_depth or len(y) < self.min_samples:
            return Counter(y).most_common(1)[0][0]

        n_features = X.shape[1]
        best_feature, best_threshold, best_score = None, None, 1
        best_splits = None

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                X_left, y_left, X_right, y_right = split_dataset(X, y, feature, threshold)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                score = (len(y_left) * gini(y_left) + len(y_right) * gini(y_right)) / len(y)
                if score < best_score:
                    best_feature, best_threshold, best_score = feature, threshold, score
                    best_splits = (X_left, y_left, X_right, y_right)

        if best_feature is None:
            return Counter(y).most_common(1)[0][0]

        left_tree = self.fit(best_splits[0], best_splits[1], depth+1)
        right_tree = self.fit(best_splits[2], best_splits[3], depth+1)
        return (best_feature, best_threshold, left_tree, right_tree)

    def train(self, X, y):
        self.tree = self.fit(X, y)

    def predict_one(self, x, node):
        if not isinstance(node, tuple):
            return node
        feature, threshold, left, right = node
        if x[feature] <= threshold:
            return self.predict_one(x, left)
        else:
            return self.predict_one(x, right)

    def predict(self, X):
        return np.array([self.predict_one(x, self.tree) for x in X])

# Random Forest
class RandomForest:
    def __init__(self, n_estimators=5, max_depth=5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def bootstrap_sample(self, X, y):
        idxs = np.random.choice(len(y), len(y), replace=True)
        return X[idxs], y[idxs]

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            X_samp, y_samp = self.bootstrap_sample(X, y)
            tree = DecisionTree(max_depth=self.max_depth)
            tree.train(X_samp, y_samp)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        y_pred = []
        for col in tree_preds.T:
            y_pred.append(Counter(col).most_common(1)[0][0])
        return np.array(y_pred)

# Example dataset
X = df[['x1','x2']].values
y = df['y'].values

# Train Random Forest
rf = RandomForest(n_estimators=5, max_depth=5)
rf.fit(X, y)
y_pred = rf.predict(X)

print("Predictions:", y_pred)
```

---

## How to Implement the Algorithm (Using sklearn)

### Libraries

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
```

---

### Load Dataset

```python
df = pd.read_csv('data.csv')
print(df)
print('\nNumber of rows and columns in the dataset: ', df.shape)
```

---

### Features and Output (Label)

```python
input_df  = df.drop(columns='y')
target_df = df['y']
```

---

### Encoding Categorical Columns

```python
# Replace with your own column names
columns_to_encode = ["X1", "X2", "X3"]
le = LabelEncoder()
for col in columns_to_encode:
    input_df["encoded_" + col] = le.fit_transform(input_df[col])

# Drop original categorical columns
drop_columns = ["X1", "X2", "X3"]
input_df = input_df.drop(drop_columns, axis=1)
```

---

### Scaling Features

```python
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(input_df)
input_df      = pd.DataFrame(scaled_features, columns=input_df.columns)
```

---

### Split Data

```python
X_train, X_test, y_train, y_test = train_test_split(
    input_df, target_df, test_size=0.3, random_state=1
)
```

---

### Apply Random Forest

```python
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
rf.fit(X_train, y_train)
```

---

### Testing / Prediction

```python
prediction_test = rf.predict(X_test)
print(y_test.values, prediction_test)
```

---

### Calculate Accuracy

```python
accuracy = accuracy_score(y_test, prediction_test)
print("Accuracy:", accuracy)
```

---

### Classification Report

```python
print("\nClassification Report:\n", classification_report(y_test, prediction_test))
```

---

### Confusion Matrix

```python
conf_matrix = confusion_matrix(y_test, prediction_test)
print("\nConfusion Matrix:\n", conf_matrix)
```

---

### Visualization

```python
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```
---