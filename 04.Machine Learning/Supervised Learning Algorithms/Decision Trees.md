# Decision Tree Algorithm

A **Decision Tree** is a **supervised learning algorithm** used for both **classification** and **regression** tasks.
It works by **splitting the dataset into subsets** based on the most significant attribute at each step, forming a tree-like structure of decisions.

---

## How a Decision Tree Works

1. **Select the Best Feature to Split**:

   * For classification → based on **Information Gain**, **Entropy**, or **Gini Index**.
   * For regression → based on **variance reduction** or **mean squared error (MSE)**.

2. **Create Decision Nodes**:

   * Each split forms a decision node.

3. **Recursive Splitting**:

   * Continue splitting until a stopping condition is met (e.g., max depth, min samples per node).

4. **Leaf Nodes**:

   * Final predictions are made at the leaves.

---

### Types of Decision Trees

* **Classification Trees** → predict categorical outcomes.
* **Regression Trees** → predict continuous outcomes.

---

### Why Use Decision Trees?

* Easy to interpret and visualize.
* Handles both **numerical** and **categorical** data.
* No need for feature scaling (normalization or standardization).
* Works well with **non-linear relationships**.

---

### When to Use Decision Trees

* When you want a **simple and interpretable** model.
* When your data has **non-linear boundaries**.
* When handling datasets with a mix of **categorical and numerical features**.

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

## Visualize Class Distribution (for classification)

```python
plt.scatter(df.x1, df.x2, c=df.y, cmap='bwr')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Scatter Plot of x1 vs x2')
plt.show()
```

---

## Algorithm Implementation From Scratch )

### Classification (using Gini Index)

```python
# Gini Index calculation
def gini_index(groups, classes):
    n_instances = float(sum([len(group) for group in groups]))
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        gini += (1.0 - score) * (size / n_instances)
    return gini
```

### Regression (using Variance Reduction)

```python
def variance_reduction(groups, y):
    total_var = np.var(y)
    total_len = len(y)
    weighted_var = 0.0
    for group in groups:
        if len(group) == 0:
            continue
        group_var = np.var([row[-1] for row in group])
        weighted_var += (len(group) / total_len) * group_var
    return total_var - weighted_var
```

> (Full tree-building implementation from scratch is lengthy. Usually sklearn is used in practice.)

---

## How to Implement the Algorithm (Using sklearn)

### Libraries

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
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
input_df = pd.DataFrame(scaled_features, columns=input_df.columns)
```

---

### Split Data

```python
X_train, X_test, y_train, y_test = train_test_split(
    input_df, target_df, test_size=0.3, random_state=1
)
```

---

## Decision Tree – Classification

```python
clf = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=1)
clf.fit(X_train, y_train)

# Prediction
y_pred_clf = clf.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred_clf)
print("Classification Accuracy:", accuracy)

# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred_clf))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_clf)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Classification)')
plt.show()

# Visualize Tree
plt.figure(figsize=(12,8))
plot_tree(clf, filled=True, feature_names=input_df.columns, class_names=True)
plt.show()
```

---

## Decision Tree – Regression

```python
reg = DecisionTreeRegressor(criterion="squared_error", max_depth=5, random_state=1)
reg.fit(X_train, y_train)

# Prediction
y_pred_reg = reg.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred_reg)
r2  = r2_score(y_test, y_pred_reg)

print("Regression MSE:", mse)
print("Regression R2 Score:", r2)

# Plot Actual vs Predicted
plt.scatter(y_test, y_pred_reg, color="blue")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Decision Tree Regression: Actual vs Predicted")
plt.show()
```
