# K-Nearest Neighbors (KNN)

K-Nearest Neighbors is a **supervised learning algorithm** used for **classification** and **regression**.
It is based on the idea that a data point’s label is determined by the **labels (classification)** or **values (regression)** of its nearest neighbors in the feature space.

---

### How KNN Works

1. Choose the number of neighbors **k**.
2. Calculate the **distance** (e.g., Euclidean) between the new data point and all training points.
3. Select the **k-nearest neighbors**.
4. For classification: assign the class with the **majority vote** among neighbors.
   For regression: take the **average value** of the neighbors.

---

### Why Use KNN?

* Simple and intuitive to understand.
* Works for both **classification** and **regression**.
* No assumptions about data distribution (non-parametric).
* Effective when the decision boundary is irregular.

---

### When to Use KNN

* When you have a **small to medium-sized dataset** (since large datasets make it slow).
* When data is **not too high-dimensional** (curse of dimensionality).
* When you want an **easy-to-implement** algorithm for quick prediction.

---

# Algorithm Implementation From Scratch

## KNN Class
```python
class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Majority vote, most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
```

## Example Usage

### Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
```

### Generate a Dataset
```python
X,y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
```

### Visualize Relation Between Data
```python
plt.scatter(X, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter Plot of x vs y')
plt.show()
```

### Apply KNN
```python
knn = KNN(k=3)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
```

### Visualization
```python
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(10,5))
m1 = plt.scatter(X_test, y_test, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, predictions, color=cmap(0.5), s=10)
plt.show()
```

---

## How to Implement the Algorithm Using Scikit-Learn Library

### Libraries

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
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
# For classification (assuming 'y' is categorical)
X_class = df.drop(columns='y')
y_class = df['y']

# For regression (assuming 'target' is continuous)
X_reg = df.drop(columns='target')
y_reg = df['target']
```

---

### Encoding Categorical Columns (if any)

```python
# Replace with your own column names
columns_to_encode = ["X1", "X2", "X3"]
le = LabelEncoder()
for col in columns_to_encode:
    if col in X_class.columns:
        X_class["encoded_" + col] = le.fit_transform(X_class[col])
        X_class = X_class.drop(col, axis=1)
```

---

### Scaling Features

```python
scaler = MinMaxScaler()
X_class = pd.DataFrame(scaler.fit_transform(X_class), columns=X_class.columns)
X_reg   = pd.DataFrame(scaler.fit_transform(X_reg), columns=X_reg.columns)
```

---

## KNN for Classification

### Split Data

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_class, y_class, test_size=0.3, random_state=1
)
```

### Apply KNN Classifier

```python
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train, y_train)
```

### Testing / Prediction

```python
prediction_test = knn_clf.predict(X_test)
print(y_test.values, prediction_test)
```

### Evaluate Performance

```python
accuracy = accuracy_score(y_test, prediction_test)
print("Classification Accuracy:", accuracy)

print("\nClassification Report:\n", classification_report(y_test, prediction_test))

conf_matrix = confusion_matrix(y_test, prediction_test)
print("\nConfusion Matrix:\n", conf_matrix)

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

---

## KNN for Regression

### Split Data

```python
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=1
)
```

### Apply KNN Regressor

```python
knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_train_reg, y_train_reg)
```

### Testing / Prediction

```python
prediction_test_reg = knn_reg.predict(X_test_reg)
```

### Evaluate Performance

```python
mse = mean_squared_error(y_test_reg, prediction_test_reg)
r2 = r2_score(y_test_reg, prediction_test_reg)

print("Regression Mean Squared Error:", mse)
print("Regression R2 Score:", r2)
```

### Visualization

```python
plt.scatter(y_test_reg, prediction_test_reg, color='blue')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('KNN Regression: Actual vs Predicted')
plt.show()
```
