# Naive Bayes

Naive Bayes is a **supervised learning algorithm** used primarily for **classification** tasks. It is based on **Bayes’ Theorem**, with the **naive assumption** that all features are independent of each other given the class label.

Bayes’ Theorem is defined as:

$$
P(y|x) = \frac{P(x|y)\,P(y)}{P(x)}
$$

Where:

* $P(y|x)$: Posterior probability (probability of class $y$ given data $x$)
* $P(x|y)$: Likelihood (probability of data $x$ given class $y$)
* $P(y)$: Prior (probability of class $y$)
* $P(x)$: Evidence (probability of data $x$)

Because $P(x)$ is constant for all classes, the algorithm focuses on maximizing:

$$
P(x|y)P(y)
$$

---

### Why Use Naive Bayes?

* Very fast and efficient for large datasets.
* Performs well with **text data** (e.g., spam detection, sentiment analysis).
* Works even with a small amount of training data.
* Handles **multi-class** classification problems easily.

---

### When to Use Naive Bayes

* When features are **conditionally independent** (or approximately independent).
* When your input is **categorical** or text-based.
* When speed and low computation cost are important.

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
# Calculate prior probabilities
def calculate_prior(y):
    classes = np.unique(y)
    priors = {}
    for cls in classes:
        priors[cls] = np.mean(y == cls)
    return priors

# Calculate likelihoods assuming Gaussian distribution
def calculate_likelihoods(X, y):
    classes = np.unique(y)
    likelihoods = {}
    for cls in classes:
        X_cls = X[y == cls]
        likelihoods[cls] = {
            "mean": np.mean(X_cls, axis=0),
            "var" : np.var(X_cls, axis=0)
        }
    return likelihoods

# Gaussian probability density function
def gaussian_prob(x, mean, var):
    eps = 1e-6  # to avoid division by zero
    numerator   = np.exp(-(x - mean)**2 / (2 * (var + eps)))
    denominator = np.sqrt(2 * np.pi * (var + eps))
    return numerator / denominator

# Predict class
def predict(X, priors, likelihoods):
    y_pred = []
    for x in X:
        posteriors = {}
        for cls in priors:
            prior = priors[cls]
            likelihood = np.prod(
                gaussian_prob(x, likelihoods[cls]['mean'], likelihoods[cls]['var'])
            )
            posteriors[cls] = prior * likelihood
        y_pred.append(max(posteriors, key=posteriors.get))
    return np.array(y_pred)

# Example dataset
X = df[['x1','x2']].values
y = df['y'].values

# Train model
priors = calculate_prior(y)
likelihoods = calculate_likelihoods(X, y)

# Predict
y_pred = predict(X, priors, likelihoods)

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
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns
```

### Load Dataset

```python
df = pd.read_csv('data.csv')
print(df)
print('\nNumber of rows and columns in the dataset: ', df.shape)
```

### Show First 5 Rows

```python
df.head()
```

### Dataset Description

```python
df.describe()
```

### Check for Missing Values

```python
missing_values = df.isnull().sum()
print(missing_values)
```

### Features and Output (Label)

```python
input_df  = df.drop(columns='y')
target_df = df['y']
```

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

### Scaling Features

```python
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(input_df)
input_df      = pd.DataFrame(scaled_features, columns=input_df.columns)
```

### Split Data

```python
X_train, X_test, y_train, y_test = train_test_split(
    input_df, target_df, test_size=0.3, random_state=1
)
```

### Apply Naive Bayes

```python
nb = GaussianNB()
nb.fit(X_train, y_train)
```

### Testing / Prediction

```python
prediction_test = nb.predict(X_test)
print(y_test.values, prediction_test)
```

### Calculate Accuracy

```python
accuracy = accuracy_score(y_test, prediction_test)
print("Accuracy:", accuracy)
```

### Classification Report

```python
print("\nClassification Report:\n", classification_report(y_test, prediction_test))
```

### Confusion Matrix

```python
conf_matrix = confusion_matrix(y_test, prediction_test)
print("\nConfusion Matrix:\n", conf_matrix)
```

### Visualization

```python
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

---
