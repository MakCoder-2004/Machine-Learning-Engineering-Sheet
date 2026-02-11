# Anomaly Detection

**Anomaly Detection** (or **Outlier Detection**) is an **unsupervised learning technique** used to identify **data points that deviate significantly** from the majority of the data.

These anomalies often represent critical information such as **fraudulent activity**, **network intrusions**, **manufacturing defects**, or **rare diseases**.

---

### Why Use Anomaly Detection?

* Detect **rare or abnormal patterns** in data.
* Useful in **fraud detection**, **network security**, and **monitoring systems**.
* Works well even when data is **unlabeled**.
* Helps improve model reliability by removing noisy or abnormal samples.

---

### When to Use Anomaly Detection

* When the **normal behavior** of data is well-defined.
* When **anomalies are rare** but important.
* When **labeled data** for abnormal cases is unavailable.
* When you need **early detection** of unusual patterns.

---

## Required Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

---

## Loading Data

```python
path = "data.csv"
df = pd.read_csv(path)
print(df.head())
```

---

## Visualize Data (2D Example)

```python
plt.scatter(df['x1'], df['x2'])
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Data Visualization')
plt.show()
```

---

## Algorithm Implementation From Scratch (Gaussian Model)

A common approach to anomaly detection assumes that **normal data follows a Gaussian distribution**.
We model each feature with mean (μ) and variance (σ²) and compute the **probability of each data point**.

### 1. Estimate Gaussian Parameters

```python
def estimate_gaussian(X):
    mu = np.mean(X, axis=0)
    sigma2 = np.var(X, axis=0)
    return mu, sigma2
```

### 2. Compute Probability Density

```python
def multivariate_gaussian(X, mu, sigma2):
    k = len(mu)
    sigma2 = np.diag(sigma2)
    X = X - mu
    return (1 / ((2 * np.pi) ** (k / 2) * np.linalg.det(sigma2) ** 0.5)) * \
           np.exp(-0.5 * np.sum(X @ np.linalg.inv(sigma2) * X, axis=1))
```

### 3. Select Threshold (ε)

```python
def select_threshold(y_val, p_val):
    best_epsilon = 0
    best_f1 = 0
    step = (max(p_val) - min(p_val)) / 1000
    
    for epsilon in np.arange(min(p_val), max(p_val), step):
        preds = p_val < epsilon
        tp = np.sum((preds == 1) & (y_val == 1))
        fp = np.sum((preds == 1) & (y_val == 0))
        fn = np.sum((preds == 0) & (y_val == 1))
        
        if tp + fp == 0 or tp + fn == 0:
            continue
        
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec)
        
        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = epsilon
    return best_epsilon, best_f1
```

### 4. Putting It All Together

```python
X = df[['x1', 'x2']].values

# Estimate parameters
mu, sigma2 = estimate_gaussian(X)

# Compute probabilities
p = multivariate_gaussian(X, mu, sigma2)

# Assume some validation data and labels for threshold selection
# (y_val = 1 for anomaly, 0 for normal)
# best_epsilon, best_f1 = select_threshold(y_val, p_val)
# anomalies = X[p < best_epsilon]
```

### 5. Visualize Anomalies

```python
plt.scatter(X[:, 0], X[:, 1], label='Normal Data')
plt.scatter(anomalies[:, 0], anomalies[:, 1], c='r', marker='x', s=100, label='Anomalies')
plt.legend()
plt.title('Anomaly Detection Results')
plt.show()
```

---

## Anomaly Detection Using Scikit-Learn

The **Scikit-Learn** library provides multiple anomaly detection models such as:

* **Isolation Forest**
* **One-Class SVM**
* **Local Outlier Factor (LOF)**

Below is an example using **Isolation Forest**.

---

### Libraries

```python
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
```

---

### Data Preparation

```python
df = pd.read_csv("data.csv")
X = df[['x1', 'x2']].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

### Apply Isolation Forest

```python
iso = IsolationForest(contamination=0.05, random_state=1)
y_pred = iso.fit_predict(X_scaled)
```

---

### Identify Anomalies

```python
df['anomaly'] = y_pred
normal = df[df['anomaly'] == 1]
anomalies = df[df['anomaly'] == -1]

print("Number of anomalies:", len(anomalies))
```

---

### Visualization

```python
plt.scatter(normal['x1'], normal['x2'], c='blue', label='Normal Data')
plt.scatter(anomalies['x1'], anomalies['x2'], c='red', label='Anomalies', marker='x')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Anomaly Detection using Isolation Forest')
plt.legend()
plt.show()
```

---

## Alternative Models for Anomaly Detection

| Algorithm                      | Description                           | Suitable For                         |
| ------------------------------ | ------------------------------------- | ------------------------------------ |
| **Isolation Forest**           | Detects anomalies by isolating points | Large datasets                       |
| **One-Class SVM**              | Learns the boundary of normal data    | Small to medium datasets             |
| **Local Outlier Factor (LOF)** | Detects local density anomalies       | Datasets with local variations       |
| **Elliptic Envelope**          | Assumes Gaussian distribution         | Data following a normal distribution |

---

## Evaluation Metrics

For labeled data (when available):

* **Precision** – How many detected anomalies are actual anomalies.
* **Recall** – How many real anomalies were detected.
* **F1-Score** – Balance between precision and recall.

```python
from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_true, y_pred == -1)
recall = recall_score(y_true, y_pred == -1)
f1 = f1_score(y_true, y_pred == -1)

print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
```

---

## Summary

- **Anomaly Detection** helps uncover unusual data patterns.
- Works well in **unsupervised** or **semi-supervised** settings.
- Common algorithms:
  * **Gaussian model**
  * **Isolation Forest**
  * **One-Class SVM**
  * **Local Outlier Factor**
