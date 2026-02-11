# K-Means Clustering

K-Means is an **unsupervised learning algorithm** used for **clustering** tasks.
It groups data into **k clusters**, where each data point belongs to the cluster with the nearest mean (centroid).

---

### How K-Means Works

1. Choose the number of clusters **k**.
2. Randomly initialize **k centroids**.
3. Assign each data point to the nearest centroid.
4. Recalculate centroids as the mean of the assigned points.
5. Repeat steps 3–4 until convergence (centroids do not change significantly).

---

### Why Use K-Means?

* Simple and efficient for clustering large datasets.
* Works well when clusters are spherical and evenly sized.
* Easy to implement and interpret.

---

### When to Use K-Means

* When you want to **group unlabeled data**.
* When clusters are roughly circular and similar in size.
* When you need **dimensionality reduction** or preprocessing before classification.

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

## Visualize Data (if 2D)

```python
plt.scatter(df.x1, df.x2)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Scatter Plot of Data')
plt.show()
```

---

## Algorithm Implementation From Scratch

```python
# Random initialization of centroids
def initialize_centroids(X, k):
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices]

# Assign clusters
def assign_clusters(X, centroids):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

# Update centroids
def update_centroids(X, labels, k):
    centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
    return centroids

# K-Means main function
def kmeans(X, k, max_iters=100):
    centroids = initialize_centroids(X, k)
    for _ in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return labels, centroids

# Example dataset
X = df[['x1','x2']].values
labels, centroids = kmeans(X, k=3)

print("Cluster assignments:", labels)
print("Centroids:\n", centroids)
```

---

## How to Implement the Algorithm (Using sklearn)

### Libraries

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
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

### Features (No Labels in Unsupervised Learning)

```python
input_df = df.copy()
```

---

### Scaling Features

```python
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(input_df)
input_df = pd.DataFrame(scaled_features, columns=input_df.columns)
```

---

### Apply K-Means

```python
kmeans = KMeans(n_clusters=3, random_state=1)
kmeans.fit(input_df)
```

---

### Get Cluster Assignments and Centroids

```python
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

print("Cluster assignments:", labels)
print("Centroids:\n", centroids)
```

---

### Visualization (2D Example)

```python
plt.scatter(input_df.iloc[:,0], input_df.iloc[:,1], c=labels, cmap='viridis')
plt.scatter(centroids[:,0], centroids[:,1], c='red', marker='x', s=200, label='Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering')
plt.legend()
plt.show()
```

---

## Elbow Method (Choosing Optimal k)

The **Elbow Method** helps to determine the optimal number of clusters by plotting the inertia (sum of squared distances of samples to their closest cluster center) against different values of *k*.

```python
inertia = []
k_values = range(1, 11)

for k in k_values:
    km = KMeans(n_clusters=k, random_state=1)
    km.fit(input_df)
    inertia.append(km.inertia_)

plt.plot(k_values, inertia, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()
```
---