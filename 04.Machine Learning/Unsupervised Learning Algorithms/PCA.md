# Principal Component Analysis (PCA)

PCA is an **unsupervised learning algorithm** used for **dimensionality reduction**.
It transforms the data into a new coordinate system where the greatest variances lie on the first components.

---

### How PCA Works

1. Standardize the dataset.
2. Compute the covariance matrix.
3. Calculate the eigenvalues and eigenvectors of the covariance matrix.
4. Sort eigenvectors by decreasing eigenvalues and choose the top *k*.
5. Transform the data into the new subspace using the selected components.

---

### Why Use PCA?

* Reduces dimensionality while preserving variance.
* Helps visualize high-dimensional data.
* Reduces overfitting and improves algorithm performance.

---

### When to Use PCA

* When you have **high-dimensional datasets**.
* When features are highly correlated.
* As a preprocessing step before clustering or classification.

---

## Required Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
```

---

## Loading Data

```python
path = "data.csv"
df = pd.read_csv(path)
print(df)
```

---

## Visualize Data (if 2D or 3D)

```python
if df.shape[1] == 2:
    plt.scatter(df.iloc[:,0], df.iloc[:,1])
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Scatter Plot of Data')
    plt.show()
```

---

## PCA Implementation From Scratch (Conceptual)

```python
# Standardize data
X = df.values
X_meaned = X - np.mean(X, axis=0)

# Covariance matrix
cov_mat = np.cov(X_meaned, rowvar=False)

# Eigen decomposition
eigen_vals, eigen_vecs = np.linalg.eigh(cov_mat)

# Sort eigenvalues and eigenvectors
sorted_index = np.argsort(eigen_vals)[::-1]
sorted_eigenvectors = eigen_vecs[:,sorted_index]
sorted_eigenvalues = eigen_vals[sorted_index]

# Select top k eigenvectors
n_components = 2
eigenvector_subset = sorted_eigenvectors[:,0:n_components]

# Transform the data
X_reduced = np.dot(eigenvector_subset.transpose(), X_meaned.transpose()).transpose()

print("Reduced Data:\n", X_reduced)
```

---

## PCA Using sklearn

### Libraries

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

### Features (No Labels in PCA)

```python
input_df = df.copy()
```

---

### Scaling Features

```python
scaler = StandardScaler()
scaled_features = scaler.fit_transform(input_df)
input_df = pd.DataFrame(scaled_features, columns=input_df.columns)
```

---

### Apply PCA

```python
pca = PCA(n_components=2)  # Reduce to 2 components for visualization
principal_components = pca.fit_transform(input_df)
```

---

### Create PCA DataFrame

```python
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
print(pca_df.head())
```

---

### Explained Variance Ratio

```python
print("Explained variance ratio:", pca.explained_variance_ratio_)
```

---

### Visualization (2D PCA Projection)

```python
plt.scatter(pca_df['PC1'], pca_df['PC2'])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Projection')
plt.show()
```

---

## Screen Plot (Choosing Optimal Number of Components)

The **Scree Plot** helps determine the optimal number of components by plotting the explained variance ratio against the number of components.

```python
pca_full = PCA()
pca_full.fit(input_df)
explained_variance = np.cumsum(pca_full.explained_variance_ratio_)

plt.plot(range(1, len(explained_variance)+1), explained_variance, 'bx-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Scree Plot for PCA')
plt.show()
```

---
