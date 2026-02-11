# Support Vector Machine (SVM)

Support Vector Machine is a **supervised learning algorithm** used for both **classification** and **regression**, but it is mostly used for classification problems. SVM aims to find the best decision boundary (called the **hyperplane**) that separates different classes with the maximum margin.

---

### Why Use Support Vector Machine?

* Effective in **high-dimensional spaces**.
* Works well with clear margin of separation.
* Robust to overfitting, especially in high-dimensional data.
* Can be used for linear and non-linear classification (with kernels).

---

### When to Use SVM

* When the dataset has **clear boundaries** between classes.
* When you want a **maximum-margin classifier**.
* When the number of features is large compared to the number of samples.
* When non-linear relationships exist (using kernel tricks).

---

### Decision Boundary in SVM

For a linear SVM, the decision boundary is defined as:

$w^T x + b = 0$

where:

* \$w\$ is the weight vector.
* \$b\$ is the bias.

The goal is to maximize the margin between the two classes.

---

### Loss Function (Hinge Loss)

The hinge loss is used in SVM:

$J(w, b) = \frac{1}{2} ||w||^2 + C \sum_{i=1}^n \max(0, 1 - y_i (w^T x_i + b))$

where:

* \$C\$ is the regularization parameter.
* \$y\_i \in {-1, +1}\$ are the true labels.

---

# How to Implement the Algorithm (Using sklearn)

## SVM for Classification (Support Vector Classification - SVC)

### Libraries

```python
import pandas as pd
import numpy as np
from sklearn import datasets, svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
```

### Load Dataset

```python
df = pd.read_csv('data.csv')
print(df)
print('\nNumber of rows and columns in the data set: ', df.shape)
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
input_df = df.drop(columns='y')  # drop the label column
target_df = df['y']
```

### Encoding Categorical Columns

```python
# Replace with your own column names
columns_to_encode = ["X1", "X2", "X3"]
le = LabelEncoder()
for col in columns_to_encode:
    input_df["encoded_" + col] = le.fit_transform(input_df[col])

# Drop the original categorical columns
drop_columns = ["X1", "X2", "X3"]
input_df = input_df.drop(drop_columns, axis=1)
```

### Scaling Features

```python
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(input_df)
input_df = pd.DataFrame(scaled_features, columns=input_df.columns)
```

### Split Data

```python
X_train, X_test, y_train, y_test = train_test_split(input_df, target_df, test_size=0.3, random_state=1)
```

### Apply Support Vector Machine

```python
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)
```

### Testing / Prediction

```python
prediction_test = clf.predict(X_test)
print("True labels:     ", y_test.values)
print("Predicted labels:", prediction_test)
```

### Calculate Accuracy

```python
accuracy = accuracy_score(y_test, prediction_test)
print("Accuracy:", accuracy)
```

### Calculate Classification Report

```python
print("\nClassification Report:\n", classification_report(y_test, prediction_test))
```

### Calculate the Confusion Matrix

```python
conf_matrix = confusion_matrix(y_test, prediction_test)
print("\nConfusion Matrix:\n", conf_matrix)
```

### Calculate Precision/ Recall/ F1-score

```python
precision = precision_score(y_test, prediction_test, average='weighted') * 100
recall = recall_score(y_test, prediction_test, average='weighted') * 100
f1 = f1_score(y_test, prediction_test, average='weighted') * 100

# Print results
print(f"Precision: {precision:.2f}%")
print(f"Recall: {recall:.2f}%")
print(f"F1 Score: {f1:.2f}%")
```

### Visualization

```python
# Title for the plot
title = "SVC with linear kernel"

x0 = X_train['X'] 
x1 = X_train['Y'] 

# Create the plot
fig, ax = plt.subplots(figsize=(8, 6))
disp = DecisionBoundaryDisplay.from_estimator(
    clf,
    X_train,
    response_method="predict",
    cmap=plt.cm.coolwarm,
    alpha=0.75,
    ax=ax,
    xlabel="Feature 1",
    ylabel="Feature 2",
)

# Scatter plot of training points
ax.scatter(x0, x1, c=y_train, edgecolors="k")

ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)

plt.show()
```

---

## K-Fold Cross Validation with SVM

```python
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
```

### Ensure that x and y are numpy arrays if they are pandas DataFrames
```pyhton
x = np.array(input_df)  # Features
y = np.array(target_df)  # Labels
```

### Initialize 5-fold cross-validation
```pyhton
kf = KFold(n_splits=5, shuffle=True, random_state=42)
```

### Initialize lists to store metrics for each fold
```python
accuracies = []
precisions = []
recalls = []
f1_scores = []
```

### Loop through each fold
```python
for fold, (train_index, test_index) in enumerate(kf.split(x)):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Initialize and train SVC
    model = SVC(kernel='linear', random_state=42)
    model.fit(x_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(x_test)
    
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Store metrics
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    
    print(f"Fold {fold + 1}:")
    print(f"  Accuracy: {accuracy:.2f}")
    print(f"  Precision: {precision:.2f}")
    print(f"  Recall: {recall:.2f}")
    print(f"  F1 Score: {f1:.2f}")
    print()
```

### Print average metrics across all folds
```python
print("Average Metrics Across All Folds:")
print(f"  Accuracy: {np.mean(accuracies):.2f}")
print(f"  Precision: {np.mean(precisions):.2f}")
print(f"  Recall: {np.mean(recalls):.2f}")
print(f"  F1 Score: {np.mean(f1_scores):.2f}")
```
---

## Support Vector Image Classification 

### Libraries
```python
import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from skimage.feature import local_binary_pattern
```

### Setting Our Categories
```pyhton
categories = ['cats', 'dogs']
```

### Extracting Features From The Image
```python
def extract_lbp_features(img):
    # Parameters for LBP (should be defined outside the function)
    radius = 3
    n_points = 8 * radius  # Number of circularly symmetric neighbor set points
    METHOD = 'uniform'  # LBP method
    
    # Convert to grayscale
    gray = rgb2gray(img)
    # Compute LBP - all arguments are positional here
    lbp = local_binary_pattern(gray, P=n_points, R=radius, method=METHOD)
    # Calculate histogram of LBP
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    # Normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)  # Add small value to avoid division by zero
    return hist

flat_data_arr = []
target_arr = []
```

### Uploading The Images To The Algorithm
```python
data_path = r'D:\Programming\Material\Machine Learning\NTI materials\DS'
for i in categories:
    print(f'Loading {i} images...')
    category_path = os.path.join(data_path, i)
    print(category_path)

    for img_name in os.listdir(category_path):
        try:
            img_arr = imread(os.path.join(category_path, img_name))
            img_resized = resize(img_arr, (150, 150, 3))
            
            # Extract LBP features instead of raw pixels
            lbp_features = extract_lbp_features(img_resized)
            
            flat_data_arr.append(lbp_features)
            target_arr.append(categories.index(i))
        except Exception as e:
            print(f"Error loading {img_name}: {str(e)}")
    print(f'{i} Loaded successfully')

flat_data = np.array(flat_data_arr)
target = np.array(target_arr)
print(f"Data shape: {flat_data.shape}")
```

### Create DataFrame
```python
df = pd.DataFrame(flat_data)
df['target'] = target
print(df.head())
```

### Split data
```python
features = df.iloc[:, :-1]
target = df['target']
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)
```

### Train SVM
```python
svc_model = svm.SVC(kernel='linear', random_state=32)
svc_model.fit(x_train, y_train)
```

### Evaluate
```python
predicted = svc_model.predict(x_test) 
print("Predictions:", predicted)
```

### Calculating Accuracy
```python
accuracy = accuracy_score(y_test, predicted)
print(f"Accuracy: {accuracy*100:.2f}%")
print(classification_report(y_test, predicted))
```

### Test on a single image
```python
test_path = r'D:\Programming\Material\Machine Learning\NTI materials\DS\cats\cat.12.jpg'
try:
    img = imread(test_path)
    plt.imshow(img)
    plt.show()
    img_resized = resize(img, (150, 150, 3))
    lbp_features = extract_lbp_features(img_resized)
    predicted_value = svc_model.predict([lbp_features])
    
    print(f"Prediction: {categories[int(predicted_value[0])]}")
    print("Raw predicted value:", predicted_value)
except Exception as e:
    print(f"Error processing test image: {str(e)}")
```

---

## SVM for Regression (Support Vector Regression - SVR)

Unlike classification, **Support Vector Regression (SVR)** is used when the target variable is continuous. Instead of trying to find a hyperplane that separates classes, SVR tries to fit the data within a margin of tolerance (called the **epsilon-tube**).

---

### Key Concepts of SVR

* **Epsilon (ε):** Defines the margin of tolerance where errors are ignored.
* **C (Regularization parameter):** Controls the trade-off between flatness of the regression line and tolerance of deviations larger than ε.
* **Kernels:** Allow SVR to model non-linear relationships (e.g., RBF, polynomial).

---

### SVR Objective Function

The optimization problem in SVR minimizes:

\$\frac{1}{2} ||w||^2 + C \sum\_{i=1}^n (\xi\_i + \xi\_i^\*)\$

subject to:

\$y\_i - (w^T x\_i + b) \leq \epsilon + \xi\_i\$

\$(w^T x\_i + b) - y\_i \leq \epsilon + \xi\_i^\*\$

\$\xi\_i, \xi\_i^\* \geq 0\$

where \$\xi\_i, \xi\_i^\*\$ are slack variables for errors outside the epsilon margin.

---

### When to Use SVR

* When you want a **robust regression model**.
* When the data has **non-linear relationships** (using kernel functions).
* When you want to control the **margin of error** (ε) and the **penalty for errors** (C).

---

## SVR Implementation in Python (Using sklearn)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

### Example dataset
```python
data = pd.read_csv('regression_data.csv')
X = data[['feature']].values
y = data['target'].values
```

### Scaling features
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Train-test split
```python
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```

### Initialize and train SVR
```python
svr_model = SVR(kernel='rbf', C=100, epsilon=0.1)
svr_model.fit(X_train, y_train)
```

### Predict
```python
predictions = svr_model.predict(X_test)
```

### Metrics
```python
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R² Score:", r2)
```

### Plot results
```python
plt.scatter(X_test, y_test, color='blue', label='True values')
plt.scatter(X_test, predictions, color='red', label='Predicted values')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()
```

---

## Summary

* **SVM Classification** → Finds the best hyperplane separating classes with maximum margin.
* **SVM Regression (SVR)** → Fits a function within a margin of tolerance (epsilon-tube), ignoring small errors and penalizing large deviations.

Both approaches can be extended to **non-linear problems** using **kernel tricks**.

---
 

