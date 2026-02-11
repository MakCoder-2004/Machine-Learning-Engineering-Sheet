# Logistic Regression

Logistic Regression is a supervised learning algorithm used for **classification problems**. Unlike linear regression, which predicts continuous values, logistic regression predicts the probability that a given input belongs to a particular class.

The hypothesis function is the **sigmoid function**:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

where

$$
z = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b
$$

The output of the sigmoid function is always between 0 and 1, which makes it suitable for classification.

---

### Why Use Logistic Regression?

* To solve **binary classification** problems (e.g., spam vs. not spam, disease vs. no disease).
* Interpretable model: coefficients represent the log-odds.
* Provides probability estimates, not just hard classifications.

---

### When to Use Logistic Regression

* When the target variable is **binary** (0 or 1).
* When the relationship between features and the log-odds of the target is approximately linear.
* When the dataset is not extremely large.

---

### Loss Function (Log Loss / Cross Entropy)

To measure how well the model fits the data, we use **Log Loss**:

$$
J(\theta) = - \frac{1}{n} \sum_{i=1}^n \Big[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \Big]
$$

where

* \$y\_i \in {0,1}\$ is the true label.
* \$\hat{y}\_i = \sigma(z\_i)\$ is the predicted probability.

---

### Gradient Descent

To minimize the log loss, we update weights using gradient descent:

$$
w_j = w_j - \alpha \frac{\partial J}{\partial w_j}
$$

The derivative:

$$
\frac{\partial J}{\partial w_j} = \frac{1}{n} \sum_{i=1}^n ( \hat{y}_i - y_i ) x_{ij}
$$

and

$$
\frac{\partial J}{\partial b} = \frac{1}{n} \sum_{i=1}^n ( \hat{y}_i - y_i )
$$

---

# Algorithm Implementation From Scratch

## Logistic Regression Class
```python
class LogisticRegression:
    def __init__(self, learning_rate=0.001, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # gradient descent
        for _ in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)
        
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))     
```

## Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
```

### Generate a Dataset
```python
bc = datasets.load_breast_cancer()
X,y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
```

## Visualize Data Distribution

```python
plt.scatter(df.x, df.y, c=df.y, cmap='bwr')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter Plot of x vs y')
plt.show()
```

### Apply Logistic Regression
```python
from LogisticRegression import LogisticRegression
reg = LogisticRegression()
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)
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

## How to Implement the Algorithm (Using sklearn)

### Libraries

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
```

### Load Dataset

```python
df = pd.read_csv('data.csv')
print(df)
print('\nNumber of rows and columns in the data set: ', df.shape)
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

### Apply Logistic Regression

```python
clf = LogisticRegression()
clf.fit(X_train, y_train)
```

### Testing / Prediction

```python
prediction_test = clf.predict(X_test)
print(y_test.values, prediction_test)
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

### Calculate Confusion Matrix
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