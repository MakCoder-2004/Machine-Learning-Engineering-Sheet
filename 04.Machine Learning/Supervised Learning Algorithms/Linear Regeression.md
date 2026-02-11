# Linear Regression

Linear regression is a supervised learning algorithm used to model the relationship between a dependent variable $y$ and one or more independent variables $x_1, x_2, \dots, x_n$. The goal is to find the best-fitting linear equation:

$$
y = m x + b
$$

for a single variable (simple linear regression) or

$$
y = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b
$$

for multiple variables (multiple linear regression).

## Why Use Linear Regression?

* To understand the strength of the relationship between variables.
* To predict future values based on existing data.
* It is one of the simplest and most interpretable models.

## When to Use Linear Regression

* When the relationship between variables is approximately linear.
* When the residuals (errors) are normally distributed and independent.
* When the dataset does not contain extreme multicollinearity.

### Loss Function

To measure how well a line fits the data, we use the **Mean Squared Error (MSE)**:

$$
MSE = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
$$

### Cost Function

The **Cost Function** $J(\theta)$ used in linear regression is defined as:

$$
J(\theta) = \frac{1}{2n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
$$

Where:

* $n$ = number of data points
* $y_i$ = actual values
* $\hat{y}_i$ = predicted values (from the line $\hat{y} = (\theta_0 + \theta_1 x_i$)
* $\theta$ = parameters (weights) of the model

* The factor $\frac{1}{2}$ simplifies the computation of derivatives when applying gradient descent.
* The  Goal is **minimize** the cost function **$J(\theta)$**.

## Gradient Descent

Gradient descent is an optimization algorithm used to minimize the MSE by iteratively updating the parameters:

$$
m_{new} = m_{old} - \alpha \frac{\partial MSE}{\partial m}
$$

$$
b_{new} = b_{old} - \alpha \frac{\partial MSE}{\partial b}
$$

## Learning Rate

The learning rate $\alpha$ controls the step size during gradient descent. A small learning rate ensures stable convergence but may require more iterations, while a large learning rate can cause the algorithm to converge quickly but may lead to unstable or non-convergent behavior.

- If $\alpha$ is too small, the Gradient Descent may take a long time to converge.
- If $\alpha$ is too large, the Gradient Descent may never reach minimum and fail to converge/diverge.
---

# Algorithm Implementation From Scratch

## Linear Regression Class
```python
class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
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
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted
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

### Generate a Regression Dataset
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

### Apply Linear Regression
```python
from LinearRegression import LinearRegression
reg = LinearRegression()
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

## How to Implement the Algorithm Using Scikit-Learn Library

### Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
```

### Load Dataset

```python
df = pd.read_csv('FileName.csv')
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
input_df = df.drop(columns='x')  # drop the unused columns for the model
target_df = df['x']
```

### Encoding Categorical Columns

```python
columns_to_encode = ["X1", "X2", "X3"]  # Replace with your own column names
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
X_train, X_test, y_train, y_test = train_test_split(input_df, target_df, test_size=0.2, random_state=42)
```

### Apply Linear Regression

```python
reg = LinearRegression()
reg.fit(X_train, y_train)
```

### Testing / Prediction

```python
prediction_test = reg.predict(X_test)
print(y_test.values, prediction_test)
```

### Calculate Errors

```python
MAEValue = mean_absolute_error(y_test, prediction_test)
MSEValue = mean_squared_error(y_test, prediction_test)
MdSEValue = median_absolute_error(y_test, prediction_test)

print('Mean Absolute Error Value is : ', MAEValue)
print('Mean Squared Error Value is : ', MSEValue)
print('Median Absolute Error Value is : ', MdSEValue)
```

### Visualization
```python
plt.figure(figsize=(10,5))
sns.scatterplot(x=y_test, y=prediction_test, color="blue", alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel("Actual Performance Index")
plt.ylabel("Predicted Performance Index")
plt.title("Actual vs Predicted Values")
plt.show()
```

---
