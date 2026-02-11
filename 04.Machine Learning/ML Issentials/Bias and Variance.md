# Bias and Variance

Understanding **bias** and **variance** is essential for diagnosing and improving machine learning models. They represent two components of model error that affect generalization. Balancing them correctly is the key to building accurate and robust models.

---

## Diagnosing bias and variance

### What is bias?

**Bias** measures the error introduced by approximating a real-world problem (which may be complex) by a much simpler model.
A model with **high bias** makes strong assumptions about the form of the mapping from inputs to outputs and therefore **underfits** the data.

* Characteristic: the model is too simple to capture the true relationship.
* Symptoms: both training and validation errors are high.
* Notation example:

  $$
  J_{\text{train}} \text{ is large and } J_{\text{cv}} \text{ is also large.}
  $$

**Intuition:** A linear model trying to fit strongly non-linear data will systematically miss the pattern → systematic error (bias).

---

### What is variance?

**Variance** measures how much the model's predictions would change if we estimated it using a different training dataset.
A model with **high variance** fits the training data very closely (including noise), and therefore **overfits**.

* Characteristic: the model is too flexible / complex.

* Symptoms: training error is low, validation error is high:

  $$
  J_{\text{cv}} > J_{\text{train}}
  $$

* Notation example:

  $$
  J_{\text{train}} \text{ is small and } J_{\text{cv}} \text{ is noticeably larger.}
  $$

**Intuition:** A very deep tree or a high-degree polynomial that passes through every training point will have low training error but poor performance on new data.

---

### Bias–Variance decomposition (mathematical)

For squared error loss, the expected prediction error at a point $$x$$ can be decomposed as:

$$
\mathbb{E}\big[ (y - \hat{f}(x))^2\big] = \big(\text{Bias}[\hat{f}(x)]\big)^2 + \text{Var}[\hat{f}(x)] + \sigma^2
$$

where:

$$
\text{Bias}[\hat{f}(x)] = \mathbb{E}[\hat{f}(x)] - f(x)
$$

$$
\text{Var}[\hat{f}(x)] = \mathbb{E}\big[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2\big]
$$

$$
\sigma^2 = \text{irreducible noise}
$$

Hence:

$$
\text{Total error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible error}
$$

---

### How bias and variance are affected

| Condition               | Bias | Variance | $$J_{\text{train}}$$ | $$J_{\text{cv}}$$ | Model behaviour              |
| ----------------------- | :--: | :------: | :------------------: | :---------------: | :--------------------------- |
| High bias               | High |    Low   |         High         |        High       | Underfitting                 |
| High variance           |  Low |   High   |          Low         |    Much higher    | Overfitting                  |
| High bias & variance    | High |   High   |         High         |     Very high     | Poor fit / bad data or model |
| Low bias & low variance |  Low |    Low   |          Low         |        Low        | Good generalization          |

Heuristics:

* **High Bias → Underfit:**
  $$
  J_{\text{train}} \text{ large.}
  $$

* **High Variance → Overfit:**
  $$
  J_{\text{train}} \text{ small and } J_{\text{cv}} \gg J_{\text{train}}.
  $$

* **Both High → Bad data / wrong assumptions.**

---

## Regularization and bias/variance

Regularization penalizes model complexity to control variance at the cost of adding some bias.

**L2 (Ridge) regularization:**

$$
J(\mathbf{w}) = \frac{1}{n}\sum_{i=1}^n L(y_i, \hat{y}*i) + \lambda \sum*{j} w_j^2
$$

**L1 (Lasso) regularization:**

$$
J(\mathbf{w}) = \frac{1}{n}\sum_{i=1}^n L(y_i, \hat{y}*i) + \lambda \sum*{j} |w_j|
$$

Effect of regularization parameter (lambda): 

* Small lambda → weak regularization → lower bias, higher variance.
* Large lambda → strong regularization → higher bias, lower variance.

---

## Establishing a baseline level of performance

A baseline is a simple, easy-to-compute reference that any useful model should outperform.

* Regression baseline: predict mean
  $$
  \hat{y} = \bar{y}
  $$

* Classification baseline: predict the most frequent class.

Use `DummyClassifier` / `DummyRegressor` from scikit-learn.

---

## Learning curves

**Learning curves** plot training and validation error (or accuracy) as a function of training set size.

* Both training and validation errors high and close → **high bias**.
* Training error low, validation high → **high variance**.
* Curves converge to low error → good generalization.

---

## Debugging a learning algorithm (practical checklist)

1. Plot learning curves.
2. Compare $$J_{\text{train}}$$ and $$J_{\text{cv}}$$
3. Tune lambda (regularization strength).
4. Improve features or data quality.
5. Use ensembles (bagging, boosting) to reduce variance.

---

## Quick scikit-learn examples

**Baseline classifier:**

```python
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
print("Baseline accuracy:", accuracy_score(y_test, baseline.predict(X_test)))
```

**Regularization effect (Ridge):**

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

for alpha in [0.01, 0.1, 1, 10, 100]:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    train_mse = mean_squared_error(y_train, model.predict(X_train))
    test_mse = mean_squared_error(y_test, model.predict(X_test))
    print(f"alpha={alpha:>6} | train_mse={train_mse:.3f} | test_mse={test_mse:.3f}")
```

---

## Summary / Key equations

Main decomposition:

$$
\mathbb{E}\big[ (y - \hat{f}(x))^2\big] = (\text{Bias}[\hat{f}(x)])^2 + \text{Var}[\hat{f}(x)] + \sigma^2
$$

Regularized cost function:

$$
J(\mathbf{w}) = \frac{1}{n}\sum_{i=1}^n L(y_i, \hat{y}_i) + \lambda |\mathbf{w}|_2^2
$$

Diagnostics summary:

* High bias → <br>large $$J_{\text{train}}$$ 
* High variance → <br>small $$J_{\text{train}}$$ large $$J_{\text{cv}} \gg J_{\text{train}}$$

---
