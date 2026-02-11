# Hyperparameter Tuning

**Hyperparameter tuning** is the process of optimizing a model’s configuration parameters that control its learning behavior.
Unlike model parameters (learned during training), **hyperparameters** are set **before** training and can significantly affect model performance.

---

## What Is Hyperparameter Tuning?

Every machine learning algorithm has **hyperparameters** — settings that define how the model learns.
Examples include the learning rate in neural networks, the number of trees in Random Forests, or the value of *k* in K-Nearest Neighbors.

Hyperparameter tuning involves **systematically searching** for the combination that yields the **best model performance** (accuracy, F1-score, etc.) on validation data.

---

## Why Is Hyperparameter Tuning Important?

Good hyperparameter choices can:

* Improve **model accuracy and generalization**
* Reduce **overfitting** or **underfitting**
* Enhance **training efficiency**
* Unlock the **full potential** of a model architecture

Without proper tuning, even strong models can perform poorly.

---

## Theoretical Perspective

Let the model be represented by:

$$
f(x; \theta, \lambda)
$$

where:

* **(\theta)** are learned parameters (weights)
* **(\lambda)** are hyperparameters (learning rate, number of layers, etc.)

Hyperparameter tuning aims to find:

$$
\lambda^* = \arg\min_\lambda \mathbb{E}*{(x, y) \sim \mathcal{D}*{val}} ; L\big(y, f(x; \theta^*(\lambda))\big)
$$

where (\theta^*(\lambda)) are the trained parameters given hyperparameters (\lambda).
This process is often **nested** — training models under different (\lambda) and evaluating them on validation data.

---

## Common Hyperparameters

| Model Type          | Common Hyperparameters                                               |
| :------------------ | :------------------------------------------------------------------- |
| **Linear Models**   | Regularization strength ((\alpha), (\lambda)), penalty type (L1, L2) |
| **Decision Trees**  | Max depth, min samples split, min samples leaf                       |
| **Random Forests**  | Number of trees, max features, max depth                             |
| **SVM**             | Kernel type, (C) (regularization), (\gamma)                          |
| **Neural Networks** | Learning rate, batch size, number of layers/neurons, dropout rate    |
| **KNN**             | Number of neighbors ((k)), distance metric                           |

---

## Methods of Hyperparameter Tuning

### 1. **Manual Search**

Trying different combinations based on intuition and experience.

**Pros:** Simple to apply
**Cons:** Time-consuming and subjective

---

### 2. **Grid Search**

Exhaustively tests all combinations from a predefined grid of hyperparameter values.

**Pros:** Systematic and thorough
**Cons:** Computationally expensive for large grids

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
                                                    test_size=0.2, random_state=42)

# Define model and parameter grid
model = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 8, None],
    'min_samples_split': [2, 4, 6]
}

# Perform grid search
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)
```

---

### 3. **Random Search**

Randomly samples hyperparameter combinations within specified ranges.

**Pros:** Faster and more efficient for high-dimensional spaces
**Cons:** Might miss optimal combinations

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': [3, 5, 8, None],
    'min_samples_split': randint(2, 10)
}

random_search = RandomizedSearchCV(model, param_distributions=param_dist,
                                   n_iter=10, cv=5, scoring='accuracy', n_jobs=-1)
random_search.fit(X_train, y_train)

print("Best Parameters:", random_search.best_params_)
```

---

### 4. **Bayesian Optimization**

Uses probabilistic models (e.g., Gaussian Processes) to model the objective function and choose the most promising hyperparameters based on past evaluations.

**Pros:** More sample-efficient
**Cons:** Complex to implement; requires specialized libraries (Optuna, Hyperopt, or Scikit-Optimize)

---

### 5. **Automated Machine Learning (AutoML)**

Tools like **Auto-sklearn**, **TPOT**, and **Google AutoML** automatically explore hyperparameter spaces and model types.

**Pros:** Hands-free and optimized search
**Cons:** Less control, may require high computation

---

## Cross-Validation in Tuning

To ensure robust evaluation during tuning, use **k-fold cross-validation**.
This splits the training data into *k* folds, training on *k–1* folds and validating on the remaining one.

This helps prevent **overfitting** to a specific validation set and yields a more reliable performance estimate.

---

## Visualization – Hyperparameter Tuning Workflow

```
+------------------------------+
|   Define Hyperparameter Grid |
+--------------+---------------+
               ↓
      Train model on folds
               ↓
  Evaluate and compare scores
               ↓
 Select best hyperparameters
               ↓
   Retrain on full training set
               ↓
       Test final model
```

---

## Practical Tips

* Start with **RandomizedSearchCV** for large spaces, then refine using **GridSearchCV**
* Use **smaller subsets** of data for initial exploration
* Monitor **validation curves** to detect overfitting
* Use **early stopping** in deep learning to prevent wasted computation

---

## Common Pitfalls

| Issue                     | Cause                             | Solution                            |
| :------------------------ | :-------------------------------- | :---------------------------------- |
| Overfitting on validation | Reusing validation data too often | Keep a separate final test set      |
| High computation cost     | Large search space                | Use randomized or Bayesian search   |
| Poor results              | Inappropriate parameter ranges    | Start with broad, reasonable ranges |
| Unstable performance      | Too few CV folds or small data    | Increase folds or dataset size      |

---

## Summary

Hyperparameter tuning is the **core step** to optimize a model’s performance and generalization ability.
It transforms a decent model into a high-performing one by finding the best configuration through structured experimentation.

**Key takeaway:**

> Don’t just train — **tune smartly**. The right hyperparameters can make the difference between a good model and a great one.

---
