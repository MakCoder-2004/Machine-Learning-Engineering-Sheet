# Ensemble Learning

**Ensemble Learning Algorithms:** Bagging, Boosting & Random Forest

---

## What is Ensemble Learning

**Definition:**
Ensemble learning combines multiple models (often called **"weak learners"**) to produce a better performing **"strong learner"**.

**Why use Ensemble Learning?**

* Reduces errors
* Improves accuracy
* Increases robustness

## Types of Ensemble Methods

* **Bagging (Bootstrap Aggregating)**
* **Boosting**
* **Random Forest** (uses Bagging + Decision Trees)

---

## Bagging

**What is Bagging?**

* Bagging = Bootstrap Aggregating
* Create many random samples from the training data (with replacement)
* Train a separate model on each sample
* Combine their outputs (e.g., by voting for classification or averaging for regression)
* **Goal:** Reduce variance (make model more stable)

### Bagging — Real-World Applications

* **Retail:** Forecasting customer demand (predicting sales using regression)
* **Healthcare:** Predicting patient readmission risks
* **Banking:** Detecting anomalies in transaction patterns

**Often used with:**

* Decision Trees
* KNN
* SVM (less common)

---

## Random Forest: Built on Bagging

* Random Forest = Bagging + Decision Trees
* Creates many decision trees
* Each tree gets a random sample of data (bagging)
* Each tree also uses a random subset of features
* **Final prediction:** majority vote (classification) or average (regression)

**Why it's good:**

* Fast to train
* Hard to overfit
* Great baseline model

### Random Forest Advantages

* Handles missing values
* Works well with categorical & numerical data
* Fast and easy to tune

### Random Forest — Real-World Applications

* **Social Media:** Spam and fake account detection
* **Healthcare:** Predicting disease diagnosis based on symptoms
* **Finance:** Credit scoring, loan approval systems
* **Environmental Science:** Predicting air quality, rainfall, or forest cover loss

---

## Boosting

**What is Boosting?**

* Models are trained sequentially
* Each new model focuses on the errors made by the previous ones
* Combines them into a strong model
* **Goal:** Reduce bias and improve prediction

### Boosting — Real-World Applications

* **Self-Driving Cars:** Object detection and decision-making
* **Email Providers:** Spam detection with high accuracy
* **Finance & Insurance:** Fraud detection (e.g., XGBoost for real-time alerts)
* **Gaming:** Predicting user churn or behavior

**Used in:**

* XGBoost
* LightGBM
* AdaBoost
* CatBoost

---

## Popular Boosting Algorithms

Boosting is an **ensemble learning technique**

| Algorithm    | Speed     | Overfitting Risk | Handles Categorical | Best For                |
| ------------ | --------- | ---------------- | ------------------- | ----------------------- |
| **AdaBoost** | Medium    | High (outliers)  | ❌ No                | Simple, small data      |
| **GBM**      | Slow      | High             | ❌ No                | Flexible loss functions |
| **XGBoost**  | Fast      | Medium           | ❌ No                | Accuracy, competitions  |
| **LightGBM** | Very Fast | Medium-High      | ❌ No                | Large datasets          |
| **CatBoost** | Medium    | Low              | ✅ Yes               | Mixed/categorical data  |

---

## Implementation of Boosting Algorithms (Using Scikit-learn)

### 1. Dataset Preparation

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

### 2. AdaBoost

```python
from sklearn.ensemble import AdaBoostClassifier

ada_model = AdaBoostClassifier(n_estimators=100, learning_rate=0.8, random_state=42)
ada_model.fit(X_train, y_train)

y_pred_ada = ada_model.predict(X_test)
print("AdaBoost Accuracy:", accuracy_score(y_test, y_pred_ada))
```

---

### 3. Gradient Boosting (GBM)

```python
from sklearn.ensemble import GradientBoostingClassifier

gbm_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbm_model.fit(X_train, y_train)

y_pred_gbm = gbm_model.predict(X_test)
print("Gradient Boosting Accuracy:", accuracy_score(y_test, y_pred_gbm))
```

---

### 4. XGBoost

```python
from xgboost import XGBClassifier

xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
```

---

### 5. LightGBM

```python
from lightgbm import LGBMClassifier

lgb_model = LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=-1, random_state=42)
lgb_model.fit(X_train, y_train)

y_pred_lgb = lgb_model.predict(X_test)
print("LightGBM Accuracy:", accuracy_score(y_test, y_pred_lgb))
```

---

### 6. CatBoost

```python
from catboost import CatBoostClassifier

cat_model = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, random_state=42, verbose=0)
cat_model.fit(X_train, y_train)

y_pred_cat = cat_model.predict(X_test)
print("CatBoost Accuracy:", accuracy_score(y_test, y_pred_cat))
```

---

### 7. Visual Comparison of Model Accuracies

```python
import matplotlib.pyplot as plt

accuracies = {
    'AdaBoost': accuracy_score(y_test, y_pred_ada),
    'GBM': accuracy_score(y_test, y_pred_gbm),
    'XGBoost': accuracy_score(y_test, y_pred_xgb),
    'LightGBM': accuracy_score(y_test, y_pred_lgb),
    'CatBoost': accuracy_score(y_test, y_pred_cat)
}

plt.figure(figsize=(8, 5))
plt.bar(accuracies.keys(), accuracies.values(), color='teal')
plt.title('Boosting Algorithm Accuracies')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()
```

---

### ✅ Summary Output Example

| Algorithm     | Accuracy |
| ------------- | -------- |
| AdaBoost      | 0.93     |
| GradientBoost | 0.95     |
| XGBoost       | 0.97     |
| LightGBM      | 0.96     |
| CatBoost      | 0.97     |

*(Results may vary slightly each run.)*
