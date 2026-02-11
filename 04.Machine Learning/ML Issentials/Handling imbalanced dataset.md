# Handling Imbalanced Datasets

* **Definition:** A dataset is imbalanced if one class has significantly more or fewer instances than the other.
* **Common Causes:**

  * Natural class imbalance (e.g., fraud detection, disease diagnosis)
  * Data collection biases
  * Uneven sampling
* **Impact:**

  * Accuracy can be misleading
  * Models may favor the majority class
  * False positives and false negatives are misinterpreted

---

## Why Handling Imbalanced Data Matters

When the dataset is imbalanced, models tend to become biased toward the majority class. For example, if 95% of samples are from one class, the model could achieve 95% accuracy simply by predicting that class every time — while completely missing the minority class.

To address this, we use **data-level** and **algorithm-level** approaches.

---

## 1. Under-sampling the Majority Class

This technique reduces the number of samples in the majority class to balance the dataset.

**Advantages:**

* Reduces training time
* Simple and effective for large datasets

**Disadvantages:**

* May remove useful information

**Example:**

```python
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

X = [[i] for i in range(10)]
y = [0]*8 + [1]*2  # Imbalanced dataset
print('Original dataset shape:', Counter(y))

rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X, y)
print('Resampled dataset shape:', Counter(y_res))
```

---

## 2. Over-sampling the Minority Class

This method duplicates or synthetically generates more instances of the minority class.

**Advantages:**

* Retains all majority data
* Helps the model learn minority class features

**Disadvantages:**

* Can cause overfitting (due to duplication)

**Example:**

```python
from imblearn.over_sampling import RandomOverSampler

X = [[i] for i in range(10)]
y = [0]*8 + [1]*2
print('Original dataset shape:', Counter(y))

ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X, y)
print('Resampled dataset shape:', Counter(y_res))
```

---

## 3. SMOTE (Synthetic Minority Over-sampling Technique)

SMOTE generates new synthetic samples from the minority class using **interpolation between existing minority samples**.

**How It Works:**

1. Select a minority class instance.
2. Choose one of its nearest neighbors.
3. Create a synthetic point along the line between them.

**Advantages:**

* Creates realistic synthetic data
* Better than random oversampling

**Disadvantages:**

* May create noisy samples
* Not effective if classes are highly overlapping

**Example:**

```python
from imblearn.over_sampling import SMOTE

X = [[i] for i in range(10)]
y = [0]*8 + [1]*2
print('Original dataset shape:', Counter(y))

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
print('Resampled dataset shape:', Counter(y_res))
```

---

## 4. Ensemble Methods

These techniques combine multiple models to handle imbalance, often using **resampling + ensemble learning**.

### Example: Balanced Random Forest

Balanced Random Forest resamples data at each bootstrap step to balance classes.

```python
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Create an imbalanced dataset
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2,
                           weights=[0.9, 0.1], random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

brf = BalancedRandomForestClassifier(n_estimators=100, random_state=42)
brf.fit(X_train, y_train)
y_pred = brf.predict(X_test)

print(classification_report(y_test, y_pred))
```

### Other Ensemble Options:

* **EasyEnsemble:** Combines several under-sampled subsets
* **RUSBoost:** Boosting with random under-sampling

---

## 5. Focal Loss

Focal Loss is an **algorithm-level approach** used mainly in deep learning. It modifies cross-entropy loss to focus more on hard-to-classify examples.

**Formula:**
$$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

Where:

* ( p_t ): predicted probability of the true class
* ( \alpha_t ): balancing factor
* ( \gamma ): focusing parameter (typically 2)

**Advantages:**

* Helps neural networks learn from hard samples
* Effective for severe imbalance (e.g., object detection)

**Example (TensorFlow):**

```python
import tensorflow as tf

def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        bce_exp = tf.exp(-bce)
        loss = alpha * (1 - bce_exp) ** gamma * bce
        return tf.reduce_mean(loss)
    return focal_loss_fixed

# Usage in a model
model.compile(optimizer='adam', loss=focal_loss(gamma=2., alpha=0.25), metrics=['accuracy'])
```

---

## Choosing the Right Technique

| Method         | Type            | Best For           | Pros                    | Cons             |
| -------------- | --------------- | ------------------ | ----------------------- | ---------------- |
| Under-sampling | Data-level      | Large datasets     | Fast                    | May lose info    |
| Over-sampling  | Data-level      | Small datasets     | Keeps all data          | Overfitting risk |
| SMOTE          | Data-level      | Moderate imbalance | Synthetic data          | May add noise    |
| Ensemble       | Algorithm-level | Any size           | Robust, accurate        | Complex          |
| Focal Loss     | Algorithm-level | Deep learning      | Focuses on hard samples | Harder to tune   |

---

## Summary

* Imbalanced datasets can cause biased models.
* Balancing techniques improve generalization and fairness.
* Combining multiple methods (e.g., SMOTE + Ensemble) often gives the best results.

**Key Takeaway:** Always evaluate with metrics like **Precision, Recall, F1-score, ROC-AUC**, not just accuracy.
