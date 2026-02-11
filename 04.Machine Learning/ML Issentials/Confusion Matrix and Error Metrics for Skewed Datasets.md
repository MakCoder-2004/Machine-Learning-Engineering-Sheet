# Confusion Matrix and Error Metrics for Skewed Datasets

In machine learning, **skewed (imbalanced) datasets** occur when one class significantly outnumbers the other (e.g., fraud detection, rare diseases). In such cases, accuracy alone can be misleading, so we rely on **confusion matrices** and derived metrics like **precision**, **recall**, and **F1-score** for better evaluation.

---

## Confusion Matrix

The **Confusion Matrix** is a performance summary table comparing predicted labels with actual labels.

| **Actual / Predicted** |     **Positive**    |     **Negative**    |
| :--------------------: | :-----------------: | :-----------------: |
|      **Positive**      |  True Positive (TP) | False Negative (FN) |
|      **Negative**      | False Positive (FP) |  True Negative (TN) |

### Interpretation

* **True Positive (TP):** Correctly predicted positives.
* **True Negative (TN):** Correctly predicted negatives.
* **False Positive (FP):** Incorrectly predicted positives (Type I Error).
* **False Negative (FN):** Incorrectly predicted negatives (Type II Error).

### Example

Suppose a medical model predicts whether a patient has a disease:

|                     | **Predicted Positive** | **Predicted Negative** |
| :------------------ | :--------------------: | :--------------------: |
| **Actual Positive** |           15           |           10           |
| **Actual Negative** |            5           |           70           |

From this table:

* TP = 15
* FN = 10
* FP = 5
* TN = 70

---

## Performance Metrics

### 1. **Accuracy**

**Definition:** Fraction of correct predictions.

$$
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
$$

**Example:**

$$
Accuracy = \frac{15 + 70}{15 + 70 + 5 + 10} = 0.85
$$

**Limitation:** Misleading for imbalanced datasets (e.g., 99% accuracy if all predictions are negative).

---

### 2. **Precision**

**Definition:** Of all predicted positives, how many are actually positive?

$$
Precision = \frac{TP}{TP + FP}
$$

**Interpretation:** High precision means fewer **false positives**.

**Example:**

$$
Precision = \frac{15}{15 + 5} = 0.75
$$

---

### 3. **Recall (Sensitivity or True Positive Rate)**

**Definition:** Of all actual positives, how many did the model correctly identify?

$$
Recall = \frac{TP}{TP + FN}
$$

**Interpretation:** High recall means fewer **false negatives**.

**Example:**

$$
Recall = \frac{15}{15 + 10} = 0.6
$$

---

### 4. **F1 Score**

**Definition:** The harmonic mean of precision and recall — balances both.

$$
F_1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
$$

**Why Harmonic Mean?** It penalizes extreme values, rewarding models that perform well on both precision and recall.

**Example:**

| Metric    | Value |
| :-------- | :---- |
| Precision | 0.75  |
| Recall    | 0.60  |

$$
F_1 = 2 \times \frac{0.75 \times 0.60}{0.75 + 0.60} = 0.6667
$$

---

## Trade-off Between Precision and Recall

Precision and recall have an **inverse relationship** — increasing one may reduce the other.

* **High Precision, Low Recall:** Predict positive only when very confident (reduces false alarms).
* **Low Precision, High Recall:** Predict positive more often (captures more actual positives).

### Adjusting the Threshold

Changing the classification threshold can control this trade-off:

* Lowering the threshold increases recall but may reduce precision.
* Raising the threshold increases precision but may lower recall.

---

## Visualization Example (Python)

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap='Blues')

# Classification Report
print(classification_report(y_test, y_pred))
```

---

## Summary Table

| Metric        | Formula                             | Focus                 | Best When                       |
| :------------ | :---------------------------------- | :-------------------- | :------------------------------ |
| **Precision** | $\frac{TP}{TP + FP}$                | Fewer false positives | Cost of false positives is high |
| **Recall**    | $\frac{TP}{TP + FN}$                | Fewer false negatives | Missing positives is costly     |
| **F1 Score**  | $2 \cdot \frac{P \cdot R}{P + R}$   | Balance of both       | Need balance between P & R      |
| **Accuracy**  | $\frac{TP + TN}{TP + TN + FP + FN}$ | Overall correctness   | Dataset is balanced             |

---

## Key Takeaways

* Accuracy can be **deceptive** in skewed datasets.
* Use **confusion matrices**, **precision**, **recall**, and **F1-score** for meaningful evaluation.
* Adjust the **classification threshold** to balance between false positives and false negatives.
* Use **Precision-Recall curves** or **ROC curves** for visual comparison.

---

> **Tip:**
> When dealing with rare event prediction (e.g., fraud detection, disease diagnosis), prioritize **recall** or **F1-score** over raw accuracy.
