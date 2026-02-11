# Multicollinearity

**Multicollinearity** occurs when two or more independent variables in a regression model are highly correlated.
It means that one variable can be linearly predicted from the others with a high degree of accuracy.
This creates problems in estimating the relationship between predictors and the dependent variable.

---

## Effects of Multicollinearity

* **Inflated Standard Errors:** Coefficients become less statistically significant.
* **Unstable Coefficients:** Small changes in data can lead to large variations in coefficient estimates.
* **Reduced Model Interpretability:** Difficulty in determining the impact of individual variables.
* **Decreased Predictive Power:** The model may perform worse on new data due to instability.

---

## Detecting Multicollinearity

### 1. **Correlation Matrix**

A correlation matrix shows the pairwise correlations between features.

* **+1:** Perfect positive correlation (both variables increase together).
* **-1:** Perfect negative correlation (one variable increases as the other decreases).
* **0:** No linear relationship.

| Feature       | Feature 1 | Feature 2 | Feature 3 |
| ------------- | --------- | --------- | --------- |
| **Feature 1** | 1.0       | 0.92      | 0.5       |
| **Feature 2** | 0.92      | 1.0       | 0.3       |
| **Feature 3** | 0.5       | 0.3       | 1.0       |

If two features have a correlation coefficient close to **±1**, it indicates possible multicollinearity.

---

### 2. **Variance Inflation Factor (VIF)**

The **VIF** quantifies how much the variance of a regression coefficient is inflated due to multicollinearity.

$$
VIF_i = \frac{1}{1 - R_i^2}
$$

Where:

* ( R_i^2 ) is the coefficient of determination obtained when variable *i* is regressed on all other independent variables.

**Interpretation:**

* VIF = 1 → No correlation between features.
* 1 < VIF < 5 → Moderate correlation (usually acceptable).
* VIF > 10 → High multicollinearity (problematic).

---

### 3. **Condition Index**

The **Condition Index (CI)** is derived from the eigenvalues of the scaled independent variable matrix.

$$
CI = \sqrt{\frac{\lambda_{max}}{\lambda_j}}
$$

Where:

* ( \lambda_{max} ) = largest eigenvalue
* ( \lambda_j ) = each smaller eigenvalue

**Interpretation:**

* CI < 10 → No multicollinearity.
* 10 ≤ CI ≤ 30 → Moderate multicollinearity.
* CI > 30 → Severe multicollinearity.

---

## How to Fix Multicollinearity

| **Method**               | **Description**                                                                      |
| :----------------------- | :----------------------------------------------------------------------------------- |
| **Remove Variables**     | Drop one of the correlated predictors.                                               |
| **Combine Variables**    | Use techniques like **Principal Component Analysis (PCA)** to reduce dimensionality. |
| **Regularization**       | Apply **Ridge** or **Lasso Regression** to penalize large coefficients.              |
| **Centering Data**       | Subtract the mean from predictors to reduce non-essential multicollinearity.         |
| **Increase Sample Size** | More data can stabilize coefficient estimates.                                       |

---

## Summary

Multicollinearity can distort regression estimates, making them unreliable. Detecting it early using **VIF**, **correlation matrices**, or **condition indices** helps maintain model accuracy and interpretability.

**Key takeaway:**

> Check correlations, monitor VIF, and use regularization or dimensionality reduction when necessary.
