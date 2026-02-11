# Gradient Descent Variants

Gradient Descent is an optimization algorithm used to minimize the **loss function** by updating the model’s parameters in the opposite direction of the gradient.

There are several types of Gradient Descent depending on **how much data** is used to compute the gradient for each update.

---

## 1. Batch Gradient Descent

### Definition
- Uses **all training samples** to compute the gradient before updating the weights.
- One full pass over the entire dataset = **1 epoch**.

### Characteristics
- Smooth convergence of the cost function  
- Computationally expensive for large datasets  
- Stable updates but slower training  

### When to Use
- When the dataset is **small enough** to fit into memory  
- When you prefer **stable and smooth convergence**

### Pros
- Converges steadily toward the minimum  
- Produces a smooth cost curve (less noise)

### Cons
- Very slow for large datasets  
- Requires large memory  

### Cost Curve Example
The cost decreases smoothly over epochs:

---

## 2. Stochastic Gradient Descent (SGD)

### Definition
- Updates weights **for each single training example** (randomly picked).
- Performs updates much more frequently.

### Characteristics
- Noisy updates (cost fluctuates), but faster convergence in practice  
- Introduces randomness — helps escape local minima  

### When to Use
- When the training set is **very large**  
- When you want **faster, online updates** without full dataset passes

### Pros
- Works well for large datasets  
- Can reach good minima faster  
- Allows **online learning** (update per sample)

### Cons
- Cost function fluctuates a lot (noisy)
- Harder to find the exact global minimum

### Cost Curve Example
The cost fluctuates but gradually decreases over time:


---

## Comparison Summary

| Feature | Batch Gradient Descent | Stochastic Gradient Descent (SGD) |
|----------|------------------------|------------------------------------|
| **Data Used per Update** | All training samples | One sample |
| **Computation per Update** | High | Very low |
| **Convergence** | Smooth and stable | Noisy but faster |
| **Memory Requirement** | Large | Minimal |
| **Best for** | Small datasets | Large datasets |
| **Speed** | Slow | Fast (per update) |

---

## Key Takeaways
- **Batch GD** → precise but slow; best for small datasets.  
- **SGD** → fast but noisy; best for large datasets.  
- In practice, a middle ground is used → **Mini-Batch Gradient Descent**, which balances stability and speed.

---
