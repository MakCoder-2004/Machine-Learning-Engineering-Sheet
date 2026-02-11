# Collaborative Filtering

**Collaborative Filtering (CF)** is one of the most widely used techniques in **Recommendation Systems**.
It predicts a user's interest in items by learning from patterns of other users.

---

## Intuition

Collaborative Filtering assumes that:

> If two users have similar preferences in the past, they will likely prefer similar items in the future.

For example:
If **User A** and **User B** both liked several of the same movies, and **User A** liked *Inception*,
then CF predicts that **User B** might like *Inception* too.

---

## Types of Collaborative Filtering

### 🧍‍♂️ User-Based Collaborative Filtering

* Finds users similar to the target user (based on rating behavior).
* Recommends items that similar users liked.

### 🎬 Item-Based Collaborative Filtering

* Finds items that are similar to each other.
* Recommends items similar to what the user already liked.

---

## Mathematical Model

Instead of relying only on similarities, we can learn parameters that represent both:

* **User preferences**
* **Item features**

This is the **latent factor model** version of collaborative filtering.

---

### Cost Function for a Single User

To learn parameters $w^{(j)}$ and $b^{(j)}$ for user $j$:

$$
J(w^{(j)}, b^{(j)}) = \frac{1}{2} \sum_{i:r(i,j)=1} (w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i,j)})^2 + \frac{\lambda}{2} \sum_{k=1}^{n} (w_k^{(j)})^2
$$

Where:

* $y^{(i,j)}$: Rating given by user $j$ to item $i$
* $x^{(i)}$: Feature vector of item $i$
* $w^{(j)}$: Preference vector of user $j$
* $b^{(j)}$: Bias term (captures average rating tendency)
* $\lambda$: Regularization parameter (to prevent overfitting)
* $r(i,j) = 1$: Indicates user $j$ rated item $i$

---

### Cost Function for All Users

To learn parameters for all users simultaneously:

$$
J(w^{(1)}, ..., w^{(n_u)}, b^{(1)}, ..., b^{(n_u)}) = \frac{1}{2} \sum_{j=1}^{n_u} \sum_{i:r(i,j)=1} (w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i,j)})^2 + \frac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^{n} (w_k^{(j)})^2
$$

---

## Learning Item Features

To learn the item feature vectors $x^{(i)}$:

$$
J(x^{(1)}, ..., x^{(n_m)}) = \frac{1}{2} \sum_{i=1}^{n_m} \sum_{j:r(i,j)=1} (w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i,j)})^2 + \frac{\lambda}{2} \sum_{i=1}^{n_m} \sum_{k=1}^{n} (x_k^{(i)})^2
$$

---

## Combined Cost Function

By combining both user and item cost functions, we get the **full collaborative filtering cost function**:

$$
J = \frac{1}{2} \sum_{(i,j):r(i,j)=1} (w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i,j)})^2 + \frac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^{n} (w_k^{(j)})^2 + \frac{\lambda}{2} \sum_{i=1}^{n_m} \sum_{k=1}^{n} (x_k^{(i)})^2
$$

---

## Optimization

We aim to **minimize the cost function** $J$ with respect to both $w^{(j)}, b^{(j)}$, and $x^{(i)}$.

### Gradient Descent Updates

For each rating $y^{(i,j)}$:

$$
\text{error}_{i,j} = (w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i,j)})
$$

Update rules:

$$
x^{(i)} := x^{(i)} - \alpha (\text{error}_{i,j} \cdot w^{(j)} + \lambda x^{(i)})
$$

$$
w^{(j)} := w^{(j)} - \alpha (\text{error}_{i,j} \cdot x^{(i)} + \lambda w^{(j)})
$$

$$
b^{(j)} := b^{(j)} - \alpha (\text{error}_{i,j})
$$

Where $\alpha$ is the **learning rate**.

---

## Prediction

Once trained, the predicted rating of user $j$ for item $i$ is:

$$
\hat{y}^{(i,j)} = w^{(j)} \cdot x^{(i)} + b^{(j)}
$$

---

## Intuitive Interpretation

* $x^{(i)}$: Describes **item features** (e.g., genre, popularity, complexity)
* $w^{(j)}$: Describes **user preferences** (how much each feature matters)
* $b^{(j)}$: Captures how generous or harsh the user is when rating

Collaborative Filtering learns **hidden (latent) factors** that explain interactions between users and items.

---

## Regularization

Regularization helps control **overfitting**:
Without it, the model may memorize training ratings rather than learn underlying patterns.

---

## Practical Steps

1. Initialize $x^{(i)}, w^{(j)}, b^{(j)}$ randomly.
2. Compute predictions $\hat{y}^{(i,j)}$.
3. Compute the cost $J$.
4. Perform gradient descent updates.
5. Repeat until convergence.

---

## Matrix Factorization View

In matrix form:

$$
Y \approx W X^T + B
$$

Where:

* $Y$: User–Item rating matrix
* $W$: User preference matrix
* $X$: Item feature matrix
* $B$: Bias vector for users

This approach is known as **Matrix Factorization**, a key formulation of collaborative filtering.

---

## TensorFlow Implementation

Below is a practical TensorFlow 2.x implementation of matrix factorization with per-user and per-item biases, mean normalization, and a training loop. This follows the mathematical model above and is ready to run with your user/item arrays.

```python
# requirements: tensorflow>=2.8, numpy, pandas
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# --- Replace these with your real arrays (mapped to contiguous ints) ---
# user_ids: array of user indices (0..n_users-1)
# item_ids: array of item indices (0..n_items-1)
# ratings: array of corresponding ratings (floats)
user_ids = np.array([0,0,1,1,2,2], dtype=np.int32)
item_ids = np.array([0,1,0,2,1,2], dtype=np.int32)
ratings = np.array([5.,4.,4.,5.,1.,2.], dtype=np.float32)

n_users = int(user_ids.max()) + 1
n_items = int(item_ids.max()) + 1

# Mean normalization per item
item_sum = np.zeros(n_items, dtype=np.float32)
item_count = np.zeros(n_items, dtype=np.int32)
for iid, r in zip(item_ids, ratings):
    item_sum[iid] += r
    item_count[iid] += 1

item_mean = np.zeros(n_items, dtype=np.float32)
nonzero = item_count > 0
item_mean[nonzero] = item_sum[nonzero] / item_count[nonzero]

ratings_centered = ratings - item_mean[item_ids]

# Hyperparameters
embedding_dim = 32
l2_reg = 1e-6
learning_rate = 1e-2
batch_size = 64
epochs = 50

# Build model
class MatrixFactorization(tf.keras.Model):
    def __init__(self, n_users, n_items, dim, l2_reg=1e-6):
        super().__init__()
        self.user_embed = layers.Embedding(
            input_dim=n_users, output_dim=dim,
            embeddings_regularizer=tf.keras.regularizers.l2(l2_reg)
        )
        self.item_embed = layers.Embedding(
            input_dim=n_items, output_dim=dim,
            embeddings_regularizer=tf.keras.regularizers.l2(l2_reg)
        )
        self.user_bias = layers.Embedding(input_dim=n_users, output_dim=1)
        self.item_bias = layers.Embedding(input_dim=n_items, output_dim=1)

    def call(self, inputs):
        user, item = inputs
        u = self.user_embed(user)
        v = self.item_embed(item)
        ub = tf.squeeze(self.user_bias(user), -1)
        ib = tf.squeeze(self.item_bias(item), -1)
        dot = tf.reduce_sum(u * v, axis=1)
        return dot + ub + ib

# Prepare dataset
dataset = tf.data.Dataset.from_tensor_slices(((user_ids, item_ids), ratings_centered))
dataset = dataset.shuffle(2048).batch(batch_size)

model = MatrixFactorization(n_users, n_items, embedding_dim, l2_reg=l2_reg)
optimizer = tf.keras.optimizers.Adam(learning_rate)

mse = tf.keras.losses.MeanSquaredError()

# Training loop
for epoch in range(epochs):
    epoch_loss = 0.0
    n_batches = 0
    for (u_batch, i_batch), y_batch in dataset:
        with tf.GradientTape() as tape:
            preds = tf.squeeze(model((u_batch, i_batch)), axis=-1)
            loss = mse(y_batch, preds)
            loss += sum(model.losses)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        epoch_loss += loss.numpy()
        n_batches += 1
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss / n_batches:.4f}")

# Extract embeddings and biases
user_embeddings = model.user_embed.get_weights()[0]
item_embeddings = model.item_embed.get_weights()[0]
user_biases = np.squeeze(model.user_bias.get_weights()[0])
item_biases = np.squeeze(model.item_bias.get_weights()[0])

# Prediction helper (add item mean back)
def predict_rating(user_id, item_id):
    pred_centered = np.dot(user_embeddings[user_id], item_embeddings[item_id]) + user_biases[user_id] + item_biases[item_id]
    return float(pred_centered + item_mean[item_id])

print('Predicted rating for user 0 on item 2:', predict_rating(0, 2))
```

---

## Finding Related Items

After training, use item embeddings to find similar items (item-to-item recommendations). Compute cosine similarity on the learned item embeddings:

```python
import numpy as np

# normalize item embeddings
item_emb_norm = item_embeddings / np.linalg.norm(item_embeddings, axis=1, keepdims=True)

# cosine similarity matrix
cos_sim = item_emb_norm.dot(item_emb_norm.T)

# top-k similar items for a given item (exclude itself)
def top_k_similar(item_id, k=5):
    sims = cos_sim[item_id].copy()
    sims[item_id] = -1.0
    topk = np.argsort(sims)[-k:][::-1]
    return [(int(i), float(sims[i])) for i in topk]

print('Top related items to item 0:', top_k_similar(0, k=3))
```

---

## Practical Notes & Extensions

* Use `tf.data` pipelines to read large datasets from CSV/Parquet and to perform mapping (id mapping, filtering).
* Tune `embedding_dim`, `l2_reg`, and `learning_rate` for best performance. Larger embeddings may help with complex datasets.
* Add global mean $\mu$ and perform mean-normalization using both user and item biases for better convergence.
* For implicit feedback (clicks), use pairwise losses like BPR or weighted pointwise losses.
* For very large item catalogs use approximate nearest neighbors (Faiss, Annoy) for fast item retrieval.

---

## Summary Table

| Concept                | Symbol           | Description                       |
| ---------------------- | ---------------- | --------------------------------- |
| User preference vector | $w^{(j)}$        | Captures what the user likes      |
| Item feature vector    | $x^{(i)}$        | Captures what the item is like    |
| Bias term              | $b^{(j)}$        | Average rating tendency of a user |
| Regularization         | $\lambda$        | Prevents overfitting              |
| Learning method        | Gradient Descent | Used to optimize parameters       |

---
