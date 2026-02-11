# Content-Based Filtering

**Content-Based Filtering (CBF)** is a type of **Recommendation System** that recommends items similar to those a user has liked in the past — based solely on the **content (features)** of the items themselves.

---

## Intuition

Content-Based Filtering assumes that:

> If a user liked an item with certain features, they will likely like other items with similar features.

For example:
If **User A** liked *Inception* (a sci-fi, action movie), the system might recommend *Interstellar* or *The Matrix*, because they share similar features such as **genre**, **director**, or **themes**.

---

## Key Idea

Each item is represented as a **feature vector** describing its characteristics — such as:

* Genre
* Keywords
* Director
* Actors
* Description (embedded via NLP models like TF–IDF or BERT)

Each user is also represented by a **profile vector** that summarizes their preferences based on previously liked or rated items.

---

## Mathematical Model

Let:

* $x^{(i)}$: Feature vector of item $i$
* $w^{(u)}$: Preference vector for user $u$
* $y^{(u,i)}$: Rating or preference of user $u$ for item $i$

Then, the **predicted rating** for user $u$ and item $i$ is:

$$
\hat{y}^{(u,i)} = w^{(u)} \cdot x^{(i)}
$$

where $\cdot$ denotes the **dot product** — measuring similarity between the user’s preferences and the item’s features.

---

## Building the User Profile

If user $u$ has rated items $i_1, i_2, ..., i_m$, their profile vector is computed as:

$$
w^{(u)} = \frac{1}{\sum_{k=1}^{m} y^{(u,i_k)}} \sum_{k=1}^{m} y^{(u,i_k)} , x^{(i_k)}
$$

This means that the user’s preferences are the **weighted average** of the features of items they liked.

---

## Similarity Measure

To find how similar an item is to a user’s profile, we can use **cosine similarity**:

$$
\text{sim}(w^{(u)}, x^{(i)}) = \frac{w^{(u)} \cdot x^{(i)}}{|w^{(u)}| , |x^{(i)}|}
$$

Items with higher similarity scores are more likely to be recommended.

---

## Normalization

It’s often useful to normalize both vectors before computing similarity to avoid bias due to vector magnitude:

$$
\tilde{w}^{(u)} = \frac{w^{(u)}}{|w^{(u)}|}, \quad \tilde{x}^{(i)} = \frac{x^{(i)}}{|x^{(i)}|}
$$

Then, similarity simplifies to:

$$
\text{sim}(\tilde{w}^{(u)}, \tilde{x}^{(i)}) = \tilde{w}^{(u)} \cdot \tilde{x}^{(i)}
$$

---

## Recommendation Process

1. Represent all items by their feature vectors $x^{(i)}$.
2. Construct user profile $w^{(u)}$ from past interactions.
3. Compute similarity $\text{sim}(w^{(u)}, x^{(i)})$ for all items.
4. Rank items based on similarity scores.
5. Recommend the top $N$ most similar items.

---

## Example: TF–IDF Representation

If the item descriptions are text-based, we can represent them as TF–IDF vectors:

$$
x^{(i)} = [\text{tfidf}(t_1, i), \text{tfidf}(t_2, i), ..., \text{tfidf}(t_n, i)]
$$

Where:

* $\text{tfidf}(t_k, i)$ = TF–IDF score of term $t_k$ in item $i$

Then, user preferences and item similarities can be computed in this high-dimensional vector space.

---

## Regularization and Learning

Sometimes, instead of manually constructing $w^{(u)}$, we can learn it using regression:

$$
J(w^{(u)}) = \frac{1}{2} \sum_{i:r(u,i)=1} (w^{(u)} \cdot x^{(i)} - y^{(u,i)})^2 + \frac{\lambda}{2} |w^{(u)}|^2
$$

Then we update $w^{(u)}$ via **gradient descent**:

$$
w^{(u)} := w^{(u)} - \alpha , \nabla_{w^{(u)}} J(w^{(u)})
$$

Where $\alpha$ is the learning rate and $\lambda$ controls overfitting.

---

## Advantages

✅ Personalized to each user’s preferences <br>
✅ Doesn’t require data from other users <br>
✅ Transparent — we can explain *why* a recommendation was made (e.g., shared features)

---

## Limitations

⚠️ Can’t recommend new items without features (**cold start**) <br>
⚠️ Narrow scope — only recommends items similar to what the user already liked <br>
⚠️ Feature engineering or embeddings are crucial for good performance

---

## Matrix View

If we represent all users and items as matrices:

$$
\hat{Y} = W X^T
$$

Where:

* $W$: User preference matrix
* $X$: Item feature matrix
* $\hat{Y}$: Predicted user–item ratings

---

## Summary Table

| Concept                | Symbol                               | Description                          |
| ---------------------- | ------------------------------------ | ------------------------------------ |
| User preference vector | $w^{(u)}$                            | Describes what the user likes        |
| Item feature vector    | $x^{(i)}$                            | Describes characteristics of an item |
| Predicted rating       | $\hat{y}^{(u,i)}$                    | Similarity between user and item     |
| Regularization         | $\lambda$                            | Prevents overfitting                 |
| Learning method        | Gradient Descent / Cosine Similarity | Used for optimization or matching    |

---

---

## TensorFlow Implementation

Below is a practical TensorFlow 2.x implementation of a **content-based recommender**. It assumes you already have item feature vectors (e.g., TF–IDF vectors, BERT embeddings, or other dense features). The model learns a *user preference vector* (in a low-dimensional latent space) and projects item features into that same space, then predicts a rating by dot product.

### Approach

1. Project high-dimensional item features to a lower-dimensional embedding with a Dense layer.
2. Learn a per-user embedding (preference vector) in the same projection space.
3. Score = dot(projected_item_features, user_embedding) + optional bias terms.

This lets us keep a compact user representation while using rich item features.

### Requirements

```text
tensorflow>=2.8
scikit-learn (for TF-IDF example)
numpy, pandas
```

### Example code (TF–IDF -> shallow CBF model)

```python
# Example: content-based recommender using TF-IDF item features
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# --- Toy dataset ---
# items dataframe with 'item_id' and 'description'
items = pd.DataFrame({
    'item_id': [0,1,2,3],
    'description': [
        'space sci-fi adventure with black hole',
        'romantic comedy with heartfelt moments',
        'sci-fi thriller about dreams',
        'light comedy about friendship'
    ]
})

# interactions: user_id, item_id, rating
interactions = pd.DataFrame({
    'user_id': [0,0,1,1,2],
    'item_id': [0,1,0,2,3],
    'rating': [5.0, 2.0, 4.0, 5.0, 3.0]
})

# --- Precompute item features (TF-IDF) ---
vectorizer = TfidfVectorizer(max_features=2000)
item_texts = items['description'].tolist()
X_tfidf = vectorizer.fit_transform(item_texts).toarray().astype(np.float32)  # shape (n_items, n_tfidf)

# Optional: scale TF-IDF vectors (helps learning)
scaler = StandardScaler(with_mean=False)
X_tfidf = scaler.fit_transform(X_tfidf)

n_items, tfidf_dim = X_tfidf.shape
n_users = interactions['user_id'].nunique()

# Map item ids to row indices (if not already 0..n_items-1)
item_id_to_idx = {iid: idx for idx, iid in enumerate(items['item_id'].tolist())}
interactions['item_idx'] = interactions['item_id'].map(item_id_to_idx)

# Training arrays
user_ids = interactions['user_id'].values.astype(np.int32)
item_idxs = interactions['item_idx'].values.astype(np.int32)
ratings = interactions['rating'].values.astype(np.float32)

# Convert item features to a TensorFlow constant for fast lookups
item_features_tf = tf.constant(X_tfidf)  # shape (n_items, tfidf_dim)

# Hyperparameters
proj_dim = 64  # dimensionality of the projected item space / user embedding
l2_reg = 1e-6
learning_rate = 1e-3
batch_size = 32
epochs = 100

# Build the model
# We will build a Keras model that takes (user_id, item_idx) as input.
class ContentBasedModel(Model):
    def __init__(self, n_users, tfidf_dim, proj_dim, l2_reg=1e-6):
        super().__init__()
        # projection from TF-IDF -> proj_dim
        self.item_projector = layers.Dense(
            proj_dim,
            activation='linear',
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
        )
        # learned user embeddings (preference vectors) in proj_dim
        self.user_embedding = layers.Embedding(
            input_dim=n_users,
            output_dim=proj_dim,
            embeddings_initializer='normal',
            embeddings_regularizer=tf.keras.regularizers.l2(l2_reg)
        )
        # optional user bias
        self.user_bias = layers.Embedding(input_dim=n_users, output_dim=1)
        # optional item bias (learned as scalar per item)
        self.item_bias = layers.Embedding(input_dim=n_items, output_dim=1)

    def call(self, inputs, training=False):
        user_ids, item_idxs = inputs
        # gather item features from the precomputed item_features_tf
        item_feats = tf.gather(item_features_tf, item_idxs)  # shape (batch, tfidf_dim)
        proj = self.item_projector(item_feats)  # shape (batch, proj_dim)
        user_vec = self.user_embedding(user_ids)  # shape (batch, proj_dim)
        user_b = tf.squeeze(self.user_bias(user_ids), axis=-1)
        item_b = tf.squeeze(self.item_bias(item_idxs), axis=-1)
        dot = tf.reduce_sum(proj * user_vec, axis=1)  # (batch,)
        return dot + user_b + item_b

# Instantiate and compile
model = ContentBasedModel(n_users=n_users, tfidf_dim=tfidf_dim, proj_dim=proj_dim, l2_reg=l2_reg)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='mse')

# Prepare tf.data dataset
dataset = tf.data.Dataset.from_tensor_slices(((user_ids, item_idxs), ratings))
dataset = dataset.shuffle(1024).batch(batch_size)

# Train
model.fit(dataset, epochs=epochs, verbose=1)

# After training: extract learned components
user_embeddings = model.user_embedding.get_weights()[0]  # (n_users, proj_dim)
item_projection_weights = model.item_projector.get_weights()[0]  # shape (tfidf_dim, proj_dim)
item_biases = np.squeeze(model.item_bias.get_weights()[0])
user_biases = np.squeeze(model.user_bias.get_weights()[0])

# Prediction helper
def predict_rating(user_id, item_id):
    item_idx = item_id_to_idx[item_id]
    item_feat = X_tfidf[item_idx]  # numpy vector
    proj = item_feat.dot(item_projection_weights)  # (proj_dim,)
    score = proj.dot(user_embeddings[user_id]) + user_biases[user_id] + item_biases[item_idx]
    return float(score)

print('Predicted rating (user 0, item 2):', predict_rating(0, 2))
```

### Explanation

* `item_features_tf` contains precomputed item vectors (TF–IDF or embeddings). The model projects these into a lower-dimensional `proj_dim` space with a Dense layer (`item_projector`).
* `user_embedding` contains learnt user preference vectors in that `proj_dim` space.
* Prediction = dot(projected_item, user_embedding) + user_bias + item_bias.

This setup is flexible: you can replace TF–IDF with BERT / SentenceTransformers embeddings (dense vectors); then skip the `item_projector` and feed item embeddings directly to the model (set `proj_dim` equal to the embedding dim).

---

## Finding Similar Items (Content-Based)

With item features (or projected item embeddings), you can find items similar to a given item using cosine similarity.

```python
# Use the projected item embeddings for similarity
# Compute projected embeddings for all items
item_feats_np = X_tfidf  # (n_items, tfidf_dim)
projected_all = item_feats_np.dot(item_projection_weights)  # (n_items, proj_dim)

# normalize
proj_norm = projected_all / np.linalg.norm(projected_all, axis=1, keepdims=True)

# cosine similarity matrix (n_items, n_items)
cos_sim = proj_norm.dot(proj_norm.T)

# top-k similar (excluding self)
def top_k_similar(item_id, k=5):
    idx = item_id_to_idx[item_id]
    sims = cos_sim[idx].copy()
    sims[idx] = -1
    topk = np.argsort(sims)[-k:][::-1]
    return [(int(items['item_id'].iloc[i]), float(sims[i])) for i in topk]

print('Top related items to item 0:', top_k_similar(0, k=3))
```

### Practical notes

* For large TF–IDF vectors you may want to compress item representations (e.g., PCA, SVD) before training.
* If item features are dense embeddings (from BERT / SentenceTransformers), you can skip the `item_projector` and directly use those vectors.
* You can also learn a small neural network on the concatenation `[user_embedding, projected_item]` for non-linear interactions instead of a dot product.

---
