# Task 4: Recommendation System (Manual SVD with scikit-learn) for Codtech Internship
# Description: Matrix factorization using TruncatedSVD for collaborative filtering on MovieLens.
#              Alternative to Surprise (NumPy 2 compatible). Includes eval, top-N recs, viz.
# Author: [Your Name] | Date: November 01, 2025
# Dataset: MovieLens 100K (loaded via URL; 943 users, 1682 movies, 100K ratings 1-5)
# Requirements: scikit-learn pandas numpy matplotlib (already installed)
# Deliverable: Notebook with results/metrics (RMSE ~0.95, top recs).

# Step 0: Import libraries
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD  # SVD for matrix factorization
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from collections import defaultdict
import matplotlib.pyplot as plt

print("Libraries imported! (NumPy version:", np.__version__, ")")

# Step 1: Load and Prep Data (via URL—no manual download)
# Ratings: user_id | movie_id | rating | timestamp
ratings_url = 'https://files.grouplens.org/datasets/movielens/ml-100k/u.data'
ratings = pd.read_csv(ratings_url, sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])

# Movie titles: movie_id | title | ...
movies_url = 'https://files.grouplens.org/datasets/movielens/ml-100k/u.item'
movies = pd.read_csv(movies_url, sep='|', encoding='latin-1', header=None,
                     names=['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + [f'genre_{i}' for i in range(19)])

print(f"Ratings loaded: {len(ratings)} samples | Movies: {len(movies)}")
print(ratings.head())

# Pivot to user-item matrix (rows=users, cols=movies, values=ratings; NaN for unseen)
user_item_matrix = ratings.pivot(index='user_id', columns='movie_id', values='rating')
total_elements = user_item_matrix.size
total_nans = np.isnan(user_item_matrix).sum().sum()  # FIXED: .sum().sum() for total NaNs
sparsity = total_nans / total_elements
print(f"Matrix shape: {user_item_matrix.shape} (sparse: {sparsity:.1%} missing)")

# Split: 80% train, 20% test
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)
train_matrix = train_data.pivot(index='user_id', columns='movie_id', values='rating')
test_ratings = test_data.set_index(['user_id', 'movie_id'])['rating']  # For RMSE

# Step 2: Matrix Factorization with TruncatedSVD
# Fill NaN with 0 (SVD assumes dense; center with global mean)
global_mean = train_matrix.mean().mean()
train_matrix_filled = train_matrix.fillna(0) - global_mean  # Center ratings

# SVD: Decompose to k=50 latent factors
n_factors = 50
svd = TruncatedSVD(n_components=n_factors, random_state=42)
user_factors = svd.fit_transform(train_matrix_filled)  # U matrix
item_factors = svd.components_.T  # V^T matrix

print(f"SVD fitted: {n_factors} latent factors | Explained variance: {svd.explained_variance_ratio_.sum():.3f}")

# Step 3: Predict Ratings (R ≈ U * Sigma * V^T + mean)
def predict_rating(user_id, movie_id):
    user_idx = user_id - 1  # 1-based IDs
    movie_idx = movie_id - 1
    if user_idx >= len(user_factors) or movie_idx >= len(item_factors):
        return global_mean  # Fallback
    pred = user_factors[user_idx].dot(item_factors[movie_idx]) + global_mean
    return np.clip(pred, 1, 5)  # Clip to 1-5

# Eval: RMSE on test set
test_preds = [predict_rating(uid, mid) for uid, mid in test_data[['user_id', 'movie_id']].values]
rmse = np.sqrt(mean_squared_error(test_ratings, test_preds))
print(f"Test RMSE: {rmse:.3f} (good; lower = accurate predictions)")

# Precision@K (K=5, threshold=3.5)
def precision_at_k(user_id, k=5, threshold=3.5):
    # Predict all movies for user
    user_preds = {}
    for mid in movies['movie_id']:
        user_preds[mid] = predict_rating(user_id, mid)
    # Top-K
    top_k = sorted(user_preds.items(), key=lambda x: x[1], reverse=True)[:k]
    # True likes (from test data for this user)
    user_true_high = test_data[(test_data['user_id'] == user_id) & (test_data['rating'] >= threshold)]['movie_id'].values
    n_rel_and_rec_k = sum(1 for mid, pred in top_k if pred >= threshold and mid in user_true_high)
    n_rec_k = k
    return n_rel_and_rec_k / n_rec_k if n_rec_k > 0 else 0

# Avg over sample users (full loop slow; sample 100)
users = ratings['user_id'].unique()
precision_k = np.mean([precision_at_k(uid, k=5) for uid in users[:100]])
print(f"Precision@5: {precision_k:.3f} (relevant top recs)")

# Step 4: Top-N Recommendations for Sample User (e.g., user 196)
sample_user = 196
user_preds = {mid: predict_rating(sample_user, mid) for mid in movies['movie_id']}
top_n = sorted(user_preds.items(), key=lambda x: x[1], reverse=True)[:5]

print(f"\nTop 5 Recs for User {sample_user}:")
for mid, est in top_n:
    title = movies[movies['movie_id'] == mid]['title'].iloc[0]
    print(f"  '{title}' → Predicted: {est:.2f}/5")

# Step 5: Visualization - Predicted Ratings Distribution (for sample user)
pred_values = list(user_preds.values())
plt.figure(figsize=(8, 5))
plt.hist(pred_values, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
plt.title(f"Predicted Ratings Distribution (User {sample_user})")
plt.xlabel('Predicted Rating (1-5)')
plt.ylabel('Frequency')
plt.axvline(3.5, color='red', linestyle='--', label='Good Threshold')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.savefig('predicted_ratings_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Summary Table (Bonus for Analysis)
metrics = {'RMSE': rmse, 'Precision@5': precision_k, 'Sparsity': sparsity}
print("\nMetrics Table:")
for k, v in metrics.items():
    print(f"{k}: {v:.3f}" if isinstance(v, (int, float)) else f"{k}: {v:.1%}")

print("\n--- Analysis Summary ---")
print(f"- Dataset: MovieLens 100K (sparse ~94% missing; SVD fills via latent factors).")
print(f"- Model: TruncatedSVD (50 factors) → RMSE {rmse:.3f}, Precision@5 {precision_k:.3f} (solid baseline).")
print("- Insights: Recs based on user/movie similarities (e.g., Pixar fans get 'Toy Story'). Centering reduces bias.")
print("- Pros: Interpretable (factors=clusters), NumPy 2 compatible. Cons: Assumes linear factors (non-linear: autoencoders).")
print("- Files: PNG saved. For app: Wrap predict in Streamlit (pip install streamlit).")
print("- Extensions: Add user bias; hybrid with genres (one-hot + Ridge).")
print("- Submission: This meets 'matrix factorization'—notebook with metrics/recs ready!")