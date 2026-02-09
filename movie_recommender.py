import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def resolve_data_dir() -> str:
    """Resolve the MovieLens 100K data directory."""
    candidates = [
        os.path.join(os.getcwd(), "data", "ml-100k"),
        os.path.join(os.getcwd(), "ml-100k"),
        os.path.join(os.getcwd(), "..", "..", "ml-100k"),
        os.path.join(os.getcwd(), "..", "ml-100k"),
    ]
    for path in candidates:
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "u.data")):
            return os.path.abspath(path)
    raise FileNotFoundError(
        "Could not locate ml-100k dataset. Place it in data/ml-100k or ../ml-100k."
    )


def load_movielens_100k(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load MovieLens 100K ratings and movie metadata."""
    ratings_cols = ["user_id", "item_id", "rating", "timestamp"]
    base_path = os.path.join(data_dir, "u1.base")
    test_path = os.path.join(data_dir, "u1.test")
    movies_path = os.path.join(data_dir, "u.item")

    train = pd.read_csv(base_path, sep="\t", names=ratings_cols, encoding="latin-1")
    test = pd.read_csv(test_path, sep="\t", names=ratings_cols, encoding="latin-1")

    movie_cols = [
        "item_id",
        "title",
        "release_date",
        "video_release_date",
        "imdb_url",
    ] + [f"genre_{i}" for i in range(19)]
    movies = pd.read_csv(movies_path, sep="|", names=movie_cols, encoding="latin-1")

    return train, test, movies[["item_id", "title"]]


def build_user_item_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    """Create a user-item rating matrix."""
    matrix = ratings.pivot_table(
        index="user_id", columns="item_id", values="rating", fill_value=0
    )
    return matrix


def compute_user_similarity(user_item_matrix: pd.DataFrame) -> np.ndarray:
    """Compute cosine similarity between users."""
    return cosine_similarity(user_item_matrix.values)


def predict_scores_for_user(
    user_id: int,
    user_item_matrix: pd.DataFrame,
    similarity_matrix: np.ndarray,
) -> pd.Series:
    """Predict scores for all items for a given user using weighted user similarity."""
    if user_id not in user_item_matrix.index:
        raise ValueError(f"User {user_id} not found in user-item matrix.")

    user_index = user_item_matrix.index.get_loc(user_id)
    similarities = similarity_matrix[user_index].copy()
    similarities[user_index] = 0.0

    ratings = user_item_matrix.values
    weighted_sum = similarities @ ratings
    denom = (np.abs(similarities)[:, None] * (ratings > 0)).sum(axis=0)

    scores = np.divide(
        weighted_sum,
        denom,
        out=np.zeros_like(weighted_sum, dtype=float),
        where=denom != 0,
    )
    return pd.Series(scores, index=user_item_matrix.columns)


def recommend_top_k(
    user_id: int,
    user_item_matrix: pd.DataFrame,
    similarity_matrix: np.ndarray,
    movies: pd.DataFrame,
    k: int = 10,
) -> pd.DataFrame:
    """Recommend top-k unseen movies for a user."""
    scores = predict_scores_for_user(user_id, user_item_matrix, similarity_matrix)
    seen_items = user_item_matrix.loc[user_id]
    unseen_scores = scores[seen_items == 0]

    top_items = unseen_scores.sort_values(ascending=False).head(k)
    recommendations = (
        pd.DataFrame({"item_id": top_items.index, "predicted_score": top_items.values})
        .merge(movies, on="item_id", how="left")
        .loc[:, ["item_id", "title", "predicted_score"]]
    )
    return recommendations


def precision_at_k(
    user_item_matrix: pd.DataFrame,
    similarity_matrix: np.ndarray,
    test_ratings: pd.DataFrame,
    k: int = 10,
    positive_threshold: int = 4,
) -> float:
    """Compute mean Precision@K over users with at least one positive test item."""
    precisions: List[float] = []
    test_grouped = test_ratings.groupby("user_id")

    for user_id, user_test in test_grouped:
        positive_items = set(
            user_test[user_test["rating"] >= positive_threshold]["item_id"].tolist()
        )
        if not positive_items:
            continue

        recommendations = recommend_top_k(
            user_id=user_id,
            user_item_matrix=user_item_matrix,
            similarity_matrix=similarity_matrix,
            movies=pd.DataFrame({"item_id": user_item_matrix.columns}),
            k=k,
        )
        recommended_items = set(recommendations["item_id"].tolist())
        hit_count = len(recommended_items & positive_items)
        precisions.append(hit_count / k)

    if not precisions:
        return 0.0
    return float(np.mean(precisions))


def main() -> None:
    data_dir = resolve_data_dir()
    train, test, movies = load_movielens_100k(data_dir)

    user_item_matrix = build_user_item_matrix(train)
    similarity_matrix = compute_user_similarity(user_item_matrix)

    user_id = 1
    recommendations = recommend_top_k(
        user_id=user_id,
        user_item_matrix=user_item_matrix,
        similarity_matrix=similarity_matrix,
        movies=movies,
        k=10,
    )

    print("Top-10 Recommendations for User 1")
    print("=" * 40)
    print(recommendations.to_string(index=False))

    p_at_10 = precision_at_k(
        user_item_matrix=user_item_matrix,
        similarity_matrix=similarity_matrix,
        test_ratings=test,
        k=10,
        positive_threshold=4,
    )

    print("\nEvaluation")
    print("=" * 40)
    print(f"Precision@10: {p_at_10:.4f}")


if __name__ == "__main__":
    main()
