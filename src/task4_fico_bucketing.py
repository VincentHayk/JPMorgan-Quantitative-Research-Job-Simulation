"""Task 4 - FICO score bucketing / quantization.

Two approaches are implemented:
1. K-means bucketing, which minimizes within-cluster squared error.
2. Dynamic programming, which maximizes bucket-level Bernoulli log-likelihood.

The dynamic-programming implementation first aggregates observations by unique
FICO score, making the exact optimization practical on the full dataset.

Run from the repository root with:
    python -m src.task4_fico_bucketing
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Loan_Data.csv"
OUTPUT_DIR = ROOT / "outputs"


def load_fico_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load FICO score and default observations."""
    data = pd.read_csv(path)
    required = {"fico_score", "default"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return data[["fico_score", "default"]].copy()


def kmeans_quantization(
    fico_scores,
    n_buckets: int = 5,
    random_state: int = 42,
):
    """Bucket one-dimensional FICO scores using K-means."""
    values = np.asarray(fico_scores, dtype=float).reshape(-1, 1)

    model = KMeans(
        n_clusters=n_buckets,
        random_state=random_state,
        n_init=20,
    )
    labels = model.fit_predict(values)

    centroids = np.sort(model.cluster_centers_.ravel())
    thresholds = [
        float((centroids[i] + centroids[i + 1]) / 2)
        for i in range(len(centroids) - 1)
    ]

    return thresholds, centroids.tolist(), labels, model.inertia_


def elbow_inertias(
    fico_scores,
    max_buckets: int = 12,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute K-means inertia over a range of bucket counts."""
    values = np.asarray(fico_scores, dtype=float).reshape(-1, 1)
    rows = []

    for k in range(1, max_buckets + 1):
        model = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=20,
        )
        model.fit(values)
        rows.append({"n_buckets": k, "inertia": model.inertia_})

    return pd.DataFrame(rows)


def _aggregate_by_score(fico_scores, defaults):
    """Aggregate counts and defaults at each unique FICO score."""
    frame = pd.DataFrame(
        {
            "fico_score": np.asarray(fico_scores, dtype=int),
            "default": np.asarray(defaults, dtype=int),
        }
    )

    grouped = (
        frame.groupby("fico_score")["default"]
        .agg(observations="count", defaults="sum")
        .reset_index()
        .sort_values("fico_score")
        .reset_index(drop=True)
    )
    return grouped


def dynamic_programming_quantization(
    fico_scores,
    defaults,
    n_buckets: int = 5,
    eps: float = 1e-12,
):
    """Find contiguous FICO buckets maximizing Bernoulli log-likelihood.

    Let DP[i, k] be the best log-likelihood obtained by partitioning the first
    i unique FICO levels into k contiguous buckets.

    Recurrence:
        DP[i, k] = max_j { DP[j, k-1] + LL(j+1, i) }

    where LL(a, b) is the Bernoulli log-likelihood of one bucket.
    """
    grouped = _aggregate_by_score(fico_scores, defaults)

    scores = grouped["fico_score"].to_numpy(dtype=float)
    counts = grouped["observations"].to_numpy(dtype=int)
    default_counts = grouped["defaults"].to_numpy(dtype=int)

    m = len(scores)
    if n_buckets < 1 or n_buckets > m:
        raise ValueError("n_buckets must be between 1 and the number of unique scores.")

    cumulative_n = np.concatenate([[0], np.cumsum(counts)])
    cumulative_k = np.concatenate([[0], np.cumsum(default_counts)])

    def segment_ll(start: int, end: int) -> float:
        """Log-likelihood for score-level indices start..end, inclusive."""
        n = cumulative_n[end + 1] - cumulative_n[start]
        k = cumulative_k[end + 1] - cumulative_k[start]
        p = k / n
        return (
            k * np.log(p + eps)
            + (n - k) * np.log(1.0 - p + eps)
        )

    dp = np.full((m + 1, n_buckets + 1), -np.inf)
    backtrack = np.full((m + 1, n_buckets + 1), -1, dtype=int)
    dp[0, 0] = 0.0

    for k in range(1, n_buckets + 1):
        for i in range(k, m + 1):
            for j in range(k - 1, i):
                candidate = dp[j, k - 1] + segment_ll(j, i - 1)
                if candidate > dp[i, k]:
                    dp[i, k] = candidate
                    backtrack[i, k] = j

    cut_indices = []
    i = m

    for k in range(n_buckets, 1, -1):
        j = backtrack[i, k]
        if j <= 0:
            raise RuntimeError("Backtracking failed to recover the optimal buckets.")
        cut_indices.append(j)
        i = j

    cut_indices = sorted(cut_indices)

    thresholds = [
        float((scores[j - 1] + scores[j]) / 2)
        for j in cut_indices
    ]

    return thresholds, float(dp[m, n_buckets]), grouped


def assign_buckets(fico_scores, thresholds):
    """Assign integer risk-bucket labels from ordered score thresholds."""
    return np.digitize(np.asarray(fico_scores, dtype=float), thresholds, right=False)


def bucket_summary(data: pd.DataFrame, thresholds) -> pd.DataFrame:
    """Summarize observations and empirical default rates by bucket."""
    result = data.copy()
    result["bucket"] = assign_buckets(result["fico_score"], thresholds)

    summary = (
        result.groupby("bucket")
        .agg(
            min_fico=("fico_score", "min"),
            max_fico=("fico_score", "max"),
            observations=("default", "size"),
            defaults=("default", "sum"),
            default_rate=("default", "mean"),
        )
        .reset_index()
    )

    return summary


def plot_elbow(elbow: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(elbow["n_buckets"], elbow["inertia"], marker="o")
    plt.xlabel("Number of buckets")
    plt.ylabel("K-means inertia")
    plt.title("FICO Score K-Means - Elbow Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_thresholds(
    fico_scores,
    thresholds,
    title: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(10, 3))
    values = np.asarray(fico_scores, dtype=float)
    plt.scatter(values, np.zeros_like(values), alpha=0.15, s=10)

    for threshold in thresholds:
        plt.axvline(threshold, linestyle="--")
        plt.text(
            threshold,
            0.02,
            f"{threshold:.1f}",
            ha="center",
            va="bottom",
        )

    plt.xlabel("FICO score")
    plt.yticks([])
    plt.title(title)
    plt.grid(True, axis="x", alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    data = load_fico_data()

    elbow = elbow_inertias(data["fico_score"], max_buckets=12)
    elbow.to_csv(OUTPUT_DIR / "fico_kmeans_elbow.csv", index=False)
    plot_elbow(elbow, OUTPUT_DIR / "fico_kmeans_elbow.png")

    kmeans_thresholds, centroids, _, inertia = kmeans_quantization(
        data["fico_score"],
        n_buckets=5,
    )
    kmeans_summary = bucket_summary(data, kmeans_thresholds)
    kmeans_summary.to_csv(
        OUTPUT_DIR / "fico_kmeans_bucket_summary.csv",
        index=False,
    )
    plot_thresholds(
        data["fico_score"],
        kmeans_thresholds,
        "FICO Buckets - K-Means",
        OUTPUT_DIR / "fico_kmeans_thresholds.png",
    )

    dp_thresholds, best_ll, _ = dynamic_programming_quantization(
        data["fico_score"],
        data["default"],
        n_buckets=5,
    )
    dp_summary = bucket_summary(data, dp_thresholds)
    dp_summary.to_csv(
        OUTPUT_DIR / "fico_loglikelihood_bucket_summary.csv",
        index=False,
    )
    plot_thresholds(
        data["fico_score"],
        dp_thresholds,
        "FICO Buckets - Maximum Log-Likelihood Dynamic Programming",
        OUTPUT_DIR / "fico_loglikelihood_thresholds.png",
    )

    print("K-means thresholds:")
    print(kmeans_thresholds)
    print("K-means centroids:")
    print(centroids)
    print(f"K-means inertia: {inertia:,.2f}")

    print("\nDynamic-programming thresholds:")
    print(dp_thresholds)
    print(f"Maximum bucket log-likelihood: {best_ll:,.4f}")

    print("\nDynamic-programming bucket summary:")
    print(dp_summary.to_string(index=False))


if __name__ == "__main__":
    main()
