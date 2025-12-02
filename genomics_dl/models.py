import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

def evaluate_clustering(X, labels):
    """
    Calcula métricas internas para un clustering dado.
    Devuelve dict con silhouette, calinski_harabasz, davies_bouldin.
    """
    labels = np.asarray(labels)

    # Si solo hay un cluster o todos son ruido, no se puede evaluar bien
    if len(np.unique(labels)) < 2:
        return {
            "silhouette": np.nan,
            "calinski_harabasz": np.nan,
            "davies_bouldin": np.nan,
        }

    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)

    return {
        "silhouette": sil,
        "calinski_harabasz": ch,
        "davies_bouldin": db,
    }

def search_best_clustering(
    X,
    n_clusters_range=range(2, 11),
    random_state=42,
    verbose=True,
):
    """
    Prueba distintos algoritmos y valores de n_clusters.
    Selecciona el mejor por silhouette.

    Algoritmos incluidos:
      - KMeans
      - GaussianMixture (GMM)
      - AgglomerativeClustering (linkage='ward')
    """

    results = []
    best_result = None

    for n_clusters in n_clusters_range:
        # --- 1) KMeans ---
        km = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init="auto",
        )
        labels_km = km.fit_predict(X)
        metrics_km = evaluate_clustering(X, labels_km)

        res_km = {
            "modelo": "kmeans",
            "n_clusters": n_clusters,
            "params": {"n_clusters": n_clusters},
            "labels": labels_km,
            "model": km,
            **metrics_km,
        }
        results.append(res_km)

        # --- 2) GMM ---
        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type="full",
            random_state=random_state,
        )
        gmm.fit(X)
        labels_gmm = gmm.predict(X)
        metrics_gmm = evaluate_clustering(X, labels_gmm)

        res_gmm = {
            "modelo": "gmm",
            "n_clusters": n_clusters,
            "params": {
                "n_components": n_clusters,
                "covariance_type": "full",
            },
            "labels": labels_gmm,
            "model": gmm,
            **metrics_gmm,
        }
        results.append(res_gmm)

        # --- 3) Agglomerative: ward (solo euclidean) ---
        agg_ward = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage="ward",
        )
        labels_aw = agg_ward.fit_predict(X)
        metrics_aw = evaluate_clustering(X, labels_aw)

        res_aw = {
            "modelo": "agglomerative_ward",
            "n_clusters": n_clusters,
            "params": {"n_clusters": n_clusters, "linkage": "ward"},
            "labels": labels_aw,
            "model": agg_ward,
            **metrics_aw,
        }
        results.append(res_aw)


        if verbose:
            print(
                f"n={n_clusters} | "
                f"kmeans sil={metrics_km['silhouette']:.3f}, "
                f"gmm sil={metrics_gmm['silhouette']:.3f}, "
                f"agg_ward sil={metrics_aw['silhouette']:.3f} "
            )

    # Convertimos resultados a DataFrame para inspección
    df_results = pd.DataFrame([
        {
            k: v for k, v in res.items()
            if k not in ("labels", "model")
        }
        for res in results
    ])

    # Seleccionar mejor por silhouette (máximo)
    # Si hay NaN, se ignoran.
    valid = df_results["silhouette"].notna()
    if not valid.any():
        raise RuntimeError("No se pudo evaluar ninguna configuración de clustering.")

    best_idx = df_results.loc[valid, "silhouette"].idxmax()
    best_summary = df_results.loc[best_idx].to_dict()
    best_full = results[best_idx]

    if verbose:
        print("\nMejor configuración:")
        print(best_summary)

    # Añadimos labels y modelo al resumen
    best_summary["labels"] = best_full["labels"]
    best_summary["model"] = best_full["model"]

    return best_summary, df_results, results
