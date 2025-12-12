from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

from pathlib import Path
from typing import Sequence, Tuple, Optional, Union

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import umap
import matplotlib.cm as cm
import matplotlib.colors as mcolors


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


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """
    Corrección FDR Benjamini–Hochberg.
    Devuelve q-values en el mismo orden que pvals.
    """
    pvals = np.asarray(pvals, dtype=float)
    n = pvals.size

    order = np.argsort(pvals)
    ranked = pvals[order]

    qvals_ranked = np.empty_like(ranked)
    prev_q = 1.0

    for i in range(n - 1, -1, -1):
        rank = i + 1
        q = ranked[i] * n / rank
        if q > prev_q:
            q = prev_q
        prev_q = q
        qvals_ranked[i] = q

    qvals = np.empty_like(qvals_ranked)
    qvals[order] = qvals_ranked
    return qvals


def differential_expression_two_clusters(
    X_log: pd.DataFrame,
    cluster_labels: pd.Series,
    min_samples_per_cluster: int = 10,
) -> pd.DataFrame:
    """
    Comparación de expresión (log(TPM)) entre dos clusters.

    X_log: matriz de expresión log1p(TPM) ya filtrada (n_muestras x n_genes)
    cluster_labels: etiquetas de cluster (exactamente 2 valores distintos)
    """
    cluster_labels = cluster_labels.loc[X_log.index]
    unique_clusters = np.sort(cluster_labels.unique())

    if unique_clusters.size != 2:
        raise ValueError(
            f"Se esperaban exactamente 2 clusters, encontrados: {unique_clusters}"
        )

    c0, c1 = unique_clusters
    mask0 = cluster_labels == c0
    mask1 = cluster_labels == c1

    n0 = mask0.sum()
    n1 = mask1.sum()

    if n0 < min_samples_per_cluster or n1 < min_samples_per_cluster:
        raise ValueError(
            f"Clusters demasiado pequeños: n0={n0}, n1={n1}, "
            f"mínimo requerido={min_samples_per_cluster}."
        )

    X0 = X_log.loc[mask0]
    X1 = X_log.loc[mask1]

    mean0 = X0.mean(axis=0)
    mean1 = X1.mean(axis=0)

    # Welch t-test por gen
    _, pvals = ttest_ind(
        X0.values,
        X1.values,
        axis=0,
        equal_var=False,
        nan_policy="omit",
    )

    qvals = benjamini_hochberg(pvals)

    diff = mean1 - mean0
    diff_abs = diff.abs()

    stats_df = pd.DataFrame(
        {
            "mean_0": mean0,
            "mean_1": mean1,
            "diff": diff,
            "diff_abs": diff_abs,
            "pval": pvals,
            "qval": qvals,
        },
        index=X_log.columns,
    ).sort_values("diff_abs", ascending=False)

    return stats_df


def plot_cluster_heatmap(
    X_log: pd.DataFrame,
    cluster_labels: pd.Series,
    gene_stats: pd.DataFrame,
    top_n: int = 50,
    zscore: bool = True,
    figsize: Tuple[int, int] = (10, 8),
    output_path: Optional[Path | str] = None,
) -> None:
    """
    Heatmap de los genes más diferenciales entre dos clusters.
    """
    cluster_labels = cluster_labels.loc[X_log.index]

    selected_genes = gene_stats.head(top_n).index.tolist()

    # Ordenar muestras por cluster
    ordered_idx = cluster_labels.sort_values().index
    X_plot = X_log.loc[ordered_idx, selected_genes]
    clusters_ordered = cluster_labels.loc[ordered_idx]

    if zscore:
        X_plot = (X_plot - X_plot.mean(axis=0)) / (X_plot.std(axis=0) + 1e-8)

    plt.figure(figsize=figsize)
    sns.heatmap(
        X_plot,
        cmap="viridis",
        yticklabels=False,
        cbar_kws={"label": "z-score" if zscore else "log(TPM)"},
    )
    plt.title(f"Heatmap top {top_n} genes diferenciales por cluster")
    plt.xlabel("Genes")
    plt.ylabel("Muestras (ordenadas por cluster)")
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close()


def run_nonmalignant_cluster_characterization(
    X_log: pd.DataFrame,
    meta: pd.DataFrame,
    cluster_col: str,
    output_dir: Path | str,
    top_n_genes: int = 50,
    save_csv: bool = True,
    save_heatmap: bool = True,
) -> pd.DataFrame:
    """
    Ejecuta caracterización de clusters nonMalignant:
      - DE entre clusters
      - opcionalmente guarda CSV y heatmap

    Devuelve el DataFrame de DE.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cluster_labels = meta[cluster_col]

    de_df = differential_expression_two_clusters(X_log, cluster_labels)

    if save_csv:
        csv_path = output_dir / "de_clusters_nonmalignant.csv"
        de_df.to_csv(csv_path, index=True)

    if save_heatmap:
        heatmap_path = output_dir / f"heatmap_top{top_n_genes}_genes_nonmalignant.png"
        plot_cluster_heatmap(
            X_log=X_log,
            cluster_labels=cluster_labels,
            gene_stats=de_df,
            top_n=top_n_genes,
            output_path=heatmap_path,
        )

    return de_df


def build_supervised_matrix(
    X: pd.DataFrame,
    meta: pd.DataFrame,
    target_col: str,
    cluster_col: Optional[str] = None,
    extra_meta_cols: Optional[Sequence[str]] = None,
    one_hot_cluster: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Construye matriz de características X_supervised e y a partir de:
      - matriz de genes (o PCs) X
      - metadatos meta (incluyendo target y, opcionalmente, cluster)
    """
    meta = meta.loc[X.index]

    y = meta[target_col].copy()

    feature_parts = [X]

    if cluster_col is not None:
        cluster_series = meta[cluster_col].astype("category")
        if one_hot_cluster:
            cluster_dummies = pd.get_dummies(cluster_series, prefix=cluster_col)
            feature_parts.append(cluster_dummies)
        else:
            feature_parts.append(
                pd.DataFrame(
                    {cluster_col: cluster_series.cat.codes},
                    index=X.index,
                )
            )

    if extra_meta_cols:
        extra_df = meta[list(extra_meta_cols)].copy()
        feature_parts.append(extra_df)

    X_sup = pd.concat(feature_parts, axis=1)

    return X_sup, y

def flag_pca_distance_outliers(
    X_pca: pd.DataFrame,
    quantile: float = 0.99,
    center: str = "mean",
) -> pd.Series:
    """
    Marca como outliers las muestras con distancia euclídea al centro
    por encima del cuantil dado.

    Parámetros
    ----------
    X_pca : DataFrame
        Embedding PCA (filas = muestras, columnas = PC1, PC2, ...).
    quantile : float
        Cuantil de distancia a partir del cual se marca outlier (0.99 -> top 1%).
    center : {"mean", "median"}
        Tipo de centro a usar (media o mediana).

    Devuelve
    --------
    outliers : Series booleana indexada por las muestras.
    """
    if center == "mean":
        center_vec = X_pca.mean(axis=0).values
    elif center == "median":
        center_vec = X_pca.median(axis=0).values
    else:
        raise ValueError("center debe ser 'mean' o 'median'.")

    dists = np.linalg.norm(X_pca.values - center_vec, axis=1)
    thresh = np.quantile(dists, quantile)
    flags = dists >= thresh

    return pd.Series(flags, index=X_pca.index, name="is_outlier_pca")


def compute_umap_from_pca(
    X_pca,
    n_neighbors=15,
    min_dist=0.3,
    random_state=42
):
    """
    Aplica UMAP sobre el espacio PCA.
    X_pca: array (n_samples, n_components_pca)
    Devuelve: X_umap (n_samples, 2)
    """
    umap_model = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=random_state,
    )
    X_umap = umap_model.fit_transform(X_pca)
    return X_umap, umap_model


def plot_umap(
    X_umap: Union[np.ndarray, pd.DataFrame],
    color: Optional[Union[pd.Series, np.ndarray, list]] = None,
    title: str = "UMAP",
    xlabel: str = "UMAP 1",
    ylabel: str = "UMAP 2",
    cmap: Optional[str] = None,
    is_categorical: Optional[bool] = None,
    figsize: Tuple[int, int] = (6, 5),
    legend: bool = True,
) -> None:
    """
    Scatter de UMAP que admite tanto variables numéricas como categóricas para colorear.

    Parámetros
    ----------
    X_umap : array-like (n_samples, 2)
        Coordenadas UMAP (2D).
    color : Series/array/list, opcional
        Variable para colorear los puntos (numérica o categórica).
    title, xlabel, ylabel : str
        Títulos y etiquetas de los ejes.
    cmap : str, opcional
        Nombre del colormap de matplotlib a usar.
        - Numérica: se usa como mapa continuo.
        - Categórica: se usa como mapa discreto (ej. 'tab10', 'tab20').
        Si None, se elige un valor por defecto razonable.
    is_categorical : bool, opcional
        Forzar tratamiento como categórica (True) o numérica (False).
        Si None, se intenta inferir automáticamente.
    figsize : (int, int)
        Tamaño de la figura.
    legend : bool
        Si True, muestra leyenda para variables categóricas.
    """
    # Asegurar array 2D
    if isinstance(X_umap, pd.DataFrame):
        coords = X_umap.values
    else:
        coords = np.asarray(X_umap)

    if coords.shape[1] != 2:
        raise ValueError(f"X_umap debe tener shape (n_samples, 2), recibido {coords.shape}")

    plt.figure(figsize=figsize)

    if color is None:
        # Caso sin color: todo en un solo color
        plt.scatter(coords[:, 0], coords[:, 1], alpha=0.7)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.show()
        return

    # Convertir color a Series para facilitar manejo
    if isinstance(color, pd.Series):
        color_series = color
    else:
        color_series = pd.Series(color, index=None)

    # Inferir si es categórica o numérica si no se especifica
    if is_categorical is None:
        if pd.api.types.is_numeric_dtype(color_series):
            # Numérica pura
            is_categorical = False
        else:
            # No numérica (object, category, etc.)
            is_categorical = True

        # Heurística adicional: pocas categorías distintas => tratamos como categórica
        n_unique = color_series.nunique(dropna=True)
        if n_unique <= 20:
            # Sobre-escribimos a categórica
            is_categorical = True

    # Caso 1: variable numérica

    if not is_categorical:
        if cmap is None:
            cmap = "viridis"

        scatter = plt.scatter(
            coords[:, 0],
            coords[:, 1],
            c=color_series.values,
            alpha=0.7,
            cmap=cmap,
        )
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        cbar = plt.colorbar(scatter)
        cbar.set_label(str(color_series.name) if color_series.name is not None else "")
        plt.tight_layout()
        plt.show()
        return

    # Caso 2: variable categórica

    # Convertimos a categoría explícita
    cat = color_series.astype("category")
    categories = cat.cat.categories
    codes = cat.cat.codes.values  # -1 para NaN

    n_cats = len(categories)

    # Colormap discreto
    if cmap is None:
        cmap = "tab10" if n_cats <= 10 else "tab20"
    base_cmap = cm.get_cmap(cmap, n_cats)

    # Mapear códigos a colores (ignorando -1 de NaN)
    norm = mcolors.Normalize(vmin=0, vmax=max(n_cats - 1, 1))
    colors_mapped = np.full_like(codes, fill_value=np.nan, dtype=float)
    mask_valid = codes >= 0
    colors_mapped[mask_valid] = codes[mask_valid]

    scatter = plt.scatter(
        coords[:, 0],
        coords[:, 1],
        c=colors_mapped,
        alpha=0.7,
        cmap=base_cmap,
        norm=norm,
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Leyenda para categorías
    if legend:
        handles = []
        labels = []
        for i, cat_name in enumerate(categories):
            color_i = base_cmap(norm(i))
            handle = plt.Line2D(
                [], [], marker="o", linestyle="",
                markersize=6, markerfacecolor=color_i, markeredgecolor="none"
            )
            handles.append(handle)
            labels.append(str(cat_name))
        plt.legend(handles, labels, title=color_series.name, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.show()
