from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


@dataclass
class ExpressionPreprocessor:
    """
    Preprocesado de expresión:
      - selección de genes de alta varianza
      - log1p
      - escalado estándar
    """
    genes: list[str]
    scaler: StandardScaler


@dataclass
class PcaModel:
    """
    Modelo PCA ajustado sobre la matriz escalada.
    """
    pca: PCA


def select_high_var_genes(
    X: pd.DataFrame,
    var_quantile: float = 0.2,
) -> list[str]:
    """
    Selecciona genes por encima de un cuantil de varianza.

    Parámetros
    ----------
    X : DataFrame (n_muestras, n_genes)
    var_quantile : float
        Cuantil de varianza (0.2 -> descarta ~20% genes menos variables).

    Devuelve
    --------
    genes_keep : lista de nombres de genes a mantener.
    """
    var = X.var(axis=0)
    thr = np.quantile(var, var_quantile)
    genes_keep = var[var > thr].index.tolist()
    return genes_keep


def fit_expression_preprocessor(
    X_train: pd.DataFrame,
    var_quantile: float = 0.2,
) -> ExpressionPreprocessor:
    """
    Ajusta preprocesado de expresión sobre el train:
      - selección de genes de alta varianza
      - log1p
      - StandardScaler (por gen)
    """
    genes = select_high_var_genes(X_train, var_quantile=var_quantile)
    X_sel = X_train[genes]
    X_log = np.log1p(X_sel)

    scaler = StandardScaler()
    scaler.fit(X_log)

    return ExpressionPreprocessor(genes=genes, scaler=scaler)


def transform_expression_preprocessor(
    X: pd.DataFrame,
    preproc: ExpressionPreprocessor,
) -> pd.DataFrame:
    """
    Aplica el preprocesado de expresión ajustado en train a un nuevo X.
    """
    X_sel = X[preproc.genes]
    X_log = np.log1p(X_sel)
    X_scaled = preproc.scaler.transform(X_log)

    return pd.DataFrame(
        X_scaled,
        index=X.index,
        columns=preproc.genes,
    )


def fit_pca_auto(
    X_scaled: pd.DataFrame,
    var_threshold: float = 0.9,
    max_components: Optional[int] = None,
    random_state: int = 42,
) -> PcaModel:
    """
    Ajusta PCA eligiendo automáticamente el nº de componentes
    para superar var_threshold (ej. 0.9 = 90% varianza explicada).

    Opcionalmente limita el nº máximo de componentes.
    """
    pca_full = PCA(random_state=random_state)
    pca_full.fit(X_scaled)

    cum = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = int(np.searchsorted(cum, var_threshold) + 1)

    if max_components is not None:
        n_comp = min(n_comp, max_components)

    pca = PCA(n_components=n_comp, random_state=random_state)
    pca.fit(X_scaled)

    return PcaModel(pca=pca)


def transform_pca(
    X_scaled: pd.DataFrame,
    pca_model: PcaModel,
) -> pd.DataFrame:
    """
    Proyecta X_scaled en el espacio PCA ajustado en train.
    """
    X_pca = pca_model.pca.transform(X_scaled)
    n_comp = X_pca.shape[1]
    cols = [f"PC{i+1}" for i in range(n_comp)]

    return pd.DataFrame(
        X_pca,
        index=X_scaled.index,
        columns=cols,
    )
