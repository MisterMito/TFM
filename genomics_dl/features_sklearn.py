from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


def _to_df(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    raise TypeError(
        "Este pipeline espera pandas.DataFrame para mantener nombres de genes/columnas."
    )


class FeatureColumnSelector(BaseEstimator, TransformerMixin):
    """Asegura que las columnas de entrada (genes) estén presentes y en el orden esperado."""

    def __init__(self, feature_cols: Iterable[str]):
        self.feature_cols = feature_cols

    def fit(self, X, y=None):
        X_df = _to_df(X)
        if self.feature_cols is None:
            raise ValueError("FeatureColumnSelector requiere feature_cols definidos.")

        feature_cols = list(self.feature_cols)
        if not feature_cols:
            raise ValueError("FeatureColumnSelector recibió feature_cols vacíos.")

        missing = [c for c in feature_cols if c not in X_df.columns]
        if missing:
            raise ValueError(
                "FeatureColumnSelector: faltan columnas en X: "
                f"{missing[:10]} ..."
            )
        self.feature_names_in_ = feature_cols
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        X_df = _to_df(X)
        missing = [c for c in self.feature_names_in_ if c not in X_df.columns]
        if missing:
            raise ValueError(
                "FeatureColumnSelector: faltan columnas en X: "
                f"{missing[:10]} ..."
            )
        return X_df.loc[:, self.feature_names_in_]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        return np.array(self.feature_names_in_, dtype=object)


class HighVarGeneSelector(BaseEstimator, TransformerMixin):
    """Selecciona columnas (genes) por encima de un cuantil de varianza."""

    def __init__(self, var_quantile: float = 0.2):
        self.var_quantile = var_quantile

    def fit(self, X, y=None):
        X_df = _to_df(X)
        var = X_df.var(axis=0)
        thr = np.quantile(var.to_numpy(), self.var_quantile)
        genes = var[var > thr].index.tolist()
        if not genes:
            raise ValueError(
                "HighVarGeneSelector: no se seleccionó ningún gen. "
                "Revisa var_quantile o la varianza de tus columnas."
            )
        self.genes_ = genes
        return self

    def transform(self, X):
        check_is_fitted(self, "genes_")
        X_df = _to_df(X)
        missing = [g for g in self.genes_ if g not in X_df.columns]
        if missing:
            raise ValueError(f"Faltan columnas en X: {missing[:10]} ...")
        return X_df.loc[:, self.genes_]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "genes_")
        return np.array(self.genes_, dtype=object)


class Log1pTransformer(BaseEstimator, TransformerMixin):
    """Aplica log1p a todas las columnas."""

    def fit(self, X, y=None):
        _ = _to_df(X)
        return self

    def transform(self, X):
        X_df = _to_df(X)
        arr = np.log1p(X_df.to_numpy())
        return pd.DataFrame(arr, index=X_df.index, columns=X_df.columns)

    def get_feature_names_out(self, input_features=None):
        X_df = _to_df(input_features) if isinstance(input_features, pd.DataFrame) else None
        if X_df is not None:
            return np.array(X_df.columns, dtype=object)
        return input_features


class PandasStandardScaler(BaseEstimator, TransformerMixin):
    """StandardScaler que devuelve DataFrame conservando index/columns."""

    def __init__(self, with_mean: bool = True, with_std: bool = True):
        self.with_mean = with_mean
        self.with_std = with_std
        self._scaler = StandardScaler(with_mean=with_mean, with_std=with_std)

    def fit(self, X, y=None):
        X_df = _to_df(X)
        self.feature_names_in_ = list(X_df.columns)
        self._scaler.fit(X_df.to_numpy())
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        X_df = _to_df(X)
        X_df = X_df.loc[:, self.feature_names_in_]
        arr = self._scaler.transform(X_df.to_numpy())
        return pd.DataFrame(arr, index=X_df.index, columns=self.feature_names_in_)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        return np.array(self.feature_names_in_, dtype=object)


class PCAAuto(BaseEstimator, TransformerMixin):
    """
    PCA con selección automática de componentes por varianza explicada acumulada.
    Devuelve DataFrame con columnas PC1..PCk
    """

    def __init__(
        self,
        var_threshold: float = 0.9,
        max_components: Optional[int] = None,
        random_state: int = 42,
    ):
        self.var_threshold = var_threshold
        self.max_components = max_components
        self.random_state = random_state

    def fit(self, X, y=None):
        X_df = _to_df(X)
        self.feature_names_in_ = list(X_df.columns)

        pca_full = PCA(random_state=self.random_state)
        pca_full.fit(X_df.to_numpy())

        cum = np.cumsum(pca_full.explained_variance_ratio_)
        n_comp = int(np.searchsorted(cum, self.var_threshold) + 1)
        if self.max_components is not None:
            n_comp = min(n_comp, self.max_components)

        self.n_components_ = n_comp
        self.pca_ = PCA(n_components=n_comp, random_state=self.random_state)
        self.pca_.fit(X_df.to_numpy())

        return self

    def transform(self, X):
        check_is_fitted(self, "pca_")
        X_df = _to_df(X)
        X_df = X_df.loc[:, self.feature_names_in_]
        arr = self.pca_.transform(X_df.to_numpy())
        cols = [f"PC{i+1}" for i in range(arr.shape[1])]
        return pd.DataFrame(arr, index=X_df.index, columns=cols)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "n_components_")
        return np.array([f"PC{i+1}" for i in range(self.n_components_)], dtype=object)
