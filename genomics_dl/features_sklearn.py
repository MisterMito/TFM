from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

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


class VarianceThresholdFilter(BaseEstimator, TransformerMixin):
    """
    Removes features with variance below a threshold.
    Prevents NaN in StandardScaler when features become constant after transformations.

    Parameters
    ----------
    threshold : float, default=1e-6
        Features with variance below this value are removed.
    """

    def __init__(self, threshold: float = 1e-6):
        self.threshold = threshold

    def fit(self, X, y=None):
        X_df = _to_df(X)
        var = X_df.var(axis=0)
        self.kept_features_ = var[var >= self.threshold].index.tolist()

        if not self.kept_features_:
            raise ValueError(
                f"VarianceThresholdFilter: All features have variance < {self.threshold}. "
                "Consider lowering the threshold."
            )

        removed_count = len(X_df.columns) - len(self.kept_features_)
        if removed_count > 0:
            # Features with insufficient variance were removed
            pass

        return self

    def transform(self, X):
        check_is_fitted(self, "kept_features_")
        X_df = _to_df(X)
        missing = [f for f in self.kept_features_ if f not in X_df.columns]
        if missing:
            raise ValueError(
                f"VarianceThresholdFilter: Missing features in X: {missing[:10]} ..."
            )
        return X_df.loc[:, self.kept_features_]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "kept_features_")
        return np.array(self.kept_features_, dtype=object)


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


class ClinicalFeaturePreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocesa features clínicas (Age, Sex) y cluster dummies dentro del pipeline.

    Aprende estadísticos del conjunto de entrenamiento (fit) y los aplica
    en transform, evitando data leakage.

    Parameters
    ----------
    age_col : str or None
        Nombre de la columna de edad. None para no incluir.
    sex_col : str or None
        Nombre de la columna de sexo. None para no incluir.
    cluster_cols : sequence of str or None
        Nombres de columnas de cluster (ya one-hot encoded). Se pasan tal cual.
    age_impute_strategy : str
        Estrategia de imputación de Age: "median" o "mean".
    sex_missing_value : str
        Valor que indica dato faltante en Sex (default "n.a.").
    """

    def __init__(
        self,
        age_col: Optional[str] = None,
        sex_col: Optional[str] = None,
        cluster_cols: Optional[Sequence[str]] = None,
        age_impute_strategy: str = "median",
        sex_missing_value: str = "n.a.",
    ):
        self.age_col = age_col
        self.sex_col = sex_col
        self.cluster_cols = cluster_cols
        self.age_impute_strategy = age_impute_strategy
        self.sex_missing_value = sex_missing_value

    def fit(self, X, y=None):
        X_df = _to_df(X)

        # --- Age ---
        if self.age_col is not None:
            age = X_df[self.age_col].astype(float)
            if self.age_impute_strategy == "median":
                self.age_fill_ = float(age.median())
            elif self.age_impute_strategy == "mean":
                self.age_fill_ = float(age.mean())
            else:
                raise ValueError(
                    f"age_impute_strategy desconocido: {self.age_impute_strategy}"
                )
            age_filled = age.fillna(self.age_fill_)
            self.age_mean_ = float(age_filled.mean())
            self.age_std_ = float(age_filled.std())
            if self.age_std_ < 1e-8:
                self.age_std_ = 1.0

        # --- Sex ---
        if self.sex_col is not None:
            sex = X_df[self.sex_col].astype(str)
            valid = sex[sex != self.sex_missing_value]
            self.sex_mode_ = str(valid.mode().iloc[0])
            self.sex_categories_ = sorted(valid.unique().tolist())

        # --- Cluster dummies ---
        if self.cluster_cols is not None:
            self.cluster_cols_ = list(self.cluster_cols)
        else:
            self.cluster_cols_ = []

        # Build output column list
        self.output_cols_: list[str] = []
        if self.age_col is not None:
            self.output_cols_.append("Age_scaled")
        if self.sex_col is not None:
            self.output_cols_.extend(
                [f"Sex_{cat}" for cat in self.sex_categories_]
            )
        self.output_cols_.extend(self.cluster_cols_)

        return self

    def transform(self, X):
        check_is_fitted(self, "output_cols_")
        X_df = _to_df(X)
        parts: list[pd.Series | pd.DataFrame] = []

        # --- Age ---
        if self.age_col is not None:
            age = X_df[self.age_col].astype(float).fillna(self.age_fill_)
            age_scaled = (age - self.age_mean_) / self.age_std_
            parts.append(age_scaled.rename("Age_scaled"))

        # --- Sex ---
        if self.sex_col is not None:
            sex = X_df[self.sex_col].astype(str).replace(
                self.sex_missing_value, self.sex_mode_
            )
            dummies = pd.DataFrame(0, index=X_df.index,
                                   columns=[f"Sex_{c}" for c in self.sex_categories_])
            for cat in self.sex_categories_:
                dummies[f"Sex_{cat}"] = (sex == cat).astype(int)
            parts.append(dummies)

        # --- Cluster dummies ---
        if self.cluster_cols_:
            cluster_df = X_df[self.cluster_cols_].fillna(0).astype(float)
            parts.append(cluster_df)

        if not parts:
            return pd.DataFrame(index=X_df.index)

        result = pd.concat(parts, axis=1)
        return result

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "output_cols_")
        return np.array(self.output_cols_, dtype=object)
