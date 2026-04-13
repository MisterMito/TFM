from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Optional

import joblib
import matplotlib.pyplot as plt
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
import yaml

from sklearn.compose import ColumnTransformer

from genomics_dl.features_sklearn import (
    ClinicalFeaturePreprocessor,
    FeatureColumnSelector,
    HighVarGeneSelector,
    Log1pTransformer,
    PandasStandardScaler,
    PCAAuto,
    VarianceThresholdFilter,
)
from genomics_dl.models.heterogeneity import build_supervised_matrix


def _to_serializable(obj):
    """Convierte tipos problemáticos (Path, numpy, etc.) a tipos serializables (str/int/float/list/dict)."""
    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_to_serializable(v) for v in obj]

    return obj


def find_repo_root(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else Path(start)
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    return start


def resolve_under_repo(path_like: str | Path, repo_root: Path) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (repo_root / p)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def build_y_multiclass(
    df: pd.DataFrame,
    class_group_col: str,
    patient_group_col: str,
    nonmalignant_label: str = "nonMalignant",
    malignant_label: str = "Malignant",
) -> np.ndarray:
    """
    Construye etiquetas multiclass:
      - Si Class_group == nonMalignant -> etiqueta única: nonMalignant_label
      - Si Class_group == Malignant    -> etiqueta = Patient_group (tipo de cáncer)
    """
    cg = df[class_group_col].astype(str)
    pg = df[patient_group_col].astype(str)

    y = np.where(cg == str(nonmalignant_label), str(nonmalignant_label), pg)

    unknown_mask = ~cg.isin([str(nonmalignant_label), str(malignant_label)])
    if unknown_mask.any():
        bad = df.loc[unknown_mask, class_group_col].unique().tolist()
        raise ValueError(
            f"Valores inesperados en {class_group_col}: {bad}. "
            f"Esperados: [{nonmalignant_label}, {malignant_label}]"
        )

    # Si es Malignant pero Patient_group viene vacío/"nan", lo marcamos para no perder la muestra.
    bad_pg = (cg == str(malignant_label)) & (pg.isin(["nan", "None", ""]))
    if bad_pg.any():
        y[bad_pg.to_numpy()] = "Malignant_unknown"

    return y.astype(str)


def build_xy(
    df: pd.DataFrame,
    feature_cols: list[str],
    class_group_col: str = "Class_group",
    patient_group_col: str = "Patient_group",
    nonmalignant_label: str = "nonMalignant",
    malignant_label: str = "Malignant",
) -> tuple[pd.DataFrame, np.ndarray]:
    X = df.loc[:, feature_cols].copy()
    y = build_y_multiclass(
        df,
        class_group_col=class_group_col,
        patient_group_col=patient_group_col,
        nonmalignant_label=nonmalignant_label,
        malignant_label=malignant_label,
    )
    return X, y


# ---------------------------------------------------------------------------
# Helpers para soporte de nonMalignant subdividido (clusters)
# ---------------------------------------------------------------------------


def _is_nonmalignant(label: str, nonmalignant_label: str = "nonMalignant") -> bool:
    """Devuelve True si *label* es la clase nonMalignant o un subtipo de cluster."""
    s = str(label)
    return s == nonmalignant_label or s.startswith(nonmalignant_label + "_")


def _is_nonmalignant_mask(y, nonmalignant_label: str = "nonMalignant") -> np.ndarray:
    """Versión vectorizada de ``_is_nonmalignant``."""
    y_str = pd.Series(y).astype(str)
    return (
        (y_str == nonmalignant_label) | y_str.str.startswith(nonmalignant_label + "_")
    ).to_numpy()


def build_y_multiclass_with_clusters(
    df: pd.DataFrame,
    class_group_col: str,
    patient_group_col: str,
    cluster_col: str = "nm_cluster",
    nonmalignant_label: str = "nonMalignant",
    malignant_label: str = "Malignant",
) -> np.ndarray:
    """
    Construye etiquetas multiclass con subtipos de nonMalignant:
      - Class_group == Malignant    → Patient_group (tipo de cáncer)
      - Class_group == nonMalignant → nonMalignant_{cluster}
    """
    y_base = build_y_multiclass(
        df,
        class_group_col=class_group_col,
        patient_group_col=patient_group_col,
        nonmalignant_label=nonmalignant_label,
        malignant_label=malignant_label,
    )

    cluster = df[cluster_col]
    nm_mask = y_base == str(nonmalignant_label)

    y_out = y_base.copy()
    for idx in np.where(nm_mask)[0]:
        cl = cluster.iloc[idx]
        if pd.notna(cl):
            y_out[idx] = f"{nonmalignant_label}_{int(cl)}"
    return y_out


def build_xy_with_clusters(
    df: pd.DataFrame,
    feature_cols: list[str],
    cluster_col: str = "nm_cluster",
    class_group_col: str = "Class_group",
    patient_group_col: str = "Patient_group",
    nonmalignant_label: str = "nonMalignant",
    malignant_label: str = "Malignant",
) -> tuple[pd.DataFrame, np.ndarray]:
    """Equivalente a ``build_xy`` pero usa ``build_y_multiclass_with_clusters``."""
    X = df.loc[:, feature_cols].copy()
    y = build_y_multiclass_with_clusters(
        df,
        class_group_col=class_group_col,
        patient_group_col=patient_group_col,
        cluster_col=cluster_col,
        nonmalignant_label=nonmalignant_label,
        malignant_label=malignant_label,
    )
    return X, y


# ---------------------------------------------------------------------------
# Helpers para soporte de Malignant subdividido (clusters)
# ---------------------------------------------------------------------------

MALIGNANT_CLUSTER_PREFIX = "Malignant"


def _is_malignant_cluster(label: str) -> bool:
    """Devuelve True si *label* es un cluster malignant (``Malignant_0``, ``Malignant_1``, ...)."""
    s = str(label)
    return s.startswith(MALIGNANT_CLUSTER_PREFIX + "_") and s[len(MALIGNANT_CLUSTER_PREFIX) + 1:].isdigit()


def _is_malignant_cluster_mask(y) -> np.ndarray:
    """Versión vectorizada de ``_is_malignant_cluster``."""
    y_str = pd.Series(y).astype(str)
    return y_str.str.match(r"^Malignant_\d+$").to_numpy()


def build_y_multiclass_with_malignant_clusters(
    df: pd.DataFrame,
    class_group_col: str,
    patient_group_col: str,
    cluster_col: str = "mal_cluster",
    nonmalignant_label: str = "nonMalignant",
    malignant_label: str = "Malignant",
) -> np.ndarray:
    """
    Construye etiquetas multiclass con clusters malignos:
      - Class_group == nonMalignant → nonMalignant (sin cambios)
      - Class_group == Malignant    → Malignant_{cluster} (reemplaza Patient_group)
    """
    cg = df[class_group_col].astype(str)
    cluster = df[cluster_col]

    # Validar Class_group
    unknown_mask = ~cg.isin([str(nonmalignant_label), str(malignant_label)])
    if unknown_mask.any():
        bad = df.loc[unknown_mask, class_group_col].unique().tolist()
        raise ValueError(
            f"Valores inesperados en {class_group_col}: {bad}. "
            f"Esperados: [{nonmalignant_label}, {malignant_label}]"
        )

    y = np.full(len(df), str(nonmalignant_label), dtype=object)
    mal_mask = cg == str(malignant_label)
    for idx in np.where(mal_mask.to_numpy())[0]:
        cl = cluster.iloc[idx]
        if pd.notna(cl):
            y[idx] = f"{MALIGNANT_CLUSTER_PREFIX}_{int(cl)}"
        else:
            y[idx] = "Malignant_unknown"

    return y.astype(str)


def build_xy_with_malignant_clusters(
    df: pd.DataFrame,
    feature_cols: list[str],
    cluster_col: str = "mal_cluster",
    class_group_col: str = "Class_group",
    patient_group_col: str = "Patient_group",
    nonmalignant_label: str = "nonMalignant",
    malignant_label: str = "Malignant",
) -> tuple[pd.DataFrame, np.ndarray]:
    """Equivalente a ``build_xy`` pero usa ``build_y_multiclass_with_malignant_clusters``."""
    X = df.loc[:, feature_cols].copy()
    y = build_y_multiclass_with_malignant_clusters(
        df,
        class_group_col=class_group_col,
        patient_group_col=patient_group_col,
        cluster_col=cluster_col,
        nonmalignant_label=nonmalignant_label,
        malignant_label=malignant_label,
    )
    return X, y


# ---------------------------------------------------------------------------
# Enriched X: cluster labels + clinical variables as features
# ---------------------------------------------------------------------------


def build_xy_enriched(
    df: pd.DataFrame,
    gene_cols: list[str],
    class_group_col: str = "Class_group",
    patient_group_col: str = "Patient_group",
    nonmalignant_label: str = "nonMalignant",
    malignant_label: str = "Malignant",
    mal_cluster_col: Optional[str] = None,
    nm_cluster_col: Optional[str] = None,
    age_col: Optional[str] = None,
    sex_col: Optional[str] = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Construye X enriquecido (genes + cluster dummies + Age + Sex) e y original.

    Las etiquetas de cluster se añaden a X (como dummies one-hot), NO a y.
    y mantiene las 19 clases originales (18 cáncer + nonMalignant).

    Parameters
    ----------
    df : DataFrame
        Datos con genes, metadata y columnas de cluster.
    gene_cols : list[str]
        Columnas de expresión génica (ENSG*).
    mal_cluster_col : str or None
        Columna de cluster Malignant (se convierte a dummies).
    nm_cluster_col : str or None
        Columna de cluster nonMalignant (se convierte a dummies).
    age_col, sex_col : str or None
        Columnas clínicas a incluir en X.

    Returns
    -------
    (X_enriched, y) : tuple
        X_enriched contiene genes + dummies de cluster + variables clínicas.
        y contiene las 19 clases originales.
    """
    # y: etiquetas originales (sin modificar por clusters)
    y = build_y_multiclass(
        df,
        class_group_col=class_group_col,
        patient_group_col=patient_group_col,
        nonmalignant_label=nonmalignant_label,
        malignant_label=malignant_label,
    )

    # X: empezamos con genes
    X = df.loc[:, gene_cols].copy()

    # Cluster dummies (NaN → 0 en todas las columnas dummy)
    cluster_cols_to_use: list[str] = []
    for col_name, prefix in [
        (mal_cluster_col, "mal_cluster"),
        (nm_cluster_col, "nm_cluster"),
    ]:
        if col_name is not None and col_name in df.columns:
            cluster_series = df[col_name].copy()
            # One-hot: cada valor único se convierte en una columna
            unique_vals = sorted(
                cluster_series.dropna().unique().astype(int).tolist()
            )
            for val in unique_vals:
                dummy_name = f"{prefix}_{val}"
                X[dummy_name] = (cluster_series == val).astype(float)
                cluster_cols_to_use.append(dummy_name)

    # Variables clínicas (se pasan tal cual; ClinicalFeaturePreprocessor
    # las procesará dentro del pipeline)
    if age_col is not None and age_col in df.columns:
        X[age_col] = df[age_col].astype(float)
    if sex_col is not None and sex_col in df.columns:
        X[sex_col] = df[sex_col].astype(str)

    return X, y


# Config + Pipelines
@dataclass(frozen=True)
class MulticlassTrainConfig:
    train_path: str
    test_path: str

    class_group_col: str = "Class_group"
    patient_group_col: str = "Patient_group"
    nonmalignant_label: str = "nonMalignant"
    malignant_label: str = "Malignant"

    model_name: str = "logreg_multiclass"
    model_version: str = "v0.1.0"

    # Features
    use_pca: bool = False
    var_quantile: float = 0.2
    selector_on_log: bool = False  # si True: log1p antes de seleccionar varianza
    pca_var_threshold: float = 0.9
    max_pca_components: Optional[int] = None

    # Clasificador
    clf_name: str = "logreg"
    clf_params: dict[str, Any] = field(default_factory=dict)

    # Pesos (para priorizar "cáncer vs no cáncer")
    malignant_weight: float = 1.0  # peso para TODAS las clases malignas
    weighting_strategy: str = "sample_weight"  # "sample_weight" (default), "class_weight" o "none"
    class_weight_override: Optional[dict[str, float]] = None

    # CV / threshold (umbral sobre p(cáncer) = 1 - p(nonMalignant))
    cv_splits: int = 5
    min_cancer_recall_for_threshold: Optional[float] = 0.95  # None -> no aplica gating (argmax puro)
    threshold_objective: str = "specificity"  # specificity, balanced_accuracy, highest_threshold

    # MLflow
    experiment_name: str = "gse183635_multiclass"
    tracking_uri: Optional[str] = None  # si None, usa el default de MLflow

    # Outputs
    save_local_bundle: bool = True
    save_plots: bool = True
    mlflow_log_artifacts: bool = True
    mlflow_log_model: bool = True
    output_models_dir: str = "models"
    output_figures_dir: str = "reports/figures/multiclass"

    random_state: int = 42

    # Clusters no supervisados (None = comportamiento original sin clusters)
    cluster_col: Optional[str] = None  # nonMalignant clusters
    malignant_cluster_col: Optional[str] = None  # Malignant clusters

    # Features clínicas y clusters como predictores (X)
    cluster_as_feature: bool = False  # True = cluster en X (no en y)
    nm_cluster_col: Optional[str] = None  # "nm_cluster" como feature
    age_col: Optional[str] = None  # "Age" como feature
    sex_col: Optional[str] = None  # "Sex" como feature
    age_impute_strategy: str = "median"


@dataclass(frozen=True)
class HierarchicalTrainConfig:
    """Configuración para clasificación jerárquica de dos etapas."""

    # Rutas de datos
    train_path: str
    test_path: str

    # Configuración de etiquetas
    class_group_col: str = "Class_group"
    patient_group_col: str = "Patient_group"
    nonmalignant_label: str = "nonMalignant"
    malignant_label: str = "Malignant"

    # Identificación del modelo
    model_name: str = "hierarchical_multiclass"
    model_version: str = "v0.3.0"

    # Etapa 1: Detección binaria de cáncer (cancer vs nonMalignant)
    stage1_clf_name: str = "xgboost"  # xgboost, lightgbm, catboost, extratrees
    stage1_clf_params: dict[str, Any] = field(default_factory=dict)
    stage1_min_recall: float = 0.95  # Objetivo de alto recall
    stage1_threshold_objective: str = "specificity"
    stage1_pos_weight: Optional[float] = None  # Auto-calcular si es None

    # Etapa 2: Clasificación de tipo de cáncer (18 tipos)
    stage2_clf_name: str = "xgboost"
    stage2_clf_params: dict[str, Any] = field(default_factory=dict)
    stage2_class_weighting: str = "balanced"  # balanced, sqrt, log, custom
    stage2_custom_weights: Optional[dict[str, float]] = None
    stage2_use_smote: bool = False  # Experimental: para clases raras

    # Ingeniería de características (compartida)
    use_pca: bool = False
    var_quantile: float = 0.15
    selector_on_log: bool = False
    pca_var_threshold: float = 0.9
    max_pca_components: Optional[int] = None
    variance_filter_threshold: float = 1e-6  # NUEVO: prevenir NaN

    # Validación cruzada
    cv_splits: int = 8
    skip_cv_for_sweep: bool = False  # Si True, usa threshold fijo sin CV (más rápido)
    random_state: int = 42

    # MLflow
    experiment_name: str = "gse183635_hierarchical"
    tracking_uri: Optional[str] = None

    # Salidas
    save_local_bundle: bool = True
    save_plots: bool = True
    mlflow_log_artifacts: bool = True
    mlflow_log_model: bool = True
    output_models_dir: str = "models"
    output_figures_dir: str = "reports/figures/hierarchical"

    # Clusters no supervisados (None = comportamiento original sin clusters)
    cluster_col: Optional[str] = None  # nonMalignant clusters
    malignant_cluster_col: Optional[str] = None  # Malignant clusters

    # Features clínicas y clusters como predictores (X)
    cluster_as_feature: bool = False
    nm_cluster_col: Optional[str] = None
    age_col: Optional[str] = None
    sex_col: Optional[str] = None
    age_impute_strategy: str = "median"


def _build_class_weight(cfg: MulticlassTrainConfig, y_train: np.ndarray) -> Optional[dict[str, float]]:
    """
    Construye class_weight SOLO si weighting_strategy == "class_weight".

    Nota: en algunos wrappers (p.ej. CalibratedClassifierCV) scikit-learn puede codificar y a enteros internamente,
    y un dict con claves string puede provocar ValueError. Por defecto usamos sample_weight para evitarlo.
    """
    if str(cfg.weighting_strategy).lower() != "class_weight":
        return None

    if cfg.class_weight_override is not None:
        # Normalizamos a float
        return {str(k): float(v) for k, v in cfg.class_weight_override.items()}

    classes = pd.Series(y_train).astype(str).unique().tolist()
    if cfg.nonmalignant_label not in classes:
        return None

    cw = {str(cfg.nonmalignant_label): 1.0}
    for c in classes:
        if str(c) != str(cfg.nonmalignant_label):
            cw[str(c)] = float(cfg.malignant_weight)
    return cw


def _build_sample_weight(cfg: MulticlassTrainConfig, y: np.ndarray) -> Optional[np.ndarray]:
    """
    Pesos por muestra (default): 1.0 para nonMalignant; malignant_weight para cualquier clase maligna.
    Devuelve None si weighting_strategy != "sample_weight".
    """
    if str(cfg.weighting_strategy).lower() != "sample_weight":
        return None
    nm_mask = _is_nonmalignant_mask(y, cfg.nonmalignant_label)
    w = np.where(nm_mask, 1.0, float(cfg.malignant_weight)).astype(float)
    return w

def build_classifier(cfg: MulticlassTrainConfig, y_train: np.ndarray):
    cw = _build_class_weight(cfg, y_train)

    if cfg.clf_name == "logreg":
        # Por defecto multinomial (mejor para multiclass) si solver lo soporta
        solver = cfg.clf_params.get("solver", "lbfgs")
        return LogisticRegression(
            C=cfg.clf_params.get("C", 1.0),
            solver=solver,
            class_weight=cw,
            max_iter=cfg.clf_params.get("max_iter", 4000),
            random_state=cfg.random_state,
        )

    if cfg.clf_name == "sgd_logloss":
        return SGDClassifier(
            loss="log_loss",
            alpha=cfg.clf_params.get("alpha", 1e-4),
            penalty=cfg.clf_params.get("penalty", "l2"),
            class_weight=cw,
            max_iter=cfg.clf_params.get("max_iter", 4000),
            tol=cfg.clf_params.get("tol", 1e-3),
            early_stopping=cfg.clf_params.get("early_stopping", True),
            validation_fraction=cfg.clf_params.get("validation_fraction", 0.1),
            n_iter_no_change=cfg.clf_params.get("n_iter_no_change", 5),
            average=cfg.clf_params.get("average", True),
            random_state=cfg.random_state,
        )

    if cfg.clf_name == "linear_svc_calibrated":
        base = LinearSVC(
            C=cfg.clf_params.get("C", 1.0),
            class_weight=cw,
            random_state=cfg.random_state,
        )
        # CalibratedClassifierCV aporta predict_proba
        return CalibratedClassifierCV(base, method="sigmoid", cv=3, n_jobs=-1)

    if cfg.clf_name == "rf":
        return RandomForestClassifier(
            n_estimators=cfg.clf_params.get("n_estimators", 500),
            max_depth=cfg.clf_params.get("max_depth", None),
            class_weight=cw,
            n_jobs=-1,
            random_state=cfg.random_state,
        )

    if cfg.clf_name == "extratrees":
        return ExtraTreesClassifier(
            n_estimators=cfg.clf_params.get("n_estimators", 500),
            max_depth=cfg.clf_params.get("max_depth", None),
            class_weight=cw,
            n_jobs=-1,
            random_state=cfg.random_state,
        )

    raise ValueError(f"clf_name desconocido: {cfg.clf_name}")


def _build_gene_pipeline_steps(
    cfg: MulticlassTrainConfig | HierarchicalTrainConfig,
) -> list[tuple[str, Any]]:
    """Construye los pasos de preprocesamiento para features genómicas."""
    selector = HighVarGeneSelector(var_quantile=cfg.var_quantile)
    log = Log1pTransformer()
    threshold = getattr(cfg, "variance_filter_threshold", 1e-6)
    variance_filter = VarianceThresholdFilter(threshold=threshold)
    scale = PandasStandardScaler()
    pca = PCAAuto(
        var_threshold=cfg.pca_var_threshold,
        max_components=cfg.max_pca_components,
        random_state=cfg.random_state,
    )

    steps: list[tuple[str, Any]] = []
    if cfg.selector_on_log:
        steps.extend([
            ("log1p", log),
            ("select", selector),
            ("variance_filter", variance_filter),
            ("scale", scale),
        ])
    else:
        steps.extend([
            ("select", selector),
            ("log1p", log),
            ("variance_filter", variance_filter),
            ("scale", scale),
        ])
    steps.append(("pca", pca if cfg.use_pca else "passthrough"))
    return steps


def build_pipeline(
    cfg: MulticlassTrainConfig,
    feature_cols: list[str],
    y_train: np.ndarray,
    clinical_cols: Optional[list[str]] = None,
) -> Pipeline:
    """
    Construye pipeline de preprocesamiento + clasificador.

    Si clinical_cols está presente, usa ColumnTransformer con dos ramas:
    - "genes": pipeline genómico (HighVar → Log1p → VarFilter → Scaler → [PCA])
    - "clinical": ClinicalFeaturePreprocessor (Age, Sex, cluster dummies)

    Sin clinical_cols, mantiene el pipeline original (retrocompatible).
    """
    clf = build_classifier(cfg, y_train=y_train)

    if clinical_cols:
        # Separar columnas genómicas de clínicas
        gene_only_cols = [c for c in feature_cols if c not in clinical_cols]

        # Rama genómica
        gene_steps = _build_gene_pipeline_steps(cfg)
        gene_pipe = Pipeline(
            [("ensure_features", FeatureColumnSelector(gene_only_cols))]
            + gene_steps
        )

        # Rama clínica
        cluster_dummy_cols = [
            c for c in clinical_cols
            if c.startswith("mal_cluster_") or c.startswith("nm_cluster_")
        ]
        age_col = cfg.age_col if cfg.age_col in clinical_cols else None
        sex_col = cfg.sex_col if cfg.sex_col in clinical_cols else None

        clinical_prep = ClinicalFeaturePreprocessor(
            age_col=age_col,
            sex_col=sex_col,
            cluster_cols=cluster_dummy_cols,
            age_impute_strategy=cfg.age_impute_strategy,
        )

        col_transformer = ColumnTransformer(
            transformers=[
                ("genes", gene_pipe, gene_only_cols),
                ("clinical", clinical_prep, clinical_cols),
            ],
            remainder="drop",
        )

        steps: list[tuple[str, Any]] = [
            ("features", col_transformer),
            ("clf", clf),
        ]
    else:
        # Pipeline original (retrocompatible)
        gene_steps = _build_gene_pipeline_steps(cfg)
        steps = (
            [("ensure_features", FeatureColumnSelector(feature_cols))]
            + gene_steps
            + [("clf", clf)]
        )

    return Pipeline(steps=steps)


# Threshold + Predicción
def choose_threshold_for_min_recall(
    y_true: np.ndarray,
    y_score: np.ndarray,
    min_recall: Optional[float],
    objective: str = "specificity",
) -> float:
    """
    Igual que en binario: selecciona umbral (sobre un score) maximizando el objetivo con recall >= min_recall.
    objective:
        - 'specificity': maximiza especificidad.
        - 'balanced_accuracy': maximiza balanced accuracy.
        - 'highest_threshold': el umbral más alto que cumple.
    """
    if min_recall is None:
        # no se usa; dejamos un valor que equivale a "siempre cáncer"
        return 0.0

    min_recall = float(min_recall)
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    specificities = 1.0 - fpr
    balanced = 0.5 * (tpr + specificities)

    mask = tpr >= min_recall
    if not np.any(mask):
        # fallback conservador
        return 0.5

    candidate_idx = np.where(mask)[0]

    if objective == "highest_threshold":
        candidate_thresholds = thresholds[candidate_idx]
        best_rel_idx = np.argmax(candidate_thresholds)
    elif objective == "balanced_accuracy":
        candidate_scores = balanced[candidate_idx]
        best_rel_idx = np.argmax(candidate_scores)
    elif objective == "specificity":
        candidate_scores = specificities[candidate_idx]
        best_rel_idx = np.argmax(candidate_scores)
    else:
        raise ValueError(f"Objetivo de threshold desconocido: {objective}")

    best_idx = candidate_idx[best_rel_idx]
    chosen = thresholds[best_idx]
    if not np.isfinite(chosen):
        return 1.0
    return float(chosen)


def _predict_with_cancer_gating(
    proba: np.ndarray,
    classes: np.ndarray,
    cancer_threshold: Optional[float],
    nonmalignant_label: str,
) -> np.ndarray:
    """
    Regla de decisión:
      - p_cancer = 1 - p(nonMalignant)
      - si p_cancer < threshold -> pred = nonMalignant
      - si p_cancer >= threshold -> pred = argmax entre clases malignas
    Si cancer_threshold es None -> argmax clásico (multiclass puro).
    """
    classes = np.asarray(classes, dtype=str)

    if cancer_threshold is None:
        return classes[np.argmax(proba, axis=1)]

    nonmal_idx = np.array(
        [i for i, c in enumerate(classes) if _is_nonmalignant(c, nonmalignant_label)],
        dtype=int,
    )
    if len(nonmal_idx) == 0:
        return classes[np.argmax(proba, axis=1)]

    p_non = proba[:, nonmal_idx].sum(axis=1)
    p_cancer = 1.0 - p_non

    malignant_idx = np.array(
        [i for i, c in enumerate(classes) if not _is_nonmalignant(c, nonmalignant_label)],
        dtype=int,
    )
    malignant_best = malignant_idx[np.argmax(proba[:, malignant_idx], axis=1)]

    # Para muestras clasificadas como nonMalignant, asignar la subclase nonMalignant más probable
    nonmal_best = nonmal_idx[np.argmax(proba[:, nonmal_idx], axis=1)]

    y_pred = np.where(p_cancer >= float(cancer_threshold), classes[malignant_best], classes[nonmal_best])
    return y_pred.astype(str)


def compute_multiclass_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    classes: np.ndarray,
    cancer_threshold: Optional[float],
    nonmalignant_label: str,
) -> dict[str, Any]:
    y_true = y_true.astype(str)
    y_pred = _predict_with_cancer_gating(proba, classes, cancer_threshold, nonmalignant_label)

    # Métricas multiclass estándar
    acc = float(np.mean(y_pred == y_true))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Log-loss (si hay clases ausentes puede fallar; lo defendemos)
    try:
        ll = float(log_loss(y_true, proba, labels=list(classes)))
    except Exception:
        ll = float("nan")

    # Métricas foco clínico: cáncer vs no cáncer
    y_true_cancer = (~_is_nonmalignant_mask(y_true, nonmalignant_label)).astype(int)
    y_pred_cancer = (~_is_nonmalignant_mask(y_pred, nonmalignant_label)).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_cancer, y_pred_cancer, labels=[0, 1]).ravel()
    cancer_recall = float(recall_score(y_true_cancer, y_pred_cancer, zero_division=0))
    cancer_precision = float(precision_score(y_true_cancer, y_pred_cancer, zero_division=0))
    cancer_fnr = float(fn / (fn + tp)) if (fn + tp) else 0.0
    cancer_specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0

    # AUC/PR-AUC en modo cáncer vs no cáncer (usando score p_cancer)
    nonmal_idx = [i for i, c in enumerate(classes) if _is_nonmalignant(c, nonmalignant_label)]
    if len(nonmal_idx) > 0:
        p_cancer = 1.0 - proba[:, nonmal_idx].sum(axis=1)
        try:
            cancer_roc_auc = float(roc_auc_score(y_true_cancer, p_cancer))
        except Exception:
            cancer_roc_auc = float("nan")
        try:
            cancer_pr_auc = float(average_precision_score(y_true_cancer, p_cancer))
        except Exception:
            cancer_pr_auc = float("nan")
    else:
        cancer_roc_auc = float("nan")
        cancer_pr_auc = float("nan")

    # Report por clase (artefacto; no lo logueamos como métricas numéricas en MLflow)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "log_loss": ll,
        "cancer_threshold": (None if cancer_threshold is None else float(cancer_threshold)),
        "cancer_tn": int(tn),
        "cancer_fp": int(fp),
        "cancer_fn": int(fn),
        "cancer_tp": int(tp),
        "cancer_fnr": cancer_fnr,
        "cancer_recall_sensitivity": cancer_recall,
        "cancer_specificity": cancer_specificity,
        "cancer_precision": cancer_precision,
        "cancer_roc_auc": cancer_roc_auc,
        "cancer_pr_auc": cancer_pr_auc,
        "per_class_report": report,
    }


# Plots
def plot_confusion_multiclass(
    y_true: np.ndarray,
    proba: np.ndarray,
    classes: np.ndarray,
    cancer_threshold: Optional[float],
    nonmalignant_label: str,
    outpath: Path,
) -> None:
    y_pred = _predict_with_cancer_gating(proba, classes, cancer_threshold, nonmalignant_label)

    cm = confusion_matrix(y_true, y_pred, labels=list(classes))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion matrix (multiclass)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90)
    ax.set_yticklabels(classes)

    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=7)

    plt.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_pr_cancer(
    y_true: np.ndarray,
    proba: np.ndarray,
    classes: np.ndarray,
    nonmalignant_label: str,
    outpath: Path,
) -> None:
    nonmal_idx = [i for i, c in enumerate(classes) if _is_nonmalignant(c, nonmalignant_label)]
    if len(nonmal_idx) == 0:
        return

    p_cancer = 1.0 - proba[:, nonmal_idx].sum(axis=1)
    y_true_cancer = (~_is_nonmalignant_mask(y_true.astype(str), nonmalignant_label)).astype(int)

    precision, recall, _ = precision_recall_curve(y_true_cancer, p_cancer)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.plot(recall, precision)
    ax.set_xlabel("Recall (cancer)")
    ax.set_ylabel("Precision (cancer)")
    ax.set_title("PR curve: cancer vs nonMalignant")
    plt.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


# Persistencia / Bundle
def _persist_mlflow_local_model(pipe: Pipeline, outdir: Path, example_X: pd.DataFrame) -> None:
    """Guarda un modelo MLflow (formato local) en outdir."""
    tmp = Path(tempfile.mkdtemp())
    try:
        signature = infer_signature(example_X, pipe.predict(example_X))
        mlflow.sklearn.save_model(
            sk_model=pipe,
            path=str(tmp),
            signature=signature,
            input_example=example_X,
        )
        _ensure_dir(outdir)
        # copiamos el contenido entero al destino
        for item in tmp.iterdir():
            dst = outdir / item.name
            if item.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(item, dst)
            else:
                shutil.copy2(item, dst)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def save_model_bundle(
    pipe: Pipeline,
    cfg: MulticlassTrainConfig,
    metrics: dict[str, Any],
    params: dict[str, Any],
    feature_cols: list[str],
    output_dir: Path,
    input_example: pd.DataFrame,
) -> Path:
    """
    Estructura:
    models/
      <model_name>/
        <model_version>/
          model.pkl
          metrics.json
          params.yaml
          signature.json
          input_example.json
          MLmodel, conda.yaml, ...
    """
    model_dir = output_dir / cfg.model_name / cfg.model_version
    _ensure_dir(model_dir)

    # model.pkl (pipeline completo)
    joblib.dump(pipe, model_dir / "model.pkl")

    # metrics.json
    with (model_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(_to_serializable(metrics), f, indent=2)

    # params.yaml
    with (model_dir / "params.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(_to_serializable(params), f, sort_keys=False)

    # signature.json (simple)
    sig = {
        "input_columns": list(input_example.columns),
        "output_type": "multiclass_label",
        "classes": list(getattr(pipe, "classes_", [])),
        "nonmalignant_label": cfg.nonmalignant_label,
        "decision_rule": (
            "argmax"
            if metrics.get("chosen_cancer_threshold") is None
            else "cancer_gating_then_argmax_malignant"
        ),
    }
    with (model_dir / "signature.json").open("w", encoding="utf-8") as f:
        json.dump(_to_serializable(sig), f, indent=2)

    # input_example.json
    input_example.to_json(model_dir / "input_example.json", orient="records", indent=2)

    # MLflow local model (carpeta)
    _persist_mlflow_local_model(pipe, model_dir, input_example)

    return model_dir


# Entrenamiento
def run_training(cfg: MulticlassTrainConfig, feature_cols: list[str]) -> dict[str, Any]:
    if cfg.tracking_uri:
        mlflow.set_tracking_uri(cfg.tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)

    df_train = load_parquet(cfg.train_path)
    df_test = load_parquet(cfg.test_path)

    # --- Construir X e y ---
    clinical_cols: Optional[list[str]] = None

    if cfg.cluster_as_feature:
        # Nuevo modo: clusters + clínicas como features en X, y original
        X_train, y_train = build_xy_enriched(
            df_train, gene_cols=feature_cols,
            class_group_col=cfg.class_group_col,
            patient_group_col=cfg.patient_group_col,
            nonmalignant_label=cfg.nonmalignant_label,
            malignant_label=cfg.malignant_label,
            mal_cluster_col=cfg.malignant_cluster_col,
            nm_cluster_col=cfg.nm_cluster_col,
            age_col=cfg.age_col, sex_col=cfg.sex_col,
        )
        X_test, y_test = build_xy_enriched(
            df_test, gene_cols=feature_cols,
            class_group_col=cfg.class_group_col,
            patient_group_col=cfg.patient_group_col,
            nonmalignant_label=cfg.nonmalignant_label,
            malignant_label=cfg.malignant_label,
            mal_cluster_col=cfg.malignant_cluster_col,
            nm_cluster_col=cfg.nm_cluster_col,
            age_col=cfg.age_col, sex_col=cfg.sex_col,
        )
        # Detectar columnas clínicas (no-ENSG)
        clinical_cols = [c for c in X_train.columns if not str(c).startswith("ENSG")]
        all_feature_cols = list(X_train.columns)
    elif cfg.malignant_cluster_col:
        _build_xy = build_xy_with_malignant_clusters
        _extra_kw: dict[str, Any] = {"cluster_col": cfg.malignant_cluster_col}
        X_train, y_train = _build_xy(
            df_train, feature_cols=feature_cols,
            class_group_col=cfg.class_group_col, patient_group_col=cfg.patient_group_col,
            nonmalignant_label=cfg.nonmalignant_label, malignant_label=cfg.malignant_label,
            **_extra_kw,
        )
        X_test, y_test = _build_xy(
            df_test, feature_cols=feature_cols,
            class_group_col=cfg.class_group_col, patient_group_col=cfg.patient_group_col,
            nonmalignant_label=cfg.nonmalignant_label, malignant_label=cfg.malignant_label,
            **_extra_kw,
        )
        all_feature_cols = feature_cols
    elif cfg.cluster_col:
        _build_xy = build_xy_with_clusters
        _extra_kw = {"cluster_col": cfg.cluster_col}
        X_train, y_train = _build_xy(
            df_train, feature_cols=feature_cols,
            class_group_col=cfg.class_group_col, patient_group_col=cfg.patient_group_col,
            nonmalignant_label=cfg.nonmalignant_label, malignant_label=cfg.malignant_label,
            **_extra_kw,
        )
        X_test, y_test = _build_xy(
            df_test, feature_cols=feature_cols,
            class_group_col=cfg.class_group_col, patient_group_col=cfg.patient_group_col,
            nonmalignant_label=cfg.nonmalignant_label, malignant_label=cfg.malignant_label,
            **_extra_kw,
        )
        all_feature_cols = feature_cols
    else:
        X_train, y_train = build_xy(
            df_train, feature_cols=feature_cols,
            class_group_col=cfg.class_group_col, patient_group_col=cfg.patient_group_col,
            nonmalignant_label=cfg.nonmalignant_label, malignant_label=cfg.malignant_label,
        )
        X_test, y_test = build_xy(
            df_test, feature_cols=feature_cols,
            class_group_col=cfg.class_group_col, patient_group_col=cfg.patient_group_col,
            nonmalignant_label=cfg.nonmalignant_label, malignant_label=cfg.malignant_label,
        )
        all_feature_cols = feature_cols

    pipe = build_pipeline(cfg, feature_cols=all_feature_cols, y_train=y_train, clinical_cols=clinical_cols)

    # Pesos por muestra (para priorizar clases malignas sin depender de class_weight dict)
    sample_weight = _build_sample_weight(cfg, y_train)
    cv_params = {"clf__sample_weight": sample_weight} if sample_weight is not None else None


    cv = StratifiedKFold(n_splits=cfg.cv_splits, shuffle=True, random_state=cfg.random_state)

    repo_root = find_repo_root()
    models_root = resolve_under_repo(cfg.output_models_dir, repo_root)
    figs_root = resolve_under_repo(cfg.output_figures_dir, repo_root)

    run_name = f"{cfg.model_name}_{cfg.model_version}"

    with mlflow.start_run(run_name=run_name) as run:
        # Params (config)
        params = asdict(cfg)
        params.pop("train_path", None)
        params.pop("test_path", None)
        mlflow.log_params({k: _to_serializable(v) for k, v in params.items()})

        # OOF proba (CV)
        oof_proba = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba", params=cv_params, n_jobs=4)
        classes = np.unique(y_train.astype(str))
        if oof_proba.shape[1] != len(classes):
            # fallback: entrenamos una vez para obtener el orden real
            tmp_pipe = build_pipeline(cfg, feature_cols=feature_cols, y_train=y_train)
            tmp_pipe.fit(X_train, y_train)
            classes = np.asarray(getattr(tmp_pipe, "classes_", classes), dtype=str)

        # Threshold en modo cancer vs non
        chosen_thr: Optional[float]
        if cfg.min_cancer_recall_for_threshold is None:
            chosen_thr = None
        else:
            nonmal_idx = [i for i, c in enumerate(classes) if _is_nonmalignant(c, cfg.nonmalignant_label)]
            if len(nonmal_idx) == 0:
                chosen_thr = None
            else:
                p_cancer_oof = 1.0 - oof_proba[:, nonmal_idx].sum(axis=1)
                y_true_cancer = (~_is_nonmalignant_mask(y_train, cfg.nonmalignant_label)).astype(int)
                chosen_thr = choose_threshold_for_min_recall(
                    y_true=y_true_cancer,
                    y_score=p_cancer_oof,
                    min_recall=cfg.min_cancer_recall_for_threshold,
                    objective=cfg.threshold_objective,
                )

        cv_metrics = compute_multiclass_metrics(
            y_true=y_train,
            proba=oof_proba,
            classes=classes,
            cancer_threshold=chosen_thr,
            nonmalignant_label=cfg.nonmalignant_label,
        )

        # Fit final en train
        pipe.fit(X_train, y_train, clf__sample_weight=sample_weight) if sample_weight is not None else pipe.fit(X_train, y_train)

        # Test proba
        test_proba = pipe.predict_proba(X_test)
        classes_fitted = np.asarray(getattr(pipe, "classes_", classes), dtype=str)

        test_metrics = compute_multiclass_metrics(
            y_true=y_test,
            proba=test_proba,
            classes=classes_fitted,
            cancer_threshold=chosen_thr,
            nonmalignant_label=cfg.nonmalignant_label,
        )

        # Logueamos métricas "planas" en MLflow (floats + ints)
        def _flat_metrics(prefix: str, d: dict[str, Any]) -> dict[str, float]:
            out: dict[str, float] = {}
            for k, v in d.items():
                if k == "per_class_report":
                    continue
                if isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(float(v)):
                    out[f"{prefix}{k}"] = float(v)
                elif isinstance(v, (int, np.integer)):
                    out[f"{prefix}{k}"] = float(v)
            return out

        mlflow.log_metrics(_flat_metrics("cv_", cv_metrics))
        mlflow.log_metrics(_flat_metrics("test_", test_metrics))
        mlflow.log_metric("chosen_cancer_threshold", float(chosen_thr) if chosen_thr is not None else -1.0)

        # Artefactos (plots)
        cm_path = figs_root / f"{cfg.model_name}_{cfg.model_version}_cm.png"
        pr_path = figs_root / f"{cfg.model_name}_{cfg.model_version}_pr_cancer.png"

        if cfg.save_plots:
            _ensure_dir(figs_root)
            plot_confusion_multiclass(
                y_true=y_test,
                proba=test_proba,
                classes=classes_fitted,
                cancer_threshold=chosen_thr,
                nonmalignant_label=cfg.nonmalignant_label,
                outpath=cm_path,
            )
            plot_pr_cancer(
                y_true=y_test,
                proba=test_proba,
                classes=classes_fitted,
                nonmalignant_label=cfg.nonmalignant_label,
                outpath=pr_path,
            )

        if cfg.mlflow_log_artifacts and cfg.save_plots:
            mlflow.log_artifact(str(cm_path))
            mlflow.log_artifact(str(pr_path))

        # Log del modelo (MLflow registry/local tracking)
        if cfg.mlflow_log_model:
            example_X = X_train.iloc[:5].copy()
            signature = infer_signature(example_X, pipe.predict(example_X))
            mlflow.sklearn.log_model(
                sk_model=pipe,
                artifact_path="model",
                signature=signature,
                input_example=example_X,
            )

        # Bundle local (models/)
        out_bundle_dir: Optional[Path] = None
        if cfg.save_local_bundle:
            input_example = X_train.iloc[:5].copy()
            metrics_payload = {
                "cv_metrics": cv_metrics,
                "test_metrics": test_metrics,
                "chosen_cancer_threshold": chosen_thr,
            }
            out_bundle_dir = save_model_bundle(
                pipe=pipe,
                cfg=cfg,
                metrics=metrics_payload,
                params=asdict(cfg),
                feature_cols=feature_cols,
                output_dir=models_root,
                input_example=input_example,
            )

        return {
            "mlflow_run_id": run.info.run_id,
            "cv_metrics": cv_metrics,
            "test_metrics": test_metrics,
            "chosen_cancer_threshold": chosen_thr,
            "bundle_dir": (str(out_bundle_dir) if out_bundle_dir is not None else None),
            "classes": list(classes_fitted),
        }


# ============================================================================
# FUNCIONES PARA CLASIFICACIÓN JERÁRQUICA DE DOS ETAPAS
# ============================================================================


def compute_class_weights_hierarchical(
    y: np.ndarray,
    strategy: str = "balanced",
    custom_weights: Optional[dict[str, float]] = None
) -> dict[str, float]:
    """
    Calcula pesos por clase para clasificación multiclase desbalanceada.

    Estrategias:
    - balanced: n_samples / (n_classes * class_count)
    - sqrt: sqrt(max_count / class_count)
    - log: log(1 + max_count / class_count)
    - custom: pesos proporcionados por el usuario
    """
    if strategy == "custom" and custom_weights:
        return {str(k): float(v) for k, v in custom_weights.items()}

    from collections import Counter
    counts = Counter(y)
    n_samples, n_classes = len(y), len(counts)

    if strategy == "balanced":
        return {str(cls): n_samples / (n_classes * count)
                for cls, count in counts.items()}
    elif strategy == "sqrt":
        max_count = max(counts.values())
        return {str(cls): np.sqrt(max_count / count)
                for cls, count in counts.items()}
    elif strategy == "log":
        max_count = max(counts.values())
        return {str(cls): np.log1p(max_count / count)
                for cls, count in counts.items()}
    else:
        raise ValueError(f"Estrategia de pesos desconocida: {strategy}")


def build_stage1_binary_classifier(
    cfg: HierarchicalTrainConfig,
    n_malignant: int,
    n_nonmalignant: int
) -> Any:
    """Construye clasificador binario para detección de cáncer (Etapa 1)."""

    # Auto-calcular peso positivo si no se especifica
    scale_pos_weight = (
        cfg.stage1_pos_weight if cfg.stage1_pos_weight
        else n_nonmalignant / n_malignant
    )

    if cfg.stage1_clf_name == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(
            n_estimators=cfg.stage1_clf_params.get("n_estimators", 500),
            max_depth=cfg.stage1_clf_params.get("max_depth", 6),
            learning_rate=cfg.stage1_clf_params.get("learning_rate", 0.1),
            subsample=cfg.stage1_clf_params.get("subsample", 0.8),
            colsample_bytree=cfg.stage1_clf_params.get("colsample_bytree", 0.8),
            gamma=cfg.stage1_clf_params.get("gamma", 1.0),
            min_child_weight=cfg.stage1_clf_params.get("min_child_weight", 5),
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=cfg.random_state,
            n_jobs=-1,
            tree_method="hist",
        )

    elif cfg.stage1_clf_name == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            n_estimators=cfg.stage1_clf_params.get("n_estimators", 500),
            max_depth=cfg.stage1_clf_params.get("max_depth", 6),
            learning_rate=cfg.stage1_clf_params.get("learning_rate", 0.1),
            subsample=cfg.stage1_clf_params.get("subsample", 0.8),
            colsample_bytree=cfg.stage1_clf_params.get("colsample_bytree", 0.8),
            scale_pos_weight=scale_pos_weight,
            objective="binary",
            random_state=cfg.random_state,
            n_jobs=-1,
            verbose=-1,
        )

    elif cfg.stage1_clf_name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=cfg.stage1_clf_params.get("iterations", 500),
            depth=cfg.stage1_clf_params.get("depth", 6),
            learning_rate=cfg.stage1_clf_params.get("learning_rate", 0.1),
            scale_pos_weight=scale_pos_weight,
            loss_function="Logloss",
            random_state=cfg.random_state,
            thread_count=-1,
            verbose=False,
        )

    elif cfg.stage1_clf_name == "extratrees":
        class_weight = {0: 1.0, 1: scale_pos_weight}
        return ExtraTreesClassifier(
            n_estimators=cfg.stage1_clf_params.get("n_estimators", 800),
            max_depth=cfg.stage1_clf_params.get("max_depth", None),
            class_weight=class_weight,
            n_jobs=-1,
            random_state=cfg.random_state,
        )

    else:
        raise ValueError(f"Nombre de clasificador stage1 desconocido: {cfg.stage1_clf_name}")


class XGBClassifierWithLabelEncoder(BaseEstimator, ClassifierMixin):
    """Wrapper de XGBClassifier que maneja etiquetas string automáticamente."""

    def __init__(self, class_weights: Optional[dict] = None, **xgb_params):
        self.class_weights = class_weights
        self.xgb_params = xgb_params
        self._label_encoder = LabelEncoder()
        self._xgb = None

    def fit(self, X, y, **fit_params):
        import xgboost as xgb
        # Codificar etiquetas string a números
        y_encoded = self._label_encoder.fit_transform(y)
        self.classes_ = self._label_encoder.classes_

        # Calcular sample_weight si hay class_weights
        sample_weight = None
        if self.class_weights is not None:
            sample_weight = np.array([self.class_weights.get(label, 1.0) for label in y])

        self._xgb = xgb.XGBClassifier(**self.xgb_params)
        self._xgb.fit(X, y_encoded, sample_weight=sample_weight, **fit_params)
        return self

    def predict(self, X):
        y_encoded = self._xgb.predict(X)
        return self._label_encoder.inverse_transform(y_encoded)

    def predict_proba(self, X):
        return self._xgb.predict_proba(X)


def build_stage2_multiclass_classifier(
    cfg: HierarchicalTrainConfig,
    y_cancer: np.ndarray
) -> Any:
    """Construye clasificador multiclase para tipos de cáncer (Etapa 2)."""

    class_weights = compute_class_weights_hierarchical(
        y_cancer,
        strategy=cfg.stage2_class_weighting,
        custom_weights=cfg.stage2_custom_weights
    )

    if cfg.stage2_clf_name == "xgboost":
        # Usar wrapper que maneja etiquetas string y class_weights
        return XGBClassifierWithLabelEncoder(
            class_weights=class_weights,
            n_estimators=cfg.stage2_clf_params.get("n_estimators", 500),
            max_depth=cfg.stage2_clf_params.get("max_depth", 8),
            learning_rate=cfg.stage2_clf_params.get("learning_rate", 0.05),
            subsample=cfg.stage2_clf_params.get("subsample", 0.8),
            colsample_bytree=cfg.stage2_clf_params.get("colsample_bytree", 0.8),
            gamma=cfg.stage2_clf_params.get("gamma", 1.0),
            objective="multi:softprob",
            random_state=cfg.random_state,
            n_jobs=-1,
            tree_method="hist",
        )

    elif cfg.stage2_clf_name == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            n_estimators=cfg.stage2_clf_params.get("n_estimators", 500),
            max_depth=cfg.stage2_clf_params.get("max_depth", 8),
            learning_rate=cfg.stage2_clf_params.get("learning_rate", 0.05),
            class_weight=class_weights,  # LightGBM soporta dict
            objective="multiclass",
            random_state=cfg.random_state,
            n_jobs=-1,
            verbose=-1,
        )

    elif cfg.stage2_clf_name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=cfg.stage2_clf_params.get("iterations", 500),
            depth=cfg.stage2_clf_params.get("depth", 8),
            learning_rate=cfg.stage2_clf_params.get("learning_rate", 0.05),
            loss_function="MultiClass",
            random_state=cfg.random_state,
            thread_count=-1,
            verbose=False,
        )

    elif cfg.stage2_clf_name == "extratrees":
        # class_weights ya tiene claves string (nombres de clases)
        # sklearn ExtraTrees acepta dict con etiquetas como claves directamente
        return ExtraTreesClassifier(
            n_estimators=cfg.stage2_clf_params.get("n_estimators", 1000),
            max_depth=cfg.stage2_clf_params.get("max_depth", None),
            class_weight=class_weights,  # Usar directamente, sin conversión
            n_jobs=-1,
            random_state=cfg.random_state,
        )

    else:
        raise ValueError(f"Nombre de clasificador stage2 desconocido: {cfg.stage2_clf_name}")


class HierarchicalClassifier(BaseEstimator):
    """
    Clasificador jerárquico de dos etapas.
    Etapa 1: Detección binaria de cáncer (cancer vs nonMalignant)
    Etapa 2: Clasificación multiclase de tipo de cáncer (18 tipos)

    Compatible con la API de scikit-learn.
    """

    def __init__(
        self,
        stage1_pipeline: Pipeline,
        stage2_pipeline: Pipeline,
        stage1_threshold: Optional[float] = None,
        nonmalignant_label: str = "nonMalignant",
    ):
        self.stage1_pipeline = stage1_pipeline
        self.stage2_pipeline = stage2_pipeline
        self.stage1_threshold = stage1_threshold
        self.nonmalignant_label = nonmalignant_label

    def fit(self, X, y, **fit_params):
        """Entrena ambas etapas."""
        # Etapa 1: Binario (cancer vs nonMalignant)
        nm_mask = _is_nonmalignant_mask(y, self.nonmalignant_label)
        y_binary = (~nm_mask).astype(int)

        stage1_fit_params = {
            k.replace("stage1__", ""): v
            for k, v in fit_params.items()
            if k.startswith("stage1__")
        }
        self.stage1_pipeline.fit(X, y_binary, **stage1_fit_params)

        # Etapa 2: Multiclase (solo tipos de cáncer)
        cancer_mask = ~nm_mask
        X_cancer = X[cancer_mask]
        y_cancer = y[cancer_mask]

        if len(X_cancer) == 0:
            raise ValueError("No hay muestras de cáncer para entrenar la etapa 2")

        stage2_fit_params = {
            k.replace("stage2__", ""): (v[cancer_mask] if hasattr(v, "__getitem__") else v)
            for k, v in fit_params.items()
            if k.startswith("stage2__")
        }
        self.stage2_pipeline.fit(X_cancer, y_cancer, **stage2_fit_params)

        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        """Predicción de probabilidades en dos etapas."""
        # Etapa 1: [p(nonMalignant), p(cancer)]
        stage1_proba = self.stage1_pipeline.predict_proba(X)
        p_cancer = stage1_proba[:, 1]
        p_non = stage1_proba[:, 0]

        # Etapa 2: probabilidades sobre tipos de cáncer
        stage2_proba = self.stage2_pipeline.predict_proba(X)
        stage2_classes = self.stage2_pipeline.classes_

        # Construir matriz de probabilidades final
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        final_proba = np.zeros((n_samples, n_classes))

        # Distribuir p(nonMalignant) entre subclases nonMalignant (si hay clusters)
        nonmal_indices = [
            i for i, c in enumerate(self.classes_)
            if _is_nonmalignant(c, self.nonmalignant_label)
        ]
        if len(nonmal_indices) == 1:
            final_proba[:, nonmal_indices[0]] = p_non
        else:
            # Repartir uniformemente entre subclusters nonMalignant
            for idx in nonmal_indices:
                final_proba[:, idx] = p_non / len(nonmal_indices)

        # Distribuir probabilidad de cáncer entre tipos
        for i, cancer_type in enumerate(stage2_classes):
            cancer_type_idx = np.where(self.classes_ == cancer_type)[0][0]
            final_proba[:, cancer_type_idx] = p_cancer * stage2_proba[:, i]

        # Normalizar (estabilidad numérica)
        row_sums = final_proba.sum(axis=1, keepdims=True)
        final_proba = final_proba / (row_sums + 1e-10)

        return final_proba

    def predict(self, X):
        """Predicción en dos etapas con umbral opcional."""
        if self.stage1_threshold is None:
            proba = self.predict_proba(X)
            return self.classes_[np.argmax(proba, axis=1)]

        # Predicción basada en umbral
        stage1_proba = self.stage1_pipeline.predict_proba(X)
        p_cancer = stage1_proba[:, 1]

        # Para nonMalignant, asignar la subclase más probable del proba completo
        nonmal_classes = [c for c in self.classes_ if _is_nonmalignant(c, self.nonmalignant_label)]
        default_nonmal = nonmal_classes[0] if len(nonmal_classes) == 1 else self.nonmalignant_label
        predictions = np.full(len(X), default_nonmal, dtype=object)

        # Si hay subclusters, asignar la subclase más probable
        if len(nonmal_classes) > 1:
            full_proba = self.predict_proba(X)
            nonmal_idx = [i for i, c in enumerate(self.classes_) if _is_nonmalignant(c, self.nonmalignant_label)]
            best_nonmal = np.array(nonmal_idx)[np.argmax(full_proba[:, nonmal_idx], axis=1)]
            for i in range(len(X)):
                predictions[i] = self.classes_[best_nonmal[i]]

        cancer_mask = p_cancer >= self.stage1_threshold
        if cancer_mask.any():
            X_cancer = X[cancer_mask]
            stage2_pred = self.stage2_pipeline.predict(X_cancer)
            predictions[cancer_mask] = stage2_pred

        return predictions

    def set_stage1_threshold(self, threshold: float):
        """Actualiza el umbral de detección binaria de cáncer."""
        self.stage1_threshold = threshold


# ============================================================================
# FUNCIONES PARA CLASIFICACIÓN JERÁRQUICA CON INTEGRACIÓN MLFLOW
# ============================================================================


def build_hierarchical_pipeline(
    cfg: HierarchicalTrainConfig,
    feature_cols: list[str],
    clf: Any,
    clinical_cols: Optional[list[str]] = None,
) -> Pipeline:
    """
    Construye pipeline de preprocesamiento para clasificación jerárquica.

    Si clinical_cols está presente, usa ColumnTransformer con dos ramas
    (genes y clínicas). Sin clinical_cols, mantiene el pipeline original.

    Args:
        cfg: Configuración jerárquica
        feature_cols: Lista de columnas (genes + clínicas si aplica)
        clf: Clasificador (etapa 1 o etapa 2)
        clinical_cols: Columnas clínicas/cluster (None = pipeline original)

    Returns:
        Pipeline de sklearn con preprocesamiento + clasificador
    """
    if clinical_cols:
        gene_only_cols = [c for c in feature_cols if c not in clinical_cols]

        gene_steps = _build_gene_pipeline_steps(cfg)
        gene_pipe = Pipeline(
            [("ensure_features", FeatureColumnSelector(gene_only_cols))]
            + gene_steps
        )

        cluster_dummy_cols = [
            c for c in clinical_cols
            if c.startswith("mal_cluster_") or c.startswith("nm_cluster_")
        ]
        age_col = cfg.age_col if cfg.age_col in clinical_cols else None
        sex_col = cfg.sex_col if cfg.sex_col in clinical_cols else None

        clinical_prep = ClinicalFeaturePreprocessor(
            age_col=age_col,
            sex_col=sex_col,
            cluster_cols=cluster_dummy_cols,
            age_impute_strategy=cfg.age_impute_strategy,
        )

        col_transformer = ColumnTransformer(
            transformers=[
                ("genes", gene_pipe, gene_only_cols),
                ("clinical", clinical_prep, clinical_cols),
            ],
            remainder="drop",
        )

        steps: list[tuple[str, Any]] = [
            ("features", col_transformer),
            ("clf", clf),
        ]
    else:
        gene_steps = _build_gene_pipeline_steps(cfg)
        steps = (
            [("ensure_features", FeatureColumnSelector(feature_cols))]
            + gene_steps
            + [("clf", clf)]
        )

    return Pipeline(steps=steps)


def compute_hierarchical_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
    nonmalignant_label: str,
    stage1_threshold: float,
) -> dict[str, Any]:
    """
    Calcula métricas para clasificación jerárquica.
    Incluye métricas por etapa y métricas globales.

    Args:
        y_true: Etiquetas verdaderas
        y_pred: Predicciones
        y_proba: Probabilidades predichas (n_samples, n_classes)
        classes: Array de clases
        nonmalignant_label: Etiqueta de no-maligno
        stage1_threshold: Umbral usado en etapa 1

    Returns:
        Diccionario con métricas CV y test
    """
    # Métricas globales multiclase
    acc = float(np.mean(y_pred == y_true))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Métricas binarias (cáncer vs nonMalignant)
    y_true_binary = (~_is_nonmalignant_mask(y_true, nonmalignant_label)).astype(int)
    y_pred_binary = (~_is_nonmalignant_mask(y_pred, nonmalignant_label)).astype(int)

    # Encontrar índices de nonMalignant para calcular p(cancer)
    nonmal_idx = [i for i, c in enumerate(classes) if _is_nonmalignant(c, nonmalignant_label)]
    p_cancer = 1.0 - y_proba[:, nonmal_idx].sum(axis=1)

    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1]).ravel()

    cancer_recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    cancer_precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    cancer_specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    cancer_fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    # AUC si hay varianza en las etiquetas
    try:
        cancer_roc_auc = float(roc_auc_score(y_true_binary, p_cancer))
        cancer_pr_auc = float(average_precision_score(y_true_binary, p_cancer))
    except ValueError:
        cancer_roc_auc = float("nan")
        cancer_pr_auc = float("nan")

    # Reporte por clase
    per_class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "cancer_tn": int(tn),
        "cancer_fp": int(fp),
        "cancer_fn": int(fn),
        "cancer_tp": int(tp),
        "cancer_fnr": cancer_fnr,
        "cancer_recall_sensitivity": cancer_recall,
        "cancer_specificity": cancer_specificity,
        "cancer_precision": cancer_precision,
        "cancer_roc_auc": cancer_roc_auc,
        "cancer_pr_auc": cancer_pr_auc,
        "stage1_threshold": float(stage1_threshold),
        "per_class_report": per_class_report,
    }


def plot_hierarchical_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: np.ndarray,
    title: str,
    outpath: Path,
) -> None:
    """
    Genera matriz de confusión para clasificación jerárquica.

    Args:
        y_true: Etiquetas verdaderas
        y_pred: Predicciones
        classes: Array de clases
        title: Título del gráfico
        outpath: Ruta de salida para guardar la imagen
    """
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="Etiqueta Real",
        xlabel="Predicción",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Añadir valores en cada celda
    thresh = cm.max() / 2.0
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_hierarchical_bundle(
    model: HierarchicalClassifier,
    cfg: HierarchicalTrainConfig,
    metrics: dict[str, Any],
    params: dict[str, Any],
    feature_cols: list[str],
    output_dir: Path,
    input_example: pd.DataFrame,
) -> Path:
    """
    Guarda el modelo jerárquico en formato bundle.
    Estructura similar a save_model_bundle pero para dos etapas.

    Args:
        model: Clasificador jerárquico entrenado
        cfg: Configuración jerárquica
        metrics: Diccionario de métricas
        params: Diccionario de parámetros
        feature_cols: Lista de columnas de genes
        output_dir: Directorio base de salida
        input_example: Ejemplo de entrada para signature

    Returns:
        Path del directorio del bundle guardado
    """
    model_dir = output_dir / cfg.model_name / cfg.model_version
    _ensure_dir(model_dir)

    # Modelo completo (pickle)
    joblib.dump(model, model_dir / "model.pkl")

    # Pipelines individuales (para debugging)
    joblib.dump(model.stage1_pipeline, model_dir / "stage1_pipeline.pkl")
    joblib.dump(model.stage2_pipeline, model_dir / "stage2_pipeline.pkl")

    # Métricas
    with (model_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(_to_serializable(metrics), f, indent=2)

    # Parámetros
    with (model_dir / "params.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(_to_serializable(params), f, sort_keys=False)

    # Signature (metadata)
    sig = {
        "input_columns": list(input_example.columns),
        "output_type": "hierarchical_multiclass",
        "classes": list(model.classes_) if hasattr(model, "classes_") else [],
        "nonmalignant_label": cfg.nonmalignant_label,
        "decision_rule": "stage1_gating_then_stage2_argmax",
        "stage1_clf": cfg.stage1_clf_name,
        "stage2_clf": cfg.stage2_clf_name,
        "stage1_threshold": model.stage1_threshold,
    }
    with (model_dir / "signature.json").open("w", encoding="utf-8") as f:
        json.dump(_to_serializable(sig), f, indent=2)

    # Input example
    input_example.to_json(model_dir / "input_example.json", orient="records", indent=2)

    # Requerimientos (para reproducibilidad)
    requirements = [
        "scikit-learn>=1.0.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.1.0",
        "catboost>=1.2.0",
    ]
    with (model_dir / "requirements.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(requirements))

    # MLflow local model format
    _persist_mlflow_local_model(model, model_dir, input_example)

    return model_dir


def run_hierarchical_training(
    cfg: HierarchicalTrainConfig,
    feature_cols: list[str],
) -> dict[str, Any]:
    """
    Entrena y evalúa clasificador jerárquico de dos etapas.
    Integración completa con MLflow (igual que run_training).

    Args:
        cfg: Configuración jerárquica
        feature_cols: Lista de columnas de genes

    Returns:
        Diccionario con métricas, run_id de MLflow, y ruta del bundle
    """
    # Configurar MLflow
    if cfg.tracking_uri:
        mlflow.set_tracking_uri(cfg.tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)

    # Cargar datos
    df_train = load_parquet(cfg.train_path)
    df_test = load_parquet(cfg.test_path)

    # --- Construir X e y ---
    clinical_cols: Optional[list[str]] = None

    if cfg.cluster_as_feature:
        X_train, y_train = build_xy_enriched(
            df_train, gene_cols=feature_cols,
            class_group_col=cfg.class_group_col,
            patient_group_col=cfg.patient_group_col,
            nonmalignant_label=cfg.nonmalignant_label,
            malignant_label=cfg.malignant_label,
            mal_cluster_col=cfg.malignant_cluster_col,
            nm_cluster_col=cfg.nm_cluster_col,
            age_col=cfg.age_col, sex_col=cfg.sex_col,
        )
        X_test, y_test = build_xy_enriched(
            df_test, gene_cols=feature_cols,
            class_group_col=cfg.class_group_col,
            patient_group_col=cfg.patient_group_col,
            nonmalignant_label=cfg.nonmalignant_label,
            malignant_label=cfg.malignant_label,
            mal_cluster_col=cfg.malignant_cluster_col,
            nm_cluster_col=cfg.nm_cluster_col,
            age_col=cfg.age_col, sex_col=cfg.sex_col,
        )
        clinical_cols = [c for c in X_train.columns if not str(c).startswith("ENSG")]
        all_feature_cols = list(X_train.columns)
    elif cfg.malignant_cluster_col:
        _build_xy = build_xy_with_malignant_clusters
        _extra_kw: dict[str, Any] = {"cluster_col": cfg.malignant_cluster_col}
        X_train, y_train = _build_xy(
            df_train, feature_cols=feature_cols,
            class_group_col=cfg.class_group_col, patient_group_col=cfg.patient_group_col,
            nonmalignant_label=cfg.nonmalignant_label, malignant_label=cfg.malignant_label,
            **_extra_kw,
        )
        X_test, y_test = _build_xy(
            df_test, feature_cols=feature_cols,
            class_group_col=cfg.class_group_col, patient_group_col=cfg.patient_group_col,
            nonmalignant_label=cfg.nonmalignant_label, malignant_label=cfg.malignant_label,
            **_extra_kw,
        )
        all_feature_cols = feature_cols
    elif cfg.cluster_col:
        _build_xy = build_xy_with_clusters
        _extra_kw = {"cluster_col": cfg.cluster_col}
        X_train, y_train = _build_xy(
            df_train, feature_cols=feature_cols,
            class_group_col=cfg.class_group_col, patient_group_col=cfg.patient_group_col,
            nonmalignant_label=cfg.nonmalignant_label, malignant_label=cfg.malignant_label,
            **_extra_kw,
        )
        X_test, y_test = _build_xy(
            df_test, feature_cols=feature_cols,
            class_group_col=cfg.class_group_col, patient_group_col=cfg.patient_group_col,
            nonmalignant_label=cfg.nonmalignant_label, malignant_label=cfg.malignant_label,
            **_extra_kw,
        )
        all_feature_cols = feature_cols
    else:
        X_train, y_train = build_xy(
            df_train, feature_cols=feature_cols,
            class_group_col=cfg.class_group_col, patient_group_col=cfg.patient_group_col,
            nonmalignant_label=cfg.nonmalignant_label, malignant_label=cfg.malignant_label,
        )
        X_test, y_test = build_xy(
            df_test, feature_cols=feature_cols,
            class_group_col=cfg.class_group_col, patient_group_col=cfg.patient_group_col,
            nonmalignant_label=cfg.nonmalignant_label, malignant_label=cfg.malignant_label,
        )
        all_feature_cols = feature_cols

    # Estadísticas de clases para etapa 1 (binario)
    nm_mask_train = _is_nonmalignant_mask(y_train, cfg.nonmalignant_label)
    y_binary_train = (~nm_mask_train).astype(int)
    n_malignant = int(y_binary_train.sum())
    n_nonmalignant = len(y_binary_train) - n_malignant

    # Máscara y etiquetas para etapa 2 (solo cáncer)
    cancer_mask_train = ~nm_mask_train
    y_cancer_train = y_train[cancer_mask_train]

    # Rutas de salida
    repo_root = find_repo_root()
    models_root = resolve_under_repo(cfg.output_models_dir, repo_root)
    figs_root = resolve_under_repo(cfg.output_figures_dir, repo_root)

    run_name = f"{cfg.model_name}_{cfg.model_version}"

    with mlflow.start_run(run_name=run_name) as run:
        # ===== LOG PARAMS =====
        params = asdict(cfg)
        params.pop("train_path", None)
        params.pop("test_path", None)
        mlflow.log_params({k: _to_serializable(v) for k, v in params.items()})

        # ===== CONSTRUIR CLASIFICADORES =====
        stage1_clf = build_stage1_binary_classifier(cfg, n_malignant, n_nonmalignant)
        stage1_pipeline = build_hierarchical_pipeline(
            cfg, all_feature_cols, stage1_clf, clinical_cols=clinical_cols,
        )

        stage2_clf = build_stage2_multiclass_classifier(cfg, y_cancer_train)
        stage2_pipeline = build_hierarchical_pipeline(
            cfg, all_feature_cols, stage2_clf, clinical_cols=clinical_cols,
        )

        # ===== SELECCIÓN DE THRESHOLD ETAPA 1 =====
        if cfg.skip_cv_for_sweep:
            # Modo rápido: usar threshold por defecto sin CV (para sweeps exploratorios)
            chosen_thr = 0.5
        else:
            # Modo completo: cross-validation para optimizar threshold
            cv = StratifiedKFold(n_splits=cfg.cv_splits, shuffle=True, random_state=cfg.random_state)
            oof_stage1_proba = cross_val_predict(
                stage1_pipeline, X_train, y_binary_train,
                cv=cv, method="predict_proba", n_jobs=4
            )
            p_cancer_oof = oof_stage1_proba[:, 1]
            chosen_thr = choose_threshold_for_min_recall(
                y_true=y_binary_train,
                y_score=p_cancer_oof,
                min_recall=cfg.stage1_min_recall,
                objective=cfg.stage1_threshold_objective,
            )

        # ===== ENTRENAR MODELOS FINALES =====
        stage1_pipeline.fit(X_train, y_binary_train)
        stage2_pipeline.fit(X_train[cancer_mask_train], y_cancer_train)

        # Crear modelo jerárquico
        hierarchical_model = HierarchicalClassifier(
            stage1_pipeline=stage1_pipeline,
            stage2_pipeline=stage2_pipeline,
            stage1_threshold=chosen_thr,
            nonmalignant_label=cfg.nonmalignant_label,
        )
        hierarchical_model.classes_ = np.unique(y_train)

        # ===== PREDICCIONES TRAIN Y TEST =====
        # Train (usando modelo entrenado con todo el train)
        train_proba = hierarchical_model.predict_proba(X_train)
        train_pred = hierarchical_model.predict(X_train)

        # Test
        test_proba = hierarchical_model.predict_proba(X_test)
        test_pred = hierarchical_model.predict(X_test)

        # ===== CALCULAR MÉTRICAS =====
        train_metrics = compute_hierarchical_metrics(
            y_true=y_train,
            y_pred=train_pred,
            y_proba=train_proba,
            classes=hierarchical_model.classes_,
            nonmalignant_label=cfg.nonmalignant_label,
            stage1_threshold=chosen_thr,
        )

        test_metrics = compute_hierarchical_metrics(
            y_true=y_test,
            y_pred=test_pred,
            y_proba=test_proba,
            classes=hierarchical_model.classes_,
            nonmalignant_label=cfg.nonmalignant_label,
            stage1_threshold=chosen_thr,
        )

        # ===== LOG MÉTRICAS A MLFLOW =====
        def _flat_metrics(prefix: str, d: dict[str, Any]) -> dict[str, float]:
            out: dict[str, float] = {}
            for k, v in d.items():
                if k == "per_class_report":
                    continue
                if isinstance(v, (int, float, np.integer, np.floating)):
                    if np.isfinite(float(v)):
                        out[f"{prefix}{k}"] = float(v)
            return out

        mlflow.log_metrics(_flat_metrics("train_", train_metrics))
        mlflow.log_metrics(_flat_metrics("test_", test_metrics))
        mlflow.log_metric("chosen_cancer_threshold", float(chosen_thr))

        # ===== PLOTS =====
        out_bundle_dir: Optional[Path] = None

        if cfg.save_plots:
            _ensure_dir(figs_root)

            # Matriz de confusión
            cm_path = figs_root / f"{cfg.model_name}_{cfg.model_version}_cm.png"
            plot_hierarchical_confusion(
                y_true=y_test,
                y_pred=test_pred,
                classes=hierarchical_model.classes_,
                title=f"Matriz de Confusión - {cfg.model_name} {cfg.model_version}",
                outpath=cm_path,
            )

            # PR curve (cáncer vs nonMalignant)
            pr_path = figs_root / f"{cfg.model_name}_{cfg.model_version}_pr_cancer.png"
            plot_pr_cancer(
                y_true=y_test,
                proba=test_proba,
                classes=hierarchical_model.classes_,
                nonmalignant_label=cfg.nonmalignant_label,
                outpath=pr_path,
            )

            if cfg.mlflow_log_artifacts:
                mlflow.log_artifact(str(cm_path))
                mlflow.log_artifact(str(pr_path))

        # ===== GUARDAR BUNDLE =====
        if cfg.save_local_bundle:
            out_bundle_dir = save_hierarchical_bundle(
                model=hierarchical_model,
                cfg=cfg,
                metrics={"train": train_metrics, "test": test_metrics, "threshold": chosen_thr},
                params=asdict(cfg),
                feature_cols=feature_cols,
                output_dir=models_root,
                input_example=X_train.iloc[:5].copy(),
            )

        # ===== LOG MODELO A MLFLOW =====
        if cfg.mlflow_log_model:
            example_X = X_train.iloc[:5].copy()
            signature = infer_signature(example_X, hierarchical_model.predict(example_X))
            mlflow.sklearn.log_model(
                sk_model=hierarchical_model,
                artifact_path="model",
                signature=signature,
                input_example=example_X,
            )

        return {
            "mlflow_run_id": run.info.run_id,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "chosen_cancer_threshold": chosen_thr,
            "bundle_dir": str(out_bundle_dir) if out_bundle_dir else None,
            "classes": list(hierarchical_model.classes_),
        }
