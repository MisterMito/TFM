from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from genomics_dl.features_sklearn import (
    FeatureColumnSelector,
    HighVarGeneSelector,
    Log1pTransformer,
    PandasStandardScaler,
    PCAAuto,
)


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
    y_str = pd.Series(y).astype(str).to_numpy()
    w = np.where(y_str == str(cfg.nonmalignant_label), 1.0, float(cfg.malignant_weight)).astype(float)
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


def build_pipeline(cfg: MulticlassTrainConfig, feature_cols: list[str], y_train: np.ndarray) -> Pipeline:
    clf = build_classifier(cfg, y_train=y_train)

    selector = HighVarGeneSelector(var_quantile=cfg.var_quantile)
    log = Log1pTransformer()
    scale = PandasStandardScaler()
    pca = PCAAuto(
        var_threshold=cfg.pca_var_threshold,
        max_components=cfg.max_pca_components,
        random_state=cfg.random_state,
    )

    steps: list[tuple[str, Any]] = [("ensure_features", FeatureColumnSelector(feature_cols))]
    if cfg.selector_on_log:
        steps.extend([("log1p", log), ("select", selector), ("scale", scale)])
    else:
        steps.extend([("select", selector), ("log1p", log), ("scale", scale)])

    steps.append(("pca", pca if cfg.use_pca else "passthrough"))
    steps.append(("clf", clf))

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

    if nonmalignant_label not in classes:
        # fallback
        return classes[np.argmax(proba, axis=1)]

    idx_non = int(np.where(classes == nonmalignant_label)[0][0])
    p_non = proba[:, idx_non]
    p_cancer = 1.0 - p_non

    malignant_idx = np.array([i for i, c in enumerate(classes) if c != nonmalignant_label], dtype=int)
    malignant_best = malignant_idx[np.argmax(proba[:, malignant_idx], axis=1)]

    y_pred = np.where(p_cancer >= float(cancer_threshold), classes[malignant_best], nonmalignant_label)
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
    y_true_cancer = (y_true != nonmalignant_label).astype(int)
    y_pred_cancer = (y_pred != nonmalignant_label).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_cancer, y_pred_cancer, labels=[0, 1]).ravel()
    cancer_recall = float(recall_score(y_true_cancer, y_pred_cancer, zero_division=0))
    cancer_precision = float(precision_score(y_true_cancer, y_pred_cancer, zero_division=0))
    cancer_fnr = float(fn / (fn + tp)) if (fn + tp) else 0.0
    cancer_specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0

    # AUC/PR-AUC en modo cáncer vs no cáncer (usando score p_cancer)
    if nonmalignant_label in classes:
        idx_non = int(np.where(classes == nonmalignant_label)[0][0])
        p_cancer = 1.0 - proba[:, idx_non]
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
    if nonmalignant_label not in classes:
        return

    idx_non = int(np.where(classes == nonmalignant_label)[0][0])
    p_cancer = 1.0 - proba[:, idx_non]
    y_true_cancer = (y_true.astype(str) != nonmalignant_label).astype(int)

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

    X_train, y_train = build_xy(
        df_train,
        feature_cols=feature_cols,
        class_group_col=cfg.class_group_col,
        patient_group_col=cfg.patient_group_col,
        nonmalignant_label=cfg.nonmalignant_label,
        malignant_label=cfg.malignant_label,
    )
    X_test, y_test = build_xy(
        df_test,
        feature_cols=feature_cols,
        class_group_col=cfg.class_group_col,
        patient_group_col=cfg.patient_group_col,
        nonmalignant_label=cfg.nonmalignant_label,
        malignant_label=cfg.malignant_label,
    )

    pipe = build_pipeline(cfg, feature_cols=feature_cols, y_train=y_train)

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
            if cfg.nonmalignant_label not in classes:
                chosen_thr = None
            else:
                idx_non = int(np.where(classes == cfg.nonmalignant_label)[0][0])
                p_cancer_oof = 1.0 - oof_proba[:, idx_non]
                y_true_cancer = (y_train.astype(str) != cfg.nonmalignant_label).astype(int)
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
