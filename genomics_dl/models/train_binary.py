from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from genomics_dl.features_sklearn import (
    HighVarGeneSelector,
    Log1pTransformer,
    PandasStandardScaler,
    PCAAuto,
)


def _to_serializable(obj):
    """Convierte tipos problemáticos (Path, numpy, etc.) a tipos serializables (str/int/float/list/dict)."""
    if isinstance(obj, Path):
        return str(obj)

    # numpy scalars
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()

    # numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # dict / list / tuple / set
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_to_serializable(v) for v in obj]

    return obj


def find_repo_root(start: Path | None = None) -> Path:
    """
    Busca el root del repo subiendo directorios hasta encontrar `pyproject.toml` o `.git`.
    """
    cur = (start or Path.cwd()).resolve()
    for p in [cur, *cur.parents]:
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    return cur  # fallback

def resolve_under_repo(path_like: str | Path, repo_root: Path) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (repo_root / p)


@dataclass(frozen=True)
class BinaryTrainConfig:
    train_path: str
    test_path: str
    label_col: str = "Class_group"
    positive_label: str = "Malignant"
    model_name: str = "logreg_binary"
    model_version: str = "v0.1.0"

    # Features
    use_pca: bool = False
    var_quantile: float = 0.2
    selector_on_log: bool = False  # si True: log1p antes de seleccionar varianza
    pca_var_threshold: float = 0.9
    max_pca_components: Optional[int] = None

    # Clasificador (baseline)
    pos_weight: float = 1.0  # >1 penaliza más errores en clase positiva
    C: float = 1.0
    max_iter: int = 2000
    random_state: int = 42

    # CV / threshold
    cv_splits: int = 5
    min_recall_for_threshold: Optional[float] = 0.95  # Si None -> usa 0.5

    # MLflow
    experiment_name: str = "gse183635_binary"
    tracking_uri: Optional[str] = None  # si None, usa el default de MLflow

    # Outputs
    save_local_bundle: bool = True
    output_models_dir: str = "models"
    output_figures_dir: str = "reports/figures/binary"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def build_xy(
    df: pd.DataFrame,
    label_col: str,
    positive_label: str,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    if label_col not in df.columns:
        raise ValueError(f"No existe label_col='{label_col}' en el dataframe.")
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas de features: {missing[:10]} ...")

    X = df.loc[:, feature_cols].copy()
    y = (df[label_col].astype(str) == str(positive_label)).astype(int).to_numpy()
    return X, y


# Pipelines
def _class_weight_from_pos_weight(pos_weight: float) -> dict[int, float]:
    # clase 0: nonMalignant, clase 1: Malignant
    return {0: 1.0, 1: float(pos_weight)}


def build_pipeline(cfg: BinaryTrainConfig) -> Pipeline:
    clf = LogisticRegression(
        C=cfg.C,
        solver="liblinear",
        class_weight=_class_weight_from_pos_weight(cfg.pos_weight),
        max_iter=cfg.max_iter,
        random_state=cfg.random_state,
    )

    selector = HighVarGeneSelector(var_quantile=cfg.var_quantile)
    log = Log1pTransformer()
    scale = PandasStandardScaler()
    pca = PCAAuto(
        var_threshold=cfg.pca_var_threshold,
        max_components=cfg.max_pca_components,
        random_state=cfg.random_state,
    )

    steps: list[tuple[str, Any]] = []
    if cfg.selector_on_log:
        steps.extend([("log1p", log), ("select", selector), ("scale", scale)])
    else:
        steps.extend([("select", selector), ("log1p", log), ("scale", scale)])

    steps.append(("pca", pca if cfg.use_pca else "passthrough"))
    steps.append(("clf", clf))

    return Pipeline(steps=steps)


# Metrics (FN foco)
def compute_binary_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    recall = recall_score(y_true, y_pred, zero_division=0)  # sensibilidad
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    # probabilísticos
    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)

    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    fnr = fn / (fn + tp) if (fn + tp) > 0 else float("nan")  # clave para FN
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")

    return {
        "threshold": float(threshold),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
        "recall_sensitivity": float(recall),
        "fnr": float(fnr),
        "specificity": float(specificity),
        "precision": float(precision),
        "npv": float(npv),
        "f1": float(f1),
        "balanced_accuracy": float(bal_acc),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "fpr": float(fpr),
    }


def choose_threshold_for_min_recall(y_true: np.ndarray, y_proba: np.ndarray, min_recall: float) -> float:
    """
    Elige el umbral más alto que mantenga recall >= min_recall.
    (Umbral más alto -> suele reducir FP, manteniendo sensibilidad objetivo si es posible)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # thresholds tiene len = len(precision)-1
    valid = np.where(recall[:-1] >= min_recall)[0]
    if len(valid) == 0:
        # No se puede alcanzar ese recall con este modelo (según estas probabilidades)
        return 0.5
    return float(thresholds[valid].max())


# Plots (artefactos)
def plot_confusion(y_true: np.ndarray, y_proba: np.ndarray, threshold: float, outpath: Path) -> None:
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig = plt.figure()
    ax = plt.gca()
    im = ax.imshow(cm)
    ax.set_title(f"Confusion matrix (thr={threshold:.3f})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["nonMalignant", "Malignant"])
    ax.set_yticklabels(["nonMalignant", "Malignant"])

    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")

    plt.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_pr(y_true: np.ndarray, y_proba: np.ndarray, outpath: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    fig = plt.figure()
    plt.plot(recall, precision)
    plt.title(f"Precision-Recall (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


# Saving bundle (models/)
def save_model_bundle(
    out_root: Path,
    model_name: str,
    model_version: str,
    pipeline: Pipeline,
    metrics: dict[str, Any],
    params: dict[str, Any],
    signature_dict: dict[str, Any],
    mlflow_run_id: Optional[str],
) -> Path:
    out_dir = out_root / model_name / model_version
    _ensure_dir(out_dir)

    # modelo
    joblib.dump(pipeline, out_dir / "model.pkl")

    # métricas / params
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(out_dir / "params.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(params, f, sort_keys=False, allow_unicode=True)

    with open(out_dir / "signature.json", "w", encoding="utf-8") as f:
        json.dump(signature_dict, f, indent=2, ensure_ascii=False)

    # mini model card
    readme = out_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                f"# {model_name} ({model_version})",
                "",
                "## Task",
                "Binary classification: nonMalignant (0) vs Malignant (1)",
                "",
                "## Artifacts",
                "- model.pkl (sklearn Pipeline completo)",
                "- metrics.json",
                "- params.yaml",
                "- signature.json",
                "",
                "## MLflow",
                f"- run_id: {mlflow_run_id or 'N/A'}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    return out_dir


# Main train
def run_training(cfg: BinaryTrainConfig, feature_cols: list[str]) -> dict[str, Any]:
    if cfg.tracking_uri:
        mlflow.set_tracking_uri(cfg.tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)

    df_train = load_parquet(cfg.train_path)
    df_test = load_parquet(cfg.test_path)

    X_train, y_train = build_xy(df_train, cfg.label_col, cfg.positive_label, feature_cols)
    X_test, y_test = build_xy(df_test, cfg.label_col, cfg.positive_label, feature_cols)

    pipe = build_pipeline(cfg)

    cv = StratifiedKFold(n_splits=cfg.cv_splits, shuffle=True, random_state=cfg.random_state)

    with mlflow.start_run(run_name=f"{cfg.model_name}_{cfg.model_version}") as run:
        run_id = run.info.run_id

        # OOF proba (para seleccionar umbral y métricas CV sin leakage)
        oof_proba = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]

        if cfg.min_recall_for_threshold is None:
            chosen_thr = 0.5
        else:
            chosen_thr = choose_threshold_for_min_recall(
                y_train, oof_proba, min_recall=cfg.min_recall_for_threshold
            )

        cv_metrics = compute_binary_metrics(y_train, oof_proba, threshold=chosen_thr)

        # Fit final en train completo y test
        pipe.fit(X_train, y_train)
        test_proba = pipe.predict_proba(X_test)[:, 1]
        test_metrics = compute_binary_metrics(y_test, test_proba, threshold=chosen_thr)

        # Artefactos plots
        repo_root = find_repo_root()
        artifact_dir = resolve_under_repo(cfg.output_figures_dir, repo_root)
        _ensure_dir(artifact_dir)

        cm_path = artifact_dir / f"{cfg.model_name}_{cfg.model_version}_cm.png"
        pr_path = artifact_dir / f"{cfg.model_name}_{cfg.model_version}_pr.png"
        plot_confusion(y_test, test_proba, threshold=chosen_thr, outpath=cm_path)
        plot_pr(y_test, test_proba, outpath=pr_path)

        # MLflow logging
        raw_params = asdict(cfg)
        params = _to_serializable(raw_params)

        params["n_train"] = int(X_train.shape[0])
        params["n_test"] = int(X_test.shape[0])
        params["n_features_input"] = int(X_train.shape[1])

        mlflow.log_params({k: v for k, v in params.items() if v is not None})

        # prefijos para distinguir CV vs TEST
        mlflow.log_metrics({f"cv_{k}": float(v) for k, v in cv_metrics.items() if k != "threshold"})
        mlflow.log_metrics({f"test_{k}": float(v) for k, v in test_metrics.items() if k != "threshold"})
        mlflow.log_metric("chosen_threshold", float(chosen_thr))

        metrics = {
            "cv": cv_metrics,
            "test": test_metrics,
            "chosen_threshold": chosen_thr,
            }

        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(pr_path))

        # Modelo + signature (MLflow)
        input_example = X_train.head(3)
        signature = infer_signature(input_example, pipe.predict_proba(input_example)[:, 1])
        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )
        signature_dict = signature.to_dict()

        # Sanitiza
        params = _to_serializable(params)
        metrics = _to_serializable(metrics)
        signature_dict = _to_serializable(signature_dict)

        # Guardado local versionado (models/)
        saved_dir = None
        if cfg.save_local_bundle:
            saved_dir = save_model_bundle(
                out_root=resolve_under_repo(cfg.output_models_dir, repo_root),
                model_name=cfg.model_name,
                model_version=cfg.model_version,
                pipeline=pipe,
                metrics={"cv": cv_metrics, "test": test_metrics, "chosen_threshold": chosen_thr},
                params=params,
                signature_dict=signature_dict,
                mlflow_run_id=run_id,
            )

        # devuelve resumen
        return {
            "mlflow_run_id": run_id,
            "chosen_threshold": chosen_thr,
            "cv_metrics": cv_metrics,
            "test_metrics": test_metrics,
            "saved_model_dir": (str(saved_dir) if saved_dir is not None else None),
        }
