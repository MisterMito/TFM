from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

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
    threshold_objective: str = "specificity"  # specificity, balanced_accuracy, highest_threshold

    # MLflow
    experiment_name: str = "gse183635_binary"
    tracking_uri: Optional[str] = None  # si None, usa el default de MLflow

    # Outputs
    save_local_bundle: bool = True
    save_plots: bool = True
    mlflow_log_artifacts: bool = True
    mlflow_log_model: bool = True
    output_models_dir: str = "models"
    output_figures_dir: str = "reports/figures/binary"

    clf_name: str = "logreg"
    clf_params: dict[str, Any] = field(default_factory=dict)


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

def build_classifier(cfg: BinaryTrainConfig):
    cw = _class_weight_from_pos_weight(cfg.pos_weight)

    if cfg.clf_name == "logreg":
        return LogisticRegression(
            C=cfg.clf_params.get("C", cfg.C),
            solver=cfg.clf_params.get("solver", "liblinear"),
            class_weight=cw,
            max_iter=cfg.clf_params.get("max_iter", cfg.max_iter),
            random_state=cfg.random_state,
        )

    if cfg.clf_name == "sgd_logloss":
        # rápido y muy típico en alta dimensión
        return SGDClassifier(
            loss="log_loss",
            alpha=cfg.clf_params.get("alpha", 1e-4),
            penalty=cfg.clf_params.get("penalty", "l2"),
            class_weight=cw,
            max_iter=cfg.clf_params.get("max_iter", 2000),
            random_state=cfg.random_state,
        )

    if cfg.clf_name == "linear_svc_calibrated":
        base = LinearSVC(
            C=cfg.clf_params.get("C", 1.0),
            class_weight=cw,
            random_state=cfg.random_state,
        )
        return CalibratedClassifierCV(base, method="sigmoid", cv=3)

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


def build_pipeline(cfg: BinaryTrainConfig, feature_cols: list[str]) -> Pipeline:
    clf = build_classifier(cfg)

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


def choose_threshold_for_min_recall(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    min_recall: Optional[float],
    objective: str = "specificity",
) -> float:
    """
    Selecciona el umbral que maximiza la métrica solicitada cumpliendo recall >= min_recall.
    objective:
        - 'specificity': maximiza especificidad.
        - 'balanced_accuracy': maximiza balanced accuracy.
        - 'highest_threshold': replica comportamiento previo (umbral más alto disponible).
    """
    if min_recall is None:
        return 0.5

    min_recall = float(min_recall)

    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    specificities = 1.0 - fpr
    balanced = 0.5 * (tpr + specificities)

    # Filtramos por recall mínimo
    mask = tpr >= min_recall
    if not np.any(mask):
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
        # roc_curve arranca en inf, que no es útil
        return 1.0
    return float(chosen)


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
def _persist_mlflow_local_model(
    pipeline: Pipeline,
    signature,
    input_example: pd.DataFrame,
    destination: Path,
) -> None:
    """
    Guarda una copia local con formato MLflow (MLmodel + data/) sin tocar el tracking server.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_model_dir = Path(tmpdir) / "mlflow_model"
        mlflow.sklearn.save_model(
            sk_model=pipeline,
            path=str(tmp_model_dir),
            signature=signature,
            input_example=input_example,
        )
        for item in tmp_model_dir.iterdir():
            target = destination / item.name
            if item.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(item, target)
            else:
                if target.exists():
                    target.unlink()
                shutil.copy2(item, target)


def save_model_bundle(
    out_root: Path,
    model_name: str,
    model_version: str,
    pipeline: Pipeline,
    metrics: dict[str, Any],
    params: dict[str, Any],
    signature_dict: dict[str, Any],
    signature,
    input_example: pd.DataFrame,
    mlflow_run_id: Optional[str],
) -> Path:
    out_dir = out_root / model_name / model_version
    _ensure_dir(out_dir)

    # modelo
    joblib.dump(pipeline, out_dir / "model.pkl")
    _persist_mlflow_local_model(pipeline, signature, input_example, out_dir)

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
                "- MLmodel + entorno MLflow",
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

    pipe = build_pipeline(cfg, feature_cols=feature_cols)

    cv = StratifiedKFold(n_splits=cfg.cv_splits, shuffle=True, random_state=cfg.random_state)

    with mlflow.start_run(run_name=f"{cfg.model_name}_{cfg.model_version}") as run:
        run_id = run.info.run_id

        # OOF proba (para seleccionar umbral y métricas CV sin leakage)
        oof_proba = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]

        chosen_thr = choose_threshold_for_min_recall(
            y_train,
            oof_proba,
            min_recall=cfg.min_recall_for_threshold,
            objective=cfg.threshold_objective,
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

        if cfg.save_plots:
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

        if cfg.mlflow_log_artifacts and cfg.save_plots:
            mlflow.log_artifact(str(cm_path))
            mlflow.log_artifact(str(pr_path))

        # Modelo + signature (MLflow)
        input_example = X_train.head(3)
        signature = infer_signature(input_example, pipe.predict_proba(input_example)[:, 1])

        if cfg.mlflow_log_model:
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
                signature=signature,
                input_example=input_example,
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
