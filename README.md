# genomics-dl

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
  <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Repositorio del Trabajo de Fin de Máster: **“Deep Learning en Genómica”**.
Proyecto centrado en datos de **ARN mensajero de plaquetas** (*platelet RNA / tumor-educated platelets*) como biomarcadores para **clasificación de cáncer** (binaria y, posteriormente, multiclase), con foco en **reproducibilidad** (Dev Container, `uv`, Git LFS, CI).

---

## Metodología

Una metodología iterativa e incremental, inspirada en CRISP-DM y en buenas prácticas de MLOps, donde el trabajo se organiza en sprints claramente delimitadas. Cada iteración produce un resultado end-to-end (pipeline de datos, análisis exploratorio, modelo supervisado, etc.), que se documenta y versiona, garantizando reproducibilidad y trazabilidad.

---

## Contexto

Este TFM se sitúa en la intersección entre genómica y ML/DL, utilizando perfiles de expresión génica en plaquetas para estudiar su utilidad como biomarcadores de cáncer.
El trabajo parte de un proyecto previo (enfoque **binario** cáncer vs no cáncer) y busca formalizar y ampliar el flujo hacia un pipeline robusto y extensible.

---

## Objetivos

- Montar un **pipeline reproducible** de preparación de datos:
  - Carga de conteos.
  - Cálculo/obtención de longitudes de gen.
  - Normalización a **TPM**.
  - Integración con **metadatos clínicos**.
  - Dataset final listo para modelado.
- Entrenar y analizar modelos de **clasificación binaria** (cáncer vs no cáncer).
- Extender a **clasificación multiclase** (tipos de cáncer) y estudiar la **heterogeneidad** de los datos.
- Explorar enfoques de **ML clásico** y **Deep Learning**.

---

## Datos

Dataset inicial objetivo: **GSE183635**.

Estructura de datos (Cookiecutter Data Science):
- `data/raw/`: datos originales (inmutables).
- `data/interim/`: transformaciones intermedias.
- `data/processed/`: dataset final canónico para modelado.

### Datos grandes (Git LFS)

Este repo está pensado para versionar datos grandes con **Git LFS**, especialmente bajo `data/`.

Comandos típicos:
```bash
git lfs install
git lfs pull
git lfs ls-files
```

Guía operativa completa: `docs/docs/TFM_Dev_Guide.md`.

---

## Entorno reproducible (Dev Container + uv)

El flujo recomendado es trabajar **dentro del Dev Container** en VS Code y gestionar dependencias con **uv** (`pyproject.toml` + `uv.lock`).

### Opción A) Dev Container (recomendada)

1) Clona el repo:
```bash
git clone git@github.com:MisterMito/TFM.git
cd TFM
```

2) Abre en VS Code y ejecuta: **Dev Containers: Reopen in Container**.

3) Dentro del contenedor:
```bash
uv sync
git lfs pull
pre-commit install
```

> Si cambian `.devcontainer/devcontainer.json` o el `Dockerfile`: **Rebuild and Reopen in Container**.

### Opción B) Local (sin contenedor)

Requiere `uv` y un Python compatible con el proyecto (en este repo se trabaja con **Python 3.12**):
```bash
uv sync
git lfs pull
pre-commit install
```

---

## Flujo diario (resumen)

```bash
git pull
uv sync            # si cambió pyproject.toml / uv.lock
git lfs pull       # si hay binarios nuevos
```

Al finalizar:
```bash
git add -A
git commit -m "feat/fix: descripción breve"
git push
```

---

## Calidad y tests

### Lint / formato (pre-commit)
```bash
uv run pre-commit run --all-files
```

### Tests (pytest)
```bash
uv run pytest -q
```

---

## CI (GitHub Actions)

El workflow de CI (según la guía de desarrollo) valida típicamente:
- `lint`: `pre-commit` + control de ficheros grandes / LFS.
- `tests`: `pytest`.
- `build-package`: `uv build` (wheel/sdist como artefactos).
- `notebooks-smoke`: ejecución de notebooks ligeros en `notebooks/smoke/`.
- `devcontainer-build`: build del Dev Container.

Comandos equivalentes en local (resumen):
```bash
uv sync --frozen && uv run pre-commit run --all-files
uv run pytest -q
uv build
```

Detalles: `docs/docs/TFM_Dev_Guide.md`.

---

## Modelos y MLOps

Estructura recomendada para artefactos versionados:
```text
models/
  <nombre_modelo>/
    v0.1.0/
      model.bin|.pt|.onnx|.pkl
      metrics.json
      params.yaml
      signature.json
      README.md            # mini model card
    v0.1.1/
      ...
```

Buenas prácticas mínimas:
- Guardar hiperparámetros (`params.yaml`), métricas (`metrics.json`) y firma (`signature.json`).
- Registrar `git_commit` del entrenamiento y referencia/hash de datos o versión del dataset.
- Mantener scripts de entrenamiento e inferencia en `genomics_dl/modeling/`.
- Integrar **MLflow** para tracking/registro cuando el flujo esté estable (ver guía).

---

## Documentación

- Contexto del TFM: `docs/00_contexto_tfm.md`
- Guía operativa (Dev Container, uv, LFS, CI, ramas, modelos): `docs/docs/TFM_Dev_Guide.md`
- Sitio de documentación (MkDocs): `docs/`

---

## Project Organization

```text
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for
│                         genomics_dl and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── genomics_dl        <- Source code for use in this project.
    ├── __init__.py             <- Makes genomics_dl a Python module
    ├── config.py               <- Store useful variables and configuration
    ├── dataset.py              <- Scripts to download or generate data
    ├── features.py             <- Code to create features for modeling
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py          <- Code to run model inference with trained models
    │   └── train.py            <- Code to train models
    └── plots.py                <- Code to create visualizations
```

---

## Contribución (ramas y PRs)

```bash
git checkout -b feat/nombre-claro
# trabajo + commits
git push -u origin feat/nombre-claro
```

Tras el merge:
```bash
git switch main
git pull
git branch -d feat/nombre-claro
git fetch -p
```

---

## Licencia

Ver `LICENSE`.
