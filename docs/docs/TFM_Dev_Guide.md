# Guía de Trabajo — TFM “Deep Learning para la Genómica”
**Dev Container + Git entre Windows/Mac, uv y Git LFS**

> Documento operativo para cualquier colaborador/a del proyecto. Resume el flujo de trabajo con contenedores de desarrollo, gestión de dependencias con `uv`, manejo de datos con Git LFS, control de versiones con Git y prácticas recomendadas.

---

## 0) Conceptos clave
- **Repositorio único** (GitHub). El trabajo se realiza preferentemente **dentro del Dev Container** en VS Code.
- **Dependencias**: `pyproject.toml` + `uv.lock` garantizan entornos reproducibles; sincronizar siempre con `uv sync`.
- **Datos** bajo `data/` versionados con **Git LFS** (JSON/CSV/FASTQ/BAM/VCF, etc.). Evitar subir datos sensibles fuera de LFS; para grandes volúmenes considerar DVC.
- **No simultanear máquinas**: antes de trabajar, ejecutar `git pull`; al terminar, `git push`.

---

## 1) Primer uso en cada máquina (one-time)
### macOS (zsh)
```bash
# Clonar el repo
git clone git@github.com:<usuario>/<repo>.git
cd <repo>

# (Opcional) Instalar herramientas en host
brew install gh git-lfs
curl -LsSf https://astral.sh/uv/install.sh | sh
git lfs install

# Abrir en VS Code y reabrir en contenedor
code .
# VS Code → "Dev Containers: Reopen in Container"
```

### Windows (PowerShell)
```powershell
# Clonar el repo
git clone git@github.com:<usuario>/<repo>.git
cd <repo>

# (Opcional) Instalar herramientas en host
winget install --id GitHub.GitLFS -e
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
git lfs install

# Abrir en VS Code y reabrir en contenedor
code .
# VS Code → "Dev Containers: Reopen in Container"
```

> Cuando cambien `.devcontainer/devcontainer.json` o el `Dockerfile`, ejecutar **Rebuild and Reopen in Container**.
> Dentro del contenedor, comprobar LFS: `git lfs --version`.

---

## 2) Flujo diario (en cualquier máquina)
```bash
# 1) Traer cambios
git pull

# 2) Sincronizar dependencias si cambió pyproject/uv.lock
uv sync

# 3) Obtener binarios LFS si los hay
git lfs pull

# 4) Trabajar en VS Code + contenedor (src/, notebooks/, tests/…)
#    Si cambió la imagen/base → Rebuild and Reopen in Container
```

Al finalizar:
```bash
git add -A
git commit -m "feat/fix: descripción breve"
git push
```

---

## 3) Gestión de dependencias (reproducible con uv)
Las operaciones se realizan **dentro del contenedor** para reflejar cambios en `pyproject.toml` y `uv.lock`.

**Añadir paquete**
```bash
uv add <paquete>
git add pyproject.toml uv.lock
git commit -m "deps: add <paquete>"
git push
```

**Eliminar paquete**
```bash
uv remove <paquete>
git add pyproject.toml uv.lock
git commit -m "deps: remove <paquete>"
git push
```

En la otra máquina: `git pull && uv sync`.

> Si se cambia la base del contenedor (CUDA, git-lfs, etc.), hacer **Rebuild**.

---

## 4) Datos con Git LFS
### 4.1 Configuración inicial (una vez por repositorio)
```bash
git lfs track "data/**/*.{json,csv,tsv,parquet,feather,txt}"
git lfs track "data/**/*.{fastq,fq,fastq.gz,bam,cram,vcf,vcf.gz}"
git lfs track "data/**/*.{h5,hdf5,npy,npz}"
git add .gitattributes
git commit -m "chore(lfs): track data patterns"
git push
```

### 4.2 Añadir datos
Colocar ficheros en `data/raw/…` y versionar:
```bash
git add data/raw/...
git commit -m "data(raw): descripción de la fuente/fecha"
git push
```

### 4.3 Traer datos en otra máquina
```bash
git pull
git lfs pull
```

> Si se subieron archivos grandes sin LFS y se desea corregir el historial:
> `git lfs migrate import --include="data/**"` (hacerlo en una rama y revisar antes de fusionar).

---

## 5) Trabajo con Dev Container
- Abrir el repositorio y usar **“Reopen in Container”**.
- Si se modifican archivos del contenedor (`Dockerfile`/`devcontainer.json`), ejecutar **Rebuild and Reopen**.
- Verificar `git lfs` dentro de la imagen; si no está, instalar en Dockerfile o en `postCreateCommand`.
- Kernel Jupyter dentro del contenedor (opcional): `python -m ipykernel install --user --name=genomics-dl`.

**Ejemplo de `postCreateCommand`**
```json
"postCreateCommand": "pip install uv pre-commit && uv sync && pre-commit install && python -m ipykernel install --user --name=genomics-dl"
```

---

## 6) Ramas y Pull Requests
```bash
# Nueva rama
git checkout -b feat/nombre-claro

# Trabajo → commits
git add -A
git commit -m "feat: ..."

# Publicar rama y abrir PR
git push -u origin feat/nombre-claro
# Abrir PR en la web o con gh CLI
```

Tras el merge:
```bash
git switch main
git pull
git branch -d feat/nombre-claro
git fetch -p        # limpiar referencias remotas
```

---

## 7) Notebooks y buenas prácticas
- Mantener código reutilizable en `src/`; notebooks para EDA/experimentos.
- Limitar salidas de notebooks en commits (p. ej., con `nbstripout` vía pre-commit).
- Establecer semillas aleatorias cuando aplique y documentar versiones (ver sección de modelos).

---

## 8) Pre-commit útil (opcional pero recomendado)
Ejemplo de hooks mínimos en `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: end-of-file-fixer
      - id: trailing-whitespace
      - id: mixed-line-ending
      - id: check-yaml
      - id: check-added-large-files
        args: ["--maxkb=51200"]  # 50 MB

  - repo: https://github.com/kynan/nbstripout
    rev: 0.7.1
    hooks:
      - id: nbstripout
        args: [--keep-output=False]
```

Activación:
```bash
pre-commit install
```

---

## 9) Problemas frecuentes (solución rápida)

### 9.1 Hooks de pre-commit y commits bloqueados
Si aparece:
```
No .pre-commit-config.yaml file was found
```
existe un *hook* instalado sin configuración. Opciones:

**A) Añadir configuración mínima (recomendado)**
```bash
cat > .pre-commit-config.yaml <<'YAML'
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: end-of-file-fixer
      - id: trailing-whitespace
      - id: mixed-line-ending
      - id: check-yaml
YAML
pre-commit install
git add .pre-commit-config.yaml
git commit -m "chore: add minimal pre-commit config"
```

**B) Permitir commits sin config (temporal)**
```bash
PRE_COMMIT_ALLOW_NO_CONFIG=1 git commit -m "mensaje"
```
o reinstalar el hook:
```bash
pre-commit install --allow-missing-config
```

**C) Desinstalar el hook**
```bash
pre-commit uninstall
```

### 9.2 Aviso de finales de línea: “LF will be replaced by CRLF”
Para evitar conversiones CRLF↔LF, fijar `.gitattributes`:
```gitattributes
* text=auto
*.py eol=lf
*.md eol=lf
*.yml eol=lf
*.yaml eol=lf
*.sh eol=lf
*.ipynb -diff
```
y/o configurar Git:
```bash
git config core.autocrlf input
```
Volver a intentar el commit/push después.

---

## 10) Chuleta de comandos
```bash
# Git
git status
git log --oneline --decorate --graph -n 10
git diff
git restore <file>
git clean -fd

# uv
uv sync
uv add <pkg>
uv remove <pkg>
uv run <cmd>

# LFS
git lfs ls-files
git lfs pull
git lfs track "<patrones>"
```

---

## 11) Modelos: versionado y buenas prácticas

### 11.1 Dónde y cómo guardar modelos
- **Carpeta**: `models/` (git-ignorada por defecto) o versionada vía **Git LFS** si se necesita compartir directamente desde GitHub.
- **Estructura sugerida** (sin registry externo):
```
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
- **Buenas prácticas**:
  - PyTorch: `torch.save()` o `safetensors`; exportar **ONNX** si se requiere interoperabilidad.
  - Scikit-learn: `joblib.dump()`.
  - Incluir tokenizers/escaladores y cualquier artefacto necesario para inferencia.

### 11.2 Versionado (semver + metadatos)
- **SemVer**: `MAJOR.MINOR.PATCH` (p. ej., `v0.2.1`).
- **Metadatos mínimos** (`params.yaml`/`metrics.json`):
  - `run_id` y `timestamp`.
  - `git_commit` (hash del commit).
  - `data_hash` o referencia a la versión del dataset.
  - `params` (hiperparámetros clave).
  - `metrics` (validación/test) y, cuando aplique, curvas/figuras.
  - `python_version` y conjunto de paquetes (o referencia a `uv.lock`).
  - `hardware` (CPU/GPU) si es relevante.

### 11.3 Registro de modelos (opciones)
- **Git LFS**: simple, dentro del repo (atención a cuotas).
- **MLflow Tracking + Model Registry**:
  - Instalar: `uv add mlflow`
  - Ejemplo mínimo:
    ```python
    import mlflow, mlflow.sklearn
    with mlflow.start_run() as run:
        # ... entrenamiento sklearn
        mlflow.log_params({"C":1.0,"penalty":"l2"})
        mlflow.log_metrics({"roc_auc":0.912})
        mlflow.sklearn.log_model(model, "model", registered_model_name="genomics-baseline")
        print("run_id:", run.info.run_id)
    ```
  - Permite comparar ejecuciones, versionar artefactos y promover (Staging → Production).
- **Alternativas**: registries cloud (Azure ML), DVC/Iterative Studio, Hubs privados.

### 11.4 Reproducibilidad
- Fijar semillas (NumPy/PyTorch/Sklearn) y documentarlas.
- Versionar `uv.lock` junto con el commit del entrenamiento.
- Registrar datos, splits, hiperparámetros y métricas comparables.
- Definir y guardar `signature.json` (forma y tipos de entrada/salida).

### 11.5 Criterios de liberación (“gating”)
- Establecer umbrales mínimos (p. ej., `roc_auc ≥ x`, `f1 ≥ y`).
- Mantener un conjunto de evaluación estable (hold-out) o validación cruzada.
- Revisar sesgos y robustez básica.

### 11.6 Despliegue y compatibilidad
- Exportar formato portable (ONNX cuando proceda) y un script mínimo de inferencia (`predict.py`).
- Documentar dependencias especiales (CUDA/cuDNN; `tensorflow-macos` en Apple Silicon).
- Cuando aplique, alinear versión de imagen Docker con versión del modelo (tags coherentes).

### 11.7 Model cards
- Añadir “model cards” por versión (`models/<nombre>/vX.Y.Z/README.md` o `reports/model_cards/`), con propósito, datos, límites, métricas, riesgos y contacto.

### 11.8 Limpieza y retención
- Evitar artefactos temporales fuera de LFS.
- Definir políticas de retención (últimos N runs, versiones promovidas).
- Comprimir artefactos cuando no afecte a la carga (p. ej., `safetensors`).

---

## 12) Estructura de carpetas (resumen)
```
.
├── notebooks/
├── src/
├── data/                 # LFS (raw/interim/processed)
├── models/               # LFS o registry externo
├── reports/              # figuras y model cards
├── .devcontainer/
├── pyproject.toml
├── uv.lock
└── .gitattributes        # patrones LFS
```
