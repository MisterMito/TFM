# CLAUDE.md

Guidance for Claude Code in this repo. Keep only what cannot be inferred from the code itself.

See @docs/00_contexto_tfm.md for the thesis context (biological problem, motivation, scope).
See @README.md for project overview, data pipeline summary and CI gates.
See @docs/docs/TFM_Dev_Guide.md for the full Dev Container / uv / Git LFS workflow.
See @docs/01_guia_trabajo_tfm.md for the per-iteration log used in the thesis write-up.
See @pyproject.toml for Python version, dependencies and tool config.

## Language convention

- **Prose in Spanish** — docstrings, inline comments, log/error messages, notebook markdown, conclusions, `docs/**`, commit bodies, PR descriptions. Match the surrounding language when editing; never translate Spanish prose as a side-effect.
- **Identifiers in English** — modules, classes (`PascalCase`), functions/variables (`snake_case`), CLI flags, MLflow run names. Domain terms already in English in the literature (`Sample ID`, `Patient_group`, `Class_group`, `TPM`, `RPK`) stay in English even inside Spanish prose.

## Code placement

**All non-trivial logic lives in `genomics_dl/`, split by purpose. Notebooks are the execution surface: they import from `genomics_dl`, run trainings, evaluate results and write up conclusions — but they do not *implement* the logic they call.**

Current purpose → module map:

| Purpose | Module |
| --- | --- |
| Data transformation (TPM, gene-length coercion, generic helpers) | `genomics_dl/utils.py` |
| Metadata I/O, label derivation, sample-ID alignment | `genomics_dl/metadata.py` |
| sklearn feature transformers (selection, scaling, PCA, clinical prep) | `genomics_dl/features_sklearn.py` |
| Unsupervised analysis (clustering, UMAP, cluster-vs-clinical tests) | `genomics_dl/models/heterogeneity.py` |
| Supervised binary training | `genomics_dl/models/train_binary.py` |
| Supervised multiclass / hierarchical training | `genomics_dl/models/train_multiclass.py` |

If no existing module fits, create a new purpose-named module in `genomics_dl/` before writing the notebook that uses it.

**What notebooks do** (in-scope): import from `genomics_dl.*`, wire config dataclasses (`BinaryTrainConfig`, …), call training / clustering / evaluation entry points, run EDA, generate plots, write Spanish-text analysis and conclusions, sanity-check shapes / class balance. Notebooks **are** where models get trained and where the thesis narrative is built.

**What notebooks do not do** (out-of-scope, must be extracted to `genomics_dl/`): custom `Transformer` / `Estimator` subclasses, training / CV / threshold-selection loops, metric implementations, reusable preprocessing functions, clustering algorithms. If you find yourself writing >5 lines of reusable logic in a cell, move it to a module and re-import.

Tests under `tests/` exercise the extracted code, not notebook cells.

## Git workflow

The README codifies a branch-per-change flow — follow it strictly:

1. `git switch main && git pull`
2. Branch with a prefix: `feat/<slug>`, `fix/<slug>`, `docs/<slug>`, `refactor/<slug>`, `chore/<slug>`.
3. Commit on that branch (Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`; Spanish body when needed).
4. `git push -u origin <branch>` when ready.
5. **Stop.** The user reviews the pushed branch and opens the PR to `main` themselves — Claude must not open PRs.

Hard rules: never commit or push to `main` directly; never `--force` / `--force-with-lease` without explicit authorisation; never `--no-verify`. Before pushing, locally run `uv run pre-commit run --all-files`, `uv run ruff check`, `uv run pytest -q` (these are CI gates).

When dependencies change, stage `pyproject.toml` **and** `uv.lock` in the same commit. Never hand-edit `uv.lock`.

## Non-obvious gotchas

- **Pandas is load-bearing in the sklearn pipeline.** Every transformer in `features_sklearn.py` enforces `pd.DataFrame` input via `_to_df` so gene-name columns survive end-to-end. Do not "simplify" any of them to accept ndarrays.
- **`align_and_merge_by_sample_id` has a positional fallback.** When a `Sample ID` does not match after regex cleanup, it merges by row position to reproduce the prior student's behaviour. Read its docstring before touching alignment logic.
- **Dual-write of trained models.** `train_binary.py` / `train_multiclass.py` log to MLflow (`notebooks/mlflow.db` + `notebooks/mlruns/`) **and** persist a versioned bundle to `models/<name>/v<MAJOR>.<MINOR>.<PATCH>/` (`model.joblib`, `params.yaml`, `metrics.json`, `signature.json`, plots). Keep both in sync when adding artifacts. Existing `*_final` dirs are immutable release snapshots — bump the version, don't overwrite.
- **Path resolution helpers.** Use `find_repo_root()` / `resolve_under_repo()` from `train_*.py` instead of hard-coding paths so scripts work from notebooks and CLI.
- **`notebooks/smoke/` is referenced by CI but does not exist yet.** The job uses `shopt -s nullglob` so it silently passes — adding a notebook there enables it on every PR with a 600 s timeout.
- **CI rejects non-LFS files > 100 MB and warns > 50 MB.** Large binaries (`*.RData`, `*.gff3`, `data/processed/*.parquet`, `notebooks/mlflow.db`) belong in Git LFS. Run `git lfs pull` after fresh clones.
- **Pre-commit hooks do not include ruff.** Run ruff manually before pushing.

## Naming conventions (not derivable from one file)

- Notebooks: `<phase>.<step>-<initials>-<slug>.ipynb`. Order matters — later phases consume parquets produced by earlier ones.
- Branches: `<type>/<short-kebab-slug>` (see Git workflow above).
- Model bundles: `models/<purpose>_<variant>/v<MAJOR>.<MINOR>.<PATCH>/`.
