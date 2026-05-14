# Skill: Model training

## When to apply
When training, evaluating, or comparing classification models.

## Training flow
1. Define the config with `MulticlassTrainConfig` (dataclass).
2. Call `run_training(cfg, feature_cols=...)`.
3. Read results from `out["test_metrics"]`.
4. For a sweep: iterate over `feat_grid × clf_grid × malignant_weights`.

## Available classifiers
- `"rf"`: `RandomForest` (`n_estimators`, `max_depth`).
- `"extratrees"`: `ExtraTrees` (`n_estimators`, `max_depth`, `min_samples_leaf`).
- `"logreg"`: `LogisticRegression` (`solver`, `max_iter`, `C`).

## Reference configuration
- `cv_splits=8`
- `min_cancer_recall_for_threshold=0.9`
- `threshold_objective="specificity"`
- `experiment_name="gse183635_multiclass_optimized"`

## Model naming
Convention: `clf_pca{0/1}_log{0/1}_vq{N}_mw{N}_variant`
Example: `rf_pca0_log0_vq20_mw3p0_malclust`

## Rules
- Always collect sweep results into a DataFrame for comparison.
- When running a sweep, use `tqdm` with EMA to estimate remaining time.
- Metric priority: 1) minimise `test_cancer_fn`, 2) maximise recall,
  3) maximise `f1_macro`.
