# Skill: Genomics pipeline

## When to apply
When working on gene expression data preprocessing, normalisation,
variance-based feature selection, or `log1p` transformations.

## Standard pipeline
1. Load raw data (CSV / Parquet from `data/raw/`).
2. Variance filtering (`HighVarianceFilter` → `var_quantile`).
3. `log1p` transformation.
4. Post-log feature selection (`VarianceFilter`).
5. Scaling (`StandardScaler`).
6. Optional PCA (`pca_var_threshold=0.9`).

## Code patterns
- See `genomics_dl/features.py` for the feature transformer implementations.
- See `genomics_dl/modeling/train.py` for the full training pipeline.
- The `MulticlassTrainConfig` dataclass holds all configuration.
- Pipelines use a sklearn `Pipeline` with a `ColumnTransformer` to keep
  genomic features and clinical features in separate branches.

## Conventions
- Always place `FeatureColumnSelector` at the entry of the pipeline.
- Priority metrics: minimise `cancer_fn`, maximise `cancer_recall`.
- The decision threshold is chosen with `choose_threshold_for_min_recall()`.
- `objective` may be `"specificity"` or `"balanced_accuracy"`.
