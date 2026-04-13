"""Tests for ClinicalFeaturePreprocessor and related utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from genomics_dl.features_sklearn import ClinicalFeaturePreprocessor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_clinical_df() -> pd.DataFrame:
    """DataFrame mimicking real data with Age, Sex, and cluster dummies."""
    rng = np.random.RandomState(42)
    n = 50
    df = pd.DataFrame({
        "ENSG00000001": rng.rand(n),
        "ENSG00000002": rng.rand(n),
        "Age": rng.normal(60, 15, n).astype(float),
        "Sex": rng.choice(["F", "M", "n.a."], n, p=[0.5, 0.45, 0.05]),
        "mal_cluster_0": rng.choice([0.0, 1.0], n),
        "mal_cluster_1": 0.0,  # will be overwritten below
        "nm_cluster_0": rng.choice([0.0, 1.0], n),
        "nm_cluster_1": 0.0,
    })
    # Make cluster dummies mutually exclusive
    df["mal_cluster_1"] = 1.0 - df["mal_cluster_0"]
    df["nm_cluster_1"] = 1.0 - df["nm_cluster_0"]

    # Introduce NaN in Age for a few rows
    df.loc[0, "Age"] = np.nan
    df.loc[3, "Age"] = np.nan
    return df


# ---------------------------------------------------------------------------
# Tests: ClinicalFeaturePreprocessor
# ---------------------------------------------------------------------------

class TestClinicalFeaturePreprocessor:

    def test_fit_transform_age_only(self, sample_clinical_df):
        prep = ClinicalFeaturePreprocessor(age_col="Age")
        prep.fit(sample_clinical_df)
        result = prep.transform(sample_clinical_df)

        assert "Age_scaled" in result.columns
        assert result.shape[0] == len(sample_clinical_df)
        assert result.shape[1] == 1
        # No NaN after imputation
        assert result["Age_scaled"].isna().sum() == 0

    def test_fit_transform_sex_only(self, sample_clinical_df):
        prep = ClinicalFeaturePreprocessor(sex_col="Sex")
        prep.fit(sample_clinical_df)
        result = prep.transform(sample_clinical_df)

        assert "Sex_F" in result.columns
        assert "Sex_M" in result.columns
        assert result.shape[1] == 2
        # Every row should have exactly one 1
        assert (result.sum(axis=1) == 1).all()

    def test_fit_transform_cluster_cols(self, sample_clinical_df):
        cluster_cols = ["mal_cluster_0", "mal_cluster_1", "nm_cluster_0", "nm_cluster_1"]
        prep = ClinicalFeaturePreprocessor(cluster_cols=cluster_cols)
        prep.fit(sample_clinical_df)
        result = prep.transform(sample_clinical_df)

        assert list(result.columns) == cluster_cols
        assert result.shape[0] == len(sample_clinical_df)

    def test_fit_transform_all_features(self, sample_clinical_df):
        cluster_cols = ["mal_cluster_0", "mal_cluster_1"]
        prep = ClinicalFeaturePreprocessor(
            age_col="Age",
            sex_col="Sex",
            cluster_cols=cluster_cols,
        )
        prep.fit(sample_clinical_df)
        result = prep.transform(sample_clinical_df)

        # Expected: Age_scaled + Sex_F + Sex_M + mal_cluster_0 + mal_cluster_1
        assert result.shape[1] == 5
        assert "Age_scaled" in result.columns
        assert "Sex_F" in result.columns
        assert "Sex_M" in result.columns
        assert result.isna().sum().sum() == 0

    def test_no_data_leakage_age(self, sample_clinical_df):
        """Fit on train, transform test — uses train stats."""
        train = sample_clinical_df.iloc[:30].copy()
        test = sample_clinical_df.iloc[30:].copy()
        # Make test Age have very different values
        test["Age"] = 100.0

        prep = ClinicalFeaturePreprocessor(age_col="Age")
        prep.fit(train)

        result_train = prep.transform(train)
        result_test = prep.transform(test)

        # Test values should be scaled using train mean/std (so they'll be large)
        assert result_test["Age_scaled"].mean() > result_train["Age_scaled"].mean()
        # Verify stats come from train
        train_age = train["Age"].dropna()
        assert abs(prep.age_fill_ - train_age.median()) < 1e-6

    def test_no_data_leakage_sex(self, sample_clinical_df):
        """Sex mode is learned from train."""
        train = sample_clinical_df.iloc[:30].copy()
        prep = ClinicalFeaturePreprocessor(sex_col="Sex")
        prep.fit(train)

        # Mode should be from train only
        valid_sex = train["Sex"][train["Sex"] != "n.a."]
        expected_mode = valid_sex.mode().iloc[0]
        assert prep.sex_mode_ == expected_mode

    def test_sex_missing_imputed_with_mode(self, sample_clinical_df):
        """Rows with 'n.a.' should be imputed with the mode."""
        # Force some rows to be "n.a."
        df = sample_clinical_df.copy()
        df.loc[0, "Sex"] = "n.a."
        df.loc[1, "Sex"] = "n.a."

        prep = ClinicalFeaturePreprocessor(sex_col="Sex")
        prep.fit(df)
        result = prep.transform(df)

        # All rows should have exactly one 1 (no all-zeros from missing)
        assert (result.sum(axis=1) == 1).all()

    def test_get_feature_names_out(self, sample_clinical_df):
        cluster_cols = ["mal_cluster_0", "mal_cluster_1"]
        prep = ClinicalFeaturePreprocessor(
            age_col="Age", sex_col="Sex", cluster_cols=cluster_cols
        )
        prep.fit(sample_clinical_df)
        names = prep.get_feature_names_out()

        assert list(names) == ["Age_scaled", "Sex_F", "Sex_M", "mal_cluster_0", "mal_cluster_1"]

    def test_empty_preprocessor(self, sample_clinical_df):
        """No features configured → empty DataFrame."""
        prep = ClinicalFeaturePreprocessor()
        prep.fit(sample_clinical_df)
        result = prep.transform(sample_clinical_df)
        assert result.shape[1] == 0

    def test_age_impute_mean(self, sample_clinical_df):
        prep = ClinicalFeaturePreprocessor(age_col="Age", age_impute_strategy="mean")
        prep.fit(sample_clinical_df)

        train_age = sample_clinical_df["Age"].dropna()
        assert abs(prep.age_fill_ - train_age.mean()) < 1e-6

    def test_invalid_age_strategy_raises(self, sample_clinical_df):
        prep = ClinicalFeaturePreprocessor(age_col="Age", age_impute_strategy="invalid")
        with pytest.raises(ValueError, match="age_impute_strategy"):
            prep.fit(sample_clinical_df)

    def test_cluster_nans_filled_with_zero(self):
        """Cluster columns with NaN (opposite class) should be 0."""
        df = pd.DataFrame({
            "mal_cluster_0": [1.0, 0.0, np.nan, np.nan],
            "mal_cluster_1": [0.0, 1.0, np.nan, np.nan],
        })
        prep = ClinicalFeaturePreprocessor(cluster_cols=["mal_cluster_0", "mal_cluster_1"])
        prep.fit(df)
        result = prep.transform(df)

        assert result.isna().sum().sum() == 0
        assert result.iloc[2, 0] == 0.0
        assert result.iloc[2, 1] == 0.0


# ---------------------------------------------------------------------------
# Tests: compute_clinical_associations
# ---------------------------------------------------------------------------

class TestClinicalAssociations:

    @pytest.fixture
    def association_df(self) -> pd.DataFrame:
        rng = np.random.RandomState(42)
        n = 200
        return pd.DataFrame({
            "Patient_group": rng.choice(
                ["Breast", "Lung", "Colorectal", "nonMalignant"], n
            ),
            "Age": rng.normal(60, 15, n),
            "Sex": rng.choice(["F", "M", "n.a."], n, p=[0.5, 0.45, 0.05]),
        })

    def test_compute_returns_sex_and_age(self, association_df):
        from genomics_dl.models.heterogeneity import compute_clinical_associations

        results = compute_clinical_associations(association_df)

        assert "sex" in results
        assert "age" in results
        assert results["sex"]["test"] == "chi2_contingency"
        assert results["age"]["test"] == "kruskal_wallis"
        assert 0 <= results["sex"]["p_value"] <= 1
        assert 0 <= results["age"]["p_value"] <= 1

    def test_cramers_v_range(self, association_df):
        from genomics_dl.models.heterogeneity import compute_clinical_associations

        results = compute_clinical_associations(association_df)
        assert 0 <= results["sex"]["cramers_v"] <= 1

    def test_descriptive_stats_by_group(self, association_df):
        from genomics_dl.models.heterogeneity import compute_clinical_associations

        results = compute_clinical_associations(association_df)
        desc = results["age"]["descriptive_by_group"]
        assert "mean" in desc.columns
        assert "median" in desc.columns
        assert len(desc) == association_df["Patient_group"].nunique()


# ---------------------------------------------------------------------------
# Tests: build_xy_enriched
# ---------------------------------------------------------------------------

class TestBuildXYEnriched:

    @pytest.fixture
    def enriched_df(self) -> pd.DataFrame:
        rng = np.random.RandomState(42)
        n = 40
        df = pd.DataFrame({
            "ENSG00000001": rng.rand(n),
            "ENSG00000002": rng.rand(n),
            "Class_group": ["Malignant"] * 30 + ["nonMalignant"] * 10,
            "Patient_group": (
                rng.choice(["Breast", "Lung", "Colorectal"], 30).tolist()
                + ["Asymptomatic controls"] * 10
            ),
            "Age": rng.normal(60, 15, n),
            "Sex": rng.choice(["F", "M"], n),
            "mal_cluster": [0.0] * 15 + [1.0] * 15 + [np.nan] * 10,
            "nm_cluster": [np.nan] * 30 + [0.0] * 5 + [1.0] * 5,
        })
        return df

    def test_y_has_19_original_classes(self, enriched_df):
        from genomics_dl.models.train_multiclass import build_xy_enriched

        gene_cols = ["ENSG00000001", "ENSG00000002"]
        X, y = build_xy_enriched(
            enriched_df, gene_cols=gene_cols,
            mal_cluster_col="mal_cluster",
            nm_cluster_col="nm_cluster",
            age_col="Age", sex_col="Sex",
        )

        # y should NOT contain any cluster labels
        assert not any("mal_cluster" in str(label) for label in y)
        assert not any("nm_cluster" in str(label) for label in y)
        # y should contain original labels
        assert "nonMalignant" in y
        assert any(label in y for label in ["Breast", "Lung", "Colorectal"])

    def test_x_contains_enriched_columns(self, enriched_df):
        from genomics_dl.models.train_multiclass import build_xy_enriched

        gene_cols = ["ENSG00000001", "ENSG00000002"]
        X, y = build_xy_enriched(
            enriched_df, gene_cols=gene_cols,
            mal_cluster_col="mal_cluster",
            nm_cluster_col="nm_cluster",
            age_col="Age", sex_col="Sex",
        )

        # Should have genes + cluster dummies + Age + Sex
        assert "ENSG00000001" in X.columns
        assert "ENSG00000002" in X.columns
        assert "mal_cluster_0" in X.columns
        assert "mal_cluster_1" in X.columns
        assert "nm_cluster_0" in X.columns
        assert "nm_cluster_1" in X.columns
        assert "Age" in X.columns
        assert "Sex" in X.columns

    def test_cluster_dummies_no_nans(self, enriched_df):
        from genomics_dl.models.train_multiclass import build_xy_enriched

        gene_cols = ["ENSG00000001", "ENSG00000002"]
        X, y = build_xy_enriched(
            enriched_df, gene_cols=gene_cols,
            mal_cluster_col="mal_cluster",
            nm_cluster_col="nm_cluster",
        )

        # Cluster dummies for opposite class samples should be 0, not NaN
        cluster_cols = [c for c in X.columns if "cluster_" in c]
        for col in cluster_cols:
            # build_xy_enriched creates dummies via (series == val).astype(float)
            # NaN == val → False → 0.0, so no NaN
            assert X[col].isna().sum() == 0

    def test_without_clusters_or_clinical(self, enriched_df):
        from genomics_dl.models.train_multiclass import build_xy_enriched

        gene_cols = ["ENSG00000001", "ENSG00000002"]
        X, y = build_xy_enriched(enriched_df, gene_cols=gene_cols)

        assert list(X.columns) == gene_cols
        assert len(y) == len(enriched_df)
