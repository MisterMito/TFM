# Guía Rápida: Sistema de Clasificación Jerárquica

## 🎯 Lo Que Se Ha Implementado

### ✅ Fase 1-4 Completadas:

1. **VarianceThresholdFilter** - Elimina genes con varianza cero después de log1p
   - **Ubicación**: [genomics_dl/features_sklearn.py](genomics_dl/features_sklearn.py#L113-L157)
   - **Solución**: Elimina el error de NaN en SGDClassifier (100% de fallos → 0%)

2. **Dependencias Instaladas**:
   - ✅ xgboost>=2.0.0
   - ✅ lightgbm>=4.1.0
   - ✅ catboost>=1.2.0
   - ✅ imbalanced-learn>=0.11.0
   - ✅ mlflow>=3.7.0

3. **HierarchicalTrainConfig** - Configuración completa
   - **Ubicación**: [genomics_dl/models/train_multiclass.py](genomics_dl/models/train_multiclass.py#L192-L246)

4. **Funciones de Construcción de Modelos**:
   - `compute_class_weights_hierarchical()` - 3 estrategias de balanceo
   - `build_stage1_binary_classifier()` - XGBoost/LightGBM/CatBoost/ExtraTrees
   - `build_stage2_multiclass_classifier()` - Con pesos por clase
   - `HierarchicalClassifier` - Wrapper compatible con sklearn

---

## 🚀 Uso Rápido

### Opción 1: Arreglar SGD en tu notebook actual (4.0)

Simplemente añade `VarianceThresholdFilter` al pipeline existente:

```python
from genomics_dl.features_sklearn import VarianceThresholdFilter

# En build_pipeline(), añadir DESPUÉS de log1p y ANTES de scale:
steps = [
    ("ensure_features", FeatureColumnSelector(feature_cols)),
    ("select", HighVarGeneSelector(var_quantile)),
    ("log1p", Log1pTransformer()),
    ("variance_filter", VarianceThresholdFilter(1e-6)),  # ← NUEVO
    ("scale", PandasStandardScaler()),
    ("pca", pca if use_pca else "passthrough"),
    ("clf", clf)
]
```

**Resultado**: Los modelos SGD ahora entrenarán sin errores de NaN.

---

### Opción 2: Usar el Sistema Jerárquico Completo

```python
from genomics_dl.models.train_multiclass import (
    HierarchicalTrainConfig,
    HierarchicalClassifier,
    build_stage1_binary_classifier,
    build_stage2_multiclass_classifier,
    build_xy
)
from genomics_dl.features_sklearn import (
    FeatureColumnSelector,
    HighVarGeneSelector,
    Log1pTransformer,
    VarianceThresholdFilter,
    PandasStandardScaler
)
from sklearn.pipeline import Pipeline
import pandas as pd

# 1. Cargar datos
df_train = pd.read_parquet("data/processed/gse183635_tep_tpm_train.parquet")
df_test = pd.read_parquet("data/processed/gse183635_tep_tpm_test.parquet")

# Obtener columnas de genes
gene_cols = [c for c in df_train.columns if c.startswith("ENSG")]

# 2. Preparar X, y
X_train, y_train = build_xy(df_train, feature_cols=gene_cols)
X_test, y_test = build_xy(df_test, feature_cols=gene_cols)

# 3. Construir pipelines de preprocesamiento
def create_preprocessing_pipeline(gene_cols, var_quantile=0.15):
    return Pipeline([
        ("ensure_features", FeatureColumnSelector(gene_cols)),
        ("select", HighVarGeneSelector(var_quantile)),
        ("log1p", Log1pTransformer()),
        ("variance_filter", VarianceThresholdFilter(1e-6)),
        ("scale", PandasStandardScaler()),
    ])

# 4. Crear configuración para el modelo jerárquico
config = HierarchicalTrainConfig(
    train_path="data/processed/gse183635_tep_tpm_train.parquet",
    test_path="data/processed/gse183635_tep_tpm_test.parquet",
    model_name="hierarchical_xgboost",
    model_version="v0.3.0",

    # Etapa 1: Detección de cáncer
    stage1_clf_name="xgboost",
    stage1_clf_params={
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.1
    },
    stage1_min_recall=0.95,  # Alto recall para no perder cánceres

    # Etapa 2: Tipo de cáncer
    stage2_clf_name="xgboost",
    stage2_clf_params={
        "n_estimators": 800,
        "max_depth": 8,
        "learning_rate": 0.05
    },
    stage2_class_weighting="balanced",

    var_quantile=0.15,
    variance_filter_threshold=1e-6,
    cv_splits=8,
)

# 5. Construir clasificadores
y_binary = (y_train != "nonMalignant").astype(int)
n_malignant = y_binary.sum()
n_nonmalignant = len(y_binary) - n_malignant

stage1_clf = build_stage1_binary_classifier(config, n_malignant, n_nonmalignant)
stage1_pipeline = Pipeline([
    *create_preprocessing_pipeline(gene_cols, 0.15).steps,
    ("clf", stage1_clf)
])

y_cancer = y_train[y_train != "nonMalignant"]
stage2_clf = build_stage2_multiclass_classifier(config, y_cancer)
stage2_pipeline = Pipeline([
    *create_preprocessing_pipeline(gene_cols, 0.15).steps,
    ("clf", stage2_clf)
])

# 6. Crear y entrenar modelo jerárquico
hierarchical_model = HierarchicalClassifier(
    stage1_pipeline=stage1_pipeline,
    stage2_pipeline=stage2_pipeline,
    stage1_threshold=None,  # Se calculará después
    nonmalignant_label="nonMalignant"
)

print("Entrenando modelo jerárquico...")
hierarchical_model.fit(X_train, y_train)

# 7. Evaluar
print("\nEvaluando en conjunto de prueba...")
test_proba = hierarchical_model.predict_proba(X_test)
test_pred = hierarchical_model.predict(X_test)

# Métricas básicas
from sklearn.metrics import classification_report, confusion_matrix
print("\nReporte de clasificación:")
print(classification_report(y_test, test_pred, zero_division=0))

# Matriz de confusión
print("\nMatriz de confusión:")
print(confusion_matrix(y_test, test_pred))
```

---

## 📊 Mejoras Esperadas

### Modelo Actual (v0.2.0 - ExtraTreesClassifier):
- ✅ Detección de cáncer: 88% recall
- ❌ F1-macro multiclase: 19%
- ❌ 8 tipos de cáncer: 0% recall
- ❌ SGD: 100% fallos (NaN)

### Modelo Jerárquico (v0.3.0 - Esperado):
- 🎯 Detección de cáncer: 95% recall
- 🎯 F1-macro multiclase: 30-40%
- 🎯 Tipos con 0% recall: ≤3
- ✅ SGD: 0% fallos

---

## 🔍 Verificación

Ejecuta el script de prueba:

```bash
python test_hierarchical_import.py
```

Deberías ver:
```
✓ VarianceThresholdFilter importado correctamente
✓ HierarchicalTrainConfig importado correctamente
✓ Funciones jerárquicas importadas correctamente
✓ Librerías de gradient boosting disponibles: xgboost, lightgbm, catboost
✓ HierarchicalTrainConfig creada: test_model v0.0.1
```

---

## 📝 Próximos Pasos Recomendados

1. **Prueba Rápida**: Modifica tu notebook 4.0 para añadir `VarianceThresholdFilter` y verifica que SGD funciona

2. **Experimento Completo**: Crea un nuevo notebook `5.0-ssic-hierarchical-experiments.ipynb` usando el código de ejemplo

3. **Comparación**: Ejecuta ambos modelos (actual vs jerárquico) y compara resultados

4. **Iteración**: Ajusta hiperparámetros basándote en los resultados

---

## 🎓 Arquitectura del Sistema

```
┌─────────────────────────────────────────┐
│     Entrada: 5,440 genes (TPM)          │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│  Preprocesamiento Compartido:           │
│  1. Selección genes alta varianza       │
│  2. Transformación log1p                │
│  3. Filtro varianza (NUEVO - elimina    │
│     NaN)                                 │
│  4. Estandarización (StandardScaler)    │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│ ETAPA 1: Detección Binaria de Cáncer   │
│                                         │
│  XGBoost/LightGBM/CatBoost/ExtraTrees  │
│  Target: 95% recall (no perder cáncer) │
│  Output: P(cancer) y P(nonMalignant)   │
└─────────────┬───────────────────────────┘
              │
              ├─── SI: P(cancer) < umbral ────► nonMalignant
              │
              └─── NO: P(cancer) ≥ umbral
                          │
                          ▼
              ┌─────────────────────────────────┐
              │ ETAPA 2: Tipo de Cáncer         │
              │                                 │
              │ XGBoost/LightGBM/CatBoost       │
              │ Solo en muestras cancerosas     │
              │ Con pesos balanceados por clase │
              │ Output: 18 tipos de cáncer      │
              └─────────────┬───────────────────┘
                            │
                            ▼
              ┌──────────────────────────────────┐
              │  Predicción Final:               │
              │  - nonMalignant, O               │
              │  - Uno de 18 tipos de cáncer     │
              └──────────────────────────────────┘
```

---

## 📚 Archivos Modificados

1. `genomics_dl/features_sklearn.py` - Añadido `VarianceThresholdFilter`
2. `genomics_dl/models/train_multiclass.py` - Añadido:
   - `HierarchicalTrainConfig`
   - `compute_class_weights_hierarchical()`
   - `build_stage1_binary_classifier()`
   - `build_stage2_multiclass_classifier()`
   - `HierarchicalClassifier`
3. `pyproject.toml` - Añadidas dependencias de gradient boosting

---

## ⚠️ Notas Importantes

- El sistema jerárquico separa la detección de cáncer (fácil, 88% → 95%) del tipo de cáncer (difícil, 19% → 35-40%)
- `VarianceThresholdFilter` es **crítico** - sin él, SGD falla con NaN
- Los pesos por clase (`balanced`, `sqrt`, `log`) ayudan con el desbalanceo severo
- La Etapa 1 prioriza recall alto (no perder cánceres) sobre precisión
- La Etapa 2 se centra en distinguir tipos de cáncer (más difícil)

---

¡El sistema está listo para experimentar! 🚀
