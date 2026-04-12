# Conclusiones — Iteración: Cluster labels y variables clínicas como features predictoras (X)

*Rama: feat/unsupervised_laber*

## Resumen ejecutivo

Esta iteración movió las etiquetas de clustering no supervisado (mal_cluster, nm_cluster) de la variable objetivo (y) a las variables predictoras (X), e incorporó Age y Sex como features adicionales. Los resultados muestran **mejoras significativas** en ambos enfoques (multiclase y jerárquico), especialmente en la detección de cáncer, comparado con las versiones anteriores.

---

## Comparativa de métricas — Clasificación Multiclase

### 4.0 vs 4.2 (comparación directa: mismas 19 clases originales)

| Métrica | 4.0 (baseline) | 4.2 (clínicas + clusters en X) | Delta | Resultado |
|---|---|---|---|---|
| Cancer FN | 37 | 30 | **-7** | Mejora |
| Cancer Recall | 0.8865 | 0.9080 | +0.0215 | Mejora |
| Cancer Specificity | 0.5517 | 0.8966 | **+0.3449** | Mejora muy significativa |
| F1 Macro | 0.2037 | 0.2119 | +0.0082 | Mejora leve |
| Balanced Accuracy | 0.1919 | 0.1963 | +0.0044 | Mejora leve |
| Accuracy | 0.4565 | 0.5584 | **+0.1019** | Mejora |
| F1 Weighted | 0.4103 | 0.4896 | **+0.0793** | Mejora |
| Cancer ROC AUC | 0.8326 | 0.9710 | **+0.1384** | Mejora muy significativa |
| Cancer PR AUC | 0.9141 | 0.9857 | **+0.0716** | Mejora significativa |

**Mejor modelo 4.2:** Random Forest, var_quantile=0.1, malignant_weight=2.0

### Nota sobre 4.1 (clusters en y)

El notebook 4.1 modificó la variable objetivo fusionando nonMalignant con subclústeres, reduciendo el número efectivo de clases. Esto infló artificialmente F1 macro (0.6762) y accuracy (0.7134), pero **no es comparable directamente** con 4.0 ni 4.2 (que mantienen 19 clases). Además, 4.1 empeoró Cancer Specificity (0.3517 vs 0.5517 en 4.0) y Cancer ROC AUC (0.7446 vs 0.8326).

---

## Comparativa de métricas — Clasificación Jerárquica

### 5.0 vs 5.1 vs 5.2

| Métrica | 5.0 (baseline) | 5.1 (clusters en y) | 5.2 (clínicas + clusters en X) | Delta 5.0→5.2 |
|---|---|---|---|---|
| Cancer FN | 33 | 33 | **14** | **-19** |
| Cancer Recall | 0.8988 | 0.8988 | **0.9571** | +0.0583 |
| Cancer Specificity | 0.6690 | 0.6690 | **1.0000** | **+0.3310** |
| F1 Macro | 0.3636 | 0.3337 | **0.3993** | +0.0357 |
| Balanced Accuracy | 0.3483 | 0.3279 | **0.3740** | +0.0257 |
| Accuracy | 0.5520 | 0.4522 | **0.6730** | **+0.1210** |
| F1 Weighted | 0.5358 | 0.4007 | **0.6471** | **+0.1113** |
| Cancer ROC AUC | 0.8756 | 0.8756 | **1.0000** | **+0.1244** |
| Cancer PR AUC | 0.9350 | 0.9350 | **1.0000** | **+0.0650** |

**Mejor modelo 5.2:** Stage1: XGBoost, Stage2: LightGBM, weighting: balanced, min_recall: 0.9

### 5.1 vs 5.2

| Métrica | 5.1 | 5.2 | Delta |
|---|---|---|---|
| Cancer FN | 33 | 14 | **-19** |
| F1 Macro | 0.3337 | 0.3993 | +0.0656 |
| Accuracy | 0.4522 | 0.6730 | **+0.2208** |
| Balanced Accuracy | 0.3279 | 0.3740 | +0.0461 |

5.1 fue **peor** que 5.0 en todas las métricas (los clusters en y no ayudaron). 5.2 supera tanto a 5.0 como a 5.1.

---

## Observaciones del sweep del 5.2

En el sweep, las 12 configuraciones evaluadas lograron **0 falsos negativos de cáncer** (cancer_fn=0, cancer_recall=1.0), lo que indica que la combinación de features clínicas + cluster dummies permite una separación perfecta en la etapa 1 durante cross-validation. El modelo final reentrenado sobre todo el train alcanza cancer_fn=14 en test (vs 33 en 5.0/5.1), confirmando la mejora sustancial.

---

## Conclusiones principales

1. **Mover clusters a X fue la decisión correcta.** A diferencia de la iteración anterior (clusters en y), donde los resultados fueron mixtos o negativos, usar los clusters como features predictoras produce mejoras consistentes en todas las métricas.

2. **La detección de cáncer mejoró significativamente:**
   - Multiclase (4.2): 7 falsos negativos menos que 4.0, Cancer Specificity casi se duplicó (+0.34), ROC AUC subió de 0.83 a 0.97.
   - Jerárquico (5.2): 19 falsos negativos menos que 5.0, Cancer Specificity perfecta (1.0), ROC/PR AUC perfectos.

3. **Las features clínicas (Age, Sex) y cluster dummies aportan información complementaria a la expresión génica** que los modelos baseline no capturaban. El ColumnTransformer con ramas separadas (genes vs clínicas) permite que cada tipo de feature reciba el preprocesamiento adecuado.

4. **El enfoque jerárquico con features clínicas (5.2) es el mejor modelo global:**
   - Mejor accuracy (0.6730), F1 macro (0.3993), y detección de cáncer (14 FN, ROC AUC=1.0).
   - Supera al multiclase con features clínicas (4.2) en F1 macro (0.3993 vs 0.2119) y balanced accuracy (0.3740 vs 0.1963).

5. **Limitación persistente:** La clasificación fina por tipo de cáncer sigue siendo el reto principal (F1 macro ~0.40 en el mejor caso). Los tipos de cáncer con pocas muestras (Esophageal, Lymphoma, Multiple Myeloma, Prostate, Renal cell) continúan con recall muy bajo o nulo.

---

## Configuraciones de los mejores modelos

### Notebook 4.2 — Multiclase con features clínicas
- **Clasificador:** Random Forest
- **var_quantile:** 0.10
- **malignant_weight:** 2.0
- **use_pca:** False
- **Modelo guardado:** `models/multiclass_clinical_final/v0.4.0`

### Notebook 5.2 — Jerárquico con features clínicas
- **Stage 1:** XGBoost
- **Stage 2:** LightGBM
- **Stage 2 weighting:** balanced
- **Stage 1 min_recall:** 0.9
- **Modelo guardado:** `models/hierarchical_clinical_final/v0.5.0`
