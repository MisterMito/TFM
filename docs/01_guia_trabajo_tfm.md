# Guía de trabajo del TFM y registro de iteraciones

Este documento recoge, de forma resumida, lo que se ha ido haciendo en el proyecto a medida que se avanza en el código y en las distintas ramas/pull requests del repositorio.

La idea es que sirva como:

- Guía temporal de las decisiones que se han tomado.
- Resumen de los datos utilizados en cada fase.
- Registro de las fuentes consultadas.
- Apoyo directo para la redacción de la memoria final.

En futuras iteraciones se pueden ir añadiendo nuevas secciones siguiendo la misma estructura.

---

## Iteración 1: Pipeline de preprocesado para GSE183635 rama (feat-exploratory_analysis)

### 1. Objetivo de la iteración

Construir un primer pipeline reproducible para el estudio basado en datos de plaquetas correspondiente a la serie GEO:

- **GSE183635**: [https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE183635](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE183635)

Estos datos se utilizan en el artículo:

> *Detection and localization of early- and late-stage cancers using platelet RNA*
> Sjors G.J.G. In ’t Veld et al.

El objetivo de esta iteración es ir desde:

- La matriz de conteos original de expresión de ARN mensajero de plaquetas,
hasta

- Un DataFrame final con expresión normalizada (TPM) + etiquetas de clase + metadatos clínicos, listo para análisis y modelado.

Estos mismos datos (conteos y metadatos) son los que utilizó la alumna anterior cuyo trabajo se está continuando. En este TFM se reutilizan como punto de partida, con la intención de:

1. Montar y validar el pipeline de datos y los primeros modelos con este estudio.
2. Una vez que el flujo y los modelos sean razonablemente estables, **buscar más datasets similares** y ampliar el conjunto de datos original.

### 2. Datos utilizados

#### 2.1. Datos de plaquetas (expresión y metadatos)

- **Serie GEO GSE183635**
  - URL: [https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE183635](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE183635)
  - Estudio asociado: *Detection and localization of early- and late-stage cancers using platelet RNA* (Sjors G.J.G. In ’t Veld et al.).
  - De aquí se obtienen:
    - La matriz de conteos de expresión de ARN mensajero de plaquetas por gen y muestra.
    - Los metadatos clínicos y de grupo de cada muestra (información de pacientes, grupos de enfermedad/controles, etc.).

- **Matriz de conteos de plaquetas**
  - Archivo en el proyecto: `data/raw/GSE183635_TEP_Count_Matrix.RData`
  - Contenido: matriz de conteos de expresión por gen y muestra, cargada en Python a partir de los datos descargados de GEO.

- **Metadatos clínicos y de grupo de muestras**
  - Proceden de los ficheros suplementarios asociados al artículo (tablas vinculadas a GSE183635), por ejemplo:
    - Identificador de muestra (Sample ID).
    - Grupo o condición clínica.
    - Edad, sexo, estadio, centro de procedencia.
    - Información de series de entrenamiento/evaluación/validación, etc.

Estos datos (conteos + metadatos) son los mismos que utilizó la alumna anterior. En este TFM se reorganizan y procesan de forma más sistemática para construir un conjunto de datos listo para ser reutilizado en distintos modelos.

#### 2.2. Datos de longitudes de genes

- **Anotación GENCODE v19 (genoma humano)**
  - Página de descarga: [https://www.gencodegenes.org/human/release_19.html](https://www.gencodegenes.org/human/release_19.html)
  - A partir de esta página se obtiene la anotación GENCODE v19 en formato GFF3, que se utiliza para calcular la longitud de los genes.

- **Fichero en el proyecto**
  - Archivo: `data/raw/gencode.v19.annotation.gff3`
  - Contenido: anotación GENCODE v19 (formato GFF3) sobre la que se calcula la longitud de los genes a partir de los exones.
  - Este fichero es grande y se mantiene solo en local, ignorado por Git mediante `.gitignore`.

Las longitudes de gen calculadas a partir de esta anotación se usan para normalizar la matriz de conteos de GSE183635 a TPM.

### 3. Estructura de notebooks y módulos

En esta iteración se organizaron los pasos principales en:

- Un notebook dedicado al cálculo de longitudes de gen a partir del GFF3 descargado desde la página de GENCODE v19.
- Un notebook centrado en:
  - La carga de la matriz de conteos de GSE183635.
  - El control de calidad básico.
  - El cálculo de TPM usando las longitudes de gen derivadas de GENCODE v19.
  - La integración con los metadatos obtenidos a partir de los ficheros asociados a GSE183635.

Además, se implementaron funciones reutilizables dentro del paquete del proyecto (por ejemplo, en `genomics_dl`) para que este flujo de datos se pueda aplicar a otros datasets similares en el futuro.

### 4. Cálculo de longitudes de gen y normalización a TPM

**Longitudes de gen**

- A partir de `data/raw/gencode.v19.annotation.gff3` se procesa el fichero en streaming (línea a línea) para evitar problemas de memoria.
- Se filtran las filas correspondientes a exones.
- Con las coordenadas de cada exón se calcula su longitud y se acumulan las longitudes por transcrito y por gen.
- Para cada gen se escoge la longitud del transcrito más largo como longitud representativa.
- Se eliminan las versiones de los identificadores de gen (por ejemplo, `ENSG00000123456.7` pasa a `ENSG00000123456`).
- El resultado se guarda en `data/interim/gene_lengths_gencode_v19_grch37.csv` como una tabla con `gene_id` y longitud en pares de bases.

**Normalización a TPM**

- Se carga la matriz de conteos de `data/raw/GSE183635_TEP_Count_Matrix.RData`.
- Se alinean los genes de la matriz de conteos con los genes para los que se ha calculado longitud.
- Con estas longitudes se calcula TPM:
  - Conversión de longitudes a kilobases.
  - Cálculo de RPK (lecturas por kilobase).
  - Normalización por la suma de RPK por muestra y escalado a 1e6 para obtener TPM.
- El resultado se reestructura de forma que cada fila sea una muestra y cada columna un gen, con los valores de expresión en TPM.

### 5. Metadatos, etiquetas y alineación con la expresión

**Metadatos y definición de clases**

- Se cargan los metadatos clínicos asociados a GSE183635 (por ejemplo, a partir de las tablas suplementarias del artículo).
- Estos metadatos incluyen:
  - Sample ID.
  - Grupo de paciente o condición (tipos de cáncer y controles).
  - Variables clínicas (edad, sexo, estadio, etc.).

- Se define una especificación de metadatos parametrizable (mediante una estructura tipo `MetadataSpec`) en la que se indica:
  - Ruta y formato del fichero de metadatos (Excel, CSV…).
  - Nombres de las columnas relevantes.
  - Valores especiales que deben tratarse como NA.
  - Mapeo de los distintos grupos de paciente a una etiqueta de clase binaria (cáncer / no cáncer).

- A partir de ese mapeo se construyen:
  - Una columna de clase binaria (cáncer vs no cáncer).
  - Una columna de grupo de paciente más detallada, que refleja el tipo de tumor o condición concreta.

**Alineación expresión–metadatos**

- La matriz de expresión en TPM y los metadatos obtenidos a partir de GSE183635 no están alineados de forma directa por el formato de los identificadores de muestra.
- Se implementa una lógica de alineación que:
  - Limpia y transforma los identificadores de la matriz de expresión para que puedan compararse con Sample ID.
  - Comprueba coincidencias exactas con la columna de Sample ID.
  - En los casos problemáticos, reproduce la siguiente estrategia: asumir que el orden de las muestras en la matriz de expresión y en los metadatos coincide y emparejarlas por posición.

- Una vez alineados, se fusionan expresión y metadatos en un único DataFrame con:
  - TPM por gen.
  - Sample ID.
  - Grupo de paciente.
  - Clase binaria.
  - Variables clínicas relevantes.

### 6. Guardado del dataset final y tratamiento de tipos

- El DataFrame final (TPM + etiquetas + metadatos) se guarda en `data/processed/` con un nombre claro: `gse183635_tep_tpm_labels.parquet`

- En este proceso se resuelven cuestiones de tipos de datos, como:
  - Valores textuales como “n.a.” en columnas numéricas (por ejemplo, edad) que se convierten a NA antes de transformarlas a tipo numérico.
  - Conversión de columnas numéricas que venían como texto a tipos numéricos adecuados.

### 7. Gestión de archivos grandes y Git

- El fichero `data/raw/gencode.v19.annotation.gff3` es de gran tamaño y no se versiona en Git.
- Inicialmente se llegó a añadir al repositorio, pero el push falló por el límite de tamaño de fichero.
- Se revirtió el commit y se configuró `data/raw/.gitignore` para excluir explícitamente `gencode.v19.annotation.gff3`.
- El archivo se mantiene únicamente en local, mientras que los derivados (como `gene_lengths_gencode_v19_grch37.csv`) sí se versionan.

### 8. Resultado de la iteración y plan a futuro

Al final de esta iteración se dispone de:

- Un pipeline reproducible para el estudio GSE183635 que va desde:
  - La matriz de conteos de expresión de ARN mensajero de plaquetas descargada de GEO
  hasta
  - Un dataset con TPM por gen, etiquetas de clase (binaria y de grupo de paciente) y metadatos clínicos.

- Código organizado y reutilizable en el proyecto para:
  - Calcular TPM a partir de longitudes de gen.
  - Cargar y estandarizar metadatos.
  - Alinear expresión y metadatos.

Estos datos son los mismos que utilizó la alumna anterior, y se usan aquí como base para:

1. Desarrollar y validar el pipeline de datos.
2. Entrenar los primeros modelos de clasificación (binaria y, posteriormente, multiclase).
3. Una vez consolidado este flujo con GSE183635, **buscar y añadir nuevos datasets de características similares** para ampliar el conjunto de datos, mejorar la robustez de los modelos y explorar mejor la generalización del enfoque.

---

## Iteración 2: Estudio de heterogeneidad en nonMalignant y preparación del feature engineering rama (feat-feature_engineering)

### 1. Objetivo de la iteración

En esta iteración se ha cerrado el estudio de heterogeneidad en el grupo **nonMalignant** (análisis no supervisado) y se ha dejado preparado el **pipeline de feature engineering** para los futuros modelos supervisados de **clasificación binaria** (cáncer vs nonMalignant).

La idea es que, a partir de esta base:

- Quede documentada la estructura interna del grupo nonMalignant (clusters, outliers, relación con metadatos).
- Esté definido y probado un flujo de transformación de la expresión (selección de genes, log, escalado, PCA) que se pueda reutilizar en la fase de modelos.


### 2. Punto de partida y split train/test

Se parte del dataset procesado:

- `data/processed/gse183635_tep_tpm_labels.parquet`
  (filas = muestras, columnas = genes en TPM + metadatos + etiquetas).

Sobre este dataset se ha realizado:

- Un **split train/test estratificado** por `Class_group`, generando:
  - `data/processed/gse183635_tep_tpm_train.parquet`
  - `data/processed/gse183635_tep_tpm_test.parquet`
- Un fichero auxiliar con los IDs de muestra y el split asignado, para trazabilidad.

A partir de este punto, **todo el análisis de heterogeneidad** se ha realizado exclusivamente sobre el **conjunto de entrenamiento**, respetando la separación train/test de cara a los futuros modelos supervisados.


### 3. Estudio de heterogeneidad en nonMalignant (EDA no supervisado)

Se ha filtrado el subconjunto de muestras con:

- `Class_group == "nonMalignant"` sobre el **train**.

Sobre este subconjunto se ha implementado un pipeline de análisis no supervisado que incluye:

1. **Transformación de expresión**
   - Aplicación de `log1p(TPM)` sobre los genes seleccionados para estabilizar la varianza.

2. **Estandarización por gen**
   - Uso de un `StandardScaler` por gen, centrando y escalando cada columna.

3. **PCA con selección automática de componentes**
   - Se ajusta un PCA completo.
   - Se elige el menor número de componentes tal que la varianza explicada acumulada ≥ 90 %.
   - Este espacio PCA sirve como base para:
     - Representaciones globales (PC1–PC2).
     - Puntos de partida para métodos no lineales.

4. **Embeddings para visualización**
   - Proyección en:
     - PC1–PC2 para una primera visión global.
     - UMAP sobre el espacio PCA (2D), para explorar la estructura local y posibles subgrupos dentro de nonMalignant.


### 4. Clustering no supervisado y selección de modelo

Sobre el embedding (PCA / PCA+UMAP) se ha definido una búsqueda de modelos de clustering probando:

- **KMeans** para distintos valores de `k`.
- **Gaussian Mixture Models (GMM)** para los mismos valores de `k`.
- **AgglomerativeClustering** (linkage `"ward"`) como referencia adicional.

Cada combinación algoritmo–k se evalúa con varias métricas internas:

- Coeficiente de **silhouette**.
- Índice de **Calinski–Harabasz**.
- Índice de **Davies–Bouldin**.

Durante la exploración se observó que:

- Una configuración de **Agglomerative** con `linkage="average"` generaba una partición con un cluster muy grande (577 muestras) y otro de solo 1 muestra, con buenas métricas pero estructura claramente degenerada (cluster de “outlier”).
- Esta variante se ha descartado del grid de modelos.

Tras restringir la búsqueda a configuraciones razonables, la solución seleccionada ha sido:

- **KMeans con k = 2**, con silhouette medio ≈ 0.28.
- Tamaños de cluster:
  - `cluster_auto = 0`: 385 muestras.
  - `cluster_auto = 1`: 193 muestras.

Se ha añadido la columna `cluster_auto` a los metadatos de nonMalignant (`meta_nm`) y se ha propagado a los metadatos de entrenamiento (`meta_train`) como información descriptiva sobre la estructura interna de nonMalignant.


### 5. Asociación de clusters con metadatos

Para interpretar los clusters nonMalignant, se han generado tablas de contingencia y tests estadísticos entre `cluster_auto` y distintas variables.

De forma cualitativa:

- Un cluster está enriquecido en **controles asintomáticos**, mientras que el otro agrupa una mayor proporción de pacientes con **patologías no malignas específicas** (angina, antiguos sarcomas, etc.).
- Existe una asimetría clara en la distribución por **institución**, lo que apunta a una combinación de:
  - Heterogeneidad clínica real.
  - Posibles efectos de centro/lote.
- Edad y sexo no explican la separación de clusters.


### 6. Caracterización de los clusters nonMalignant

Se ha creado un pequeño módulo reutilizable en:

- `genomics_dl/models/heterogeneity.py`

con funciones para caracterizar los clusters a nivel de expresión:

1. **`differential_expression_two_clusters(X_log, cluster_labels)`**
   - Trabaja sobre `X_log = log1p(TPM)` para nonMalignant.
   - Asume exactamente dos clusters.
   - Para cada gen, calcula:
     - Media de expresión en cada cluster.
     - Diferencia de medias y diferencia absoluta.
     - p-valor (t-test de Welch).
     - q-valor (corrección FDR de Benjamini–Hochberg).
   - Devuelve un DataFrame ordenado por la diferencia absoluta de expresión.

2. **`run_nonmalignant_cluster_characterization(...)`**
   - Función envoltorio que:
     - Ejecuta la comparación de expresión entre los dos clusters.
     - Opcionalmente guarda un CSV con los resultados.
     - Opcionalmente genera un heatmap con los genes más diferenciales (top N), ordenando las muestras por `cluster_auto`.

Se ha ejecutado este flujo sobre `X_nm_log` y `meta_nm`, obteniendo:

- Un DataFrame con los resultados de expresión diferencial entre los dos clusters nonMalignant.
- Figuras/archivos en `reports/figures/nonmalignant/`, pensados para documentar el análisis en la memoria del TFM.


### 7. Detección de outliers en el espacio PCA de nonMalignant

En el mismo módulo de heterogeneidad se ha añadido una función para marcar muestras extremas en el embedding PCA:

- **`flag_pca_distance_outliers(X_pca, quantile=0.99, center="mean")`**

Esta función:

- Calcula la distancia euclídea de cada muestra nonMalignant al centro del embedding PCA (media o mediana).
- Marca como outliers las muestras cuya distancia está por encima de un cuantil dado (por defecto, 99 %, es decir, top 1 % de distancias).

Con esto se ha generado:

- Un flag booleano `is_outlier_pca` en los metadatos de nonMalignant (`meta_nm`).

### 8. Pipeline de feature engineering preparado (aún no utilizado en modelos)

Se ha dejado definido un módulo genérico de **feature engineering** para expresión génica en:

- `genomics_dl/features.py`

con un pequeño pipeline que encapsula:

1. **Selección de genes de alta varianza**
   - Función tipo `select_high_var_genes` para filtrar genes según un cuantil de varianza.

2. **Ajuste del preprocesado en train**

   - `fit_expression_preprocessor(X_train, var_quantile=0.2)`:
     - Selecciona genes por varianza (por encima de un cuantil dado).
     - Aplica `log1p`.
     - Ajusta un `StandardScaler` sobre los genes seleccionados.
     - Devuelve la información necesaria para aplicar la misma transformación después.

3. **Aplicación del preprocesado en cualquier conjunto**

   - `transform_expression_preprocessor(X, preproc)`:
     - Aplica exactamente el mismo filtrado de genes, `log1p` y escalado a cualquier matriz (por ejemplo, test).

4. **PCA con selección automática de componentes**

   - `fit_pca_auto(X_scaled, var_threshold=0.9, max_components=...)`:
     - Ajusta un PCA y selecciona el número de componentes para alcanzar un umbral de varianza explicada (por defecto, 90 %).
   - `transform_pca(X_scaled, pca_model)`:
     - Proyecta nuevos datos en el espacio PCA.

Este pipeline está listo para ser utilizado de forma consistente en:

- Datos de entrenamiento (train).
- Datos de validación/test.

Todavía **no se ha integrado** en los notebooks de clasificación binaria: se utilizará en la siguiente fase, cuando se monten los primeros modelos supervisados.


### 9. Decisiones de diseño y cierre de la iteración

- Las variables `cluster_auto` e `is_outlier_nonmalignant_pca` se han añadido a los metadatos de entrenamiento como información descriptiva y para documentar la heterogeneidad interna del grupo nonMalignant.
- No está previsto utilizarlas como features de entrada en el **primer modelo binario** (cáncer vs nonMalignant), ya que:
  - La clusterización se ha definido solo sobre nonMalignant.
  - El objetivo principal de esta parte es exploratorio y descriptivo.

- Todo el análisis no supervisado (clustering, outliers, DE entre clusters) se ha realizado únicamente sobre el **conjunto de entrenamiento**, manteniendo la separación train/test para la futura evaluación de los modelos.

Con estos pasos se da por **cerrada la parte de heterogeneidad en el grupo nonMalignant** y queda preparado el diseño del **feature engineering**. La siguiente etapa del proyecto se centrará en la construcción y evaluación de los modelos de **clasificación binaria**.

---

## Iteración 3: Modelos binarios (Malignant vs nonMalignant) con pipeline reproducible y tracking en MLflow (feat/binary_model)

### 1. Objetivo de la iteración

En esta iteración se ha construido y dejado operativo el flujo completo para entrenar y evaluar modelos supervisados binarios (cáncer vs no cáncer) sobre el dataset procesado, integrando:

- El **feature engineering** dentro de un pipeline reproducible estilo scikit-learn.
- El **tracking de experimentos** con MLflow.
- Una lógica explícita de **selección de umbral** orientada a reducir falsos negativos.

El objetivo es dejar preparado un “marco” estable de entrenamiento binario sobre el que se puedan iterar modelos y, posteriormente, extender el enfoque a clasificación multiclase.


### 2. Definición del problema y etiquetas

Se fijó el problema binario como:

- `y = 1` si `Class_group == "Malignant"`
- `y = 0` si `Class_group == "nonMalignant"`

Se mantuvo la **separación train/test** definida en iteraciones anteriores, sin modificar los splits ya establecidos.


### 3. Integración del feature engineering en un pipeline end-to-end

El preprocesado de expresión se migró a un enfoque completamente integrado en un **scikit-learn Pipeline**, con el objetivo de:

- Evitar fugas de información (leakage) durante validación cruzada.
- Poder entrenar, validar y guardar el modelo como un único objeto serializable.
- Facilitar el despliegue y la reutilización del modelo en otros entornos.

El pipeline incluye etapas para:

- **Selección de genes** (por ejemplo, por varianza).
- **Transformación logarítmica** de la expresión (`log1p`).
- **Estandarización** (StandardScaler).
- **PCA opcional**, con selección automática del número de componentes cuando se usa esta reducción de dimensionalidad.

Dado que el dataset mezcla **genes** y **metadatos**, se añadió además una etapa explícita para fijar qué columnas de entrada se consideran features de expresión.


### 4. Control explícito de columnas (genes vs metadatos)

Para evitar ambigüedades y errores en la construcción de features, se definió de forma clara:

- Una lista `metadata_cols` con las columnas de metadatos/variables clínicas y de control.
- Un criterio sistemático para identificar columnas de genes (por ejemplo, identificadores ENSG…).

A partir de esto:

- Se asegura que el pipeline trabaje siempre con la lista correcta de columnas de expresión.
- Se añadió un transformador específico (`FeatureColumnSelector`) dentro del pipeline para **fijar las columnas de entrada** y evitar problemas en validación cruzada (por ejemplo, con `cross_val_predict`), donde las estructuras de DataFrame pueden variar si no se controlan bien las columnas.


### 5. Métricas y foco clínico: minimizar falsos negativos

Se definió como prioridad del problema binario el **control de falsos negativos**, es decir:

- Minimizar la tasa de falsos negativos (FNR = 1 − recall/sensibilidad).

Además de métricas estándar (ROC-AUC, PR-AUC, F1, balanced accuracy), se incorporó un conjunto de métricas más alineadas con un uso clínico:

- Número de FN y FNR.
- Sensibilidad (recall) y especificidad.
- Matriz de confusión y métricas derivadas (precision, NPV, FPR, etc.).

Esto permite evaluar no solo la “calidad global” del modelo, sino también su comportamiento en escenarios donde detectar correctamente casos Malignant es más importante que reducir falsos positivos.


### 6. Selección de umbral orientada a sensibilidad y especificidad

En lugar de utilizar simplemente el umbral estándar de 0.5, se introdujo un mecanismo de **selección de umbral configurable** (`threshold_objective`), basado en la curva ROC.

La lógica permite:

- Explorar distintos puntos sobre la curva ROC.
- Elegir umbrales que **maximicen la especificidad** bajo una condición mínima de recall (priorizando así el control de falsos negativos).
- Separar claramente:
  - La evaluación de la calidad del ranking del modelo (AUC).
  - La elección del punto de decisión clínicamente relevante (threshold).

De esta forma, el modelo puede tener un buen ranking global, pero el punto operativo concreto se ajusta explícitamente a las necesidades del problema.


### 7. Entrenamiento y comparación de múltiples clasificadores

Se amplió el enfoque desde un baseline inicial (por ejemplo, una regresión logística) a un conjunto más amplio de clasificadores, permitiendo barrer:

- Distintos algoritmos de clasificación.
- Diferentes **pesos de clase** o penalización de la clase positiva (Malignant).
- Variantes de preprocesado (con y sin PCA, distintos quantiles de varianza, etc.).
- Hiperparámetros específicos de cada modelo.

El barrido se configuró con **validación cruzada** (por defecto, `cv_splits = 8`) para:

- Estimar la estabilidad de las métricas.
- Evitar sobreajuste a una única partición train/val.
- Comparar modelos bajo un procedimiento homogéneo.


### 8. Gestión de outputs locales y reducción de “ruido”

Durante esta iteración se revisó qué artefactos se generan localmente al entrenar y hacer tracking. Se distinguió entre:

- Información que interesa preservar y versionar (modelo final, métricas agregadas, configuración).
- Ficheros de trabajo internos (por ejemplo, una base de datos SQLite local tipo `mlflow.db`) que no deben acabar en el control de versiones ni en `notebooks/`.

Se trabajó para:

- Centralizar la **trazabilidad de experimentos en MLflow**.
- Mantener el repositorio más limpio, guardando únicamente lo necesario como resultado estable (bundle del modelo).


### 9. Estructura de guardado del modelo y bundle final

El guardado del resultado se consolidó en un **bundle de modelo**, que incluye:

- El **pipeline completo** (preprocesado + clasificador) serializado en un único fichero, por ejemplo `model.pkl`.
- Métricas y parámetros relevantes del experimento.
- Información de firma y ejemplos de entrada cuando aplica.

Además, se añadió una **copia completa del modelo en formato MLflow**, incluyendo:

- `MLmodel`
- `conda.yaml` u otro descriptor de entorno
- `input_example`

Este bundle dual (pickle + MLflow) facilita:

- El uso directo del modelo en el propio proyecto.
- La posibilidad de desplegarlo en entornos compatibles con MLflow.

Se añadió una validación en notebook para comprobar que el bundle final contiene tanto el `MLmodel` como el `model.pkl` esperados.


### 10. Notebook de baselines y pruebas de robustez

Se actualizó el notebook:

- `notebooks/3.0-ssic-binary-baselines.ipynb`

para:

- Ejecutar una **malla más amplia** de experimentos binarios.
- Recopilar, ordenar y mostrar los resultados en un **DataFrame resumen** que permita seleccionar el mejor candidato.
- Incluir una prueba más exigente (“pesada”):
  - Reentrenar el mejor modelo con **validación cruzada más estricta** (por ejemplo, `CV = 10`).
  - Aumentar límites como `max_iter` para asegurar convergencia.
  - Comparar las métricas de esta versión “refinada” con la versión estándar usada en el bundle final.


### 11. Resultado final de la iteración y siguientes pasos

Al finalizar esta iteración queda definido un flujo reproducible para:

- Preparar correctamente las features (separando genes de metadatos).
- Entrenar múltiples modelos binarios con validación cruzada.
- Elegir un umbral de decisión orientado a **reducir falsos negativos** y controlar la especificidad.
- Registrar experimentos y resultados en **MLflow**.
- Guardar un único **artefacto final** (bundle) listo para versionado y uso posterior.

Con esto, la Iteración 3 deja establecido el **marco de entrenamiento binario** y el sistema de tracking/versionado. Los siguientes pasos se centrarán en:

- Consolidar el mejor modelo binario según criterios clínicos definidos.
- Extender la misma metodología (pipeline + MLflow + selección de umbral) a problemas de **clasificación multiclase** (tipos de cáncer).

---
