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
