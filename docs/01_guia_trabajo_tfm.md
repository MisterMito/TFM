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
