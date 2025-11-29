# Contexto del TFM: Deep Learning en Genómica

## 1. Planteamiento general

Este Trabajo de Fin de Máster se sitúa en la intersección entre la genómica y el aprendizaje automático/aprendizaje profundo, utilizando datos de transcripción de ARN mensajero de plaquetas para estudiar su utilidad como biomarcadores de cáncer.

El trabajo continúa la línea iniciada por una alumna anterior. En ese proyecto se partía de datos de transcripciones de ARN mensajero de plaquetas y se obtenían conteos de expresión por gen. La idea de fondo es que, en pacientes sanos, el patrón de expresión génica en plaquetas debería ser aproximadamente cuasi-aleatorio, mientras que en pacientes con cáncer se observan perfiles distintos.

Estos cambios se relacionan con el hecho de que los tumores generan nuevos vasos sanguíneos a los que se adhieren las plaquetas y, a través de un proceso conocido como **educación por tumores**, las plaquetas adquieren “información” del entorno tumoral. Esto da lugar a un perfil característico asociado al tipo de cáncer, que en principio podría permitir determinar el tipo de cáncer incluso antes de que las imágenes médicas lo muestren con claridad.

En la práctica, estos perfiles son muy ruidosos y los patrones no son triviales de extraer. La limpieza, el preprocesado y la modelización de los datos se convierten, por tanto, en el núcleo de la dificultad del trabajo.

## 2. Trabajo previo y punto de partida

En el TFM previo se planteó un problema de **clasificación binaria**: distinguir entre pacientes con cáncer y pacientes sin cáncer a partir de datos de expresión de ARN de plaquetas.

Ese trabajo definió un flujo básico de obtención de datos, preprocesado y construcción de un modelo de clasificación binaria (cáncer / no cáncer). La presente memoria toma como punto de partida esa idea y su implementación general, con el objetivo de ampliarla y generalizarla.

## 3. Objetivo general del TFM

El objetivo general de este TFM es explorar y desarrollar modelos de clasificación, tanto de **machine learning** como de **deep learning**, que aprovechen datos de expresión génica de plaquetas para:

- Distinguir entre pacientes con cáncer y pacientes sin cáncer (clasificación binaria).
- Diferenciar entre varios tipos de cáncer (clasificación multiclase), cuando la calidad y la cantidad de datos lo permitan.

## 4. Objetivos específicos

A partir de este objetivo general, se plantean los siguientes objetivos específicos:

1. **Revisión del trabajo previo**
   - Leer y analizar la memoria de la alumna anterior.
   - Entender cómo se construyeron los datos, cómo se definieron las etiquetas y qué modelos se utilizaron en la clasificación binaria inicial.

2. **Actualización del estado del arte**
   - Revisar la literatura reciente sobre plaquetas “educadas por tumores” como biomarcadores.
   - Revisar trabajos que aplican modelos de aprendizaje automático y profundo a datos de expresión génica para la clasificación de cáncer.
   - Situar el TFM dentro de este contexto.

3. **Búsqueda y selección de datasets**
   - Identificar datasets de RNA-seq de plaquetas relacionados con cáncer.
   - Valorar qué datasets son más adecuados para el objetivo del trabajo y cómo se pueden combinar o comparar entre sí.

4. **Preparación y limpieza de datos**
   - Cargar las matrices de conteos de expresión por gen.
   - Obtener y aplicar longitudes de gen adecuadas para normalizar los datos (por ejemplo, mediante TPM).
   - Integrar los conteos normalizados con metadatos clínicos (edad, sexo, estadio, grupo de paciente, etc.).
   - Construir un conjunto de datos final listo para el modelado, con expresión, etiquetas de clase y metadatos relevantes.

5. **Estudio de la heterogeneidad de los datos**
   - Analizar la variabilidad entre muestras dentro de un mismo grupo (por ejemplo, mismo tipo de cáncer).
   - Analizar las diferencias entre grupos (cáncer vs no cáncer, distintos tipos de cáncer).
   - Identificar posibles fuentes de ruido o sesgo debidas al diseño experimental o a diferencias entre estudios.

6. **Clasificación binaria (cáncer vs no cáncer)**
   - Plantear el problema de clasificación binaria con distintos modelos de ML y DL.
   - Evaluar su rendimiento con métricas adecuadas y analizar sus limitaciones.

7. **Clasificación multiclase (tipos de cáncer)**
   - Extender el enfoque hacia un problema multiclase en el que se intente distinguir entre diferentes tipos de cáncer a partir de los perfiles de plaquetas.
   - Estudiar la dificultad añadida por la heterogeneidad entre tipos de tumor y por posibles desbalances de clases.

8. **Documentación y reproducibilidad**
   - Mantener el código organizado en una estructura clara y modular.
   - Documentar los pasos de preprocesado, construcción de características y modelado, de manera que el pipeline pueda reutilizarse con otros datasets y servir de base para trabajos posteriores.

## 5. Dificultades principales esperadas

Las principales dificultades previstas en el desarrollo del TFM son:

- **Calidad y limpieza de los datos**: los perfiles de expresión de plaquetas son ruidosos, y las decisiones de preprocesado (filtros, normalización, selección de genes, etc.) tienen un impacto fuerte en los resultados.
- **Heterogeneidad entre pacientes y entre estudios**: pueden aparecer diferencias importantes debidas a protocolos experimentales, centros de origen, tipos de cáncer y otros factores clínicos.
- **Formulación del problema multiclase**: pasar de un enfoque binario (cáncer / no cáncer) a uno multiclase incrementa la complejidad, especialmente si hay clases poco representadas o muy similares entre sí.

Este documento sirve como marco general para situar el TFM y para guiar las decisiones de diseño de datos y modelos a lo largo del proyecto.
