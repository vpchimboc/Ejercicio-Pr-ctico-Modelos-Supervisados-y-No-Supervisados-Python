# Ejercicio Práctico de Machine Learning - Análisis de Rendimiento Académico

## Descripción General

Este ejercicio práctico demuestra la aplicación de dos tipos fundamentales de algoritmos de Machine Learning sobre un conjunto de datos de rendimiento académico:

1. **Modelo Supervisado (Clasificación):** Regresión Logística para predecir la aprobación de asignaturas
2. **Modelo No Supervisado (Clustering):** K-Means para segmentar patrones de rendimiento académico

### Dataset
- **Fuente:** `academic_performance_master.csv`
- **Registros Válidos:** 44,915 después del preprocesamiento
- **Período:** 2020-2024
- **Características Clave:** Nota 1 (0-100), Nota Final (0-10), Carrera, Nivel, Estado de Aprobación

---

## Artefactos Generados

### 1. **Modelos Entrenados**
- `logistic_regression_model.pkl` - Modelo de Regresión Logística entrenado
- `kmeans_model.pkl` - Modelo K-Means entrenado (K=4)
- `scaler_kmeans.pkl` - Escalador StandardScaler para K-Means

### 2. **Métricas y Resultados**
- `logistic_regression_metrics.json` - Métricas de evaluación del modelo LR (Accuracy: 98.19%)
- `kmeans_centroids.csv` - Centroides de los 4 clusters de K-Means

### 3. **Datos Preprocesados**
- `df_clasificacion_preprocessed.csv` - Dataset preparado para Regresión Logística
- `df_clustering_preprocessed.csv` - Dataset preparado para K-Means

### 4. **Visualizaciones**
- `kmeans_elbow_method.png` - Gráfico del Método del Codo para selección de K
- `kmeans_clusters.png` - Visualización de los 4 clusters en el espacio de notas

### 5. **Documentación**
- `ejercicio_practico_ml.md` - Notebook de explicación completo en Markdown
- `README.md` - Este archivo

### 6. **Aplicación Interactiva**
- `streamlit_app.py` - Aplicación Streamlit para exploración interactiva de los modelos

### 7. **Presentación**
- `/ml_slides_project/` - Directorio con 10 diapositivas HTML sobre los algoritmos
  - `slide_1_titulo.html` - Título y contexto
  - `slide_2_dataset.html` - Descripción del dataset
  - `slide_3_lr_intro.html` - Introducción a Regresión Logística
  - `slide_4_lr_performance.html` - Rendimiento del modelo LR
  - `slide_5_lr_confusion.html` - Matriz de confusión
  - `slide_6_kmeans_intro.html` - Introducción a K-Means
  - `slide_7_kmeans_centroids.html` - Centroides de clusters
  - `slide_8_kmeans_analysis.html` - Análisis de clusters y estrategias
  - `slide_9_conclusions.html` - Conclusiones
  - `slide_10_questions.html` - Preguntas y discusión

---

## Cómo Usar

### Opción 1: Ejecutar la Aplicación Streamlit

```bash
streamlit run streamlit_app.py
```

La aplicación se abrirá en `http://localhost:8501` con las siguientes pestañas:
- **Inicio:** Descripción general del ejercicio
- **Regresión Logística:** Métricas y matriz de confusión del modelo de clasificación
- **K-Means:** Centroides y descripción de clusters
- **Visualizaciones:** Gráficos del método del codo y clusters

### Opción 2: Revisar el Notebook de Explicación

Abrir `ejercicio_practico_ml.md` en cualquier editor de Markdown o navegador para una explicación detallada de:
- Análisis Exploratorio de Datos (EDA)
- Preprocesamiento
- Implementación de Regresión Logística
- Implementación de K-Means
- Resultados y análisis

### Opción 3: Ver la Presentación

Abrir cualquiera de los archivos HTML en `/ml_slides_project/` en un navegador web para ver las diapositivas de presentación.

---

## Resultados Principales

### Regresión Logística
- **Precisión (Accuracy):** 98.19%
- **F1-Score Ponderado:** 0.9814
- **Recall (Clase Aprobado):** 0.9965
- **Conclusión:** Las notas son predictores extremadamente fuertes del estado de aprobación final.

### K-Means (K=4)

| Cluster | Centroide Nota 1 | Centroide Nota Final | Descripción |
| :--- | :--- | :--- | :--- |
| **0** | 97.40 | 8.49 | Alto Rendimiento |
| **1** | 15.95 | 0.43 | Bajo Rendimiento/Abandono |
| **2** | 0.60 | 9.28 | Recuperación Exitosa |
| **3** | 82.54 | 2.78 | Recuperación Fallida |

---

## Archivos de Código Fuente

### Scripts Python Utilizados
- `eda_preprocessing.py` - Análisis exploratorio y preprocesamiento de datos
- `logistic_regression_model.py` - Entrenamiento del modelo de Regresión Logística
- `kmeans_model.py` - Entrenamiento del modelo K-Means

Estos scripts pueden ser ejecutados nuevamente para reproducir los resultados:

```bash
python3.11 eda_preprocessing.py
python3.11 logistic_regression_model.py
python3.11 kmeans_model.py
```

---

## Requisitos de Instalación

```bash
pip install pandas numpy scikit-learn joblib matplotlib streamlit
```

---

## Estructura de Directorios

```
/home/ubuntu/
├── README.md
├── ejercicio_practico_ml.md
├── streamlit_app.py
├── eda_preprocessing.py
├── logistic_regression_model.py
├── kmeans_model.py
├── logistic_regression_model.pkl
├── logistic_regression_metrics.json
├── kmeans_model.pkl
├── kmeans_centroids.csv
├── scaler_kmeans.pkl
├── kmeans_elbow_method.png
├── kmeans_clusters.png
├── df_clasificacion_preprocessed.csv
├── df_clustering_preprocessed.csv
└── ml_slides_project/
    ├── slide_1_titulo.html
    ├── slide_2_dataset.html
    ├── slide_3_lr_intro.html
    ├── slide_4_lr_performance.html
    ├── slide_5_lr_confusion.html
    ├── slide_6_kmeans_intro.html
    ├── slide_7_kmeans_centroids.html
    ├── slide_8_kmeans_analysis.html
    ├── slide_9_conclusions.html
    └── slide_10_questions.html
```

---

## Conclusiones

Este ejercicio práctico demuestra cómo combinar modelos supervisados y no supervisados para obtener una visión completa del rendimiento académico:

- **Regresión Logística** proporciona predicciones precisas sobre quién está en riesgo de no aprobar
- **K-Means** identifica perfiles de estudiantes que requieren intervenciones específicas
- La combinación de ambos modelos permite tomar decisiones educativas informadas y personalizadas

---

## Autor


Diciembre 2025

---

## Licencia

Este ejercicio práctico es proporcionado con fines educativos.
