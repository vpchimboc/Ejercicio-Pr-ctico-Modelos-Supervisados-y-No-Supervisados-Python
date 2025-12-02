# Presentación: Modelos de Machine Learning en Rendimiento Académico

## Diapositiva 1: Título y Contexto
**Título:** Aplicación de Modelos de Machine Learning en el Análisis de Rendimiento Académico
**Subtítulo:** Regresión Logística (Clasificación) y K-Means (Clustering)
**Fecha:** Diciembre 2025
**Contexto:** Análisis de datos de rendimiento académico (`academic_performance_master.csv`) para predecir el éxito y segmentar patrones de estudio.

## Diapositiva 2: El Dataset y el Problema
**Título:** El Dataset: Un Vistazo al Rendimiento Académico
**Puntos Clave:**
*   **Fuente de Datos:** Registros de asignaturas con notas, estado de aprobación, carrera y nivel.
*   **Registros Válidos:** 44,915 entradas después del preprocesamiento.
*   **Problema de Clasificación:** Predecir el estado de aprobación (Aprobado vs. No Aprobado).
*   **Problema de Clustering:** Identificar grupos naturales de estudiantes basados en sus notas.
*   **Características Clave:** Nota 1 (0-100) y Nota Final (0-10).

## Diapositiva 3: Modelo 1 - Regresión Logística (Clasificación)
**Título:** Regresión Logística: Predicción del Éxito Académico
**Puntos Clave:**
*   **Concepto:** Algoritmo de clasificación lineal que utiliza la función sigmoide para estimar la probabilidad de pertenencia a una clase.
*   **Aplicación:** Modelar la probabilidad de que un estudiante **Apruebe** una asignatura (Clase 1).
*   **Características (Features):** Notas (Nota 1, Nota Final) y variables categóricas codificadas (Carrera, Nivel).
*   **División de Datos:** 80% Entrenamiento, 20% Prueba.

## Diapositiva 4: Regresión Logística - Alto Rendimiento Predictivo
**Título:** El Modelo de Regresión Logística Alcanza una Alta Precisión
**Puntos Clave:**
*   **Precisión Global (Accuracy):** 98.19%
*   **F1-Score Ponderado:** 0.9814
*   **Conclusión:** Las notas son predictores extremadamente fuertes del estado de aprobación final.
*   **Métrica Clave:** El *Recall* para la clase "Aprobado" (Clase 1) es de 0.9965, indicando que el modelo casi nunca falla al identificar una asignatura aprobada.

## Diapositiva 5: Regresión Logística - Análisis de Errores
**Título:** Matriz de Confusión: Identificando Falsos Positivos
**Puntos Clave:**
*   **Verdaderos Positivos (VP):** 8014 (Aprobado correctamente predicho).
*   **Verdaderos Negativos (VN):** 806 (No Aprobado correctamente predicho).
*   **Falsos Positivos (FP):** 135 (Predijo Aprobado, pero fue No Aprobado).
*   **Falsos Negativos (FN):** 28 (Predijo No Aprobado, pero fue Aprobado).
*   **Interpretación:** El modelo tiene un sesgo hacia la predicción de "Aprobado", lo que resulta en 135 Falsos Positivos. Esto sugiere que la relación entre las notas y el estado final es casi determinística.

## Diapositiva 6: Modelo 2 - K-Means (Clustering)
**Título:** K-Means: Segmentación de Patrones de Rendimiento
**Puntos Clave:**
*   **Concepto:** Algoritmo no supervisado que particiona $N$ observaciones en $K$ clusters, minimizando la varianza dentro de cada cluster.
*   **Aplicación:** Descubrir grupos naturales de estudiantes con patrones de notas similares.
*   **Características:** Únicamente `Nota 1` y `Nota Final` (escaladas para igual peso).
*   **Selección de K:** El **Método del Codo** sugirió que $K=4$ es el número óptimo de clusters para este dataset.

## Diapositiva 7: K-Means - Los Cuatro Perfiles de Rendimiento
**Título:** Centroides: Definiendo los Cuatro Perfiles de Estudiantes
**Tabla de Centroides (Notas Originales):**
| Cluster | Centroide Nota 1 | Centroide Nota Final |
| :--- | :--- | :--- |
| **0** | 97.40 | 8.49 |
| **1** | 15.95 | 0.43 |
| **2** | 0.60 | 9.28 |
| **3** | 82.54 | 2.78 |
**Visualización:** Se adjunta un gráfico de dispersión de los clusters.

## Diapositiva 8: K-Means - Análisis Detallado de los Clusters
**Título:** Interpretación de los Clusters: Estrategias de Intervención
**Puntos Clave:**
*   **Cluster 0 (Alto Rendimiento):** Estudiantes consistentemente excelentes. Requieren programas de enriquecimiento.
*   **Cluster 1 (Bajo Rendimiento/Abandono):** Notas muy bajas en ambas evaluaciones. Necesitan intervención urgente y tutorías intensivas.
*   **Cluster 2 (Recuperación Exitosa):** Mal inicio (Nota 1 baja) pero excelente cierre (Nota Final alta). Indican potencial de recuperación con apoyo focalizado.
*   **Cluster 3 (Recuperación Fallida):** Buen inicio (Nota 1 alta) pero fracaso en la etapa final (Nota Final baja). Sugiere problemas de motivación o preparación para el examen final.

## Diapositiva 9: Conclusiones y Aplicaciones
**Título:** Conclusiones: De la Predicción a la Segmentación
**Puntos Clave:**
*   **Regresión Logística:** Proporciona una herramienta de alerta temprana con alta fiabilidad para predecir el estado de aprobación.
*   **K-Means:** Permite la creación de perfiles de estudiantes para personalizar las estrategias de apoyo académico.
*   **Valor Práctico:** La combinación de ambos modelos ofrece una visión completa: *quién* está en riesgo (LR) y *por qué* (K-Means).
*   **Siguiente Paso:** Uso de la aplicación Streamlit para la exploración interactiva de los modelos.

## Diapositiva 10: Preguntas
**Título:** Preguntas y Discusión
**Puntos Clave:**
*   Gracias por su atención.
*   Preguntas.
