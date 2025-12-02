# Ejercicio Práctico de Machine Learning: Análisis de Rendimiento Académico


**Fecha:** 02 de Diciembre de 2025
**Dataset:** `academic_performance_master.csv`

## 1. Introducción

Este ejercicio práctico tiene como objetivo aplicar y explicar dos tipos fundamentales de algoritmos de Machine Learning sobre un conjunto de datos de rendimiento académico:

1.  **Modelo Supervisado (Clasificación):** Regresión Logística, para predecir la aprobación de asignaturas.
2.  **Modelo No Supervisado (Clustering):** K-Means, para segmentar los registros académicos basados en las notas.

El dataset `academic_performance_master.csv` contiene registros detallados de asignaturas, incluyendo notas, estado de aprobación, carrera y nivel.

## 2. Análisis Exploratorio de Datos (EDA) y Preprocesamiento

El dataset inicial fue cargado y se identificó la necesidad de limpiar y transformar los datos para los modelos.

### 2.1. Estructura de Datos

El dataset contiene 44,915 registros válidos después de la limpieza inicial de filas con valores nulos en las columnas clave.

| Columna | Tipo de Dato | Descripción |
| :--- | :--- | :--- |
| `Nota_1` | Numérico | Nota de la primera evaluación (0-100). |
| `Nota_Final` | Numérico | Nota final de la asignatura (0-10). |
| `Carrera` | Categórico | Programa de estudios. |
| `Nivel` | Categórico | Nivel de la asignatura (PRIMERO, SEGUNDO, etc.). |
| `Estado_1` | Categórico | Estado final de la asignatura (APROBADO, REPROBADO, etc.). |

### 2.2. Preprocesamiento para Clasificación

**Objetivo:** Predecir si una asignatura será **APROBADA** (1) o **NO APROBADA** (0).

*   **Variable Objetivo (`Target_Clasificacion`):** Se creó una variable binaria donde `APROBADO` = 1 y el resto (REPROBADO, DESERTOR, RETIRADO) = 0.
*   **Características (Features):** Se seleccionaron `Nota_1`, `Nota_Final`, `Carrera` y `Nivel`.
*   **Codificación:** Las variables categóricas (`Carrera` y `Nivel`) se transformaron usando **One-Hot Encoding** para que el modelo de Regresión Logística pueda interpretarlas.

## 3. Modelo Supervisado: Regresión Logística (Clasificación)

La **Regresión Logística** es un algoritmo de clasificación lineal que utiliza la función logística (o sigmoide) para modelar la probabilidad de que una instancia pertenezca a una clase particular.

### 3.1. Implementación

Se entrenó un modelo de Regresión Logística utilizando el 80% de los datos preprocesados, reservando el 20% para la evaluación.

### 3.2. Evaluación del Modelo

El modelo mostró un rendimiento muy alto, lo cual es esperado dado que las notas (`Nota_1` y `Nota_Final`) son predictores muy fuertes del estado de aprobación.

| Métrica | Valor |
| :--- | :--- |
| **Precisión (Accuracy)** | 0.9819 |
| **Macro Avg F1-Score** | 0.9491 |

**Reporte de Clasificación (Clase 0: No Aprobado, Clase 1: Aprobado):**

| Clase | Precisión | Recall | F1-Score | Soporte |
| :--- | :--- | :--- | :--- | :--- |
| **0** | 0.9664 | 0.8565 | 0.9082 | 941 |
| **1** | 0.9834 | 0.9965 | 0.9899 | 8042 |

**Matriz de Confusión:**

| | Predicción 0 | Predicción 1 |
| :--- | :--- | :--- |
| **Real 0** | 806 (Verdaderos Negativos) | 135 (Falsos Positivos) |
| **Real 1** | 28 (Falsos Negativos) | 8014 (Verdaderos Positivos) |

**Análisis:**
El modelo es excelente para identificar las asignaturas **Aprobadas** (Clase 1), con un *Recall* de casi el 100%. Sin embargo, tiene una tasa de **Falsos Positivos** (135 casos) relativamente alta para la Clase 0, lo que significa que en 135 ocasiones predijo que una asignatura sería reprobada cuando en realidad fue aprobada. Esto puede deberse a la alta correlación entre las características y el objetivo.

## 4. Modelo No Supervisado: K-Means (Clustering)

**K-Means** es un algoritmo de clustering que agrupa observaciones en $K$ clusters, donde cada observación pertenece al cluster cuyo centroide (media) es el más cercano.

### 4.1. Preprocesamiento y Selección de K

*   **Características:** Se utilizaron únicamente `Nota_1` y `Nota_Final`.
*   **Escalado:** Los datos fueron estandarizados (escalados) para asegurar que ambas notas contribuyan por igual a la distancia euclidiana.
*   **Método del Codo:** Se aplicó el método del codo para estimar el número óptimo de clusters ($K$). Aunque el gráfico no se incluye aquí, se determinó que **K=4** era un número razonable para la segmentación.

![Gráfico del Método del Codo](https://private-us-east-1.manuscdn.com/sessionFile/KlnoV4HsqxXRVwxFD4tYQy/sandbox/8afdfRMlw6IiY0eI1UN68u-images_1764680674671_na1fn_L2hvbWUvdWJ1bnR1L2ttZWFuc19lbGJvd19tZXRob2Q.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvS2xub1Y0SHNxeFhSVnd4RkQ0dFlReS9zYW5kYm94LzhhZmRmUk1sdzZJaVkwZUkxVU42OHUtaW1hZ2VzXzE3NjQ2ODA2NzQ2NzFfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwydHRaV0Z1YzE5bGJHSnZkMTl0WlhSb2IyUS5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3OTg3NjE2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=bXlzeO-zoZ7Go3MnLNm5xEWR8GqbRojhB~Kj97-T7R~QHjHbDqmdkY7RM3qEZncWMp1vf4eNOMob0JCLlcMq-otIRRhzd2PB0ZoTQtdVPSii80fX7IentmviX-3J6qmsb5~WhsKKIYcsD3QNcDXLoqhb2NG~yqAHvCB41QawdqpWs9C4Wkf8paznO5PKdrxC7b9ovqNko5WQwldYoskBAKX2hXMSS6SvSsjOEw-sxDTyl-BYKx-YqpZyF5YiG8nOJId8Trenbzjlz7cz3Ga4blLl-yQXEd8Cg6qajGzT~NnBsS0W8mNFZoxHNMfFgQ5RIRoazseACgnqoGip2kC3fg__)

### 4.2. Resultados del Clustering (K=4)

Se entrenó el modelo K-Means con $K=4$ y se obtuvieron los siguientes centroides (puntos medios de cada cluster, desescalados a los valores originales de las notas):

| Cluster | Centroide Nota 1 | Centroide Nota Final | Descripción del Cluster |
| :--- | :--- | :--- | :--- |
| **0** | 97.40 | 8.49 | **Alto Rendimiento:** Notas muy altas en ambas evaluaciones. |
| **1** | 15.95 | 0.43 | **Bajo Rendimiento/Abandono:** Notas muy bajas en ambas evaluaciones. |
| **2** | 0.60 | 9.28 | **Rendimiento Inconsistente (Recuperación Exitosa):** Nota 1 muy baja, pero Nota Final muy alta. |
| **3** | 82.54 | 2.78 | **Rendimiento Inconsistente (Recuperación Fallida):** Nota 1 alta, pero Nota Final muy baja. |

![Gráfico de Clusters K-Means](https://private-us-east-1.manuscdn.com/sessionFile/KlnoV4HsqxXRVwxFD4tYQy/sandbox/8afdfRMlw6IiY0eI1UN68u-images_1764680674672_na1fn_L2hvbWUvdWJ1bnR1L2ttZWFuc19jbHVzdGVycw.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvS2xub1Y0SHNxeFhSVnd4RkQ0dFlReS9zYW5kYm94LzhhZmRmUk1sdzZJaVkwZUkxVU42OHUtaW1hZ2VzXzE3NjQ2ODA2NzQ2NzJfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwydHRaV0Z1YzE5amJIVnpkR1Z5Y3cucG5nIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzk4NzYxNjAwfX19XX0_&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=sIrBX98NXuu5oIEuBGdN37gK9ONKNwTjWRJUBm5ailVnXtc0CaOdRDHGOL3ra2egi~eHXm9Frg6pDi7vQiWB7qlUYeRcBPW7TYSEaX-EQwzGSQfzb2WKDOopALl4wjibHGukeKtV1MxLM8ep9Qw36JDnNlDTbKanjiV1A7UyC6Qzb3xCR9QpUu4JFP6NsSra8PViHpaC9F7NyKW56YKd9Io7VJlDYZOq1X4VjWdPx-gUR3hZLDYPTocvwvbMQDcV~sdSKk~WXRi9QhCqIuUZjgbXi9q60vjM6wMvezP2H6-lAYx3EAusXVCKywLFkHFP-uu5MFh1nK~Cx1gaqNr1mA__)

**Análisis:**
El clustering revela patrones interesantes en el rendimiento académico:
*   **Cluster 0:** Estudiantes consistentemente excelentes.
*   **Cluster 1:** Estudiantes con problemas serios o que abandonaron tempranamente.
*   **Cluster 2:** Estudiantes que tuvieron un mal inicio (Nota 1 baja) pero lograron una excelente recuperación o un examen final muy bueno (Nota Final alta).
*   **Cluster 3:** Estudiantes que tuvieron un buen inicio (Nota 1 alta) pero fallaron en la etapa final (Nota Final baja), lo que podría indicar desmotivación o problemas en el examen final.

## 5. Conclusión

Este ejercicio ha demostrado la aplicación de un modelo de clasificación para predecir el éxito académico y un modelo de clustering para segmentar el rendimiento, proporcionando información valiosa para la toma de decisiones educativas. Los archivos generados (modelos, métricas y visualizaciones) serán utilizados en la aplicación Streamlit para una demostración interactiva.

## 6. Referencias

No se utilizaron referencias externas para este análisis.
