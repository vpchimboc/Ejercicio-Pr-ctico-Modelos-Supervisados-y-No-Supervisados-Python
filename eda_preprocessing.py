import pandas as pd
import numpy as np

# Definir el path del archivo
file_path = 'academic_performance_master.csv'

# Nombres de columna inferidos del análisis inicial (15 columnas de datos)
column_names = [
    'Periodo', 'Tipo', 'ID_Estudiante', 'Nombre_Estudiante', 'Carrera', 'Nivel',
    'Asignatura', 'Creditos', 'Nota_1', 'Nota_Final', 'Estado_1', 'Estado_2',
    'Estado_3', 'ID_Profesor', 'Nombre_Profesor'
]

# Cargar el dataset. Intentaremos leerlo con el separador de coma, asumiendo que la primera columna es un índice.
try:
    # Leer el archivo sin especificar separador, y luego usar la primera columna como índice.
    # Esto debería manejar la mezcla de separadores si la primera columna es el índice.
    df = pd.read_csv(file_path, header=None, index_col=0, names=column_names)
    
except Exception as e:
    print(f"Error al cargar o preprocesar el archivo: {e}")
    # Si falla, intentamos una carga más robusta asumiendo que el separador es la coma,
    # y que la primera columna es un índice que se debe eliminar.
    try:
        df = pd.read_csv(file_path, header=None, sep=',')
        # Asumimos que la primera columna es el índice y la eliminamos.
        df = df.iloc[:, 1:]
        df.columns = column_names
    except Exception as e2:
        print(f"Error en el segundo intento de carga: {e2}")
        exit()

print("--- Información Inicial del DataFrame ---")
print(df.info())
print("\n--- Primeras 5 filas ---")
print(df.head())
print("\n--- Conteo de valores únicos en columnas clave ---")
print(df['Estado_1'].value_counts())
print(df['Carrera'].value_counts().head())
print(df['Nivel'].value_counts())

# --- Preprocesamiento para Clasificación (Regresión Logística) ---
# Objetivo: Predecir si el estudiante 'APROBADO' o 'REPROBADO' (usando Estado_1)

# 1. Crear la variable objetivo binaria
# Asumiremos que 'APROBADO' es la clase positiva (1) y cualquier otra cosa es negativa (0).
df['Target_Clasificacion'] = df['Estado_1'].apply(lambda x: 1 if x == 'APROBADO' else 0)
print("\n--- Conteo de la variable objetivo de Clasificación ---")
print(df['Target_Clasificacion'].value_counts())

# 2. Selección de características para Clasificación
# Usaremos 'Nota_1', 'Nota_Final' y codificaremos 'Carrera' y 'Nivel'.
features_clasificacion = ['Nota_1', 'Nota_Final', 'Carrera', 'Nivel']
df_clasificacion = df.dropna(subset=features_clasificacion + ['Target_Clasificacion']).copy()

# Convertir las notas a numérico, forzando errores a NaN para limpieza
df_clasificacion['Nota_1'] = pd.to_numeric(df_clasificacion['Nota_1'], errors='coerce')
df_clasificacion['Nota_Final'] = pd.to_numeric(df_clasificacion['Nota_Final'], errors='coerce')
df_clasificacion.dropna(subset=['Nota_1', 'Nota_Final'], inplace=True)

# Codificación One-Hot para variables categóricas
df_clasificacion = pd.get_dummies(df_clasificacion, columns=['Carrera', 'Nivel'], drop_first=True)

# --- Preprocesamiento para Clustering (K-Means) ---
# Objetivo: Agrupar las asignaturas/registros basándose en las notas.

# 1. Selección de características para Clustering
features_clustering = ['Nota_1', 'Nota_Final']
df_clustering = df.dropna(subset=features_clustering).copy()

# Convertir las notas a numérico, forzando errores a NaN para limpieza
df_clustering['Nota_1'] = pd.to_numeric(df_clustering['Nota_1'], errors='coerce')
df_clustering['Nota_Final'] = pd.to_numeric(df_clustering['Nota_Final'], errors='coerce')
df_clustering.dropna(subset=['Nota_1', 'Nota_Final'], inplace=True)

# Guardar los DataFrames preprocesados y la lista de columnas para el siguiente paso
# Para la clasificación, guardamos las columnas de características y el target.
clasificacion_cols = [col for col in df_clasificacion.columns if col.startswith('Carrera_') or col.startswith('Nivel_') or col in ['Nota_1', 'Nota_Final', 'Target_Clasificacion']]
df_clasificacion[clasificacion_cols].to_csv('df_clasificacion_preprocessed.csv', index=False)

# Para el clustering, guardamos solo las notas.
df_clustering[features_clustering].to_csv('df_clustering_preprocessed.csv', index=False)

print("\n--- Resumen del Preprocesamiento ---")
print(f"DataFrame de Clasificación guardado en df_clasificacion_preprocessed.csv con {len(df_clasificacion)} filas.")
print(f"DataFrame de Clustering guardado en df_clustering_preprocessed.csv con {len(df_clustering)} filas.")
