import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import json

# Configurar la página de Streamlit
st.set_page_config(page_title="ML Académico", layout="wide")

# Título principal
st.title("🎓 Ejercicio Práctico de Machine Learning - Rendimiento Académico")

# Cargar los modelos y datos
@st.cache_resource
def load_models():
    lr_model = joblib.load('logistic_regression_model.pkl')
    kmeans_model = joblib.load('kmeans_model.pkl')
    scaler_kmeans = joblib.load('scaler_kmeans.pkl')
    
    with open('logistic_regression_metrics.json', 'r') as f:
        lr_metrics = json.load(f)
    
    return lr_model, kmeans_model, scaler_kmeans, lr_metrics

lr_model, kmeans_model, scaler_kmeans, lr_metrics = load_models()

# Cargar datos preprocesados
df_clasificacion = pd.read_csv('df_clasificacion_preprocessed.csv')
df_clustering = pd.read_csv('df_clustering_preprocessed.csv')

# Cargar imágenes
elbow_img = plt.imread('kmeans_elbow_method.png')
clusters_img = plt.imread('kmeans_clusters.png')

# Crear pestañas
tab1, tab2, tab3, tab4 = st.tabs(["📊 Inicio", "🔵 Regresión Logística", "🎯 K-Means", "📈 Visualizaciones"])

# ===== TAB 1: INICIO =====
with tab1:
    st.header("Bienvenido al Ejercicio Práctico de Machine Learning")
    
    st.markdown("""
    Este ejercicio práctico demuestra la aplicación de dos tipos fundamentales de algoritmos de Machine Learning:
    
    ### 1. **Modelo Supervisado: Regresión Logística (Clasificación)**
    - **Objetivo:** Predecir si una asignatura será aprobada o no.
    - **Características:** Nota 1, Nota Final, Carrera, Nivel.
    - **Rendimiento:** Precisión del 98.19%.
    
    ### 2. **Modelo No Supervisado: K-Means (Clustering)**
    - **Objetivo:** Segmentar los registros académicos en grupos homogéneos.
    - **Características:** Nota 1, Nota Final.
    - **Clusters:** 4 grupos identificados con patrones de rendimiento distintos.
    
    ### Dataset
    - **Total de registros:** 44,915 registros válidos.
    - **Fuente:** `academic_performance_master.csv`.
    - **Período:** Múltiples períodos académicos (2020-2024).
    """)
    
    # Mostrar estadísticas básicas
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Registros", len(df_clustering))
    with col2:
        st.metric("Precisión LR", f"{lr_metrics['accuracy']:.2%}")
    with col3:
        st.metric("Clusters K-Means", "4")

# ===== TAB 2: REGRESIÓN LOGÍSTICA =====
with tab2:
    st.header("🔵 Regresión Logística - Predicción de Aprobación")
    
    st.markdown("""
    La **Regresión Logística** es un algoritmo de clasificación que modela la probabilidad de que una instancia 
    pertenezca a una clase particular utilizando la función logística (sigmoide).
    
    ### Características del Modelo
    - **Algoritmo:** Regresión Logística (Solver: liblinear)
    - **Características:** Nota 1, Nota Final, Carrera (One-Hot Encoded), Nivel (One-Hot Encoded)
    - **Clases:** 0 (No Aprobado), 1 (Aprobado)
    - **Datos de Entrenamiento:** 80% (35,932 registros)
    - **Datos de Prueba:** 20% (8,983 registros)
    """)
    
    # Mostrar métricas
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Precisión (Accuracy)", f"{lr_metrics['accuracy']:.4f}")
    with col2:
        st.metric("Macro Avg F1-Score", f"{lr_metrics['classification_report']['macro avg']['f1-score']:.4f}")
    with col3:
        st.metric("Weighted Avg F1-Score", f"{lr_metrics['classification_report']['weighted avg']['f1-score']:.4f}")
    
    # Reporte de clasificación
    st.subheader("Reporte de Clasificación")
    report_data = {
        "Clase": ["0 (No Aprobado)", "1 (Aprobado)"],
        "Precisión": [
            f"{lr_metrics['classification_report']['0']['precision']:.4f}",
            f"{lr_metrics['classification_report']['1']['precision']:.4f}"
        ],
        "Recall": [
            f"{lr_metrics['classification_report']['0']['recall']:.4f}",
            f"{lr_metrics['classification_report']['1']['recall']:.4f}"
        ],
        "F1-Score": [
            f"{lr_metrics['classification_report']['0']['f1-score']:.4f}",
            f"{lr_metrics['classification_report']['1']['f1-score']:.4f}"
        ],
        "Soporte": [
            int(lr_metrics['classification_report']['0']['support']),
            int(lr_metrics['classification_report']['1']['support'])
        ]
    }
    st.dataframe(pd.DataFrame(report_data), use_container_width=True)
    
    # Matriz de confusión
    st.subheader("Matriz de Confusión")
    conf_matrix = np.array(lr_metrics['confusion_matrix'])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(conf_matrix, cmap='Blues', aspect='auto')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicción 0', 'Predicción 1'])
    ax.set_yticklabels(['Real 0', 'Real 1'])
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Real')
    ax.set_title('Matriz de Confusión - Regresión Logística')
    
    # Añadir valores en las celdas
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, conf_matrix[i, j], ha="center", va="center", color="black", fontsize=16)
    
    plt.colorbar(im, ax=ax)
    st.pyplot(fig)

# ===== TAB 3: K-MEANS =====
with tab3:
    st.header("🎯 K-Means - Clustering de Rendimiento Académico")
    
    st.markdown("""
    **K-Means** es un algoritmo de clustering que agrupa observaciones en K clusters, donde cada observación 
    pertenece al cluster cuyo centroide (media) es el más cercano.
    
    ### Características del Modelo
    - **Algoritmo:** K-Means
    - **Características:** Nota 1, Nota Final (escaladas)
    - **Número de Clusters (K):** 4
    - **Inicializaciones:** 10
    - **Semilla Aleatoria:** 42
    """)
    
    # Cargar centroides
    centroides_df = pd.read_csv('kmeans_centroids.csv')
    
    st.subheader("Centroides de los Clusters")
    st.dataframe(centroides_df, use_container_width=True)
    
    st.subheader("Descripción de los Clusters")
    cluster_descriptions = {
        "Cluster 0": {
            "Centroide Nota 1": 97.40,
            "Centroide Nota Final": 8.49,
            "Descripción": "Alto Rendimiento: Notas muy altas en ambas evaluaciones."
        },
        "Cluster 1": {
            "Centroide Nota 1": 15.95,
            "Centroide Nota Final": 0.43,
            "Descripción": "Bajo Rendimiento/Abandono: Notas muy bajas en ambas evaluaciones."
        },
        "Cluster 2": {
            "Centroide Nota 1": 0.60,
            "Centroide Nota Final": 9.28,
            "Descripción": "Rendimiento Inconsistente (Recuperación Exitosa): Nota 1 muy baja, pero Nota Final muy alta."
        },
        "Cluster 3": {
            "Centroide Nota 1": 82.54,
            "Centroide Nota Final": 2.78,
            "Descripción": "Rendimiento Inconsistente (Recuperación Fallida): Nota 1 alta, pero Nota Final muy baja."
        }
    }
    
    for cluster, info in cluster_descriptions.items():
        with st.expander(f"📌 {cluster}"):
            st.write(f"**Centroide Nota 1:** {info['Centroide Nota 1']:.2f}")
            st.write(f"**Centroide Nota Final:** {info['Centroide Nota Final']:.2f}")
            st.write(f"**Descripción:** {info['Descripción']}")

# ===== TAB 4: VISUALIZACIONES =====
with tab4:
    st.header("📈 Visualizaciones de los Modelos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Método del Codo (K-Means)")
        st.image(elbow_img, caption="Gráfico del Método del Codo para K-Means", use_container_width=True)
    
    with col2:
        st.subheader("Clusters K-Means")
        st.image(clusters_img, caption="Visualización de los Clusters K-Means", use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**Ejercicio Práctico de Machine Learning**")
