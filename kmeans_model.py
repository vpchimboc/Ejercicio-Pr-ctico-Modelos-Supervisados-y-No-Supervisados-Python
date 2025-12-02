import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib

# 1. Cargar los datos preprocesados
df = pd.read_csv('df_clustering_preprocessed.csv')

# 2. Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
X_scaled_df = pd.DataFrame(X_scaled, columns=df.columns)

# Guardar el scaler para la aplicación Streamlit
joblib.dump(scaler, 'scaler_kmeans.pkl')

# 3. Método del Codo para encontrar el número óptimo de clusters (K)
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# 4. Visualización del Método del Codo
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o')
plt.title('Método del Codo para K-Means')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Inercia')
plt.grid(True)
plt.savefig('kmeans_elbow_method.png')
plt.close()
print("Gráfico del Método del Codo guardado en kmeans_elbow_method.png")

# 5. Elegir K y entrenar el modelo final
# Basado en el gráfico (que asumimos mostrará un codo en K=3 o K=4), elegiremos K=4
K_final = 4
kmeans_final = KMeans(n_clusters=K_final, random_state=42, n_init=10)
df['Cluster'] = kmeans_final.fit_predict(X_scaled)

# 6. Guardar el modelo K-Means
joblib.dump(kmeans_final, 'kmeans_model.pkl')

# 7. Visualización de los Clusters
plt.figure(figsize=(10, 7))
plt.scatter(df['Nota_1'], df['Nota_Final'], c=df['Cluster'], cmap='viridis', s=50, alpha=0.6)
plt.title(f'Clusters K-Means (K={K_final}) de Rendimiento Académico')
plt.xlabel('Nota 1 (Escalada)')
plt.ylabel('Nota Final (Escalada)')
plt.colorbar(label='Cluster ID')
plt.grid(True)
plt.savefig('kmeans_clusters.png')
plt.close()
print(f"Gráfico de Clusters K-Means guardado enkmeans_clusters.png")

# 8. Análisis de los Centroides (para el Notebook)
centroids = scaler.inverse_transform(kmeans_final.cluster_centers_)
centroids_df = pd.DataFrame(centroids, columns=['Centroide_Nota_1', 'Centroide_Nota_Final'])
centroids_df['Cluster'] = range(K_final)
centroids_df.to_csv('kmeans_centroids.csv', index=False)

print(f"\nModelo K-Means (K={K_final}) guardado en kmeans_model.pkl")
print(f"Scaler guardado en scaler_kmeans.pkl")
print(f"Centroides guardados en kmeans_centroids.csv")
