import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import json

# 1. Cargar los datos preprocesados
df = pd.read_csv('df_clasificacion_preprocessed.csv')

# 2. Separar características (X) y objetivo (y)
# La última columna es el target (Target_Clasificacion)
X = df.drop(columns=['Target_Clasificacion'])
y = df['Target_Clasificacion']

# 3. Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Entrenar el modelo de Regresión Logística
# Usamos un solver robusto para datasets grandes
model = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# 5. Evaluar el modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred).tolist()

# 6. Guardar el modelo y las métricas
joblib.dump(model, 'logistic_regression_model.pkl')

metrics = {
    "accuracy": accuracy,
    "classification_report": report,
    "confusion_matrix": conf_matrix,
    "features": list(X.columns)
}

with open('logistic_regression_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print("--- Evaluación del Modelo de Regresión Logística ---")
print(f"Precisión (Accuracy): {accuracy:.4f}")
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))
print(f"\nModelo guardado en logistic_regression_model.pkl")
print(f"Métricas guardadas en logistic_regression_metrics.json")
