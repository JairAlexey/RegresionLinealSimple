import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importar y limpiar dataset
dataset = pd.read_csv('life_expectancy_data.csv')
dataset.columns = dataset.columns.str.strip()
dataset = dataset.dropna(subset=['GDP', 'Life expectancy'])

# Seleccionar variables
X = dataset[['GDP']]
y = dataset['Life expectancy']

# Dividir datos
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Entrenar modelo
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicciones (solo necesarias para el gráfico de prueba)
y_pred = regressor.predict(X_test)

# Métricas
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)

# Resultados
print(f'Coeficiente: {regressor.coef_[0]:.6f}')
print(f'Intercepto: {regressor.intercept_:.2f}')
print(f'R² (test): {r2:.4f}')

# --- Gráfica 1: Datos de Entrenamiento y Regresión ---
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='Entrenamiento')
plt.plot(X_train, regressor.predict(X_train), color='red', label='Regresión (Entrenamiento)')
plt.title('Esperanza de Vida vs PIB per cápita (Entrenamiento)', fontsize=14)
plt.xlabel('GDP (PIB per cápita)', fontsize=12)
plt.ylabel('Esperanza de Vida', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# --- Gráfica 2: Datos de Prueba y Regresión (basada en entrenamiento) ---
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='green', label='Prueba', alpha=0.7)
plt.plot(X_train, regressor.predict(X_train), color='red', label='Regresión (Entrenamiento)') # Usamos el modelo entrenado
plt.title('Esperanza de Vida vs PIB per cápita (Prueba)', fontsize=14)
plt.xlabel('GDP (PIB per cápita)', fontsize=12)
plt.ylabel('Esperanza de Vida', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()