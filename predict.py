import joblib
import pandas as pd

def cargar_modelo(ruta: str):
    return joblib.load(ruta)

def predecir_fs(modelo, columnas, datos: dict):
    nuevo = pd.DataFrame([[datos[col] for col in columnas]], columns=columnas)
    return float(modelo.predict(nuevo)[0])
