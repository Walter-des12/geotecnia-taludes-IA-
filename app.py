# ============================================
# APP ESTABILIDAD DE TALUDES
# IA (LightGBM) + pySlope (Bishop)
# Compatible con Streamlit Cloud
# ============================================

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.metrics import r2_score
from pyslope import Slope, Material

from load_data import cargar_dataset
from predict import cargar_modelo, predecir_fs
from utils import completar_con_medias


# ============================================
# WRAPPER pySlope seguro para Streamlit
# ============================================

def analizar_con_pyslope(H, beta, peso, cohesion, phi, ru):
    """
    Ejecuta un análisis de estabilidad con pySlope y retorna:
      - FS crítico
      - Figura Plotly (convertida a dict si hace falta)
    """

    s = Slope(height=H, angle=beta, length=None)

    m1 = Material(
        unit_weight=peso,
        friction_angle=phi,
        cohesion=cohesion,
        depth_to_bottom=H
    )
    s.set_materials(m1)

    if ru > 0:
        water_depth = max(0.1, H * (1.0 - ru))
        s.set_water_table(water_depth)

    x_top = s.get_top_coordinates()[0]
    x_bot = s.get_bottom_coordinates()[0]
    s.set_analysis_limits(x_top - H, x_bot + H)

    s.update_analysis_options(slices=50, iterations=2000)
    s.analyse_slope()

    fs_min = float(s.get_min_FOS())

    fig_plotly = s.plot_critical()

    # Si Streamlit falla, devolver figura como dict
    try:
        return fs_min, fig_plotly
    except:
        return fs_min, fig_plotly.to_dict()


# ============================================
# CARGA DE MODELO Y DATA
# ============================================

@st.cache_resource
def cargar_artifacts():
    df = cargar_dataset("data/slope_stability_dataset.csv")
    artifacts = cargar_modelo("models/best_fs_model.joblib")
    return df, artifacts

df, artifacts = cargar_artifacts()
model = artifacts["model"]
columnas = artifacts["feature_names"]
X_test = artifacts.get("X_test", None)
y_test = artifacts.get("y_test", None)


# ============================================
# CONFIG STREAMLIT
# ============================================

st.set_page_config(
    page_title="Estabilidad de Taludes - IA + pySlope",
    layout="wide"
)

st.title("🟤 Estabilidad de Taludes | IA + pySlope (Bishop)")

st.markdown("""
Esta aplicación combina:
- Un modelo de **Machine Learning (LightGBM)** que predice el Factor de Seguridad (FS).
- El software **pySlope**, que aplica el método de **Bishop** para hallar el FS crítico.
""")


# ============================================
# VALIDACIÓN DEL MODELO
# ============================================

if X_test is not None and y_test is not None:
    st.subheader("Validación del modelo IA")

    y_pred = model.predict(X_test)
    r2_val = r2_score(y_test, y_pred)

    fig_v, ax = plt.subplots(figsize=(4, 3))
    ax.scatter(y_test, y_pred, alpha=0.7)
    minv = min(min(y_test), min(y_pred))
    maxv = max(max(y_test), max(y_pred))
    ax.plot([minv, maxv], [minv, maxv], "r--")
    ax.set_xlabel("FS real")
    ax.set_ylabel("FS predicho")
    ax.set_title(f"FS real vs Predicho (R² = {r2_val:.4f})")
    ax.grid(True)

    st.pyplot(fig_v)
    plt.close(fig_v)


# ============================================
# INPUT DEL USUARIO
# ============================================

st.markdown("---")
st.subheader("Parámetros del talud")

col1, col2 = st.columns(2)

with col1:
    peso = st.number_input("Peso unitario γ (kN/m³)", 10.0, 30.0, 18.0, step=0.5)
    cohesion = st.number_input("Cohesión c (kPa)", 0.0, 500.0, 24.0, step=1.0)
    phi = st.number_input("Ángulo de fricción interna φ (°)", 0.0, 50.0, 30.0, step=1.0)

with col2:
    beta = st.number_input("Ángulo del talud β (°)", 5.0, 85.0, 45.0, step=1.0)
    altura = st.number_input("Altura del talud H (m)", 1.0, 200.0, 20.0, step=1.0)
    ru = st.number_input("Relación de presión de poros ru", 0.0, 1.0, 0.1, step=0.01)


# ============================================
# PROCESAMIENTO
# ============================================

st.markdown("---")

if st.button("Calcular FS con IA + pySlope"):
    st.subheader("Resultados del análisis")

    entrada = {
        "Peso_unitario_kN_m3": peso,
        "Cohesion_kPa": cohesion,
        "Angulo_de_friccion_interna": phi,
        "Angulo_del_talud": beta,
        "Altura_del_talud_m": altura,
        "Relacion_de_presion_de_poros": ru,
    }

    entrada_completa = completar_con_medias(df, entrada)

    # ===== IA =====
    fs_ia = predecir_fs(model, columnas, entrada_completa)
    estado_ia = "ESTABLE" if fs_ia >= 1.0 else "INESTABLE"

    # ===== pySlope =====
    fs_pyslope, fig_pyslope = analizar_con_pyslope(altura, beta, peso, cohesion, phi, ru)
    estado_py = "ESTABLE" if fs_pyslope >= 1.0 else "INESTABLE"

    c1, c2 = st.columns(2)

    with c1:
        st.metric("FS (IA)", f"{fs_ia:.3f}",
                  "Estable" if estado_ia == "ESTABLE" else "Inestable")

    with c2:
        st.metric("FS (Bishop – pySlope)", f"{fs_pyslope:.3f}",
                  "Estable" if estado_py == "ESTABLE" else "Inestable")

    st.markdown("---")

    # ===== GRAFICO PY SLOPE =====
    st.subheader("Superficie crítica de falla (Bishop – pySlope)")

    try:
        st.plotly_chart(fig_pyslope, use_container_width=True)
    except:
        st.plotly_chart(fig_pyslope.to_dict(), use_container_width=True)

    st.markdown("---")

    # ===== SENSIBILIDAD =====
    st.subheader("Análisis de sensibilidad (modelo IA)")

    variables_sens = {
        "Cohesión c (kPa)": ("Cohesion_kPa", np.linspace(0, 500, 20)),
        "Ángulo de fricción φ (°)": ("Angulo_de_friccion_interna", np.linspace(0, 50, 20)),
        "ru": ("Relacion_de_presion_de_poros", np.linspace(0, 1, 20)),
        "Ángulo del talud β (°)": ("Angulo_del_talud", np.linspace(10, 80, 20)),
    }

    for nombre, (colvar, vals) in variables_sens.items():
        fs_vals = []
        for v in vals:
            tmp = dict(entrada_completa)
            tmp[colvar] = v
            fs_vals.append(predecir_fs(model, columnas, tmp))

        fig_s, ax_s = plt.subplots(figsize=(4, 3))
        ax_s.plot(vals, fs_vals, linewidth=2)
        ax_s.set_title(nombre)
        ax_s.set_xlabel(nombre)
        ax_s.set_ylabel("FS (IA)")
        ax_s.grid(True)
        st.pyplot(fig_s)
        plt.close(fig_s)

    st.markdown("---")

    # ===== SHAP =====
    st.subheader("Importancia de variables (SHAP – IA)")

    try:
        X_background = df[columnas].sample(30, random_state=42)

        explainer = shap.KernelExplainer(
            lambda X: model.predict(pd.DataFrame(X, columns=columnas)),
            X_background,
        )

        shap_values = explainer.shap_values(pd.DataFrame([entrada_completa])[columnas])

        fig_shap = plt.figure(figsize=(5, 4))
        shap.summary_plot(
            shap_values,
            pd.DataFrame([entrada_completa])[columnas],
            show=False
        )
        st.pyplot(fig_shap)
        plt.close(fig_shap)

    except Exception as e:
        st.warning(f"No se pudo calcular SHAP: {e}")
