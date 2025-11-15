# ============================================
# APP ESTABILIDAD DE TALUDES
# IA (LightGBM) + pySlope (Bishop)
# ============================================

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from io import BytesIO
from sklearn.metrics import r2_score
from pyslope import Slope, Material

from load_data import cargar_dataset
from predict import cargar_modelo, predecir_fs
from utils import completar_con_medias


# ============================================
# FUNCIÓN PARA MOSTRAR GRÁFICOS PEQUEÑOS
# ============================================

def show_small_plot(fig, width_px=420):
    """Convierte figuras matplotlib a PNG pequeño y las muestra."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    st.image(buf, width=width_px)


# ============================================
# WRAPPER: ANÁLISIS CON pySlope
# ============================================

def analizar_con_pyslope(H, beta, peso, cohesion, phi, ru):
    """
    Ejecuta un análisis de estabilidad con pySlope usando Bishop.
    Devuelve FS crítico y la figura plotly.
    """

    s = Slope(height=H, angle=beta, length=None)  # type: ignore[arg-type]


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

    fs_min = s.get_min_FOS()
    fig = s.plot_critical()

    # Reducir tamaño del gráfico plotly
    fig.update_layout(width=450, height=350)

    return float(fs_min), fig


# ============================================
# CARGAR DATA Y MODELO
# ============================================

@st.cache_resource
def cargar_artifacts():
    df = cargar_dataset("data/slope_stability_dataset.csv")
    artifacts = cargar_modelo("models/best_fs_model.joblib")
    return df, artifacts


df, artifacts = cargar_artifacts()

model = artifacts["model"]
columnas = artifacts["feature_names"]
X_test = artifacts.get("X_test")
y_test = artifacts.get("y_test")


# ============================================
# INTERFAZ STREAMLIT
# ============================================

st.set_page_config(page_title="Estabilidad de Taludes IA + pySlope", layout="wide")

st.title("🟤 Estabilidad de Taludes | IA + pySlope (Bishop)")
st.markdown("""
Esta aplicación combina:

- **Machine Learning (LightGBM)** para predecir el Factor de Seguridad (FS).
- **pySlope** (método de rebanadas de Bishop) para obtener un FS crítico.
""")

st.markdown("---")


# ============================================
# VALIDACIÓN DEL MODELO IA
# ============================================

if X_test is not None and y_test is not None:
    st.subheader("Validación interna del modelo IA")

    y_pred = model.predict(X_test)
    r2_val = r2_score(y_test, y_pred)

    fig_val, ax = plt.subplots(figsize=(4, 3))
    ax.scatter(y_test, y_pred, alpha=0.7, s=30)

    minv = min(min(y_test), min(y_pred))
    maxv = max(max(y_test), max(y_pred))
    ax.plot([minv, maxv], [minv, maxv], "r--")

    ax.set_xlabel("FS real")
    ax.set_ylabel("FS predicho")
    ax.set_title(f"R² = {r2_val:.4f}")
    ax.grid(True)

    show_small_plot(fig_val, width_px=380)

    st.info(f"R² del modelo IA: **{r2_val:.4f}**")

st.markdown("---")


# ============================================
# ENTRADA DE PARÁMETROS
# ============================================

st.subheader("Parámetros del talud")

col1, col2 = st.columns(2)

with col1:
    peso = st.number_input("Peso unitario γ (kN/m³)", 10.0, 30.0, 18.0)
    cohesion = st.number_input("Cohesión c (kPa)", 0.0, 150.0, 24.0)
    phi = st.number_input("Ángulo de fricción interna φ (°)", 0.0, 50.0, 30.0)

with col2:
    beta = st.number_input("Ángulo del talud β (°)", 5.0, 85.0, 45.0)
    altura = st.number_input("Altura del talud H (m)", 1.0, 200.0, 20.0)
    ru = st.number_input("Relación de presión de poros ru (-)", 0.0, 1.0, 0.1)

st.markdown("---")


# ============================================
# BOTÓN PRINCIPAL
# ============================================

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

    fs_ia = predecir_fs(model, columnas, entrada_completa)
    fs_py, fig_py = analizar_con_pyslope(altura, beta, peso, cohesion, phi, ru)

    colA, colB = st.columns(2)

    with colA:
        st.metric("FS (modelo IA)", f"{fs_ia:.3f}", "Estable" if fs_ia >= 1 else "Inestable")
    with colB:
        st.metric("FS (pySlope - Bishop)", f"{fs_py:.3f}", "Estable" if fs_py >= 1 else "Inestable")

    st.markdown("---")

    # 🟤 Superficie crítica de falla
    st.subheader("Superficie crítica de falla (pySlope)")
    st.plotly_chart(fig_py, use_container_width=False)
    st.markdown("---")

    # 🟠 Sensibilidad
    st.subheader("Análisis de sensibilidad (IA)")

    variables_sens = {
        "Cohesión c (kPa)": ("Cohesion_kPa", np.linspace(0, 150, 20)),
        "Ángulo de fricción φ (°)": ("Angulo_de_friccion_interna", np.linspace(0, 50, 20)),
        "ru": ("Relacion_de_presion_de_poros", np.linspace(0, 1, 20)),
        "Ángulo del talud β (°)": ("Angulo_del_talud", np.linspace(10, 80, 20)),
    }

    for titulo, (colvar, vals) in variables_sens.items():
        fs_vals = []

        for v in vals:
            entrada_tmp = dict(entrada_completa)
            entrada_tmp[colvar] = v
            fs_vals.append(predecir_fs(model, columnas, entrada_tmp))

        fig_s, ax_s = plt.subplots(figsize=(4, 3))
        ax_s.plot(vals, fs_vals, linewidth=2)
        ax_s.set_title(titulo)
        ax_s.set_xlabel(titulo)
        ax_s.set_ylabel("FS (IA)")
        ax_s.grid(True)

        show_small_plot(fig_s, width_px=380)

    st.markdown("---")

    # 🟣 SHAP
    st.subheader("Importancia de variables (SHAP)")

    try:
        X_bg = df[columnas].sample(30, random_state=42)
        explainer = shap.KernelExplainer(lambda X: model.predict(pd.DataFrame(X, columns=columnas)), X_bg)

        shap_values = explainer.shap_values(pd.DataFrame([entrada_completa])[columnas])

        fig_shap = plt.figure(figsize=(4, 3))
        shap.summary_plot(shap_values, pd.DataFrame([entrada_completa])[columnas], show=False)
        show_small_plot(fig_shap, width_px=380)

    except Exception as e:
        st.warning(f"No se pudo calcular SHAP: {e}")
