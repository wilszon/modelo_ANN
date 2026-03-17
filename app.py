import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# ── Configuración de la página ──────────────────────────────────────────────
st.set_page_config(
    page_title="Predictor de Riesgo Crediticio",
    page_icon="💳",
    layout="centered",
)

# ── Cargar artefactos ────────────────────────────────────────────────────────
@st.cache_resource
def cargar_modelos():
    label_encoders = joblib.load("label_encoders.joblib")
    pca            = joblib.load("pca_model.joblib")
    scaler         = joblib.load("minmax_scaler.joblib")
    model          = load_model("modelo_nn.keras")
    return label_encoders, pca, scaler, model

label_encoders, pca, scaler, model = cargar_modelos()

# ── Opciones categóricas ─────────────────────────────────────────────────────
CREDIT_MIX_OPTIONS  = ["Bad", "Good", "Standard"]
PAYMENT_MIN_OPTIONS = ["NM", "No", "Yes"]

LABEL_MAP = {0: "🔴 Alto Riesgo", 1: "🟡 Riesgo Medio", 2: "🟢 Bajo Riesgo"}
COLOR_MAP  = {0: "#FF4B4B",        1: "#FFA500",         2: "#21C354"}

RECOMENDACIONES = {
    0: {
        "icono":  "🚨",
        "titulo": "Acción urgente requerida",
        "color":  "#FF4B4B",
        "puntos": [
            "Reducir inmediatamente los días de retraso en pagos — cada día adicional deteriora el historial.",
            "Evitar nuevas solicitudes de crédito por al menos 6 meses para no acumular más consultas.",
            "Pagar las deudas pendientes priorizando las de mayor tasa de interés.",
            "Cambiar el hábito de pago: al menos cubrir el monto mínimo mensual sin falta.",
            "Reducir el número de cuentas bancarias activas si supera las necesarias.",
            "Considerar asesoría financiera profesional para reestructurar las obligaciones actuales.",
        ],
    },
    1: {
        "icono":  "⚠️",
        "titulo": "Hay margen de mejora",
        "color":  "#FFA500",
        "puntos": [
            "Mantener los pagos al día: evitar cualquier retraso adicional para mejorar la categoría.",
            "Intentar reducir la deuda pendiente al menos un 20% en los próximos 3 meses.",
            "Mejorar el mix crediticio combinando diferentes tipos de productos financieros responsablemente.",
            "Limitar las consultas de crédito únicamente a las estrictamente necesarias.",
            "Aumentar la antigüedad del historial manteniendo cuentas activas con buen comportamiento.",
            "Revisar si el número de tarjetas de crédito es manejable y cerrar las que no se usen.",
        ],
    },
    2: {
        "icono":  "✅",
        "titulo": "Perfil crediticio saludable",
        "color":  "#21C354",
        "puntos": [
            "Mantener el buen comportamiento de pago — la consistencia es clave para conservar este perfil.",
            "Continuar diversificando el mix crediticio para fortalecer aún más el historial.",
            "Aprovechar la buena calificación para negociar mejores tasas de interés con entidades financieras.",
            "Mantener la deuda pendiente por debajo del 30% del límite de crédito disponible.",
            "Evitar abrir muchas cuentas nuevas al mismo tiempo para no generar múltiples consultas.",
            "Considerar productos de ahorro o inversión para optimizar el capital disponible.",
        ],
    },
}

# ── Interfaz ─────────────────────────────────────────────────────────────────
st.title("💳 Predictor de Riesgo Crediticio")
st.markdown("Ingresa los datos del cliente y el modelo calculará su nivel de riesgo crediticio.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Datos Financieros")
    num_bank_accounts      = st.number_input("N° Cuentas Bancarias",        min_value=0, max_value=20,   value=3,   step=1)
    num_credit_card        = st.number_input("N° Tarjetas de Crédito",      min_value=0, max_value=15,   value=3,   step=1)
    interest_rate          = st.number_input("Tasa de Interés (%)",         min_value=1, max_value=40,   value=14,  step=1)
    delay_from_due_date    = st.number_input("Días de Retraso en Pagos",    min_value=0, max_value=70,   value=10,  step=1)
    num_of_delayed_payment = st.number_input("N° Pagos Atrasados",          min_value=0, max_value=30,   value=8,   step=1)

with col2:
    st.subheader("📋 Historial Crediticio")
    num_credit_inquiries  = st.number_input("N° Consultas de Crédito",      min_value=0, max_value=20,   value=4,   step=1)
    credit_mix            = st.selectbox("Tipo de Mix Crediticio",          CREDIT_MIX_OPTIONS)
    outstanding_debt      = st.number_input("Deuda Pendiente (USD)",        min_value=0, max_value=6000, value=800, step=50)
    credit_history_age    = st.number_input("Antigüedad Historial (años)",  min_value=0, max_value=40,   value=15,  step=1)
    payment_of_min_amount = st.selectbox("¿Paga monto mínimo?",            PAYMENT_MIN_OPTIONS)

st.divider()

# ── Predicción ───────────────────────────────────────────────────────────────
if st.button("🔍 Predecir Riesgo", use_container_width=True, type="primary"):

    credit_mix_enc = label_encoders["Credit_Mix"].transform([credit_mix])[0]
    payment_enc    = label_encoders["Payment_of_Min_Amount"].transform([payment_of_min_amount])[0]

    row = np.array([[
        num_bank_accounts, num_credit_card, interest_rate,
        delay_from_due_date, num_of_delayed_payment, num_credit_inquiries,
        credit_mix_enc, outstanding_debt, credit_history_age, payment_enc,
    ]])

    X_pca    = pca.transform(row)
    X_scaled = scaler.transform(X_pca)
    probs    = model.predict(X_scaled, verbose=0)[0]

    p_alto, p_medio, p_bajo = float(probs[0]), float(probs[1]), float(probs[2])

    # Simplemente toma la clase con mayor probabilidad
    pred_idx = int(np.argmax([p_alto, p_medio, p_bajo]))

    label = LABEL_MAP[pred_idx]
    color = COLOR_MAP[pred_idx]

    # ── Resultado ─────────────────────────────────────────────────────────────
    st.subheader("📈 Resultado")
    st.markdown(
        f"<div style='background-color:{color}22; border-left:6px solid {color}; "
        f"padding:16px 20px; border-radius:8px; font-size:1.6rem; font-weight:700; color:{color}'>"
        f"{label}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    # ── Probabilidades ────────────────────────────────────────────────────────
    st.markdown("#### 📊 Probabilidades detalladas")
    clases = [
        ("🔴 Alto Riesgo",  p_alto,  0),
        ("🟡 Riesgo Medio", p_medio, 1),
        ("🟢 Bajo Riesgo",  p_bajo,  2),
    ]
    for nombre, prob, idx in clases:
        es_ganadora = pred_idx == idx
        sufijo = " ◀ predicción" if es_ganadora else ""
        peso   = "700" if es_ganadora else "400"
        st.markdown(
            f"<div style='margin-bottom:4px; font-weight:{peso}'>{nombre}{sufijo}</div>",
            unsafe_allow_html=True,
        )
        st.progress(prob, text=f"{prob*100:.2f}%")

    with st.expander("🔢 Ver valores numéricos"):
        tabla = pd.DataFrame({
            "Clase":        ["🔴 Alto Riesgo", "🟡 Riesgo Medio", "🟢 Bajo Riesgo"],
            "Probabilidad": [f"{p*100:.4f}%" for p in [p_alto, p_medio, p_bajo]],
            "Valor raw":    [f"{p:.6f}"       for p in [p_alto, p_medio, p_bajo]],
        })
        st.dataframe(tabla, use_container_width=True, hide_index=True)

    # ── Recomendaciones ───────────────────────────────────────────────────────
    st.markdown("#### 💡 Recomendaciones")
    rec = RECOMENDACIONES[pred_idx]

    st.markdown(
        f"<div style='background-color:{rec['color']}22; border-left:6px solid {rec['color']}; "
        f"padding:14px 18px; border-radius:8px; margin-bottom:14px'>"
        f"<span style='font-size:1.1rem; font-weight:700; color:{rec['color']}'>"
        f"{rec['icono']}  {rec['titulo']}</span></div>",
        unsafe_allow_html=True,
    )

    for i, punto in enumerate(rec["puntos"], 1):
        st.markdown(f"**{i}.** {punto}")

st.divider()
st.caption("Modelo: Red Neuronal (Keras) · Preprocesamiento: LabelEncoder → SelectKBest → PCA → MinMaxScaler")
