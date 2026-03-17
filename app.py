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

# ── Opciones categóricas (extraídas del notebook) ────────────────────────────
CREDIT_MIX_OPTIONS      = ["Bad", "Good", "Standard"]
PAYMENT_MIN_OPTIONS     = ["NM", "No", "Yes"]

# Features seleccionadas por SelectKBest (orden del notebook)
# Num_Bank_Accounts, Num_Credit_Card, Interest_Rate, Delay_from_due_date,
# Num_of_Delayed_Payment, Num_Credit_Inquiries, Credit_Mix,
# Outstanding_Debt, Credit_History_Age, Payment_of_Min_Amount
SELECTED_FEATURES = [
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Num_Credit_Inquiries",
    "Credit_Mix",
    "Outstanding_Debt",
    "Credit_History_Age",
    "Payment_of_Min_Amount",
]

LABEL_MAP = {0: "🔴 Alto Riesgo", 1: "🟡 Riesgo Medio", 2: "🟢 Bajo Riesgo"}
COLOR_MAP  = {0: "#FF4B4B",        1: "#FFA500",         2: "#21C354"}

# ── Interfaz ─────────────────────────────────────────────────────────────────
st.title("💳 Predictor de Riesgo Crediticio")
st.markdown("Ingresa los datos del cliente y el modelo calculará su nivel de riesgo crediticio.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Datos Financieros")
    num_bank_accounts      = st.number_input("N° Cuentas Bancarias",          min_value=0.0,  max_value=20.0,  value=3.0,   step=0.5)
    num_credit_card        = st.number_input("N° Tarjetas de Crédito",        min_value=0.0,  max_value=15.0,  value=3.0,   step=0.5)
    interest_rate          = st.number_input("Tasa de Interés (%)",           min_value=1,    max_value=40,    value=14)
    delay_from_due_date    = st.number_input("Días de Retraso en Pagos",      min_value=-5.0, max_value=70.0,  value=10.0,  step=0.25)
    num_of_delayed_payment = st.number_input("N° Pagos Atrasados",            min_value=0.0,  max_value=30.0,  value=8.0,   step=0.25)

with col2:
    st.subheader("📋 Historial Crediticio")
    num_credit_inquiries   = st.number_input("N° Consultas de Crédito",       min_value=0.0,  max_value=20.0,  value=4.0,   step=0.25)
    credit_mix             = st.selectbox("Tipo de Mix Crediticio",           CREDIT_MIX_OPTIONS)
    outstanding_debt       = st.number_input("Deuda Pendiente (USD)",         min_value=0.0,  max_value=6000.0, value=800.0, step=10.0)
    credit_history_age     = st.number_input("Antigüedad Historial (años)",   min_value=0.0,  max_value=40.0,  value=15.0,  step=0.1, format="%.2f")
    payment_of_min_amount  = st.selectbox("¿Paga monto mínimo?",             PAYMENT_MIN_OPTIONS)

st.divider()

# ── Predicción ───────────────────────────────────────────────────────────────
if st.button("🔍 Predecir Riesgo", use_container_width=True, type="primary"):

    # 1) Codificar variables categóricas
    credit_mix_enc = label_encoders["Credit_Mix"].transform([credit_mix])[0]
    payment_enc    = label_encoders["Payment_of_Min_Amount"].transform([payment_of_min_amount])[0]

    # 2) Armar fila con las 10 features seleccionadas (mismo orden)
    row = np.array([[
        num_bank_accounts,
        num_credit_card,
        interest_rate,
        delay_from_due_date,
        num_of_delayed_payment,
        num_credit_inquiries,
        credit_mix_enc,
        outstanding_debt,
        credit_history_age,
        payment_enc,
    ]])

    # 3) PCA (5 componentes)
    X_pca = pca.transform(row)

    # 4) MinMax Scaler
    X_scaled = scaler.transform(X_pca)

    # 5) Predicción
    probs    = model.predict(X_scaled, verbose=0)[0]
    pred_idx = int(np.argmax(probs))

    # ── Resultados ────────────────────────────────────────────────────────────
    st.subheader("📈 Resultado")

    label = LABEL_MAP[pred_idx]
    color = COLOR_MAP[pred_idx]

    st.markdown(
        f"<div style='background-color:{color}22; border-left:6px solid {color}; "
        f"padding:16px 20px; border-radius:8px; font-size:1.4rem; font-weight:700; color:{color}'>"
        f"{label}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### Probabilidades por clase")
    prob_df = pd.DataFrame({
        "Clase": ["🔴 Alto Riesgo", "🟡 Riesgo Medio", "🟢 Bajo Riesgo"],
        "Probabilidad": [f"{p*100:.1f}%" for p in probs],
        "Valor": probs,
    })

    for _, r in prob_df.iterrows():
        st.progress(float(r["Valor"]), text=f"{r['Clase']}  —  {r['Probabilidad']}")

st.divider()
st.caption("Modelo: Red Neuronal (Keras) · Preprocesamiento: LabelEncoder → SelectKBest → PCA → MinMaxScaler")