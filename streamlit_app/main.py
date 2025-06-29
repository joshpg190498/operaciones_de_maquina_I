import requests

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


# Configuración de la página y API
st.set_page_config(page_title="Dashboard de Ganancias", layout="wide")
API_URL = "http://api:8000"

# Mapeo de niveles educativos
education_map = {
    "Preschool": 1,
    "1st-4th": 2,
    "5th-6th": 3,
    "7th-8th": 4,
    "9th": 5,
    "10th": 6,
    "11th": 7,
    "12th": 8,
    "HS-grad": 9,
    "Some-college": 10,
    "Assoc-voc": 11,
    "Assoc-acdm": 12,
    "Bachelors": 13,
    "Masters": 14,
    "Doctorate": 15,
    "Prof-school": 16
}

def predictions_page():
    st.header("📊 Predicción de ingresos")

    with st.form("census_form"):
        col1, col2 = st.columns(2)

        age = col1.number_input("Edad", min_value=18, max_value=65, value=32)
        workclass = col2.selectbox("Clase de trabajo", [
            "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
            "Local-gov", "State-gov", "Without-pay", "Never-worked"
        ])

        education_level = col1.selectbox("Nivel educativo", list(education_map.keys()))

        marital_status = col2.selectbox("Estado civil", [
            "Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"
        ])

        occupation = col1.selectbox("Ocupación", [
            "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
            "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
            "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"
        ])

        relationship = col2.selectbox("Relación", [
            "Husband", "Not-in-family", "Own-child", "Unmarried", "Wife", "Other-relative"
        ])

        race = col1.selectbox("Raza", [
            "White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"
        ])

        gender = col2.selectbox("Género", ["Male", "Female"])

        hours_per_week = col1.number_input("Horas por semana", min_value=1, max_value=100, value=40)

        native_country = col2.selectbox("País de origen", [
            "United-States", "Mexico", "Philippines", "Germany", "Canada", "India", "Other"
        ])

        submit = st.form_submit_button("Predecir ingreso")

    if submit:
        payload = {
            "age": age,
            "workclass": workclass,
            "educationnum": education_map[education_level],
            "marital_status": marital_status,
            "occupation": occupation,
            "relationship": relationship,
            "race": race,
            "gender": gender,
            "hours_per_week": hours_per_week,
            "native_country": native_country
        }

        try:
            response = requests.post(f"{API_URL}/predict/", json=payload)
            response.raise_for_status()
            resultado = response.json().get("prediction")
            proba = response.json().get("proba")

            st.success(f"✅ Resultado de predicción: {resultado} (Probabilidad: {proba*100:.2f}%)")
        except Exception as e:
            st.error(f"❌ Error al consultar la API: {e}")

def history_page():
    st.header("📜 Historial de predicciones")
    try:
        resp = requests.get(f"{API_URL}/history/")
        resp.raise_for_status()
        df = pd.DataFrame(resp.json())
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        st.dataframe(df)
    except Exception as e:
        st.error(f"⚠️ No se pudo cargar el historial: {e}")

def docs_page():
    st.header("📚 Documentación de la API")
    components.iframe(f"http://localhost:8800/docs", height=800, scrolling=True)

# Definición de páginas usando st.navigation (Streamlit 1.45)
pages = [
    st.Page(predictions_page, title="📈 Predicciones"),
    st.Page(history_page,    title="📜 Historial"),
    st.Page(docs_page,       title="📚 Documentación API")
]

def main():
    selected_page = st.navigation(pages, position="sidebar", expanded=True)
    selected_page.run()

if __name__ == "__main__":
    main()
