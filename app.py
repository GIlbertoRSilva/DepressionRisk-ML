import streamlit as st
from scripts import predict  

st.set_page_config(page_title="Predição de Risco de Depressão", layout="wide")
st.title("Predição do Risco de Depressão em Universitários")

model_name = st.selectbox(
    "Escolha o modelo",
    list(predict.MODELS_INFO.keys()),
    format_func=lambda x: f"{x} - {predict.MODELS_INFO[x]['description']}"
)

st.subheader("Informe os seus dados:")

def select_option(feature):
    return st.selectbox(feature, list(predict.MAPPINGS[feature].keys()))

user_input = {}
user_input["Gender"] = select_option("Gender")
user_input["Age"] = st.slider("Age", 16, 60, 22)
user_input["Academic Pressure"] = st.slider("Academic Pressure (1-5)", 1, 5, 3)
user_input["CGPA"] = st.slider("CGPA", 0.0, 10.0, 8.0, step=0.1)
user_input["Study Satisfaction"] = st.slider("Study Satisfaction (1-5)", 1, 5, 4)
user_input["Sleep Duration"] = st.slider("Sleep Duration (hours)", 0, 12, 6)
user_input["Dietary Habits"] = select_option("Dietary Habits")
user_input["Have you ever had suicidal thoughts ?"] = select_option("Have you ever had suicidal thoughts ?")
user_input["Work/Study Hours"] = st.slider("Work/Study Hours per day", 0, 16, 5)
user_input["Financial Stress"] = st.slider("Financial Stress (1-5)", 1, 5, 2)
user_input["Family History of Mental Illness"] = select_option("Family History of Mental Illness")


threshold = st.slider("Threshold de risco", 0.0, 1.0, 0.5, step=0.01)

if st.button("🔍 Prever Risco"):
    try:
        result = predict.predict(user_input, model_name, threshold)
        st.success(result["message"])
        st.info(f"Modelo usado: {model_name}\nDescrição: {result['model_description']}")
        st.write(f"Probabilidade exata: {result['probability']*100:.2f}%")
    except Exception as e:
        st.error(f"Erro ao realizar predição: {e}")
