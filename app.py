
import streamlit as st
import pandas as pd
import time
from scripts import predict
import os

css_file = os.path.join(os.path.dirname(__file__), "style.css")
with open(css_file) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


st.set_page_config(page_title="Predição de Risco de Depressão", layout="wide")
st.title("Predição do Risco de Depressão em Universitários")

# Criação das abas
tab1, tab2 = st.tabs(["Predição Individual", "Triagem via CSV"])


with tab1:
    st.subheader("Informe os seus dados:")

    model_name = st.selectbox(
        "Escolha o modelo",
        list(predict.MODELS_INFO.keys()),
        format_func=lambda x: f"{x} - {predict.MODELS_INFO[x]['description']}"
    )

    def select_option(feature, key):
        return st.selectbox(feature, list(predict.MAPPINGS[feature].keys()), key=key)

    user_input = {}
    user_input["Gender"] = select_option("Gender", key="gender_ind")
    user_input["Age"] = st.slider("Age", 16, 60, 21, step=1,key="age_ind")
    user_input["Academic Pressure"] = st.slider("Academic Pressure (1-5)", 1, 5, 3, key="acad_ind")
    user_input["CGPA"] = st.slider("CGPA", 0.0, 10.0, 6.0, step=0.1, key="cgpa_ind")
    user_input["Study Satisfaction"] = st.slider("Study Satisfaction (1-5)", 1, 5, 3, key="study_ind")
    user_input["Sleep Duration"] = st.slider("Sleep Duration (hours)", 0, 16, 8, step=1,key="sleep_ind")
    user_input["Dietary Habits"] = select_option("Dietary Habits", key="diet_ind")
    user_input["Have you ever had suicidal thoughts ?"] = select_option(
        "Have you ever had suicidal thoughts ?", key="suicide_ind"
    )
    user_input["Work/Study Hours"] = st.slider("Work/Study Hours per day", 0, 16, 5,step=1,key="work_ind")
    user_input["Financial Stress"] = st.slider("Financial Stress (1-5)", 1, 5, 2, key="finance_ind")
    user_input["Family History of Mental Illness"] = select_option(
        "Family History of Mental Illness", key="family_ind"
    )

    threshold = st.slider("Threshold de risco", 0.0, 1.0, 0.5, step=0.01, key="threshold_ind")

    if st.button("Prever Risco", key="btn_ind"):
        try:
            result = predict.predict(user_input, model_name, threshold)
            st.success(result["message"])
            st.info(f"Modelo usado: {model_name}\nDescrição: {result['model_description']}")
            st.write(f"Probabilidade exata: {result['probability']*100:.2f}%")
        except Exception as e:
            st.error(f"Erro ao realizar predição: {e}")


with tab2:
    st.subheader("Triagem de múltiplos alunos via CSV")
    uploaded_file = st.file_uploader("Envie um arquivo CSV com os dados dos alunos", type=["csv"], key="upload_csv")

    # Mapeamento de colunas alternativas para nomes esperados pelo modelo
    COLUMN_MAPPING = {
        "Study Hours": "Work/Study Hours",
        "Work/Sleep Duration": "Sleep Duration",
    }

    REQUIRED_COLS = [
        "Gender", "Age", "Academic Pressure", "CGPA", "Study Satisfaction",
        "Sleep Duration", "Dietary Habits", "Have you ever had suicidal thoughts ?",
        "Work/Study Hours", "Financial Stress", "Family History of Mental Illness"
    ]

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Renomear colunas automaticamente se houver alternativas
            df.rename(columns=COLUMN_MAPPING, inplace=True)
            
            # Verificar colunas faltantes
            missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]
            if missing_cols:
                st.error(f"As seguintes colunas obrigatórias estão faltando: {missing_cols}")
            else:
                st.success(f"Arquivo carregado com sucesso! Total de alunos: {len(df)}")
                
                # Converter colunas categóricas se forem numéricas
                for feature in predict.MAPPINGS:
                    if feature in df.columns:
                        if pd.api.types.is_numeric_dtype(df[feature]):
                            inverse_map = {v: k for k, v in predict.MAPPINGS[feature].items()}
                            df[feature] = df[feature].apply(lambda x: inverse_map.get(x, x))
                
                # Seleção do modelo
                model_name_csv = st.selectbox(
                    "Escolha o modelo",
                    list(predict.MODELS_INFO.keys()),
                    format_func=lambda x: f"{x} - {predict.MODELS_INFO[x]['description']}",
                    key="model_csv"
                )
                
                # Threshold
                threshold_csv = st.slider("Threshold de risco", 0.0, 1.0, 0.5, step=0.01, key="threshold_csv")
                
                if st.button("Prever risco para todos os alunos", key="btn_csv"):
                    results = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    total = len(df)
                    for idx, row in df.iterrows():
                        user_input = row.to_dict()
                        try:
                            result = predict.predict(user_input, model_name_csv, threshold_csv)
                            results.append({
                                "Aluno": idx,
                                "Mensagem": result["message"],
                                "Probabilidade": result["probability"],
                                "Modelo": model_name_csv
                            })
                        except Exception as e:
                            results.append({
                                "Aluno": idx,
                                "Mensagem": f"Erro: {e}",
                                "Probabilidade": None,
                                "Modelo": model_name_csv
                            })
                        
                        progress_bar.progress((idx + 1) / total)
                        status_text.text(f"Processando aluno {idx + 1} de {total}...")
                        time.sleep(0.01)
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    results_df = pd.DataFrame(results)
                    st.success("Processamento concluído!")
                    st.dataframe(results_df)
                    
                    # Botão para download
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Baixar resultados",
                        data=csv,
                        file_name='resultados_triagem.csv',
                        mime='text/csv'
                    )
                
        except Exception as e:
            st.error(f"Erro ao processar o CSV: {e}")
