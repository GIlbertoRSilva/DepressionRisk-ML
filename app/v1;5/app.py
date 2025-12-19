import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
import joblib
import os

import shap
import matplotlib.pyplot as plt

TRANSLATIONS = {
    "pt": {
        "page_title": "Sistema de Triagem de Saúde Mental",
        "sidebar_nav_title": "Navegação",
        "nav_go_to": "Ir para:",
        "nav_options": ["Página Inicial", "Autoavaliação (Aluno)", "Portal Institucional"],
        "nav_footer": "v1.5.0 - Atualizado com SHAP e i18n",
      
        "model_svm_desc": "Apresenta a maior capacidade de identificar corretamente casos relevantes (minimiza falsos negativos). Contudo, tende a apresentar uma precisão menor (mais falsos positivos).",
        "model_knn_desc": "Exibe um desempenho equilibrado. Alta probabilidade de acerto quando classifica como relevante, mas pode deixar de identificar algumas situações.",
        "model_mlp_desc": "Apresenta o melhor desempenho geral e maior acurácia. Bom equilíbrio entre identificar casos relevantes e evitar falsas indicações. (Suporta explicação SHAP).",
       
        "error_model_unknown": "Modelo desconhecido: {}",
        "error_model_not_found": "CRÍTICO: Arquivo de modelo não encontrado em: {}. Verifique a pasta 'models/'.",
        "error_loading_model": "Erro ao carregar o modelo {}: {}",
        "error_invalid_input": "Valor de entrada inválido para {}: {}",
        "error_data_structure": "Erro de estrutura de dados. Faltando coluna: {}",
        "error_inference": "Erro durante a inferência do modelo: {}",
        "error_missing_cols": "Faltando colunas: {}",
        "error_reading_file": "Erro ao ler arquivo: {}",
        "error_invalid_values_col": "Erro: Valores inválidos encontrados na coluna '{}'. Verifique se os dados correspondem a: {}",
        "error_processing_model": "Erro durante processamento do modelo: {}",
        "warning_disclaimer_title": "Aviso de Isenção de Responsabilidade",
        "warning_disclaimer_text": "**Esta ferramenta NÃO fornece diagnóstico médico.** Os resultados são apenas indicadores estatísticos para auxiliar na triagem. **Em caso de crise, procure um profissional ou ligue 188 (CVV).**",
        "warning_sensitive_q": "Atenção: Pergunta Sensível",
        "warning_suicide_help": "Se você está passando por um momento difícil agora, procure ajuda imediata (ex: CVV 188 no Brasil).",
        "warning_restricted_area": "Área Restrita. Módulo para processamento de múltiplos alunos.",
        # Home Page
        "home_title": "Sistema de Triagem: Saúde Mental do Estudante",
        "home_welcome": "Bem-vindo(a) ao ambiente de monitoramento e cuidado.",
        "home_description": "Esta ferramenta utiliza Inteligência Artificial para auxiliar na identificação precoce de riscos relacionados à saúde mental em estudantes universitários.\n\n**Como funciona?**\nBaseado em um modelo de Machine Learning treinado com dados históricos, analisamos padrões em fatores acadêmicos e hábitos de vida.",
        "home_tech_details": "ℹ️ Detalhes Técnicos dos Modelos Disponíveis",
        "home_choose_arch": "Você pode escolher entre diferentes arquiteturas de IA para a análise:",
        # Student Assessment
        "student_title": "Autoavaliação (Triagem Individual)",
        "student_subtitle": "Preencha o formulário com atenção. Seus dados são processados em tempo real e não são armazenados.",
        "config_analysis": "⚙️ Configuração da Análise",
        "select_model": "Selecione o Modelo de IA:",
        "form_section1": "1. Perfil e Fatores Acadêmicos",
        "label_gender": "Gênero",
        "opt_gender": ["Feminino", "Masculino"],
        "label_age": "Idade",
        "label_cgpa": "CGPA (Média Acumulada 0-10)",
        "help_cgpa": "Sua média global de notas.",
        "label_acad_pressure": "Nível de Pressão Acadêmica (1=Baixa, 5=Extrema)",
        "label_study_sat": "Satisfação com os Estudos (1=Insatisfeito, 5=Muito Satisfeito)",
        "label_hours": "Horas Diárias de Estudo/Trabalho",
        "form_section2": "2. Saúde e Bem-Estar",
        "label_sleep": "Horas médias de sono por noite",
        "label_diet": "Hábitos Alimentares",
        "opt_diet": ["Saudável", "Médio", "Não Saudável"],
        "label_financial": "Estresse Financeiro (1=Baixo, 5=Alto)",
        "label_family_hist": "Histórico familiar de doença mental?",
        "opt_yes_no": ["Não", "Sim"],
        "label_suicide": "Você já teve pensamentos suicidas?",
        "opt_suicide": ["Não", "Sim", "Prefiro não responder"],
        "btn_submit": "Executar Análise de Risco",
        "spinner_processing": "Processando dados utilizando o modelo {}...",
       
        "cat_low": "Baixo Risco",
        "cat_mod": "Risco Moderado",
        "cat_high": "Alto Risco",
        "msg_low": "Seus indicadores atuais sugerem um bom equilíbrio. Mantenha seus hábitos saudáveis de sono e alimentação.",
        "msg_mod": "Alerta amarelo. Alguns fatores indicam sobrecarga ou estresse elevado. Considere rever sua rotina de sono e pressão acadêmica.",
        "msg_high": "**Recomendação de Cuidado:** O padrão de respostas indica probabilidade elevada. Recomendamos fortemente buscar o serviço de apoio psicológico da instituição.",
        "result_title": "Analysis Result",
        "gauge_title": "Probabilidade de Risco (%)",
        "model_used_caption": "Modelo utilizado: {}",
        "cat_indicated_title": "Categoria Indicada: **{}**",
        "support_contacts_expander": "🆘 Contatos de Apoio (Exemplo)",
   
        "shap_ind_title": "Relatório de Transparência da IA (SHAP - MLP)",
        "shap_ind_expander": "Entenda como o resultado foi calculado",
        "shap_ind_desc": "Além do gráfico abaixo, preparamos um resumo textual para facilitar a compreensão.",
        "shap_spinner": "Gerando explicação detalhada (SHAP)...",
        "shap_txt_risk_drivers": "**Fatores que mais contribuíram para o AUMENTO do risco:**",
        "shap_txt_risk_reducers": "**Fatores que ajudaram a REDUZIR o risco (Proteção):**",
        "shap_txt_neutral": "O impacto dos demais fatores foi neutro ou equilibrado.",
        
        "portal_title": "Portal Institucional (Triagem em Lote)",
        "config_batch": "⚙️ Configuração do Lote",
        "thresholds_title": "Definição de Limiares (Thresholds)",
        "thresholds_slider": "Ajuste de Sensibilidade",
        "thresholds_rec_title": "**Valores Recomendados:**",
        "thresholds_rec_text": "Para um equilíbrio entre identificar riscos reais e evitar alarmes falsos, sugerimos:\n• **Início do Moderado:** entre 0.40 e 0.50\n• **Início do Alto:** entre 0.70 e 0.85",
        "thresholds_current": "**Configuração Atual:**\n\nModerado: >= {:.0f}%\nAlto: >={:.0f}%",
        "data_selection_title": "### Seleção de Dados",
        "data_source_radio": "Escolha a fonte dos dados:",
        "opt_data_source": ["Carregar Planilha (.csv/.xlsx)", "Usar Dados de Demonstração (Simulação)"],
        "upload_instructions": "O arquivo deve conter as colunas: `Gender`, `Age`, `Academic Pressure`, `CGPA`, `Study Satisfaction`, `Sleep Duration`, `Dietary Habits`, `Have you ever had suicidal thoughts ?`, `Work/Study Hours`, `Financial Stress`, `Family History of Mental Illness`.",
        "upload_label": "Carregar planilha de dados",
        "upload_success": "Arquivo carregado. {} registros.",
        "btn_start_proc": "Iniciar Processamento",
        "demo_mode_info": "Este modo gera dados sintéticos (500 alunos) e executa a predição automaticamente para demonstrar o funcionamento do sistema.",
        "btn_generate_demo": "Gerar Dados e Simular Predição",
        "spinner_generating_demo": "Gerando dados de demonstração...",
        "demo_success": "Dados gerados com sucesso: {} registros simulados.",
        "demo_sample_title": "##### Amostra dos Dados Gerados:",
        "spinner_encoding": "Codificando variáveis e analisando (Otimizado)...",
 
        "progress_encoding": "1/4 Codificando variáveis e validando dados...",
        "progress_inference": "2/4 Executando modelo preditivo de IA...",
        "progress_shap": "3/4 Calculando explicabilidade SHAP (isso pode levar alguns segundos)...",
        "progress_finishing": "4/4 Finalizando e gerando visualizações...",
    
        "dash_title": "Dashboard de Resultados",
        "chart_dist_title": "Distribuição da População por Risco",
        "chart_x_axis": "Nível de Risco",
        "chart_y_axis": "Total de Alunos",
        "list_prioritized_title": "Lista Priorizada de Alunos",
       
        "shap_batch_title": "Análise Global de Fatores (SHAP - MLP)",
        "shap_batch_desc": "O gráfico abaixo (Beeswarm Plot) resume quais variáveis são mais importantes globalmente para o modelo MLP neste lote de alunos.\n- **Eixo Y:** Variáveis ordenadas por importância.\n- **Eixo X:** Impacto na probabilidade de risco (negativo = diminui risco, positivo = aumenta risco).\n- **Cor:** Valor original da variável (Vermelho = alto, Azul = baixo).",
        "shap_batch_cols_info": "ℹ️ As colunas com os valores SHAP individuais (prefixo 'SHAP_') foram adicionadas à tabela abaixo para análise detalhada.",
       
        "batch_risk_factors_col": "Principais Fatores de Risco",
        "batch_summary_title": "Resumo Executivo do Grupo (Decisão Coletiva)",
        "batch_summary_text": "Com base na análise de todos os alunos processados, os **3 principais fatores** que mais impulsionam o risco neste grupo são:",
        
        "label_male": "Masculino",
        "label_female": "Feminino",
        "risk_prob_col": "Probabilidade_Risco",
        "risk_cat_col": "Categoria_Risco"
    },
    "en": {
        "page_title": "Mental Health Screening System",
        "sidebar_nav_title": "Navigation",
        "nav_go_to": "Go to:",
        "nav_options": ["Home Page", "Self-Assessment (Student)", "Institution Portal"],
        "nav_footer": "v1.5.0 - Updated with SHAP and i18n",
       
        "model_svm_desc": "Shows the highest capacity to correctly identify relevant cases (minimizes false negatives). However, tends to have lower precision (more false positives).",
        "model_knn_desc": "Displays balanced performance. High probability of being correct when classifying as relevant, but might miss some situations requiring attention.",
        "model_mlp_desc": "Shows best overall performance and highest accuracy. Good balance between identifying relevant cases and avoiding false flags. (Supports SHAP explanation).",
    
        "error_model_unknown": "Unknown model: {}",
        "error_model_not_found": "CRITICAL: Model file not found at: {}. Check the 'models/' folder.",
        "error_loading_model": "Error loading model {}: {}",
        "error_invalid_input": "Invalid input value for {}: {}",
        "error_data_structure": "Data structure error. Missing column: {}",
        "error_inference": "Error during model inference: {}",
        "error_missing_cols": "Missing columns: {}",
        "error_reading_file": "Error reading file: {}",
        "error_invalid_values_col": "Error: Invalid values found in column '{}'. Check if data corresponds to: {}",
        "error_processing_model": "Error during model processing: {}",
        "warning_disclaimer_title": "Disclaimer Warning",
        "warning_disclaimer_text": "**This tool DOES NOT provide medical diagnosis.** The results are statistical AI-based indicators to aid initial screening. **In case of psychic distress or crisis, always seek a mental health professional or call emergency services.**",
        "warning_sensitive_q": "Attention: Sensitive Question",
        "warning_suicide_help": "If you are going through a difficult time right now, please seek immediate help.",
        "warning_restricted_area": "Restricted Area. Module for processing multiple students.",
  
        "home_title": "Screening System: Student Mental Health",
        "home_welcome": "Welcome to the monitoring and care environment.",
        "home_description": "This tool uses Artificial Intelligence to aid in the early identification of risks related to mental health in university students.\n\n**How does it work?**\nBased on a Machine Learning model trained with historical data, we analyze patterns in academic factors and lifestyle habits.",
        "home_tech_details": "ℹ️ Technical Details of Available Models",
        "home_choose_arch": "You can choose between different AI architectures for the analysis:",
        
        "student_title": "Self-Assessment (Individual Screening)",
        "student_subtitle": "Fill out the form carefully. Your data is processed in real-time and not stored after closing the page.",
        "config_analysis": "⚙️ Analysis Configuration",
        "select_model": "Select AI Model:",
        "form_section1": "1. Profile and Academic Factors",
        "label_gender": "Gender",
        "opt_gender": ["Female", "Male"],
        "label_age": "Age",
        "label_cgpa": "CGPA (Cumulative Grade Point Average 0-10)",
        "help_cgpa": "Your global grade average.",
        "label_acad_pressure": "Academic Pressure Level (1=Low, 5=Extreme)",
        "label_study_sat": "Study Satisfaction (1=Dissatisfied, 5=Very Satisfied)",
        "label_hours": "Daily Study/Work Hours",
        "form_section2": "2. Health and Well-being",
        "label_sleep": "Average sleep hours per night",
        "label_diet": "Dietary Habits",
        "opt_diet": ["Healthy", "Average", "Unhealthy"],
        "label_financial": "Financial Stress (1=Low, 5=High)",
        "label_family_hist": "Family history of mental illness?",
        "opt_yes_no": ["No", "Yes"],
        "label_suicide": "Have you ever had suicidal thoughts?",
        "opt_suicide": ["No", "Yes", "Prefer not to answer"],
        "btn_submit": "Run Risk Analysis",
        "spinner_processing": "Processing data using model {}...",
        
        "cat_low": "Low Risk",
        "cat_mod": "Moderate Risk",
        "cat_high": "High Risk",
        "msg_low": "Your current indicators suggest a good balance. Maintain your healthy sleep and eating habits.",
        "msg_mod": "Yellow alert. Some factors indicate overload or elevated stress. Consider reviewing your sleep routine and academic pressure.",
        "msg_high": "**Care Recommendation:** The pattern of responses indicates elevated probability. We strongly recommend seeking the institution's psychological support service.",
        "result_title": "Analysis Result",
        "gauge_title": "Risk Probability (%)",
        "model_used_caption": "Model used: {}",
        "cat_indicated_title": "Indicated Category: **{}**",
        "support_contacts_expander": "🆘 Support Contacts (Example)",

        "shap_ind_title": "AI Transparency Report (SHAP - MLP)",
        "shap_ind_expander": "Understand how the result was calculated",
        "shap_ind_desc": "In addition to the chart below, we have prepared a textual summary to facilitate understanding.",
        "shap_spinner": "Generating detailed explanation (SHAP)...",
        "shap_txt_risk_drivers": "**Factors that most contributed to INCREASING risk:**",
        "shap_txt_risk_reducers": "**Factors that helped REDUCE risk (Protection):**",
        "shap_txt_neutral": "The impact of other factors was neutral or balanced.",
    
        "portal_title": "Institution Portal (Batch Screening)",
        "config_batch": "⚙️ Batch Configuration",
        "thresholds_title": "Threshold Definitions",
        "thresholds_slider": "Sensitivity Adjustment",
        "thresholds_rec_title": "**Recommended Values:**",
        "thresholds_rec_text": "For a balance between identifying real risks and avoiding false alarms, we suggest:\n• **Start of Moderate:** between 0.40 and 0.50\n• **Start of High:** between 0.70 and 0.85",
        "thresholds_current": "**Current Configuration:**\n\nModerate: >= {:.0f}%\nHigh: >={:.0f}%",
        "data_selection_title": "### Data Selection",
        "data_source_radio": "Choose data source:",
        "opt_data_source": ["Load Spreadsheet (.csv/.xlsx)", "Use Demo Data (Simulation)"],
        "upload_instructions": "The file must contain columns: `Gender`, `Age`, `Academic Pressure`, `CGPA`, `Study Satisfaction`, `Sleep Duration`, `Dietary Habits`, `Have you ever had suicidal thoughts ?`, `Work/Study Hours`, `Financial Stress`, `Family History of Mental Illness`.",
        "upload_label": "Load data spreadsheet",
        "upload_success": "File loaded. {} records.",
        "btn_start_proc": "Start Processing",
        "demo_mode_info": "This mode generates synthetic data (500 students) and automatically runs prediction to demonstrate system functionality.",
        "btn_generate_demo": "Generate Data and Simulate Prediction",
        "spinner_generating_demo": "Generating demo data...",
        "demo_success": "Data generated successfully: {} simulated records.",
        "demo_sample_title": "##### Sample of Generated Data:",
        "spinner_encoding": "Encoding variables and analyzing (Optimized)...",
      
        "progress_encoding": "1/4 Encoding variables and validating data...",
        "progress_inference": "2/4 Running AI prediction model...",
        "progress_shap": "3/4 Calculating SHAP explainability (this may take a few seconds)...",
        "progress_finishing": "4/4 Finishing and generating visualizations...",

        "dash_title": "Results Dashboard",
        "chart_dist_title": "Population Distribution by Risk",
        "chart_x_axis": "Risk Level",
        "chart_y_axis": "Total Students",
        "list_prioritized_title": "Prioritized Student List",

        "shap_batch_title": "Global Factor Analysis (SHAP - MLP)",
        "shap_batch_desc": "The chart below (Beeswarm Plot) summarizes which variables are most important globally for the MLP model in this batch of students.\n- **Y-Axis:** Variables ordered by importance.\n- **X-Axis:** Impact on risk probability (negative = decreases risk, positive = increases risk).\n- **Color:** Original value of the variable (Red = high, Blue = low).",
        "shap_batch_cols_info": "ℹ️ Columns with individual SHAP values (prefix 'SHAP_') have been added to the table below for detailed analysis.",
        
        "batch_risk_factors_col": "Main Risk Factors",
        "batch_summary_title": "Group Executive Summary (Collective Decision)",
        "batch_summary_text": "Based on the analysis of all students, the **top 3 factors** driving risk in this group are:",
      
        "label_male": "Male",
        "label_female": "Female",
        "risk_prob_col": "Risk_Probability",
        "risk_cat_col": "Risk_Category"
    }
}


current_lang = "pt"


def t(key):
    return TRANSLATIONS[current_lang].get(key, f"MISSING_{key}")

st.set_page_config(
    page_title="Mental Health Screening System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


with st.sidebar:
    lang_selection = st.selectbox("Language / Idioma", ["Português", "English"], index=0)
    current_lang = "pt" if lang_selection == "Português" else "en"


MAPPINGS = {
    "Gender": {"Male": 0, "Female": 1},
    "Dietary Habits": {"Unhealthy": 0, "Average": 1, "Healthy": 2},
    "Family History of Mental Illness": {"No": 0, "Yes": 1},
    "Have you ever had suicidal thoughts ?": {"No": 0, "Yes": 1}
}

MODELS_INFO = {
    "SVM": {"path": "models/svm.joblib", "description_key": "model_svm_desc"},
    "KNN": {"path": "models/knn.joblib", "description_key": "model_knn_desc"},
    "MLP": {"path": "models/mlp.joblib", "description_key": "model_mlp_desc"}
}

@st.cache_resource
def load_model_st(model_name):
    """Carrega o modelo e armazena em cache no Streamlit."""
    if model_name not in MODELS_INFO:
        st.error(t("error_model_unknown").format(model_name))
        return None, None
    
    info = MODELS_INFO[model_name]
    model_path = info["path"]
    
    description = t(info["description_key"])

    if not os.path.exists("models"):
        os.makedirs("models")
        
    if not os.path.exists(model_path):
         st.error(t("error_model_not_found").format(model_path))
        
         try:
             from sklearn.dummy import DummyClassifier
             dummy = DummyClassifier(strategy="most_frequent")
            
             X_dummy = np.zeros((10, 11)) 
             y_dummy = np.random.randint(0, 2, 10)
             dummy.fit(X_dummy, y_dummy)
             joblib.dump(dummy, model_path)
             st.warning(f"Modelo dummy criado em {model_path} para fins de teste.")
         except:
             return None, None

    try:
        model = joblib.load(model_path)
        return model, description
    except Exception as e:
        st.error(t("error_loading_model").format(model_name, e))
        return None, None

def generate_synthetic_demo_data(n_samples=500):
    np.random.seed(42)
    data = {
        "Gender": np.random.choice(["Male", "Female"], n_samples),
        "Age": np.random.randint(18, 35, n_samples),
        "Academic Pressure": np.random.randint(1, 6, n_samples),
        "CGPA": np.round(np.random.uniform(5.0, 10.0, n_samples), 2),
        "Study Satisfaction": np.random.randint(1, 6, n_samples),
        "Sleep Duration": np.random.choice([4, 5, 6, 7, 8, 9], n_samples),
        "Dietary Habits": np.random.choice(["Healthy", "Average", "Unhealthy"], n_samples),
        "Have you ever had suicidal thoughts ?": np.random.choice(["Yes", "No"], n_samples, p=[0.1, 0.9]),
        "Work/Study Hours": np.random.randint(2, 14, n_samples),
        "Financial Stress": np.random.randint(1, 6, n_samples),
        "Family History of Mental Illness": np.random.choice(["Yes", "No"], n_samples, p=[0.2, 0.8]),
        "Student_ID": [f"STD-{1000+i}" for i in range(n_samples)]
    }
    df = pd.DataFrame(data)

    df["Age"] = df["Age"].clip(15, 45).round(0)
    df["CGPA"] = df["CGPA"].clip(0, 10).round(2)
    df["Sleep Duration"] = df["Sleep Duration"].clip(0, 12).round(1)
    df["Work/Study Hours"] = df["Work/Study Hours"].clip(0, 16).round(1)
    
    return df

def get_demo_data():
    possible_paths = [
        "../data/processed/demo.csv", 
        "data/processed/demo.csv", 
        "demo.csv"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return pd.read_csv(path)
            
    return generate_synthetic_demo_data()

def encode_input(user_input: dict):

    encoded = {}
    for feature, value in user_input.items():
        if feature in MAPPINGS:
            
           
            if value in MAPPINGS[feature]:
                 encoded[feature] = MAPPINGS[feature][value]
            else:
                 
                 found = False
                 for en_key, mapped_val in MAPPINGS[feature].items():
                    
                     val_to_en_map = {
                         "Feminino": "Female", "Masculino": "Male",
                         "Saudável": "Healthy", "Médio": "Average", "Não Saudável": "Unhealthy",
                         "Não": "No", "Sim": "Yes"
                     }
                     
                     val_in_en = val_to_en_map.get(value, value)

                     if val_in_en in MAPPINGS[feature]:
                          encoded[feature] = MAPPINGS[feature][val_in_en]
                          found = True
                          break
                 if not found:
                    st.error(t("error_invalid_input").format(feature, value))
                    return None
        else:
            encoded[feature] = value  
    return encoded

@st.cache_resource
def get_shap_explainer(_model, model_name):
    
    if model_name != "MLP":
        return None

    
    background_data_raw = generate_synthetic_demo_data(n_samples=25) 
    
    df_processed_bg = background_data_raw.copy()
    for col, mapping in MAPPINGS.items():
        if col in df_processed_bg.columns:
             df_processed_bg[col] = df_processed_bg[col].map(mapping)
    
    model_cols = ["Gender", "Age", "Academic Pressure", "CGPA", "Study Satisfaction", 
                  "Sleep Duration", "Dietary Habits", "Have you ever had suicidal thoughts ?", 
                  "Work/Study Hours", "Financial Stress", "Family History of Mental Illness"]
    
    X_background = df_processed_bg[model_cols].fillna(0)

    def custom_predict_proba(data):
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=X_background.columns)
        return _model.predict_proba(data)
    try:
        explainer = shap.KernelExplainer(custom_predict_proba, X_background)
        return explainer
    except Exception as e:
        st.warning(f"Não foi possível inicializar o SHAP Explainer para MLP: {e}")
        return None

def run_prediction(user_input_dict: dict, model_name: str):

    model, description = load_model_st(model_name)
    if model is None: return None, None, None

    encoded_input = encode_input(user_input_dict)
    if encoded_input is None: return None, None, None

    expected_order = ["Gender", "Age", "Academic Pressure", "CGPA", "Study Satisfaction", 
                      "Sleep Duration", "Dietary Habits", "Have you ever had suicidal thoughts ?", 
                      "Work/Study Hours", "Financial Stress", "Family History of Mental Illness"]
    
    df_encoded = pd.DataFrame([encoded_input])
    
    try:
        df_encoded = df_encoded[expected_order]
    except KeyError as e:
         st.error(t("error_data_structure").format(e))
         return None, None, None

    shap_values_obj = None
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df_encoded)[:, 1][0]
        else:
            pred = model.predict(df_encoded)[0]
            proba = float(pred) 
       
        if model_name == "MLP":
             explainer = get_shap_explainer(model, model_name)
             if explainer:
                
                 shap_values_raw = explainer.shap_values(df_encoded)

                 sv = None
                 ev = None
            
                 target_idx = 1
                 
                 if isinstance(shap_values_raw, list):
                    
                     idx_to_use = target_idx if len(shap_values_raw) > target_idx else 0
                     sv = shap_values_raw[idx_to_use][0] 
                    
                     ev_raw = explainer.expected_value
                     if isinstance(ev_raw, (list, np.ndarray)) and len(ev_raw) > idx_to_use:
                         ev = ev_raw[idx_to_use]
                     else:
                         ev = ev_raw

                 elif isinstance(shap_values_raw, np.ndarray):
                   
                     if len(shap_values_raw.shape) == 3: 
                         
                         idx_to_use = target_idx if shap_values_raw.shape[2] > target_idx else 0
                         sv = shap_values_raw[0, :, idx_to_use] 
                         
                         ev_raw = explainer.expected_value
                         if isinstance(ev_raw, (list, np.ndarray)) and len(ev_raw) > idx_to_use:
                             ev = ev_raw[idx_to_use]
                         else:
                             ev = ev_raw
                             
                     elif len(shap_values_raw.shape) == 2:
                        
                         sv = shap_values_raw[0]
                         ev = explainer.expected_value
                     
                     else:
                         
                         sv = shap_values_raw[0]
                         ev = explainer.expected_value

                 
                 shap_values_obj = shap.Explanation(values=sv, 
                                                  base_values=ev, 
                                                  data=df_encoded.iloc[0], 
                                                  feature_names=df_encoded.columns)

        return proba, description, shap_values_obj
    except Exception as e:
        st.error(t("error_inference").format(e))
        return None, None, None

def plot_gauge(probabilidade):

    if probabilidade < 0.40: bar_color = "#A9DFBF" 
    elif probabilidade < 0.70: bar_color = "#F9E79F" 
    else: bar_color = "#F5B7B1" 
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probabilidade * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': t("gauge_title"), 'font': {'size': 18, 'color': '#2C3E50'}},
        number = {'suffix': "%", 'font': {'color': '#2C3E50'}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#2C3E50"},
            'bar': {'color': bar_color}, 
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#F0F2F6",
            'steps': [
                {'range': [0, 40], 'color': "rgba(169, 223, 191, 0.3)"}, 
                {'range': [40, 75], 'color': "rgba(249, 231, 159, 0.3)"}, 
                {'range': [75, 100], 'color': "rgba(245, 183, 177, 0.3)"} 
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 75
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def show_home():
    st.title(t("home_title"))
    st.warning(
        f"""
        #### {t("warning_disclaimer_title")}
        {t("warning_disclaimer_text")}
        """, icon="⚠️"
    )

    st.write("---") 
    
    st.markdown(f"### {t("home_welcome")}")
    
    st.markdown(t("home_description"))
    
    with st.expander(t("home_tech_details")):
        st.write(t("home_choose_arch"))
        for model_name, info in MODELS_INFO.items():
            st.markdown(f"**• {model_name}:** {t(info['description_key'])}")

def show_student_assessment():
    st.title(t("student_title"))
    st.markdown(t("student_subtitle"))

    with st.sidebar:
        st.write("---")
        st.subheader(t("config_analysis"))
        selected_model_name = st.selectbox(t("select_model"), list(MODELS_INFO.keys()), index=2) # Padrão MLP
        st.caption(t(MODELS_INFO[selected_model_name]['description_key']))
        st.write("---")

    with st.form("assessment_form"):
        st.subheader(t("form_section1"))
        col1, col2, col3 = st.columns(3)

        user_input = {}
        
        with col1:
           
            gender_opts = t("opt_gender")
            gender_label = st.selectbox(t("label_gender"), gender_opts)
           
            user_input["Gender"] = "Female" if gender_label == gender_opts[0] else "Male"
            user_input["Age"] = st.number_input(t("label_age"), min_value=16, max_value=80, value=21)
            
        with col2:
            user_input["CGPA"] = st.number_input(t("label_cgpa"), 0.0, 10.0, 7.5, step=0.1, help=t("help_cgpa"))
            user_input["Academic Pressure"] = st.slider(t("label_acad_pressure"), 1, 5, 3)

        with col3:
            user_input["Study Satisfaction"] = st.slider(t("label_study_sat"), 1, 5, 3)
            user_input["Work/Study Hours"] = st.number_input(t("label_hours"), 0, 20, 6)

        st.subheader(t("form_section2"))
        col_h1, col_h2 = st.columns(2)
        
        with col_h1:
            user_input["Sleep Duration"] = st.number_input(t("label_sleep"), 2, 16, 7)
            diet_opts = t("opt_diet")
            diet_label = st.selectbox(t("label_diet"), diet_opts)
        
            diet_map_rev = {diet_opts[0]: "Healthy", diet_opts[1]: "Average", diet_opts[2]: "Unhealthy"}
            user_input["Dietary Habits"] = diet_map_rev[diet_label]
            
        with col_h2:
            user_input["Financial Stress"] = st.slider(t("label_financial"), 1, 5, 3)
            yes_no_opts = t("opt_yes_no")
            hist_label = st.radio(t("label_family_hist"), yes_no_opts)
            user_input["Family History of Mental Illness"] = "No" if hist_label == yes_no_opts[0] else "Yes"

        st.write("---")
        st.markdown(f"⚠️ **{t('warning_sensitive_q')}**")
        suicide_opts = t("opt_suicide")
        suicide_label = st.selectbox(t("label_suicide"), suicide_opts)

        if suicide_label in [suicide_opts[1], suicide_opts[2]]:
            user_input["Have you ever had suicidal thoughts ?"] = "Yes"
            st.warning(t("warning_suicide_help"))
        else:
            user_input["Have you ever had suicidal thoughts ?"] = "No"

        st.write("---")
        submit_button = st.form_submit_button(t("btn_submit"), type="primary")

    if submit_button:
       
        spinner_msg = t("spinner_processing").format(selected_model_name)
        if selected_model_name == "MLP":
             spinner_msg = t("shap_spinner")

        with st.spinner(spinner_msg):
           
            proba, description, shap_obj = run_prediction(user_input, selected_model_name)
            time.sleep(0.5)

        if proba is not None:
            probabilidade_valor = proba
            
            if probabilidade_valor < 0.40:
                categoria = t("cat_low")
                msg_tipo = "success"
                mensagem_final = t("msg_low")
            elif probabilidade_valor < 0.70:
                categoria = t("cat_mod")
                msg_tipo = "warning"
                mensagem_final = t("msg_mod")
            else:
                categoria = t("cat_high")
                msg_tipo = "error"
                mensagem_final = t("msg_high")

            st.subheader(t("result_title"))
            
            col_res_gauge, col_res_txt = st.columns([1, 1.5])
            
            with col_res_gauge:
                st.plotly_chart(plot_gauge(probabilidade_valor), use_container_width=True)
                st.caption(t("model_used_caption").format(selected_model_name))
            
            with col_res_txt:
                st.markdown(f"### {t('cat_indicated_title').format(categoria)}")
                st.progress(probabilidade_valor)
                
                if msg_tipo == "success":
                    st.success(mensagem_final, icon="✅")
                elif msg_tipo == "warning":
                    st.warning(mensagem_final, icon="⚠️")
                else:
                    st.error(mensagem_final, icon="🛑")
                    with st.expander(t("support_contacts_expander")):
                        st.write("- **SAP (Serviço de Apoio Psicológico):** Bloco C, Sala 2")
                        st.write("- **CVV (Nacional/Brazil):** Ligue 188")
            
        
            if selected_model_name == "MLP" and shap_obj is not None:
                st.write("---")
                st.subheader(t("shap_ind_title"))
                with st.expander(t("shap_ind_expander"), expanded=True):
                    st.markdown(t("shap_ind_desc"))
                    
                   
                    fig_shap, ax = plt.subplots()
                    
                    shap.plots.waterfall(shap_obj, show=False, max_display=12)
                    st.pyplot(fig_shap)
                    plt.close(fig_shap) 

                    vals = shap_obj.values
                    names = shap_obj.feature_names
                    
                    
                    feature_impacts = list(zip(names, vals))
                    
                    
                    risk_increasers = sorted([x for x in feature_impacts if x[1] > 0], key=lambda x: x[1], reverse=True)
                
                    risk_reducers = sorted([x for x in feature_impacts if x[1] < 0], key=lambda x: x[1])

                    col_txt1, col_txt2 = st.columns(2)
                    
                    with col_txt1:
                        st.markdown(t("shap_txt_risk_drivers"))
                        if risk_increasers:
                            for name, val in risk_increasers[:3]: # Top 3
                                st.write(f"- **{name}** (+{val:.2f})")
                        else:
                            st.write(t("shap_txt_neutral"))

                    with col_txt2:
                        st.markdown(t("shap_txt_risk_reducers"))
                        if risk_reducers:
                            for name, val in risk_reducers[:3]: # Top 3
                                st.write(f"- **{name}** ({val:.2f})")
                        else:
                             st.write(t("shap_txt_neutral"))

def show_institution_portal():
    st.title(t("portal_title"))
    st.warning(t("warning_restricted_area"))

    with st.sidebar:
        st.write("---")
        st.subheader(t("config_batch"))
        selected_model_batch = st.selectbox(t("select_model"), list(MODELS_INFO.keys()), index=2, key="batch_model_select")
        
        st.write("---")
        st.markdown(t("thresholds_title"))
    
        thresholds_batch = st.slider(
            t("thresholds_slider"),
            min_value=0.0, max_value=1.0, value=(0.40, 0.75), step=0.05, key="slider_batch"
        )
        
        st.caption(t("thresholds_rec_title"))
        st.caption(t("thresholds_rec_text"))
        
        st.write("---") 
        
        cut_mod_b, cut_high_b = thresholds_batch
    
        st.info(t("thresholds_current").format(cut_mod_b*100, cut_high_b*100))

    st.markdown(t("data_selection_title"))
    
    data_source_opts = t("opt_data_source")
    data_source = st.radio(t("data_source_radio"), data_source_opts)
    
    df_batch = None
    start_processing = False

    if data_source == data_source_opts[0]:
        st.markdown(t("upload_instructions"))
        uploaded_file = st.file_uploader(t("upload_label"), type=['csv', 'xlsx'])

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df_batch = pd.read_csv(uploaded_file)
                else:
                    df_batch = pd.read_excel(uploaded_file)
                
                st.success(t("upload_success").format(len(df_batch)))
                
                required_cols_model = list(MAPPINGS.keys()) + ["Age", "Academic Pressure", "CGPA", "Study Satisfaction", "Sleep Duration", "Dietary Habits", "Have you ever had suicidal thoughts ?`, `Work/Study Hours`, `Financial Stress`, `Family History of Mental Illness`."]
                missing_cols = [col for col in required_cols_model if col not in df_batch.columns]
                
                if missing_cols:
                    st.error(t("error_missing_cols").format(missing_cols))
                else:
                    if st.button(t("btn_start_proc"), type="primary"):
                        start_processing = True

            except Exception as e:
                st.error(t("error_reading_file").format(e))
                
    else: 
        st.info(t("demo_mode_info"))
        if st.button(t("btn_generate_demo"), type="primary"):
            with st.spinner(t("spinner_generating_demo")):
                df_batch = get_demo_data()
                st.success(t("demo_success").format(len(df_batch)))
                st.markdown(t("demo_sample_title"))
                st.dataframe(df_batch.head(3), use_container_width=True)
                start_processing = True

    if start_processing and df_batch is not None:
        model, _ = load_model_st(selected_model_batch)
        
        if model is not None:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            df_processed = df_batch.copy()
            
          
            status_text.text(t("progress_encoding"))
            progress_bar.progress(10)
            
            try:
                for col, mapping in MAPPINGS.items():
                    if col not in df_processed.columns: continue
                    
                    if pd.api.types.is_numeric_dtype(df_processed[col]):
                        continue

                    df_processed[col] = df_processed[col].map(mapping)
                    
                    if df_processed[col].isnull().any():
                        st.error(t("error_invalid_values_col").format(col, list(mapping.keys())))
                        st.stop()
            
                model_cols = ["Gender", "Age", "Academic Pressure", "CGPA", "Study Satisfaction", 
                              "Sleep Duration", "Dietary Habits", "Have you ever had suicidal thoughts ?", 
                              "Work/Study Hours", "Financial Stress", "Family History of Mental Illness"]
                
                X_batch = df_processed[model_cols]
                X_batch = X_batch.fillna(0)
                
          
                status_text.text(t("progress_inference"))
                progress_bar.progress(30)
        
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(X_batch)[:, 1]
                else:
                    probabilities = model.predict(X_batch).astype(float)

                
                col_prob_name = t("risk_prob_col")
                col_cat_name = t("risk_cat_col")

                df_batch[col_prob_name] = probabilities
        
                df_batch[col_cat_name] = pd.cut(
                    df_batch[col_prob_name],
                    bins=[-0.1, cut_mod_b, cut_high_b, 1.1], 
                    labels=[t("cat_low"), t("cat_mod"), t("cat_high")]
                )
                
                progress_bar.progress(50)
                
                shap_values_matrix = None
                if selected_model_batch == "MLP":
                    status_text.text(t("progress_shap"))
                    progress_bar.progress(60)
                    
                    explainer = get_shap_explainer(model, selected_model_batch)
                    if explainer:
                      
                        shap_vals_raw = explainer.shap_values(X_batch)

                        target_idx = 1 
                        
                        if isinstance(shap_vals_raw, list):
                            idx_to_use = target_idx if len(shap_vals_raw) > target_idx else 0
                            shap_values_matrix = shap_vals_raw[idx_to_use]
                            
                        elif isinstance(shap_vals_raw, np.ndarray):
                            if len(shap_vals_raw.shape) == 3:
                                idx_to_use = target_idx if shap_vals_raw.shape[2] > target_idx else 0
                                shap_values_matrix = shap_vals_raw[:, :, idx_to_use]
                            else:
                                shap_values_matrix = shap_vals_raw
                        else:
                            shap_values_matrix = shap_vals_raw

                       
                        if len(shap_values_matrix.shape) == 2:
                            shap_df = pd.DataFrame(shap_values_matrix, columns=[f"SHAP_{c}" for c in X_batch.columns])
                            df_batch = pd.concat([df_batch.reset_index(drop=True), shap_df.reset_index(drop=True)], axis=1)
                            
                            feature_names = X_batch.columns
                            readable_factors = []
                            
                            for i in range(shap_values_matrix.shape[0]):
                                row_vals = shap_values_matrix[i]
                                drivers = []
                                for j, val in enumerate(row_vals):
                                    if val > 0: 
                                        drivers.append((feature_names[j], val))
                                
                                drivers.sort(key=lambda x: x[1], reverse=True)
                                
                                if drivers:
                                    top_3 = [d[0] for d in drivers[:3]]
                                    readable_factors.append(", ".join(top_3))
                                else:
                                    readable_factors.append(t("shap_txt_neutral"))
                            
                            col_factors_name = t("batch_risk_factors_col")
                            df_batch[col_factors_name] = readable_factors

                        else:
                            st.warning("Aviso: Formato inesperado dos valores SHAP em lote.")
                
    
                status_text.text(t("progress_finishing"))
                progress_bar.progress(90)
                
    
                def formatar_genero(val):
                    if val == 0 or str(val).lower() == 'male': return t("label_male")
                    if val == 1 or str(val).lower() == 'female': return t("label_female")
                    return val
                
                if 'Gender' in df_batch.columns:
                    df_batch['Gender'] = df_batch['Gender'].apply(formatar_genero)

                if 'Age' in df_batch.columns:
                    df_batch['Age'] = df_batch['Age'].fillna(0).astype(int)

                progress_bar.progress(100)
                status_text.empty() 
                
                st.write("---")
                st.subheader(t("dash_title"))
                
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    if not df_batch.empty:
                        contagem = df_batch[col_cat_name].value_counts().reset_index()
                        contagem.columns = [t("chart_x_axis"), t("chart_y_axis")]
                        
                        color_map = {t("cat_low"):'#A9DFBF', t("cat_mod"):'#F9E79F', t("cat_high"):'#F5B7B1'}
                        
                        fig_bar = px.bar(contagem, x=t("chart_x_axis"), y=t("chart_y_axis"),
                                            title=t("chart_dist_title"),
                                            color=t("chart_x_axis"),
                                            text=t("chart_y_axis"),
                                            color_discrete_map=color_map)
                        fig_bar.update_layout(xaxis_title=None)
                        st.plotly_chart(fig_bar, use_container_width=True)

                with col_d2:
                    if selected_model_batch == "MLP" and shap_values_matrix is not None:
                            st.subheader(t("shap_batch_title"))
                 
                            if len(shap_values_matrix.shape) == 2:
                                mean_abs_impact = np.mean(np.abs(shap_values_matrix), axis=0)
                                top_indices = np.argsort(mean_abs_impact)[::-1][:3]
                                top_features = X_batch.columns[top_indices]
                                
                                st.markdown(f"**{t('batch_summary_title')}**")
                                st.write(t("batch_summary_text"))
                                for f in top_features:
                                    st.write(f"- {f}")
                            
                            st.markdown(t("shap_batch_desc"))
                            
                            fig_shap_batch, ax = plt.subplots()
                            shap.summary_plot(shap_values_matrix, X_batch, show=False, max_display=10)
                            st.pyplot(fig_shap_batch)
                            plt.close(fig_shap_batch)
                
            
                st.subheader(t("list_prioritized_title"))
                
                if selected_model_batch == "MLP" and shap_values_matrix is not None:
                    st.info(t("shap_batch_cols_info"))

                df_sorted = df_batch.sort_values(by=col_prob_name, ascending=False)

                cols_to_show = [col_cat_name, col_prob_name]
                if 'Student_ID' in df_batch.columns: cols_to_show.insert(0, 'Student_ID')
                
                if selected_model_batch == "MLP" and t("batch_risk_factors_col") in df_batch.columns:
                    cols_to_show.append(t("batch_risk_factors_col"))

                cols_to_show.extend([c for c in df_batch.columns if c not in cols_to_show])

                num_cells = df_sorted.shape[0] * len(cols_to_show)
                pd.set_option("styler.render.max_elements", max(num_cells + 5000, 262144))

                st.dataframe(
                    df_sorted[cols_to_show].style.background_gradient(subset=[col_prob_name], cmap='RdYlGn_r', vmin=0, vmax=1)
                                                .format({col_prob_name: "{:.2%}"}),
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(t("error_processing_model").format(e))

with st.sidebar:
    st.title(t("sidebar_nav_title"))
    nav_opts = t("nav_options")
    selected_page = st.radio(
        t("nav_go_to"),
        nav_opts,
        index=0
    )
    st.write("---")
    st.markdown(t("nav_footer"))

if selected_page == nav_opts[0]:
    show_home()
elif selected_page == nav_opts[1]:
    show_student_assessment()
elif selected_page == nav_opts[2]:
    show_institution_portal()