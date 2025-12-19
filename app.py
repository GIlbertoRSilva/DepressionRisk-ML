import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
import joblib
import os

# Configuração da Página
st.set_page_config(
    page_title="Sistema de Triagem de Saúde Mental",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAPPINGS = {
    "Gender": {"Male": 0, "Female": 1},
    "Dietary Habits": {"Unhealthy": 0, "Average": 1, "Healthy": 2},
    "Family History of Mental Illness": {"No": 0, "Yes": 1},
    "Have you ever had suicidal thoughts ?": {"No": 0, "Yes": 1}
}

MODELS_INFO = {
    "SVM": {"path": "models/svm.joblib", "description": "Apresenta a maior capacidade de identificar corretamente casos relevantes, ou seja, minimiza o risco de deixar de detectar situações importantes. Contudo, tende a apresentar uma precisão menor, indicando casos que não necessitam de acompanhamento."},
    "KNN": {"path": "models/knn.joblib", "description": "Exibe um desempenho equilibrado. Quando classifica um caso como relevante, a probabilidade de acerto é alta, mas ele deixa de identificar algumas situações que também podem exigir atenção, resultando em um desempenho moderado."},
    "MLP": {"path": "models/mlp.joblib", "description": "Apresenta o melhor desempenho geral, com maior acurácia. Esse modelo mantém um bom equilíbrio entre identificar corretamente casos relevantes e evitar falsas indicações, mostrando-se também o mais consistente nos testes."}
}

@st.cache_resource
def load_model_st(model_name):
    """Carrega o modelo e armazena em cache no Streamlit."""
    if model_name not in MODELS_INFO:
        st.error(f"Modelo desconhecido: {model_name}")
        return None, None
    
    info = MODELS_INFO[model_name]
    model_path = info["path"]
    
    if not os.path.exists("models"):
        os.makedirs("models")
        
    if not os.path.exists(model_path):
         st.error(f"CRÍTICO: Arquivo de modelo não encontrado em: {model_path}. Verifique a pasta 'models/'.")
         return None, None

    try:
        model = joblib.load(model_path)
        return model, info["description"]
    except Exception as e:
        st.error(f"Erro ao carregar o modelo {model_name}: {e}")
        return None, None

def generate_synthetic_demo_data(n_samples=500):
    """
    Gera dados sintéticos na memória caso o arquivo demo.csv não exista.
    Segue a lógica do script fornecido anteriormente.
    """
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
    """Aplica os MAPPINGS nas entradas do usuário (Modo Individual)."""
    encoded = {}
    for feature, value in user_input.items():
        if feature in MAPPINGS:
            if value not in MAPPINGS[feature]:
                st.error(f"Valor de entrada inválido para {feature}: {value}")
                return None
            encoded[feature] = MAPPINGS[feature][value]
        else:
            encoded[feature] = value  
    return encoded

def run_prediction(user_input_dict: dict, model_name: str):

    model, description = load_model_st(model_name)
    if model is None: return None 

    encoded_input = encode_input(user_input_dict)
    if encoded_input is None: return None

    expected_order = ["Gender", "Age", "Academic Pressure", "CGPA", "Study Satisfaction", 
                      "Sleep Duration", "Dietary Habits", "Have you ever had suicidal thoughts ?", 
                      "Work/Study Hours", "Financial Stress", "Family History of Mental Illness"]
    
    df = pd.DataFrame([encoded_input])
    
    try:
        df = df[expected_order]
    except KeyError as e:
         st.error(f"Erro de estrutura de dados. Faltando coluna: {e}")
         return None

    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)[:, 1][0]
        else:
            pred = model.predict(df)[0]
            proba = float(pred) 
            
        return proba, description
    except Exception as e:
        st.error(f"Erro durante a inferência do modelo: {e}")
        return None

def plot_gauge(probabilidade):

    if probabilidade < 0.40: bar_color = "#A9DFBF" # Verde
    elif probabilidade < 0.70: bar_color = "#F9E79F" # Amarelo
    else: bar_color = "#F5B7B1" # Vermelho
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probabilidade * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Probabilidade de Risco (%)", 'font': {'size': 18, 'color': '#2C3E50'}},
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
    st.title("Sistema de Triagem: Saúde Mental do Estudante")
    st.warning(
        """
        #### Aviso de Isenção de Responsabilidade
        **Esta ferramenta NÃO fornece diagnóstico médico.**
        
        Os resultados apresentados são apenas indicadores estatísticos baseados em IA para auxiliar na triagem inicial.
        
        **Em caso de sofrimento psíquico ou crise, procure sempre um profissional de saúde mental ou ligue 188 (CVV).**
        """, icon="⚠️"
    )

    st.write("---") 
    
    st.markdown("### Bem-vindo(a) ao ambiente de monitoramento e cuidado.")
    
    st.markdown("""
    Esta ferramenta utiliza Inteligência Artificial para auxiliar na identificação precoce de riscos relacionados à saúde mental, especificamente depressão, em estudantes universitários.
    
    **Como funciona?**
    Baseado em um modelo de Machine Learning treinado com dados históricos, analisamos padrões em fatores acadêmicos, hábitos de vida e histórico pessoal para fornecer um indicativo probabilístico de risco.
    """)
    
    with st.expander("ℹ️ Detalhes Técnicos dos Modelos Disponíveis"):
        st.write("Você pode escolher entre diferentes arquiteturas de IA para a análise:")
        for model_name, info in MODELS_INFO.items():
            st.markdown(f"**• {model_name}:** {info['description']}")

def show_student_assessment():
    st.title("Autoavaliação (Triagem Individual)")
    st.markdown("Preencha o formulário com atenção. Seus dados são processados em tempo real e não são armazenados após o fechamento da página.")

    with st.sidebar:
        st.write("---")
        st.subheader("⚙️ Configuração da Análise")
        selected_model_name = st.selectbox("Selecione o Modelo de IA:", list(MODELS_INFO.keys()), index=2) # Padrão MLP
        st.caption(MODELS_INFO[selected_model_name]['description'])
        st.write("---")

    with st.form("assessment_form"):
        st.subheader("1. Perfil e Fatores Acadêmicos")
        col1, col2, col3 = st.columns(3)

        user_input = {}
        
        with col1:
            gender_label = st.selectbox("Gênero", ["Feminino", "Masculino"])
            user_input["Gender"] = "Female" if gender_label == "Feminino" else "Male"
            user_input["Age"] = st.number_input("Idade", min_value=16, max_value=80, value=21)
            
        with col2:
            user_input["CGPA"] = st.number_input("CGPA (Média Acumulada 0-10)", 0.0, 10.0, 7.5, step=0.1, help="Sua média global de notas.")
            user_input["Academic Pressure"] = st.slider("Nível de Pressão Acadêmica (1=Baixa, 5=Extrema)", 1, 5, 3)

        with col3:
            user_input["Study Satisfaction"] = st.slider("Satisfação com os Estudos (1=Insatisfeito, 5=Muito Satisfeito)", 1, 5, 3)
            user_input["Work/Study Hours"] = st.number_input("Horas Diárias de Estudo/Trabalho", 0, 20, 6)

        st.subheader("2. Saúde e Bem-Estar")
        col_h1, col_h2 = st.columns(2)
        
        with col_h1:
            user_input["Sleep Duration"] = st.number_input("Horas médias de sono por noite", 2, 16, 7)
            diet_label = st.selectbox("Hábitos Alimentares", ["Saudável", "Médio", "Não Saudável"])
            diet_map = {"Saudável": "Healthy", "Médio": "Average", "Não Saudável": "Unhealthy"}
            user_input["Dietary Habits"] = diet_map[diet_label]
            
        with col_h2:
            user_input["Financial Stress"] = st.slider("Estresse Financeiro (1=Baixo, 5=Alto)", 1, 5, 3)
            hist_label = st.radio("Histórico familiar de doença mental?", ["Não", "Sim"])
            user_input["Family History of Mental Illness"] = "No" if hist_label == "Não" else "Yes"

        st.write("---")
        st.markdown("⚠️ **Atenção: Pergunta Sensível**")
        suicide_label = st.selectbox("Você já teve pensamentos suicidas?",["Não", "Sim", "Prefiro não responder"])

        if suicide_label in ["Sim", "Prefiro não responder"]:
            user_input["Have you ever had suicidal thoughts ?"] = "Yes"
            st.warning(
                "Se você está passando por um momento difícil agora, ligue para o CVV (188) ou procure ajuda imediata."
            )
        else:
            user_input["Have you ever had suicidal thoughts ?"] = "No"

        st.write("---")
        submit_button = st.form_submit_button("Executar Análise de Risco", type="primary")

    if submit_button:
        with st.spinner(f"Processando dados utilizando o modelo {selected_model_name}..."):
            proba = run_prediction(user_input, selected_model_name)
            time.sleep(0.5)

        if proba is not None:
            probabilidade_valor, description = proba
            
            if probabilidade_valor < 0.40:
                categoria = "Baixo Risco"
                msg_tipo = "success"
                mensagem_final = "Seus indicadores atuais sugerem um bom equilíbrio. Mantenha seus hábitos saudáveis de sono e alimentação."
            elif probabilidade_valor < 0.70:
                categoria = "Risco Moderado"
                msg_tipo = "warning"
                mensagem_final = "Alerta amarelo. Alguns fatores indicam sobrecarga ou estresse elevado. Considere rever sua rotina de sono e pressão acadêmica."
            else:
                categoria = "Alto Risco"
                msg_tipo = "error"
                mensagem_final = "**Recomendação de Cuidado:** O padrão de respostas indica probabilidade elevada. Recomendamos fortemente buscar o serviço de apoio psicológico da instituição."

            st.subheader("Resultado da Análise")
            
            col_res_gauge, col_res_txt = st.columns([1, 1.5])
            
            with col_res_gauge:
                st.plotly_chart(plot_gauge(probabilidade_valor), use_container_width=True)
                st.caption(f"Modelo utilizado: {selected_model_name}")
            
            with col_res_txt:
                st.markdown(f"### Categoria Indicada: **{categoria}**")
                st.progress(probabilidade_valor)
                
                if msg_tipo == "success":
                    st.success(mensagem_final, icon="✅")
                elif msg_tipo == "warning":
                    st.warning(mensagem_final, icon="⚠️")
                else:
                    st.error(mensagem_final, icon="🛑")
                    with st.expander("🆘 Contatos de Apoio (Exemplo)"):
                        st.write("- **SAP (Serviço de Apoio Psicológico):** Bloco C, Sala 2")
                        st.write("- **CVV (Nacional):** Ligue 188")

def show_institution_portal():
    st.title("Portal Institucional (Triagem em Lote)")
    st.warning("Área Restrita. Módulo para processamento de múltiplos alunos.")

    with st.sidebar:
        st.write("---")
        st.subheader("⚙️ Configuração do Lote")
        selected_model_batch = st.selectbox("Modelo:", list(MODELS_INFO.keys()), index=2, key="batch_model_select")
        
        st.write("---")
        st.markdown("**Definição de Limiares (Thresholds)**")
    
        thresholds_batch = st.slider(
            "Ajuste de Sensibilidade",
            min_value=0.0, max_value=1.0, value=(0.40, 0.75), step=0.05, key="slider_batch"
        )
        
        st.caption("**Valores Recomendados:**")
        st.caption("Para um equilíbrio entre identificar riscos reais e evitar alarmes falsos, sugerimos:")
        st.caption("• **Início do Moderado:** entre 0.40 e 0.50")
        st.caption("• **Início do Alto:** entre 0.70 e 0.85")
        
        st.write("---") 
        
        cut_mod_b, cut_high_b = thresholds_batch
    
        st.info(f"**Configuração Atual:**\n\nModerado: >= {cut_mod_b*100:.0f}%\nAlto: >={cut_high_b*100:.0f}%")

    st.markdown("### Seleção de Dados")
    
    data_source = st.radio("Escolha a fonte dos dados:", 
                           ["Carregar Planilha (.csv/.xlsx)", "Usar Dados de Demonstração (Simulação)"])
    
    df_batch = None
    start_processing = False

    if data_source == "Carregar Planilha (.csv/.xlsx)":
        st.markdown("""
        O arquivo deve conter as colunas: `Gender`, `Age`, `Academic Pressure`, `CGPA`, `Study Satisfaction`, `Sleep Duration`, `Dietary Habits`, `Have you ever had suicidal thoughts ?`, `Work/Study Hours`, `Financial Stress`, `Family History of Mental Illness`.
        """)
        uploaded_file = st.file_uploader("Carregar planilha de dados", type=['csv', 'xlsx'])

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df_batch = pd.read_csv(uploaded_file)
                else:
                    df_batch = pd.read_excel(uploaded_file)
                
                st.success(f"Arquivo carregado. {len(df_batch)} registros.")
                
                required_cols_model = list(MAPPINGS.keys()) + ["Age", "Academic Pressure", "CGPA", "Study Satisfaction", "Sleep Duration", "Work/Study Hours", "Financial Stress"]
                missing_cols = [col for col in required_cols_model if col not in df_batch.columns]
                
                if missing_cols:
                    st.error(f"Faltando colunas: {missing_cols}")
                else:
                    if st.button("Iniciar Processamento", type="primary"):
                        start_processing = True

            except Exception as e:
                st.error(f"Erro ao ler arquivo: {e}")
                
    else: 
        st.info("Este modo gera dados sintéticos (500 alunos) e executa a predição automaticamente para demonstrar o funcionamento do sistema.")
        if st.button("Gerar Dados e Simular Predição", type="primary"):
            with st.spinner("Gerando dados de demonstração..."):
                df_batch = get_demo_data()
                st.success(f"Dados gerados com sucesso: {len(df_batch)} registros simulados.")
                st.markdown("##### Amostra dos Dados Gerados:")
                st.dataframe(df_batch.head(3), use_container_width=True)
                start_processing = True

    if start_processing and df_batch is not None:
        model, _ = load_model_st(selected_model_batch)
        
        if model is not None:
            progress_bar = st.progress(0)
            
            df_processed = df_batch.copy()
            
            with st.spinner("Codificando variáveis e analisando (Otimizado)..."):
                
                for col, mapping in MAPPINGS.items():
                    if col not in df_processed.columns: continue
                    
                    if pd.api.types.is_numeric_dtype(df_processed[col]):
                        continue

                    df_processed[col] = df_processed[col].map(mapping)
                    
                    if df_processed[col].isnull().any():
                        st.error(f"Erro: Valores inválidos encontrados na coluna '{col}'. Verifique se os dados correspondem a: {list(mapping.keys())}")
                        st.stop()
            
                model_cols = ["Gender", "Age", "Academic Pressure", "CGPA", "Study Satisfaction", 
                              "Sleep Duration", "Dietary Habits", "Have you ever had suicidal thoughts ?", 
                              "Work/Study Hours", "Financial Stress", "Family History of Mental Illness"]
                
                try:
                    X_batch = df_processed[model_cols]
                    
                   
                    X_batch = X_batch.fillna(0)
         
                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(X_batch)[:, 1]
                    else:
                        probabilities = model.predict(X_batch).astype(float)

                    df_batch["Probabilidade_Risco"] = probabilities
            
                    df_batch["Categoria_Risco"] = pd.cut(
                        df_batch["Probabilidade_Risco"],
                        bins=[-0.1, cut_mod_b, cut_high_b, 1.1], 
                        labels=["Baixo", "Moderado", "Alto"]
                    )
                    
                    def formatar_genero(val):
                        if val == 0 or str(val).lower() == 'male': return "Masculino"
                        if val == 1 or str(val).lower() == 'female': return "Feminino"
                        return val
                    
                    if 'Gender' in df_batch.columns:
                        df_batch['Gender'] = df_batch['Gender'].apply(formatar_genero)

    
                    if 'Age' in df_batch.columns:
                        df_batch['Age'] = df_batch['Age'].fillna(0).astype(int)

                    progress_bar.progress(100)
                    
                    st.write("---")
                    st.subheader("Dashboard de Resultados")
                    
                    col_d1, col_d2 = st.columns(2)
                    with col_d1:
                        if not df_batch.empty:
                            contagem = df_batch['Categoria_Risco'].value_counts().reset_index()
                            contagem.columns = ['Nível de Risco', 'Total de Alunos']
                            
                            fig_bar = px.bar(contagem, x='Nível de Risco', y='Total de Alunos',
                                             title='Distribuição da População por Risco',
                                             color='Nível de Risco',
                                             text='Total de Alunos',
                                             color_discrete_map={'Baixo':'#A9DFBF', 'Moderado':'#F9E79F', 'Alto':'#F5B7B1'})
                            fig_bar.update_layout(xaxis_title=None)
                            st.plotly_chart(fig_bar, use_container_width=True)
                    
                
                    st.subheader("Lista Priorizada de Alunos")
                    df_sorted = df_batch.sort_values(by="Probabilidade_Risco", ascending=False)

                    cols_to_show = ['Categoria_Risco', 'Probabilidade_Risco']
                    if 'Student_ID' in df_batch.columns: cols_to_show.insert(0, 'Student_ID')
                    cols_to_show.extend([c for c in df_batch.columns if c not in cols_to_show])

                    num_cells = df_sorted.shape[0] * len(cols_to_show)
                    pd.set_option("styler.render.max_elements", max(num_cells + 5000, 262144))

                    st.dataframe(
                        df_sorted[cols_to_show].style.background_gradient(subset=['Probabilidade_Risco'], cmap='RdYlGn_r', vmin=0, vmax=1)
                                     .format({'Probabilidade_Risco': "{:.2%}"}),
                        use_container_width=True
                    )
                    
                except Exception as e:
                    st.error(f"Erro durante processamento do modelo: {e}")

with st.sidebar:
    st.title("Navegação")
    selected_page = st.radio(
        "Ir para:",
        ["Página Inicial", "Autoavaliação (Aluno)", "Portal Institucional"],
        index=0
    )
    st.write("---")
    st.markdown("v1.4.0 - Atualizado em Dezembro de 2025")

if selected_page == "Página Inicial":
    show_home()
elif selected_page == "Autoavaliação (Aluno)":
    show_student_assessment()
elif selected_page == "Portal Institucional":
    show_institution_portal()