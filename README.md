# DepressionRisk-ML — Machine Learning for Predicting Depression Risk in University Students

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()
[![Status](https://img.shields.io/badge/status-active-brightgreen)]()
[![DOI](https://zenodo.org/badge/1110932673.svg)](https://doi.org/10.5281/zenodo.17914238)

An end-to-end Machine Learning project designed to predict depression risk among university students using multiple classification algorithms. The project includes a complete training pipeline, exploratory notebooks, and an interactive interface built with Streamlit.

---

## Features

- **Multiple Models**: KNN, SVM, and MLP Neural Network  
- **Interactive Web App**: Streamlit interface for single and batch predictions  
- **Exploratory Data Analysis**: Dedicated notebooks for visual and statistical inspection  
- **Automated Preprocessing**: Encoding, scaling, and SMOTE balancing  
- **Robust Validation**: Stratified cross-validation with multiple performance metrics  

---

## Dataset

The project uses the **Student Depression Dataset** (Kaggle), including variables such as:

- Gender, Age, Academic Pressure, CGPA, Study Satisfaction, Sleep Duration  
- Dietary Habits, Family History of Mental Illness  
- **Target**: Binary classification (Depressed = Yes / No)

Preprocessing steps included cleaning inconsistent values, encoding categories, and normalization.

---

## Quick Start

### Requirements
- Python 3.10+
- pip

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/seu-usuario/DepressionRisk-ML.git
cd DepressionRisk-ML
```

2. **Crie o ambiente virtual**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows PowerShell
# oupython -m venv venv
.\venv\Scripts\activate        # Windows PowerShell
# or
source venv/bin/activate       # Linux/macOS
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Use

### Run the Streamlit App
```bash
streamlit run app.py
```
Acesse em `http://localhost:8501`

**Features**:

- **Individual Prediction**: Enter data and get a probability prediction
- **Screening via CSV**: Send a CSV file for batch predictions

### Treinar Modelos
```bash
python train_models.py
```

## Estrutura do Projeto

```
DepressionRisk-ML/
├── app.py
├── main.py
├── requirements.txt
├── .env
├── .gitignore
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── knn.joblib
│   ├── svm.joblib
│   ├── mlp.joblib
│   └── *_metadata.json
│
├── notebooks/
│   ├── 01-EDA.ipynb
│   ├── 02-preprocess.ipynb
│   ├── 03-model-training-lab.ipynb
│   └── 04-model-comparison.ipynb
│
├── scripts/
│   └── predict.py
│
├── src/
│   ├── config.py
│   ├── models.py
│   ├── preprocess.py
│   ├── train.py
│   └── visual.py
│
└── reports/
    └── figures/

```

## Final Train

| Modelo | Acurácia | Precisão | Recall | F1-Score |
|--------|----------|----------|--------|----------|
| KNN    | 84.12%   | 87.00%   | 86.95% | 86.36%   |
| SVM    | 84.40%   | 82.44%   | 93.23% | 87.51%   |
| MLP    | 85.56%   | 86.68%   | 89.15% | 87.90%   |


### Hyperparameter 

**KNN**
- n_neighbors: 19
- weights: distance
- metric: minkowski

**SVM**
- kernel: poly
- C: 0.001
- probability: True

**MLP**
- hidden_layer_sizes: (128, 64, 32)
- activation: logistic
- solver: adam

## Notebooks

- **01-EDA.ipynb**: Distributions, correlations, initial exploration
- **02-preprocess.ipynb**: Cleaning, encoding, balancing
- **03-model-training-lab.ipynb**: Hyperparameter search and experiments
- **04-model-comparison.ipynb**: Cross-model evaluation and visualizations

## Tecnologias

- **Data Science**: NumPy, Pandas, Scikit-learn
- **Balanceamento**: Imbalanced-learn (SMOTE)
- **Visualização**: Matplotlib, Seaborn
- **Interface**: Streamlit
- **Utilidades**: Python-dotenv, Joblib


## Pipeline de Treinamento

1. Load raw data
2. Clean inconsistent values
3. **Preprocess**: 
   - Encode categories
   - Apply SMOTE
   - Scale features
4. Train models using GridSearchCV and Validate using cross-validation
5. Evaluation Metrics
6. Save models and metadata

## Métricas de Avaliação

- **Accuracy**
- **Precisão** 
- **Recall**
- **F1-Score**
- **Confusion Matrix**
- **ROC-AUC**

## Contributions

Contributions are welcome! Please:

1. Fork the project
2. Create a branch for your feature (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 👥 Contributors

This project also benefited from support during the early prototyping phase. Three collaborators contributed with initial drafts of individual model training scripts and early comparisons between algorithm performances.

- Thaissa Evellin Soares da Silva

- Venícius de Moraes

- José Carlos 

All refinements, restructuring, methodological decisions, and development of the full v1.0 pipeline, notebooks, and documentation were conducted by the main author.

Colaboraitor 

## Licença

Este projeto está licenciado sob a MIT License - veja o arquivo [LICENSE](LICENSE) para detalhes.

## ⚠️ Disclaimer

This model is intended solely for educational and research purposes. **It must not be used for clinical or medical diagnosis.** Always seek a mental health professional for real evaluations.

## 📬 Contact

- **Autor**: [Gilberto Rodrigues da Silva]
- **Email**: [gilberto.rodrigues07@aluno.ifce.edu.br]
- **GitHub**: [GilbertoRSilva](https://github.com/seu-usuario)


