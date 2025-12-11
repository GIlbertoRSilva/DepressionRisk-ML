import joblib
import os
import numpy as np
import pandas as pd


MAPPINGS = {
    "Gender": {"Male": 0, "Female": 1},
    "Dietary Habits": {"Unhealthy": 0, "Average": 1, "Healthy": 2},
    "Family History of Mental Illness": {"No": 0, "Yes": 1},
    "Have you ever had suicidal thoughts ?": {"No": 0, "Yes": 1}
}

MODELS_INFO = {
    "SVM": {"path": "models/svm.joblib", "description": "Bom para datasets pequenos e margens claras."},
    "KNN": {"path": "models/knn.joblib", "description": "Simples e Consistente para padrões próximos."},
    "MLP": {"path": "models/mlp.joblib", "description": "Capta relações não-lineares complexas, requer mais dados."}
}

def load_model(model_name):
    if model_name not in MODELS_INFO:
        raise ValueError(f"Modelo desconhecido: {model_name}")
    info = MODELS_INFO[model_name]
    model_path = info["path"]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Arquivo de modelo não encontrado: {model_path}")
    model = joblib.load(model_path)
    return model, info["description"]


def encode_input(user_input: dict):
    encoded = {}
    for feature, value in user_input.items():
        if feature in MAPPINGS:
            if value not in MAPPINGS[feature]:
                raise ValueError(f"Valor inválido para {feature}: {value}. Opções: {list(MAPPINGS[feature].keys())}")
            encoded[feature] = MAPPINGS[feature][value]
        else:
            encoded[feature] = value  
    return encoded

def predict(user_input: dict, model_name: str, threshold=0.5):
    model, description = load_model(model_name)
    encoded_input = encode_input(user_input)

    # Garantir que é um dataframe com 1 linha
    df = pd.DataFrame([encoded_input])

    # Probabilidade de classe positiva
    proba = model.predict_proba(df)[:, 1][0]

    # Decisão binária
    pred = int(proba >= threshold)

    # Interpretar para usuário
    if pred == 1:
        if proba > 0.75:
            message = f"Risco de depressão: ALTO ({proba*100:.1f}%)"
        else:
            message = f"Risco de depressão: MODERADO ({proba*100:.1f}%)"
    else:
        message = f"Risco de depressão: BAIXO ({proba*100:.1f}%)"

    return {
        "prediction": pred,
        "probability": proba,
        "message": message,
        "model_description": description
    }


if __name__ == "__main__":
    user_input_example = {
        "Gender": "Female",
        "Age": 18,
        "Academic Pressure": 5,
        "CGPA": 5.1,
        "Study Satisfaction": 2,
        "Sleep Duration": 4,
        "Dietary Habits": "Average",
        "Have you ever had suicidal thoughts ?": "Yes",
        "Work/Study Hours": 3,
        "Financial Stress": 4,
        "Family History of Mental Illness": "Yes"
    }

    result = predict(user_input_example, "MLP", threshold=0.5)
    print(result)
