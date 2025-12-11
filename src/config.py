
"""
Configuração central de modelos, hiperparâmetros e flags de pré-processamento.
definições dos modelos que serão testados. Apenas arquivo descritivo, não para teste.

"""

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

RANDOM_STATE = 42

# Cada chave é o identificador do experimento/modelo
# Novos modelos podem ser adicionados na configuração

MODELS_CONFIG = {
    "knn": {
        "display_name": "K-Nearest Neighbors",
        "model": KNeighborsClassifier(),
        "use_scaler": True,
        "use_smote": True,
        "param_grid": {
            "model__n_neighbors": [ 7, 11,15, 19],
            "model__weights": ["uniform", "distance"]
        }
    },
    
    "svm": {
        "display_name": "Support Vector Machine",
        "model": SVC(probability=True, random_state=RANDOM_STATE),
        "use_scaler": True,
        "use_smote": True,
        "param_grid": {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["linear", "rbf"]
        }
    },


    "mlp": {
        "display_name": "MLP (Neural Net)",
        "model": MLPClassifier(max_iter=500, early_stopping=True, random_state=RANDOM_STATE),
        "use_scaler": True,
        "use_smote": True,
        "param_grid": {
            "model__hidden_layer_sizes": [(32, 16), (64,32), (128,64,32)],
            "model__alpha": [0.0001, 0.001]
        }
    }
}

def get_model_keys():
    return list(MODELS_CONFIG.keys())

def get_display_names():
    return {k: MODELS_CONFIG[k]["display_name"] for k in MODELS_CONFIG}