
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
        "model": KNeighborsClassifier(
                    n_neighbors=15,
                    weights="distance",
                    metric="minkowski",
                    p=2),
        "use_scaler": True,
        "use_smote": True,
        "param_grid": {
            "model__n_neighbors": [11, 15, 19],        
            "model__weights": ["uniform", "distance"], 
            "model__metric": ["minkowski"],            
            "model__p": [1, 2]                        
        }
    },
    
    "svm": {
        "display_name": "Support Vector Machine",
        "model": SVC(kernel='rbf', probability=True),
        "use_scaler": True,
        "use_smote": True,
        "param_grid": {
            "model__C": [0.001, 0.01],
            "model__kernel": ["rbf", "poly"]
        }
    },

    "mlp": {
        "display_name": "MLP (Neural Net)",
        "model": MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                max_iter=800,
                early_stopping=True,
                n_iter_no_change=20,
                random_state=42
            ),
        "use_scaler": True,
        "use_smote": True,
        "param_grid": {
            "model__hidden_layer_sizes": [(128, 64, 32), (64, 32), (32, 16)], # CORRIGIDO
            "model__activation": ["relu", "logistic", "tanh"],               # CORRIGIDO
            "model__alpha": [0.0001, 0.01],                                  # CORRIGIDO
            "model__learning_rate_init": [0.001, 0.0001],                    # CORRIGIDO
            "model__solver": ["adam"]                                        # CORRIGIDO
        }
    }
}


def get_model_keys():
    return list(MODELS_CONFIG.keys())

def get_display_names():
    return {k: MODELS_CONFIG[k]["display_name"] for k in MODELS_CONFIG}