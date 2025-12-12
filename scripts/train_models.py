"""
Script de Treinamento de Modelos
Treina todos os modelos (KNN, SVM, MLP) e salva em models/
"""

import sys
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Adicionar diretório pai ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import train as tr
from src.config import MODELS_CONFIG, RANDOM_STATE
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train_models():
    """Treina todos os modelos"""
  
    print("TREINAMENTO DE MODELOS")

    
    # Carregar dados processados
    print("Carregando dados...")
    train_df = pd.read_csv("data/processed/train_dataset.csv")
    test_df = pd.read_csv("data/processed/test_dataset.csv")
    
    X_train = train_df.drop(columns=["Depression"])
    y_train = train_df["Depression"]
    X_test = test_df.drop(columns=["Depression"])
    y_test = test_df["Depression"]
    
    print(f"   Treino: {X_train.shape}")
    print(f"   Teste:  {X_test.shape}\n")
    
    # Criar diretório de modelos
    Path("models").mkdir(exist_ok=True)
    
    results = {}
    
    # Treinar cada modelo
    for model_name, config in MODELS_CONFIG.items():
        print(f"{'─'*60}")
        print(f"Treinando: {config['display_name']}")
        print(f"{'─'*60}")
        
        use_smote = config.get("use_smote", False)
        use_scaler = config.get("use_scaler", True)
        
        smote = SMOTE(random_state=RANDOM_STATE) if use_smote else None
        pipeline = tr.build_pipeline(
            model=config["model"],
            use_scaler=use_scaler,
            use_smote=use_smote,
            smote=smote
        )
        
        # Treinar
        print("Treinando...")
        pipeline.fit(X_train, y_train)
        
        # Avaliar
        proba_test = pipeline.predict_proba(X_test)[:, 1]
        preds_test = (proba_test >= 0.5).astype(int)
        
        test_metrics = {
            "accuracy": accuracy_score(y_test, preds_test),
            "precision": precision_score(y_test, preds_test, zero_division=0),
            "recall": recall_score(y_test, preds_test, zero_division=0),
            "f1": f1_score(y_test, preds_test, zero_division=0)
        }
        
        print(f"   Acurácia:  {test_metrics['accuracy']:.4f}")
        print(f"   Precisão:  {test_metrics['precision']:.4f}")
        print(f"   Recall:    {test_metrics['recall']:.4f}")
        print(f"   F1-Score:  {test_metrics['f1']:.4f}")
        
        # Salvar modelo
        joblib.dump(pipeline, f"models/{model_name}.joblib")
        print(f"Salvo em models/{model_name}.joblib")
        
        # Salvar metadados
        metadata = {
            "model_name": model_name,
            "display_name": config["display_name"],
            "timestamp": datetime.now().isoformat(),
            "test_metrics": {k: float(v) for k, v in test_metrics.items()}
        }
        
        with open(f"models/{model_name}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        results[model_name] = test_metrics
        print()
    
    # Resumo
    print(f"{'─'*60}")
    print("RESUMO DOS MODELOS")
    print(f"{'─'*60}")
    print(f"{'Modelo':<20} {'Acurácia':<15} {'Precisão':<15} {'Recall':<15} {'F1-Score':<15}")
    print("─" * 80)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<20} "
              f"{metrics['accuracy']:.4f}          "
              f"{metrics['precision']:.4f}          "
              f"{metrics['recall']:.4f}          "
              f"{metrics['f1']:.4f}")
    
    best_model = max(results.items(), key=lambda x: x[1]["f1"])
    print(f"\nMelhor Modelo (F1): {best_model[0]} ({best_model[1]['f1']:.4f})\n")


if __name__ == "__main__":
    train_models()
