import os
import numpy as np

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

def load_data(train_path, target_col):
    train = pd.read_csv(train_path)
    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]

    return train, X_train, y_train

def build_pipeline(model, use_scaler=True, use_smote=False, smote=None):
    steps = []

    if use_smote:
        if smote is None:
            raise ValueError("SMOTE ativado, mas nenhum objeto foi fornecido.")
        steps.append(("smote", smote))

    if use_scaler:
        steps.append(("scaler", StandardScaler()))

    steps.append(("model", model))
    return Pipeline(steps)


def cross_validate(pipeline, X, y, threshold=0.5, n_splits=10,
                 random_state=42, verbose=True):
    
    kf = StratifiedKFold(n_splits=n_splits,shuffle=True, 
                         random_state=random_state)

    metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": []
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), start=1):
        if verbose:
            print(f"\nFOLD {fold}")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipeline.fit(X_train, y_train)

        proba = pipeline.predict_proba(X_val)[:, 1]
        preds = (proba >= threshold).astype(int)

        metrics["accuracy"].append(accuracy_score(y_val, preds))
        metrics["precision"].append(precision_score(y_val, preds, zero_division=0))
        metrics["recall"].append(recall_score(y_val, preds, zero_division=0))
        metrics["f1"].append(f1_score(y_val, preds, zero_division=0))

        if verbose:
            print(f"Acurácia: {metrics['accuracy'][-1]:.4f}")
            print(f"Precisão: {metrics['precision'][-1]:.4f}")
            print(f"Recall:   {metrics['recall'][-1]:.4f}")
            print(f"F1-score: {metrics['f1'][-1]:.4f}")

    return metrics

def summarize_cv_results(metrics):
    print("\nMÉDIAS E DESVIOS-PADRÃO")
    for metric, values in metrics.items():
        print(
            f"{metric.capitalize():<10}: "
            f"{np.mean(values):.4f} | DP: {np.std(values):.4f}"
        )
def evaluate_on_test(pipeline,X_test,y_test,threshold=0.5):

    proba = pipeline.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    print("\nAVALIAÇÃO FINAL NO TESTE")
    print(f"Acurácia:  {accuracy_score(y_test, preds):.4f}")
    print(f"Precisão:  {precision_score(y_test, preds, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, preds, zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(y_test, preds, zero_division=0):.4f}")

def train_final_model(model, X, y, use_scaler=True, use_smote=False, smote=None):
    pipeline = build_pipeline(model, use_scaler, use_smote, smote)
    pipeline.fit(X, y)
    return pipeline

def run_gridsearch(
    pipeline,
    param_grid,
    X,
    y,
    scoring="f1",
    n_splits=5,
    n_jobs=-1,
    verbose=1,
    refit=True,
    random_state=42
):
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=refit,
        return_train_score=True
    )

    grid.fit(X, y)

    results_df = pd.DataFrame(grid.cv_results_)

    return {
        "best_estimator": grid.best_estimator_,
        "best_params": grid.best_params_,
        "best_score": grid.best_score_,
        "cv_results": results_df,
        "grid_object": grid
    }



def find_best_threshold(
    model,
    X_val,
    y_val,
    thresholds=np.arange(0.05, 0.95, 0.01),
    metric=f1_score
):
    scores = []

    proba = model.predict_proba(X_val)[:, 1]

    for t in thresholds:
        preds = (proba >= t).astype(int)
        scores.append(metric(y_val, preds))

    best_idx = np.argmax(scores)

    return thresholds[best_idx], scores[best_idx]
