# preprocess.py

import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

RAW_PATH = os.path.join(
    os.path.dirname(__file__), 
    "..", "data", "raw", "student_depression_dataset.csv"
)
RAW_PATH = os.path.abspath(RAW_PATH)

PROCESSED_PATH = "../data/processed/student_depression_dataset_cleaned.csv"


# ============================================================
# 1. Carregar dataset bruto
# ============================================================

def load_raw_dataset(path: str = RAW_PATH) -> pd.DataFrame:
    print("📥 Dataset bruto carregado.")
    return pd.read_csv(path)


# ============================================================
# 2. Remover linhas com valores inconsistentes
# ============================================================

def remove_rows_with_inconsistent_values(df: pd.DataFrame) -> pd.DataFrame:
    inconsistent_values = [
        "Other", "Others", "?", "Unknown",
        "Prefer not to say", " ", "", "NA",
        "N/A", "None"
    ]

    # Remove linhas contendo QUALQUER um desses valores
    mask = ~df.isin(inconsistent_values).any(axis=1)
    df = df[mask]

    print(f"🧹 Linhas inconsistentes removidas. Restaram {len(df)} linhas.")
    return df


# ============================================================
# 3. Remover colunas irrelevantes
# ============================================================

def remove_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    print("🧨 Removendo colunas irrelevantes...")
    cols_to_remove = [
        'id', 'City', 'Profession', 'Degree',
        'Job Satisfaction', 'Work Pressure'
    ]

    df = df.drop(columns=[c for c in cols_to_remove if c in df.columns], errors="ignore")
    return df


# ============================================================
# 4. Mapear valores categóricos
# ============================================================

def treat_values(df: pd.DataFrame) -> pd.DataFrame:
    print("🔧 Tratando valores categóricos...")

    binary_map = {"Yes": 1, "No": 0}
    gender_map = {"Male": 0, "Female": 1}
    diet_map = {"Unhealthy": 0, "Moderate": 1, "Healthy": 2}
    sleep_map = {
        "'Less than 5 hours'": 4,
        "'5-6 hours'": 5.5,
        "'7-8 hours'": 7.5,
        "'More than 8 hours'": 9
    }

    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map(gender_map)

    if "Dietary Habits" in df.columns:
        df["Dietary Habits"] = df["Dietary Habits"].map(diet_map)

    if "Have you ever had suicidal thoughts ?" in df.columns:
        df["Have you ever had suicidal thoughts ?"] = df["Have you ever had suicidal thoughts ?"].map(binary_map)

    if "Family History of Mental Illness" in df.columns:
        df["Family History of Mental Illness"] = df["Family History of Mental Illness"].map(binary_map)

    if "Sleep Duration" in df.columns:
        df["Sleep Duration"] = df["Sleep Duration"].map(sleep_map)

    return df


# ============================================================
# 5. Filtrar idade
# ============================================================

def filter_age(df: pd.DataFrame) -> pd.DataFrame:
    print("🎯 Filtrando idades acima de 45...")
    return df[df["Age"] <= 45]


# ============================================================
# 6. Remover NA
# ============================================================

def drop_missing(df: pd.DataFrame) -> pd.DataFrame:
    print("🧼 Removendo linhas com valores NA...")
    return df.dropna()


# ============================================================
# 7. Split train/test
# ============================================================

def split_data(df):
    from pathlib import Path

    X = df.drop(columns=["Depression"])
    y = df["Depression"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    processed_path = Path("../data/processed")
    processed_path.mkdir(parents=True, exist_ok=True)

    output_train = processed_path / "train_dataset.csv"
    output_test = processed_path / "test_dataset.csv"

    train_df.to_csv(output_train, index=False)
    test_df.to_csv(output_test, index=False)

    print("📁 Arquivos de treino e teste gerados com sucesso!")
    return train_df, test_df


# ============================================================
# 8. Pipeline principal
# ============================================================

def preprocess_pipeline():
    print("\n🚀 Iniciando pré-processamento...\n")

    df = load_raw_dataset()
    df = remove_rows_with_inconsistent_values(df)
    df = remove_irrelevant_columns(df)
    df = treat_values(df)
    df = filter_age(df)
    df = drop_missing(df)

    print(f"💾 Salvando dataset limpo: {PROCESSED_PATH}")
    Path("../data/processed").mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    print("✂️ Realizando Split em treino e teste...")
    split_data(df)

    print("\n✅ Pré-processamento concluído com sucesso!\n")
    return df


if __name__ == "__main__":
    preprocess_pipeline()
