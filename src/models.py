# src/models.py
"""
Funções utilitárias para salvar, carregar e 
gerenciar modelos treinados.
"""
import os
import json
from datetime import datetime
from pathlib import Path
import joblib

DEFAULT_MODEL_DIR = Path("models")

def ensure_model_dir(model_dir: Path | str = DEFAULT_MODEL_DIR):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir

def _metadata_path_for(model_path: Path):
    return model_path.with_suffix(model_path.suffix + ".json").with_name(model_path.stem + "_metadata.json")

def save_model(
    model,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    model_name: str = "model.joblib",
    metadata: dict | None = None,
    overwrite: bool = False
) -> str:

    model_dir = ensure_model_dir(model_dir)
    model_path = model_dir / model_name

    if model_path.exists() and not overwrite:
        raise FileExistsError(f"Model already exists: {model_path}. Pass overwrite=True to replace.")
    
    joblib.dump(model, model_path)

    meta = {
        "model_name": model_name,
        "saved_at": datetime.utcnow().isoformat() + "Z",
    }
    if metadata:
        meta.update(metadata)

    meta_path = model_dir / (Path(model_name).stem + "_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)

    return str(model_path)

def load_model(model_dir: str | Path = DEFAULT_MODEL_DIR, model_name: str = "model.joblib"):

    model_dir = Path(model_dir)
    model_path = model_dir / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)

    meta_path = model_dir / (Path(model_name).stem + "_metadata.json")
    metadata = None
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)

    return model, metadata

def list_models(model_dir: str | Path = DEFAULT_MODEL_DIR):
    model_dir = Path(model_dir)
    if not model_dir.exists():
        return []
    return [p.name for p in model_dir.glob("*.joblib")]

def get_metadata(model_dir: str | Path = DEFAULT_MODEL_DIR, model_name: str | None = None):
    model_dir = Path(model_dir)
    if model_name:
        meta_path = model_dir / (Path(model_name).stem + "_metadata.json")
        if not meta_path.exists():
            return None
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    else:
        metas = {}
        for meta_file in model_dir.glob("*_metadata.json"):
            with open(meta_file, encoding="utf-8") as f:
                metas[meta_file.stem] = json.load(f)
        return metas
