import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import RobertaModel, RobertaTokenizer

from config import CODEBERT_MODEL_NAME, FEATURES_DIR

_MODEL_CACHE: Tuple[RobertaTokenizer, RobertaModel] | None = None


def load_analysis_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_codebert() -> Tuple[RobertaTokenizer, RobertaModel]:
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        tokenizer = RobertaTokenizer.from_pretrained(CODEBERT_MODEL_NAME)
        model = RobertaModel.from_pretrained(CODEBERT_MODEL_NAME)
        model.eval()
        _MODEL_CACHE = (tokenizer, model)
    return _MODEL_CACHE


def compute_codebert_features(code: str, top_k_tokens: int = 10) -> Dict[str, object]:
    """
    Возвращает:
    - pooled embedding (mean по токенам)
    - top_k токенов с наибольшей нормой эмбеддинга
    """
    tokenizer, model = _load_codebert()

    inputs = tokenizer(
        code,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    token_embs = outputs.last_hidden_state[0]
    pooled = token_embs.mean(dim=0)
    tokens: List[str] = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    scores = torch.norm(token_embs, dim=1)

    top_k = min(top_k_tokens, len(tokens))
    top_indices = torch.topk(scores, k=top_k).indices.tolist()
    top_tokens = [tokens[i] for i in top_indices]

    return {
        "embedding": pooled.tolist(),
        "top_tokens": top_tokens,
    }


def build_feature_package(analysis_json_path: Path) -> Path:
    data = load_analysis_json(analysis_json_path)
    code = data["code"]
    ast = data["ast"]
    cfg = data["cfg"]
    dfg = data["dfg"]

    cb = compute_codebert_features(code)

    package = {
        "file": data["file"],
        "code": code,
        "ast": ast,
        "cfg": cfg,
        "dfg": dfg,
        "codebert_embedding": cb["embedding"],
        "codebert_top_tokens": cb["top_tokens"],
    }

    out_path = FEATURES_DIR / (Path(data["file"]).stem + "_features.json")
    out_path.write_text(json.dumps(package, indent=2), encoding="utf-8")
    return out_path


if __name__ == "__main__":
    analysis_files = list(FEATURES_DIR.glob("*_analysis.json"))
    if not analysis_files:
        raise RuntimeError("Нет файлов *_analysis.json, сначала запусти run_java_analyzer.py")

    for path in analysis_files:
        out = build_feature_package(path)
        print(f"Feature package saved to: {out}")
