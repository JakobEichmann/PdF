"""Semantic anchor extraction with source-grounded spans."""

from __future__ import annotations

from typing import Dict, List, Tuple
import re

import torch
from transformers import RobertaModel, RobertaTokenizer

from config import CODEBERT_MODEL_NAME

_MODEL_CACHE: Tuple[RobertaTokenizer, RobertaModel] | None = None


def _load_codebert() -> Tuple[RobertaTokenizer, RobertaModel]:
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        tokenizer = RobertaTokenizer.from_pretrained(CODEBERT_MODEL_NAME)
        model = RobertaModel.from_pretrained(CODEBERT_MODEL_NAME)
        model.eval()
        _MODEL_CACHE = (tokenizer, model)
    return _MODEL_CACHE


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


class CodeBERTSpanRanker:
    def __init__(self) -> None:
        self.tokenizer, self.model = _load_codebert()

    def embed_span(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=96,
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        token_embs = outputs.last_hidden_state[0]
        return token_embs.mean(dim=0)

    def rank_expressions(self, expressions: List[Dict[str, object]], top_k: int = 5) -> List[Dict[str, object]]:
        valid: List[Tuple[Dict[str, object], float]] = []
        for item in expressions:
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            try:
                emb = self.embed_span(text)
                score = torch.norm(emb).item()
                valid.append((item, score))
            except Exception:
                continue
        valid.sort(key=lambda x: -x[1])
        return [item for item, _score in valid[:top_k]]


def extract_semantic_expressions(ast: Dict) -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    seen: set[tuple[str, str, int | None]] = set()

    for node in ast.get("nodes", []):
        ntype = str(node.get("type", ""))
        label = _clean(str(node.get("label", "")))
        line = node.get("line")
        if not label:
            continue

        role = None
        text = None
        if ntype == "VariableDeclarator" and "=" in label:
            role = "variable declaration"
            text = label
            rhs = label.split("=", 1)[1].strip()
            key_rhs = ("rhs expression", rhs, line)
            if rhs and key_rhs not in seen:
                seen.add(key_rhs)
                candidates.append({"role": "rhs expression", "text": rhs, "line": line})
        elif ntype == "AssignExpr":
            role = "assignment"
            text = label
        elif ntype == "NormalAnnotationExpr":
            role = "annotation"
            text = label
        elif ntype == "BinaryExpr":
            role = "binary expression"
            text = label
        elif ntype == "Parameter" and "@" in label:
            role = "annotated declaration"
            text = label
        elif ntype == "MethodDeclaration":
            role = "method context"
            text = label

        if role and text:
            key = (role, text, line)
            if key not in seen:
                seen.add(key)
                candidates.append({"role": role, "text": text, "line": line})

    return candidates


def build_semantic_cues(ast: Dict, max_items: int = 5) -> str:
    expressions = extract_semantic_expressions(ast)
    if not expressions:
        return "No semantic anchors extracted from the code."
    ranker = CodeBERTSpanRanker()
    top = ranker.rank_expressions(expressions, top_k=max_items)
    if not top:
        return "No semantic anchors extracted from the code."

    lines = ["CodeBERT-based semantic anchors (most central expressions):"]
    for item in top:
        role = item.get("role", "expression")
        text = _clean(str(item.get("text", "")))
        line = item.get("line")
        if isinstance(line, int):
            lines.append(f"- {role}: {text} (line {line})")
        else:
            lines.append(f"- {role}: {text}")
    return "\n".join(lines)
