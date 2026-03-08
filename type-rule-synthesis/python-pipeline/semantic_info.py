"""
Semantic anchor extraction using CodeBERT.

This module extracts a variety of expressions from a Java AST, embeds them
using CodeBERT and ranks them by "centrality" (approximated by the
embedding norm).  The highest‑ranked expressions form semantic anchors
that help guide the language model towards relevant constraints and
dependencies.  We also attempt to extract variable names from annotation
arguments so that parameters like ``arg0``, ``n``, ``m`` and ``k`` are
considered for ranking.
"""

from __future__ import annotations

from typing import Dict, List
import re
import torch
from transformers import RobertaTokenizer, RobertaModel

from config import CODEBERT_MODEL_NAME


def extract_semantic_expressions(ast: Dict) -> Dict[str, List[str]]:
    """
    Extract semantically interesting expressions from an AST.

    We collect assignments in their entirety (e.g. ``l = f + 1``), right‑hand
    sides of assignments, binary expressions, and the conditions of
    ``if``/``while`` statements.  Additionally, we parse annotation
    expressions (e.g. ``@EqualTo(other="arg0")``) to extract the
    parameter values (like ``arg0``).  These parameter names are
    included in the ``rhs_expressions`` list so that CodeBERT can
    evaluate their relevance.

    Returns a dictionary of lists keyed by category.
    """
    assignments: List[str] = []
    rhs_expressions: List[str] = []
    binary_ops: List[str] = []
    conditions: List[str] = []
    for node in ast.get("nodes", []):
        t = node.get("type")
        label = node.get("label", "")
        if not isinstance(label, str):
            continue
        # Normalize whitespace
        label = label.replace("\n", " ").strip()
        # Assignment: capture entire assignment and RHS
        if t == "VariableDeclarator" and "=" in label:
            cleaned = " ".join(label.split())
            assignments.append(cleaned)
            rhs = cleaned.split("=", 1)[1].strip()
            rhs_expressions.append(rhs)
        # Binary expressions
        if t == "BinaryExpr":
            cleaned = " ".join(label.split())
            binary_ops.append(cleaned)
        # Conditions in if/while
        if t in ("IfStmt", "WhileStmt") and "(" in label and ")" in label:
            start = label.find("(") + 1
            end = label.rfind(")")
            if 0 < start < end:
                cond = label[start:end].strip()
                if cond:
                    conditions.append(cond)
        # Extract annotation argument values
        if t == "NormalAnnotationExpr":
            # Find quoted strings in annotation parameters.  We match names
            # like arg0, arg1, n, m, k and ignore numeric literals.
            for val in re.findall(r'"([A-Za-z_][A-Za-z0-9_]*)"', label):
                rhs_expressions.append(val)
    return {
        "assignments": assignments,
        "rhs_expressions": rhs_expressions,
        "binary_ops": binary_ops,
        "conditions": conditions,
    }


class CodeBERTSpanRanker:
    """
    Thin wrapper around CodeBERT for ranking small code fragments.

    Expressions are embedded using mean‑pooling and scored by the norm
    of their embeddings.  The highest‑scoring expressions are returned.
    """
    def __init__(self) -> None:
        self.tokenizer = RobertaTokenizer.from_pretrained(CODEBERT_MODEL_NAME)
        self.model = RobertaModel.from_pretrained(CODEBERT_MODEL_NAME)
        self.model.eval()
    def embed_span(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=64,
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        token_embs = outputs.last_hidden_state[0]
        return token_embs.mean(dim=0)
    def rank_expressions(self, expressions: List[str], top_k: int = 5) -> List[str]:
        if not expressions:
            return []
        embs: List[torch.Tensor | None] = []
        for expr in expressions:
            try:
                embs.append(self.embed_span(expr))
            except Exception:
                embs.append(None)
        valid_pairs = [(expr, emb) for expr, emb in zip(expressions, embs) if emb is not None]
        if not valid_pairs:
            return []
        scores = [torch.norm(emb).item() for _, emb in valid_pairs]
        sorted_pairs = sorted(zip(valid_pairs, scores), key=lambda x: -x[1])
        top: List[str] = []
        for (expr, _emb), _score in sorted_pairs[:top_k]:
            top.append(expr)
        return top


def build_semantic_cues(ast: Dict, max_items: int = 5) -> str:
    """
    Construct a textual block of semantic anchors using CodeBERT.

    The function extracts a list of candidate expressions from the AST,
    removes duplicates, ranks them via CodeBERT and returns a summary
    of the top ``max_items`` items.  If no expressions are found, a
    fallback message is returned.
    """
    sx = extract_semantic_expressions(ast)
    expressions: List[str] = (
        sx["rhs_expressions"] + sx["binary_ops"] + sx["conditions"]
    )
    seen: set[str] = set()
    unique_exprs: List[str] = []
    for e in expressions:
        if e and e not in seen:
            seen.add(e)
            unique_exprs.append(e)
    if not unique_exprs:
        return "No semantic anchors extracted from the code."
    ranker = CodeBERTSpanRanker()
    top = ranker.rank_expressions(unique_exprs, top_k=max_items)
    if not top:
        return "No semantic anchors extracted from the code."
    lines = ["CodeBERT-based semantic anchors (most central expressions):"]
    for expr in top:
        lines.append(f"- {expr}")
    return "\n".join(lines)