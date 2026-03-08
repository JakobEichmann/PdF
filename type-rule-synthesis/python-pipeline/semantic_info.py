# semantic_info.py
# Универсальная семантическая подсистема на базе CodeBERT:
# - выделяет выражения из AST
# - ранжирует их по "семантической центральности"
# - возвращает текстовый блок для вставки в LLM-промпт

from typing import Dict, List
import torch
from transformers import RobertaTokenizer, RobertaModel

from config import CODEBERT_MODEL_NAME


def extract_semantic_expressions(ast: Dict) -> Dict[str, List[str]]:
    """
    Извлекает семантически интересные выражения из AST:
    - присваивания целиком: "l = f + 1"
    - правые части присваиваний: "f + 1"
    - бинарные выражения: "f + 1", "a && b", "x - y"
    - условия if/while: содержимое скобок

    Возвращает словарь списков строк.
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

        label = label.replace("\n", " ").strip()

        # Присваивание вида "l = f + 1"
        if t == "VariableDeclarator" and "=" in label:
            cleaned = " ".join(label.split())
            assignments.append(cleaned)
            rhs = cleaned.split("=", 1)[1].strip()
            rhs_expressions.append(rhs)

        # Бинарное выражение, например "f + 1"
        if t == "BinaryExpr":
            cleaned = " ".join(label.split())
            binary_ops.append(cleaned)

        # Условия if/while (грубо вытаскиваем содержимое скобок)
        if t in ("IfStmt", "WhileStmt") and "(" in label and ")" in label:
            start = label.find("(") + 1
            end = label.rfind(")")
            if 0 < start < end:
                cond = label[start:end].strip()
                if cond:
                    conditions.append(cond)

    return {
        "assignments": assignments,
        "rhs_expressions": rhs_expressions,
        "binary_ops": binary_ops,
        "conditions": conditions,
    }


class CodeBERTSpanRanker:
    """
    Обертка над CodeBERT для ранжирования небольших выражений
    по "семантической центральности".
    """

    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained(CODEBERT_MODEL_NAME)
        self.model = RobertaModel.from_pretrained(CODEBERT_MODEL_NAME)
        self.model.eval()

    def embed_span(self, text: str) -> torch.Tensor:
        """
        Возвращает mean-pool embedding для короткого фрагмента кода.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=64,
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        token_embs = outputs.last_hidden_state[0]  # [seq_len, hidden]
        return token_embs.mean(dim=0)              # [hidden]

    def rank_expressions(self, expressions: List[str], top_k: int = 5) -> List[str]:
        """
        Ранжирует выражения по норме embedding (грубая мера "выделенности").
        Возвращает top_k выражений.
        """
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
    Главная функция: собирает семантические "якоря" и
    возвращает текстовый блок для промпта LLM.

    Пример вывода:
    CodeBERT-based semantic anchors (most central expressions):
    - f + 1
    - l = f + 1
    - Interval(min = 2, max = 4)
    """
    sx = extract_semantic_expressions(ast)

    expressions: List[str] = (
        sx["rhs_expressions"]
        + sx["binary_ops"]
        + sx["conditions"]
    )

    # убрать дубликаты, сохраняя порядок
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
