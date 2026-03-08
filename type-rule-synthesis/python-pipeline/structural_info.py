"""Readable structural cues from GNN node embeddings."""

from __future__ import annotations

from typing import Dict, List
import re

import torch


WHITELIST = {
    "VariableDeclarator": "Variable declaration",
    "AssignExpr": "Assignment",
    "NormalAnnotationExpr": "Annotation",
    "Parameter": "Annotated declaration",
    "BinaryExpr": "Expression",
}


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def summarize_gnn_structural_cues_compact(
    ast: Dict,
    gnn_node_embs: torch.Tensor,
    code: str,
    top_k: int = 8,
) -> str:
    nodes = ast.get("nodes", [])
    if not nodes or gnn_node_embs is None or gnn_node_embs.numel() == 0:
        return "No high-salience structural signals from GNN."

    limit = min(len(nodes), gnn_node_embs.shape[0])
    importance = torch.norm(gnn_node_embs[:limit], dim=1)
    k = min(top_k, limit)
    idxs = torch.topk(importance, k=k).indices.tolist()

    results: List[str] = []
    seen: set[str] = set()
    for idx in idxs:
        node = nodes[idx]
        ntype = str(node.get("type", ""))
        if ntype not in WHITELIST:
            continue
        label = _clean(node.get("label", ""))
        line = node.get("line")
        prefix = WHITELIST[ntype]
        line_part = f" at line {line}" if isinstance(line, int) else ""
        item = f"{prefix}{line_part}: {label}"
        if item not in seen:
            seen.add(item)
            results.append(item)

    for node in nodes:
        ntype = str(node.get("type", ""))
        if ntype not in {"VariableDeclarator", "NormalAnnotationExpr"}:
            continue
        label = _clean(node.get("label", ""))
        line = node.get("line")
        prefix = WHITELIST.get(ntype, ntype)
        line_part = f" at line {line}" if isinstance(line, int) else ""
        item = f"{prefix}{line_part}: {label}"
        if item not in seen:
            seen.add(item)
            results.append(item)

    if not results:
        return "No high-salience structural signals from GNN."
    return "\n".join(f"- {item}" for item in results)
