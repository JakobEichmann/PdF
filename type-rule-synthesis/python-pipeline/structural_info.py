# structural_info.py

from typing import Dict
import torch


def summarize_gnn_structural_cues_compact(
    ast: Dict,
    gnn_node_embs: torch.Tensor,
    code: str,
    top_k: int = 8,
) -> str:
    """
    Компактное, человекочитаемое резюме от GNN.
    Отфильтровывает шумные типы и оставляет только аннотации/присваивания/выражения.
    """
    nodes = ast["nodes"]
    importance = torch.norm(gnn_node_embs, dim=1)
    idxs = torch.topk(importance, k=min(top_k, len(nodes))).indices.tolist()

    whitelist = {
        "NormalAnnotationExpr",
        "AssignExpr",
        "VariableDeclarator",
        "BinaryExpr",
        "MemberValuePair",
    }

    results: list[str] = []

    for i in idxs:
        n = nodes[i]
        ntype = n["type"]
        label = n.get("label", "").replace("\n", " ").strip()

        if ntype not in whitelist:
            continue

        # Небольшие хелперы для красивого текста
        if ntype == "NormalAnnotationExpr" and "Interval" in label:
            results.append(f"Interval annotation: {label}")
        elif ntype == "AssignExpr":
            results.append(f"Assignment: {label}")
        elif ntype == "VariableDeclarator" and "=" in label:
            results.append(f"Variable declaration: {label}")
        elif ntype == "BinaryExpr":
            results.append(f"Expression: {label}")
        elif ntype == "MemberValuePair":
            results.append(f"Interval bound: {label}")
        else:
            results.append(f"{ntype}: {label}")

    if not results:
        return "No high-salience structural signals from GNN."

    return "\n".join(f"- {r}" for r in results)
