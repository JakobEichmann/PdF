"""
Utilities for summarising salient structural cues from GNN embeddings.

The GNN produces a node embedding for each AST node.  We compute a
simple importance score based on the norm of each embedding, select the
top‑k nodes and convert them into a compact, human‑readable summary.
Additionally, to avoid omitting important variable declarations,
we ensure that all declarations with initialisers are included even
if they fall outside the top‑k.
"""

from __future__ import annotations

from typing import Dict
import torch


def summarize_gnn_structural_cues_compact(
    ast: Dict,
    gnn_node_embs: torch.Tensor,
    code: str,
    top_k: int = 8,
) -> str:
    """
    Summarise structural cues from a GNN.

    Parameters
    ----------
    ast : Dict
        The AST dictionary with ``nodes``.
    gnn_node_embs : torch.Tensor
        Node embeddings from the GNN, one per AST node.
    code : str
        The original source code.  Unused here but kept for API
        compatibility.
    top_k : int, optional
        Number of highest‑salience nodes to include.
    """
    nodes = ast["nodes"]
    if gnn_node_embs is None or gnn_node_embs.numel() == 0:
        return "No high-salience structural signals from GNN."
    importance = torch.norm(gnn_node_embs, dim=1)
    # Take the indices of the top‑k nodes
    k = min(top_k, len(nodes))
    idxs = torch.topk(importance, k=k).indices.tolist()
    # Whitelist of node types we care about
    whitelist = {
        "NormalAnnotationExpr",
        "AssignExpr",
        "VariableDeclarator",
        "BinaryExpr",
        "MemberValuePair",
    }
    results: list[str] = []
    # Build summaries for the top‑k nodes
    for i in idxs:
        n = nodes[i]
        ntype = n.get("type")
        label = str(n.get("label", "")).replace("\n", " ").strip()
        if ntype not in whitelist:
            continue
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
    # Supplement with all variable declarations to ensure completeness
    decls: list[str] = []
    for n in nodes:
        if n.get("type") == "VariableDeclarator":
            lbl = str(n.get("label", "")).replace("\n", " ").strip()
            if "=" in lbl:
                decls.append(lbl)
    for d in decls:
        formatted = f"Variable declaration: {d}"
        if formatted not in results:
            results.append(formatted)
    if not results:
        return "No high-salience structural signals from GNN."
    return "\n".join(f"- {r}" for r in results)