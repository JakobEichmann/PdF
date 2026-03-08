from typing import Dict

import torch


def summarize_gnn_structural_cues(ast: Dict, node_embeddings: torch.Tensor, top_k: int = 3) -> str:
    """
    Делает GNN реально полезным для LLM:
    - берёт embeddings узлов AST
    - считает "важность" узла как L2-норму embedding
    - выбирает top_k узлов
    - возвращает короткие текстовые описания этих узлов
    """
    nodes = ast["nodes"]
    num_nodes = len(nodes)
    if num_nodes == 0 or node_embeddings is None:
        return "GNN structural cues are not available."

    # защита: если embeddings меньше, чем узлов (не должно быть, но на всякий случай)
    num_emb_nodes = node_embeddings.shape[0]
    limit = min(num_nodes, num_emb_nodes)

    # важность = норма embedding
    importance = torch.norm(node_embeddings[:limit], dim=1)
    topk = min(top_k, limit)
    top_indices = torch.topk(importance, k=topk).indices.tolist()

    snippets = []
    for idx in top_indices:
        n = nodes[idx]
        n_type = n.get("type", "Unknown")
        label = n.get("label", "").replace("\n", " ").strip()
        if len(label) > 80:
            label = label[:77] + "..."
        snippets.append(f"- [{n_type}] {label}")

    # плюс немного агрегированной статистики по типам
    type_counts = {}
    for n in nodes:
        t = n.get("type", "Unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    sorted_types = sorted(type_counts.items(), key=lambda x: -x[1])[:5]
    type_summary = ", ".join(f"{t}×{c}" for t, c in sorted_types)

    text = (
        "GNN identified the following structurally important AST nodes:\n"
        + "\n".join(snippets)
        + f"\nDominant AST node types: {type_summary}."
    )
    return text
