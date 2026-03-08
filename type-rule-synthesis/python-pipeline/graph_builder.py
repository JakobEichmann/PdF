import json
from pathlib import Path
from typing import Tuple

import torch
import numpy as np
from torch_geometric.data import Data

from config import FEATURES_DIR


def load_feature_package(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_graph_from_features(features: dict) -> Tuple[Data, torch.Tensor]:
    """
    Строит граф PyG из AST/CFG/DFG и возвращает:
    - Data (x, edge_index)
    - codebert_embedding как torch.Tensor
    """
    ast = features["ast"]
    cfg = features["cfg"]
    dfg = features["dfg"]
    codebert_emb = torch.tensor(features["codebert_embedding"], dtype=torch.float)

    # Количество AST-узлов
    num_nodes = len(ast["nodes"])

    # Простые типовые признаки: тип AST-узла -> индекс
    types = [n["type"] for n in ast["nodes"]]
    unique_types = {t: i for i, t in enumerate(sorted(set(types)))}
    type_indices = [unique_types[t] for t in types]

    # One-hot признаки по типу
    num_types = len(unique_types)
    x = np.zeros((num_nodes, num_types), dtype=np.float32)
    for i, idx in enumerate(type_indices):
        x[i, idx] = 1.0

    # Собираем ребра из AST, CFG и DFG
    edges = []

    # AST edges (ориентированные, добавим в обе стороны)
    for e in ast["edges"]:
        src, dst = e
        if src < num_nodes and dst < num_nodes:
            edges.append((src, dst))
            edges.append((dst, src))

    # CFG nodes: у них свои id, но мы их мапим в AST-диапазон грубо:
    # просто добавим "виртуальные" узлы, если хочешь. Для простоты недели 5
    # можно ограничиться только AST, а CFG/DFG использовать потом.
    # В минимальном прототипе просто игнорируем CFG/DFG в графе GNN.

    # Если хочешь добавить CFG/DFG: здесь можно расширить x и num_nodes.

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    data = Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=edge_index
    )
    return data, codebert_emb


if __name__ == "__main__":
    feature_files = list(FEATURES_DIR.glob("*_features.json"))
    if not feature_files:
        raise RuntimeError("Нет *_features.json, сначала запусти feature_extraction.py")

    for path in feature_files:
        features = load_feature_package(path)
        data, emb = build_graph_from_features(features)
        print(path.name, "→", data, "codebert_emb_dim:", emb.shape)
