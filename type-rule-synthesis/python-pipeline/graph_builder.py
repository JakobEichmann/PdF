import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from config import FEATURES_DIR


EDGE_KIND_AST = 0
EDGE_KIND_CFG = 1
EDGE_KIND_DFG = 2
EDGE_KIND_ANN = 3


def load_feature_package(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_edge_pairs(edge_items: Iterable):
    for e in edge_items:
        if isinstance(e, list) and len(e) == 2:
            yield int(e[0]), int(e[1]), None
        elif isinstance(e, dict):
            src = e.get("src")
            dst = e.get("dst")
            if isinstance(src, int) and isinstance(dst, int):
                yield src, dst, e.get("type")


def _build_cfg_node_lookup(cfg: Dict) -> Dict[int, int]:
    lookup: Dict[int, int] = {}
    for node in cfg.get("nodes", []):
        node_id = node.get("id")
        ast_id = node.get("ast_id")
        if isinstance(node_id, int) and isinstance(ast_id, int):
            lookup[node_id] = ast_id
    return lookup


def _build_dfg_node_lookup(dfg: Dict) -> Dict[int, int]:
    lookup: Dict[int, int] = {}
    for node in dfg.get("nodes", []):
        node_id = node.get("id")
        ast_id = node.get("ast_id")
        if isinstance(node_id, int) and isinstance(ast_id, int):
            lookup[node_id] = ast_id
    return lookup


def build_graph_from_features(features: dict) -> Tuple[Data, torch.Tensor]:
    ast = features["ast"]
    cfg = features["cfg"]
    dfg = features["dfg"]
    codebert_emb = torch.tensor(features["codebert_embedding"], dtype=torch.float)

    ast_nodes = ast.get("nodes", [])
    num_nodes = len(ast_nodes)
    if num_nodes == 0:
        return Data(x=torch.zeros((0, 1), dtype=torch.float), edge_index=torch.zeros((2, 0), dtype=torch.long)), codebert_emb

    types = [str(n.get("type", "Unknown")) for n in ast_nodes]
    unique_types = {t: i for i, t in enumerate(sorted(set(types)))}
    type_indices = [unique_types[t] for t in types]

    num_types = len(unique_types)
    x = np.zeros((num_nodes, num_types + 3), dtype=np.float32)
    for i, idx in enumerate(type_indices):
        x[i, idx] = 1.0
        n = ast_nodes[i]
        if n.get("line") is not None:
            x[i, num_types] = 1.0
        if str(n.get("type")) in {"VariableDeclarator", "AssignExpr", "NormalAnnotationExpr", "Parameter"}:
            x[i, num_types + 1] = 1.0
        if str(n.get("type")) in {"NameExpr", "MethodCallExpr", "BinaryExpr"}:
            x[i, num_types + 2] = 1.0

    edges: list[tuple[int, int]] = []

    for src, dst, _etype in _iter_edge_pairs(ast.get("edges", [])):
        if 0 <= src < num_nodes and 0 <= dst < num_nodes:
            edges.append((src, dst))
            edges.append((dst, src))

    cfg_lookup = _build_cfg_node_lookup(cfg)
    for src_cfg, dst_cfg, _etype in _iter_edge_pairs(cfg.get("edges", [])):
        src_ast = cfg_lookup.get(src_cfg)
        dst_ast = cfg_lookup.get(dst_cfg)
        if isinstance(src_ast, int) and isinstance(dst_ast, int):
            edges.append((src_ast, dst_ast))
            edges.append((dst_ast, src_ast))

    dfg_lookup = _build_dfg_node_lookup(dfg)
    for src_dfg, dst_dfg, etype in _iter_edge_pairs(dfg.get("edges", [])):
        src_ast = dfg_lookup.get(src_dfg)
        dst_ast = dfg_lookup.get(dst_dfg)
        if isinstance(src_ast, int) and isinstance(dst_ast, int):
            edges.append((src_ast, dst_ast))
            edges.append((dst_ast, src_ast))

    if not edges:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    data = Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=edge_index,
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
