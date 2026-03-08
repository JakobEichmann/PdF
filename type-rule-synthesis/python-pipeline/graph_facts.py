"""Readable source-grounded summaries for assignments, DFG edges and annotation references."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple
import re


IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _line_text(line: int | None) -> str:
    return f"line {line}" if isinstance(line, int) else "unknown line"


def _scope(node: Dict) -> str:
    method = node.get("method")
    if isinstance(method, str) and method and method != "<global>":
        return method
    return "global"


def _scoped_var(scope: str, var: str) -> str:
    return f"{scope}:{var}"


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _parse_assignment(label: str) -> Tuple[str, str] | None:
    cleaned = _clean(label).rstrip(";")
    if "=" not in cleaned:
        return None
    lhs, rhs = cleaned.split("=", 1)
    lhs = lhs.strip()
    rhs = rhs.strip()
    if not lhs or not rhs:
        return None
    lhs = lhs.split()[-1]
    if not IDENT_RE.match(lhs):
        return None
    return lhs, rhs


def _iter_edge_dicts(edges: Iterable):
    for e in edges:
        if isinstance(e, dict):
            src = e.get("src")
            dst = e.get("dst")
            if isinstance(src, int) and isinstance(dst, int):
                yield e
        elif isinstance(e, list) and len(e) == 2:
            src, dst = e
            if isinstance(src, int) and isinstance(dst, int):
                yield {"src": src, "dst": dst, "type": None}


def _node_lookup(nodes: List[Dict]) -> Dict[int, Dict]:
    lookup: Dict[int, Dict] = {}
    for n in nodes:
        node_id = n.get("id")
        if isinstance(node_id, int):
            lookup[node_id] = n
    return lookup


def derive_assignment_facts_from_ast(ast: Dict) -> List[str]:
    results: List[str] = []
    for node in ast.get("nodes", []):
        ntype = node.get("type")
        if ntype not in {"VariableDeclarator", "AssignExpr"}:
            continue
        parsed = _parse_assignment(str(node.get("label", "")))
        if not parsed:
            continue
        lhs, rhs = parsed
        scope = _scope(node)
        results.append(f"{_scoped_var(scope, lhs)} <- {rhs} ({_line_text(node.get('line'))})")
    return results


def derive_def_use_facts(dfg: Dict) -> List[str]:
    nodes = dfg.get("nodes", [])
    lookup = _node_lookup(nodes)
    facts: List[str] = []
    seen: set[str] = set()
    for e in _iter_edge_dicts(dfg.get("edges", [])):
        if e.get("type") not in {"data_flow", "receiver_use"}:
            continue
        src = lookup.get(e["src"])
        dst = lookup.get(e["dst"])
        if not src or not dst:
            continue
        var = str(dst.get("var", src.get("var", "")))
        if not IDENT_RE.match(var):
            continue
        scope = _scope(dst)
        role = str(dst.get("role", "use"))
        src_line = src.get("line")
        dst_line = dst.get("line")
        entry = (
            f"{_scoped_var(scope, var)} defined at {_line_text(src_line)}, "
            f"used at {_line_text(dst_line)} as {role}"
        )
        if entry not in seen:
            seen.add(entry)
            facts.append(entry)
    return facts


def derive_annotation_dependency_facts(dfg: Dict) -> List[str]:
    nodes = dfg.get("nodes", [])
    lookup = _node_lookup(nodes)
    facts: List[str] = []
    seen: set[str] = set()
    for e in _iter_edge_dicts(dfg.get("edges", [])):
        if e.get("type") != "annotation_ref":
            continue
        src = lookup.get(e["src"])
        dst = lookup.get(e["dst"])
        if not src or not dst:
            continue
        var = str(dst.get("var", src.get("var", "")))
        if not IDENT_RE.match(var):
            continue
        scope = _scope(dst)
        ann = _clean(str(dst.get("annotation", dst.get("label", "annotation"))))
        owner_var = str(dst.get("owner_var", "")).strip()
        owner_suffix = f" on {_scoped_var(scope, owner_var)}" if owner_var else ""
        entry = (
            f"{_scoped_var(scope, var)} referenced by {ann}{owner_suffix} "
            f"({_line_text(dst.get('line'))})"
        )
        if entry not in seen:
            seen.add(entry)
            facts.append(entry)
    return facts


def summarize_graph_facts(ast: Dict, cfg: Dict, dfg: Dict) -> str:
    return summarize_graph_facts_compact(ast, cfg, dfg)


def summarize_graph_facts_compact(ast: Dict, cfg: Dict, dfg: Dict) -> str:
    assignments = derive_assignment_facts_from_ast(ast)
    def_uses = derive_def_use_facts(dfg)
    ann_deps = derive_annotation_dependency_facts(dfg)

    parts: List[str] = []
    if assignments:
        parts.append("Assignments:")
        parts.extend(f"- {item}" for item in assignments)
    if def_uses:
        parts.append("Def-use chains (from DFG):")
        parts.extend(f"- {item}" for item in def_uses)
    if ann_deps:
        parts.append("Annotation dependencies:")
        parts.extend(f"- {item}" for item in ann_deps)
    if not parts:
        return "No source-grounded data-flow facts extracted."
    return "\n".join(parts)
