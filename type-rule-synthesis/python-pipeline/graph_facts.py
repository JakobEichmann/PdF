"""
Utilities for summarising facts derived from abstract syntax trees (AST),
control‑flow graphs (CFG) and data‑flow graphs (DFG).

The functions in this module generate concise, human‑readable descriptions
of program structure and def‑use information for inclusion in language
model prompts.  They attempt to filter out noise such as Java keywords
and comment artefacts so that the resulting summaries focus on real
variables and assignments.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List
import re


def _clean(label: str, max_len: int = 80) -> str:
    """Normalise a label by collapsing whitespace and truncating."""
    s = label.replace("\n", " ").strip()
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s


def derive_dfg_facts(dfg: Dict) -> List[str]:
    """
    Extract simple facts from a data‑flow graph.

    For each variable that appears on more than one line, we record
    the set of lines on which it occurs.  We also derive basic
    def‑use chains by following edges in the DFG.  Variables whose
    names look like Java keywords or other noise are filtered out.
    """
    nodes = dfg.get("nodes", [])
    edges = dfg.get("edges", [])

    facts: List[str] = []

    # Build mapping var -> lines
    occ: Dict[str, set] = {}
    # Simple keyword/reserved set to exclude non‑variable names
    reserved: set[str] = {
        "public", "private", "protected", "void", "int", "float", "double", "char",
        "boolean", "class", "static", "final", "return", "error", "assignment",
        "type", "incompatible", "other",
    }
    for n in nodes:
        v = n.get("var", "?")
        line = n.get("line", -1)
        if not v or not isinstance(v, str):
            continue
        # Skip obvious keywords and malformed identifiers
        if v in reserved:
            continue
        if not re.match(r"^[a-zA-Z_]\w*$", v):
            continue
        occ.setdefault(v, set()).add(line)

    for v, lines in occ.items():
        # Include only variables that occur on multiple distinct lines
        if v.strip() and len(lines) > 1:
            ls = ", ".join(str(l) for l in sorted(lines))
            facts.append(f"Variable {v} occurs on lines {ls}.")

    # Extract data‑flow edges for variables, skipping mismatched vars
    for src, dst in edges:
        if src < 0 or src >= len(nodes) or dst < 0 or dst >= len(nodes):
            continue
        n1 = nodes[src]
        n2 = nodes[dst]
        v1 = n1.get("var")
        v2 = n2.get("var")
        if not v1 or not v2 or v1 != v2:
            continue
        if v1 in reserved or not re.match(r"^[a-zA-Z_]\w*$", v1):
            continue
        l1 = n1.get("line", -1)
        l2 = n2.get("line", -1)
        if isinstance(l1, int) and isinstance(l2, int) and l1 != -1 and l2 != -1:
            facts.append(f"Data-flow: value of {v1} flows from line {l1} to line {l2}.")

    return facts


def derive_cfg_facts(cfg: Dict) -> List[str]:
    """
    Extract simple facts from a control‑flow graph.  For each edge we
    produce a statement of the form:

        Control-flow: statement '<src>' can be followed by '<dst>'.

    Labels are cleaned and truncated to avoid verbose outputs.
    """
    nodes = cfg.get("nodes", [])
    edges = cfg.get("edges", [])
    facts: List[str] = []
    for src, dst in edges:
        if src < 0 or src >= len(nodes) or dst < 0 or dst >= len(nodes):
            continue
        s1 = _clean(nodes[src].get("label", ""))
        s2 = _clean(nodes[dst].get("label", ""))
        if s1 and s2:
            facts.append(f"Control-flow: statement '{s1}' can be followed by '{s2}'.")
    return facts


def derive_assignment_facts_from_ast(ast: Dict) -> List[str]:
    """
    Extract assignment and annotation facts from an AST.  We look for
    variable declarators with an initializer as well as Interval
    annotations on fields.  The resulting descriptions are intended
    for inclusion in prompts and are kept short.
    """
    nodes = ast.get("nodes", [])
    facts: List[str] = []
    for n in nodes:
        t = n.get("type")
        label = n.get("label", "")
        if t == "VariableDeclarator" and "=" in str(label):
            parts = str(label).split("=")
            lhs = parts[0].replace("int", "").strip()
            rhs = "=".join(parts[1:]).strip().rstrip(";")
            facts.append(f"Variable {lhs} is defined as '{_clean(rhs)}'.")
        if t == "FieldDeclaration" and "@Interval" in str(label):
            facts.append(f"Field with @Interval annotation: '{_clean(str(label))}'.")
    return facts


def summarize_graph_facts(ast: Dict, cfg: Dict, dfg: Dict) -> str:
    """
    Produce a detailed multi‑section summary of assignment/annotation facts,
    data‑flow facts and control‑flow facts.  The output is organised
    with headers and bullet points and is suitable for LLM consumption.
    """
    dfg_facts = derive_dfg_facts(dfg)
    cfg_facts = derive_cfg_facts(cfg)
    assign_facts = derive_assignment_facts_from_ast(ast)
    lines: List[str] = []
    if assign_facts:
        lines.append("Assignment and annotation facts:")
        lines.extend(f"- {f}" for f in assign_facts)
    if dfg_facts:
        lines.append("Data-flow facts (from DFG):")
        lines.extend(f"- {f}" for f in dfg_facts)
    if cfg_facts:
        lines.append("Control-flow facts (from CFG):")
        lines.extend(f"- {f}" for f in cfg_facts)
    if not lines:
        return "No explicit graph-derived facts could be extracted."
    return "\n".join(lines)


def summarize_graph_facts_compact(ast: Dict, cfg: Dict, dfg: Dict) -> str:
    """
    Produce a compact, high‑level summary of facts for inclusion in prompts.
    The summary includes:

    - Assignments (only the minimal declaration text)
    - Interval annotations
    - Def‑use chains over variables, filtered to exclude Java keywords
      and other non‑variable identifiers
    """
    # Collect data-flow assignments and interval annotations separately
    assignments: List[str] = []
    annotations: List[str] = []
    for node in ast.get("nodes", []):
        t = node.get("type")
        raw_label = node.get("label", "")
        label = str(raw_label).replace("\r", "\n")
        # Capture assignments either from VariableDeclarator or AssignExpr
        if t == "VariableDeclarator" and "=" in label:
            last_line = label.splitlines()[-1].strip()
            assignments.append(last_line)
        elif t == "AssignExpr":
            cleaned = label.replace("\n", " ").strip()
            assignments.append(cleaned)
        # Capture Interval annotations
        if t == "NormalAnnotationExpr" and "Interval" in label:
            cleaned_ann = label.replace("\n", " ").strip()
            annotations.append(cleaned_ann)
    parts: List[str] = []
    # Rename the assignments section to "Data-flow summary" to make clear
    # that these are source-level definitions, not arbitrary code lines.
    if assignments:
        parts.append("Data-flow summary:")
        for a in assignments:
            parts.append(f"- {a}")
    if annotations:
        parts.append("Interval annotations:")
        for ann in annotations:
            parts.append(f"- {ann}")
    # Build def-use chains, filtering out noise
    var_lines: dict[str, set[int]] = defaultdict(set)
    # Precompile reserved set and pattern
    reserved: set[str] = {
        "public", "private", "protected", "void", "int", "float", "double", "char",
        "boolean", "class", "static", "final", "return", "error", "assignment",
        "type", "incompatible", "other",
    }
    ident_re = re.compile(r"^[a-zA-Z_]\w*$")
    for n in dfg.get("nodes", []):
        v = n.get("var")
        line = n.get("line")
        if not v or not isinstance(v, str):
            continue
        if v in reserved or not ident_re.match(v):
            continue
        if isinstance(line, int):
            var_lines[v].add(line)
    chains: List[str] = []
    for v, lines in var_lines.items():
        if len(lines) > 1:
            seq = " -> ".join(str(x) for x in sorted(lines))
            chains.append(f"{v}: lines {seq}")
    if chains:
        parts.append("Def-use chains (from DFG):")
        for c in chains:
            parts.append(f"- {c}")
    if not parts:
        return "No high-level data-flow facts extracted."
    return "\n".join(parts)