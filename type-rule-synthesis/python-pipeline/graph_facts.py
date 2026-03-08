from typing import Dict, List
from collections import defaultdict

def _clean(label: str, max_len: int = 80) -> str:
    s = label.replace("\n", " ").strip()
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s


def derive_dfg_facts(dfg: Dict) -> List[str]:
    """
    Простые факты из DFG:
    - где встречаются переменные
    - деф-юз цепочки
    """
    nodes = dfg.get("nodes", [])
    edges = dfg.get("edges", [])

    facts: List[str] = []

    # переменная -> множество строк
    occ: Dict[str, set] = {}
    for n in nodes:
        v = n.get("var", "?")
        line = n.get("line", -1)
        occ.setdefault(v, set()).add(line)

    for v, lines in occ.items():
        if v.strip() and len(lines) > 1:
            ls = ", ".join(str(l) for l in sorted(lines))
            facts.append(f"Variable {v} occurs on lines {ls}.")

    # деф-юз по рёбрам
    for src, dst in edges:
        if src < 0 or src >= len(nodes) or dst < 0 or dst >= len(nodes):
            continue
        n1 = nodes[src]
        n2 = nodes[dst]
        if n1.get("var") != n2.get("var"):
            continue
        v = n1.get("var", "?")
        l1 = n1.get("line", -1)
        l2 = n2.get("line", -1)
        if l1 != -1 and l2 != -1:
            facts.append(f"Data-flow: value of {v} flows from line {l1} to line {l2}.")

    return facts


def derive_cfg_facts(cfg: Dict) -> List[str]:
    """
    Факты из CFG:
    - какая инструкция следует за какой
    (для нашего простого примера – линейная последовательность).
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
    Facts like:
    - variable l is defined as f + 1
    - field f has annotation @Interval(min = 1, max = 3)
    """
    nodes = ast.get("nodes", [])
    facts: List[str] = []

    for n in nodes:
        t = n.get("type")
        label = n.get("label", "")

        if t == "VariableDeclarator" and "=" in label:
            # crudely parse "l = f + 1"
            parts = label.split("=")
            lhs = parts[0].replace("int", "").strip()
            rhs = "=".join(parts[1:]).strip().rstrip(";")
            facts.append(
                f"Variable {lhs} is defined as '{_clean(rhs)}'."
            )

        if t == "FieldDeclaration" and "@Interval" in label:
            facts.append(
                f"Field with @Interval annotation: '{_clean(label)}'."
            )

    return facts



def summarize_graph_facts(ast: Dict, cfg: Dict, dfg: Dict) -> str:
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
    Компактное резюме фактов из AST/DFG:
    - присваивания (f = 3, l = f + 1)
    - Interval-аннотации
    - def-use цепочки с номерами строк (по DFG)
    Никаких сырых edge-ов.
    """
    assignments: List[str] = []
    annotations: List[str] = []

    for node in ast.get("nodes", []):
        t = node.get("type")
        raw_label = node.get("label", "")

        # Нормализуем перевод строки
        label = raw_label.replace("\r", "\n")

        # Присваивания: два случая
        if t == "VariableDeclarator" and "=" in label:
            # У тебя там комментарий на первой строке, а код на второй
            # Берём последнюю строку, чтобы избавиться от комментария
            last_line = label.splitlines()[-1].strip()
            assignments.append(last_line)

        elif t == "AssignExpr":
            cleaned = label.replace("\n", " ").strip()
            assignments.append(cleaned)

        # Interval-аннотации
        if t == "NormalAnnotationExpr" and "Interval" in label:
            cleaned_ann = label.replace("\n", " ").strip()
            annotations.append(cleaned_ann)

    parts: List[str] = []

    if assignments:
        parts.append("Assignments:")
        for a in assignments:
            parts.append(f"- {a}")

    if annotations:
        parts.append("Interval annotations:")
        for ann in annotations:
            parts.append(f"- {ann}")

    # DFG: def-use цепочки по переменным
    var_lines: dict[str, set[int]] = defaultdict(set)
    for n in dfg.get("nodes", []):
        v = n.get("var")
        line = n.get("line")
        if v is not None and isinstance(line, int):
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