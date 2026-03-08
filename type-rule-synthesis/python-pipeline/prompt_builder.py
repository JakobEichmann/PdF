# prompt_builder.py
# Research-grade, compact prompt builder for testcase-style rule synthesis
import re
from typing import Optional, Dict, List, Set
import torch

from structural_info import summarize_gnn_structural_cues_compact
from semantic_info import build_semantic_cues
from graph_facts import summarize_graph_facts_compact


# Parses:
# annotation Name(params) BaseType
#     :<==> "predicate"
#     for "wellformed";
ANNOT_RE = re.compile(
    r"annotation\s+(?P<name>\w+)\((?P<params>.*?)\)\s+(?P<base>\w+)\s*"
    r":<==>\s*\"(?P<predicate>.*?)\"\s*"
    r"for\s*\"(?P<wellformed>.*?)\"\s*;",
    re.DOTALL,
)


def extract_annotation_spec(code: str) -> Optional[Dict[str, str]]:
    m = ANNOT_RE.search(code)
    if not m:
        return None
    return {
        "name": m.group("name").strip(),
        "params": m.group("params").strip(),
        "base": m.group("base").strip(),
        "predicate": m.group("predicate").strip(),
        "wellformed": m.group("wellformed").strip(),
    }


def _format_annotation_spec_block(code: str) -> str:
    annot = extract_annotation_spec(code)
    if annot is None:
        return "Annotation spec:\n- (not found)\n"
    return (
        "Annotation spec:\n"
        f"- name: {annot['name']}\n"
        f"- params: {annot['params']}\n"
        f"- base: {annot['base']}\n"
        f"- predicate: {annot['predicate']}\n"
        f"- wellformed: {annot['wellformed']}\n"
    )


def _strip_block_comments_except_annotation_spec(code: str) -> str:
    """
    Removes all /* ... */ blocks except the one that contains 'annotation '.
    This prevents leaking 'Expected result' blocks into the prompt.
    """
    out: List[str] = []
    i = 0
    n = len(code)

    while i < n:
        j = code.find("/*", i)
        if j == -1:
            out.append(code[i:])
            break

        out.append(code[i:j])

        k = code.find("*/", j + 2)
        if k == -1:
            break

        block = code[j : k + 2]
        if "annotation " in block:
            out.append(block)

        i = k + 2

    return "".join(out)


def _build_line_mask_not_in_block_comment(code: str) -> List[bool]:
    """
    Returns a boolean mask per line: True if the line is outside /* ... */.
    We use this to ignore @Something inside block comments.
    """
    lines = code.split("\n")
    mask: List[bool] = [True] * len(lines)
    in_block = False
    for idx, line in enumerate(lines):
        if not in_block:
            if "/*" in line:
                in_block = True
                mask[idx] = False
        else:
            mask[idx] = False
            if "*/" in line:
                in_block = False
    return mask


def extract_minimal_code_fragment(code: str, ast: Dict) -> str:
    """
    Selects only relevant lines for rule synthesis:
    - annotation spec block (kept even though it's in a comment)
    - // :: error: lines and the following line
    - annotation usage @X(...) OUTSIDE block comments only
    - VariableDeclarator with '='
    - AssignExpr

    Also strips the 'Expected result' comment by removing all block comments
    except the annotation spec block.
    """
    cleaned = _strip_block_comments_except_annotation_spec(code)
    lines: List[str] = cleaned.split("\n")
    interesting: Set[int] = set()

    outside_mask = _build_line_mask_not_in_block_comment(cleaned)

    # capture around "annotation "
    for i, line in enumerate(lines):
        if "annotation " in line:
            for j in range(max(0, i - 2), min(len(lines), i + 12)):
                interesting.add(j)

    # error markers + following line
    for i, line in enumerate(lines):
        if ":: error:" in line:
            interesting.add(i)
            if i + 1 < len(lines):
                interesting.add(i + 1)

    # annotation usage outside block comments
    for i, line in enumerate(lines):
        if not outside_mask[i]:
            continue
        if "@" in line and "(" in line and ")" in line:
            interesting.add(i)

    # AST-driven selection
    for node in ast.get("nodes", []):
        ntype = node.get("type")
        raw_label = node.get("label", "")
        label = raw_label.replace("\r", "\n")

        if ntype == "VariableDeclarator" and "=" in label:
            last = label.splitlines()[-1].strip() if label else ""
            if last:
                for i, line in enumerate(lines):
                    if last in line:
                        interesting.add(i)

        if ntype == "AssignExpr":
            assign_text = label.replace("\n", " ").strip()
            if assign_text:
                for i, line in enumerate(lines):
                    if assign_text in line:
                        interesting.add(i)

    if not interesting:
        return cleaned

    window = 1
    final_idx: Set[int] = set()
    for idx in interesting:
        start = max(0, idx - window)
        end = min(len(lines) - 1, idx + window)
        for j in range(start, end + 1):
            final_idx.add(j)

    result_lines: List[str] = []
    for i in sorted(final_idx):
        if lines[i].strip() == "---":
            continue
        result_lines.append(lines[i])

    return "\n".join(result_lines).strip()


def build_llm_prompt(
    features: Dict,
    gnn_graph_embedding: torch.Tensor,
    codebert_embedding: torch.Tensor,
    gnn_node_embeddings: torch.Tensor | None = None,
) -> str:
    code = features["code"]
    ast = features["ast"]
    cfg = features["cfg"]
    dfg = features["dfg"]

    if gnn_node_embeddings is not None:
        gnn_summary = summarize_gnn_structural_cues_compact(ast, gnn_node_embeddings, code)
    else:
        gnn_summary = "No high-salience structural signals from GNN."

    semantic_summary = build_semantic_cues(ast)
    graph_facts = summarize_graph_facts_compact(ast, cfg, dfg)
    minimal_code = extract_minimal_code_fragment(code, ast)
    annot_text = _format_annotation_spec_block(code)

    prompt = f"""
You are an expert in refinement type systems and type rule synthesis.

Your goal is to synthesize GENERAL rule schemata (not facts about this specific program),
in the EXACT format used by our testcases.

{annot_text}

Code fragment:
-------------
{minimal_code}

Structural signals (GNN):
{gnn_summary}

Data-flow summary:
{graph_facts}

Semantic anchors (CodeBERT):
{semantic_summary}


Task:
Synthesize rule schemata that explain how refinement annotations propagate so that flagged assignments become type-correct,
in the testcase rule format.

Output requirements (STRICT):
- Output one or more rules.
- Each rule is a block:
  - 0 or more premise lines
  - a line containing exactly: ---
  - 1 or more conclusion lines
- Separate multiple rules by a single blank line.
- Do NOT repeat identical lines.
- Do NOT repeat identical blocks.

Allowed line kinds:
(A) Annotation line:
    <term> : @AnnName(...)
    <term> : AnnName(...)
    where <term> is either a variable (v, v0..v9) OR a call term like v1.insert(v2)
(B) Constraint line (arithmetic):
    <expr> <= <expr>
    <expr> >= <expr>
    <expr> < <expr>
    <expr> > <expr>
    <expr> = <expr>
    where <expr> uses only names a,b,c,d,n,m,k,v,v0..v9 and +,- and numerals.

- Use ONLY schematic meta-variables: v, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, a, b, c, d, n, m, k.
- Do NOT use any identifiers from the Java code (arg, arg0, l0, etc).
- Do NOT output Java code or declarations.
- The token ";" is FORBIDDEN anywhere.
- The token "=" is allowed ONLY:
  (1) in constraint lines (e.g. a = b), or
  (2) inside annotation argument lists (e.g. min="a").
  It must NEVER be used as Java assignment (e.g. v = v).
- Forbidden Java tokens: "int ", "{", "}", "public", "class", "return".
- Annotation lines must NOT contain ";" and must NOT contain " = " (spaces) outside of constraints.
- Annotation argument syntax must use quotes: other="v" (not other=v).
- Keep annotation syntax exactly like Java annotations, for example:
    v1 : @EqualTo(other="v2")
    v  : @Interval(min="a", max="b")
    v1.insert(v2) : @NonEmpty
    v1.insert(v2) : MinLength(n+1)
- If no premise is needed, output just:
  ---
  <conclusion line(s)>
- Do NOT output explanations, headers, bullets, or any extra text. Only the rules.
""".strip()

    return prompt