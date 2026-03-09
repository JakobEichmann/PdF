from pathlib import Path
import argparse
import os
import tempfile

from config import DEFAULT_JAVA_FILE, TESTCASES_DIR
from run_java_analyzer import run_java_analyzer
from feature_extraction import build_feature_package
from graph_builder import load_feature_package, build_graph_from_features
from gnn_model import encode_graph_with_nodes
from prompt_builder import build_llm_prompt, extract_annotation_spec
from llm_stub import generate_rule_with_llm
from rule_check import check_rule
from structural_info import summarize_gnn_structural_cues_compact
from semantic_info import build_semantic_cues
from graph_facts import summarize_graph_facts_compact
from collections import defaultdict
from typing import Dict, List

import re  # For extracting method names from code lines


def strip_block_comments_except_annotation_spec(code: str) -> str:
    """
    Remove all block comments from the code except the annotation spec
    itself. This helper is used to sanitize the Java source before
    analysis. It is copied here to keep the run script self-contained.
    """
    out: list[str] = []
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


def make_sanitized_temp_java_file(src: Path) -> Path:
    """
    Read a .java file, strip out block comments except the annotation spec,
    and write the sanitized code to a temporary file. Returns the path to
    the temporary file.
    """
    code = src.read_text(encoding="utf-8")
    sanitized = strip_block_comments_except_annotation_spec(code)
    fd, tmp_path = tempfile.mkstemp(prefix="sanitized_", suffix=".java")
    os.close(fd)
    Path(tmp_path).write_text(sanitized, encoding="utf-8")
    return Path(tmp_path)


def run_one(java_file: Path) -> None:
    """
    Run the entire pipeline on a single Java file. This function orchestrates
    the Java analysis, feature extraction, graph encoding, prompt building,
    rule synthesis, and Z3 checking. It prints concise debug information
    prefixed with [DEBUG] and the final synthesized rule with the Z3 status.
    """
    print(f"[1] Running Java analyzer on {java_file}.")
    sanitized_java = make_sanitized_temp_java_file(java_file)
    analysis_json_path = run_java_analyzer(sanitized_java)

    print(f"[2] Building feature package.")
    feature_package_path = build_feature_package(analysis_json_path)

    print(f"[3] Loading features and building graph.")
    features = load_feature_package(feature_package_path)
    graph_data, codebert_emb = build_graph_from_features(features)
    ast = features["ast"]
    cfg = features["cfg"]
    dfg = features["dfg"]

    # Print basic graph statistics
    print("[DEBUG] AST:", len(ast.get("nodes", [])), "nodes,", len(ast.get("edges", [])), "edges")
    print("[DEBUG] CFG:", len(cfg.get("nodes", [])), "nodes,", len(cfg.get("edges", [])), "edges")
    print("[DEBUG] DFG:", len(dfg.get("nodes", [])), "nodes,", len(dfg.get("edges", [])), "edges")
    print("[DEBUG] PyG graph:", graph_data.num_nodes, "nodes,", graph_data.num_edges, "edges")

    print(f"[4] Encoding graph with GNN.")
    gnn_emb, node_embs = encode_graph_with_nodes(graph_data)

    print(f"[5] Building LLM prompt.")
    # Construct the full prompt for the LLM. This prompt is passed to the LLM unchanged.
    prompt = build_llm_prompt(
        features,
        gnn_graph_embedding=gnn_emb,
        codebert_embedding=codebert_emb,
        gnn_node_embeddings=node_embs,
    )
    # Derive and print concise debug information instead of printing the full prompt.
    code = features["code"]
    # Annotation specification
    annot = extract_annotation_spec(code)
    print("[DEBUG] Annotation spec:")
    if annot:
        print(f"[DEBUG]- name: {annot['name']}")
        print(f"[DEBUG]- params: {annot['params']}")
        print(f"[DEBUG]- base: {annot['base']}")
        print(f"[DEBUG]- predicate: {annot['predicate']}")
        print(f"[DEBUG]- wellformed: {annot['wellformed']}")
    else:
        print("[DEBUG]- (not found)")
    # Compute method mapping and variable occurrences for debug output
    code_lines = code.splitlines()
    method_by_line: Dict[int, str] = {}
    current_method = None
    # Regex to capture method names, allowing for optional 'static' and generics in return type.
    # Example matches: 'public void foo(', 'private static int bar(', 'protected List<String> baz('
    method_pattern = re.compile(r"\b(?:public|private|protected)\s+(?:static\s+)?[\w<>\[\]]+\s+(\w+)\s*\(")
    for idx, cl in enumerate(code_lines, start=1):
        m = method_pattern.search(cl)
        if m:
            current_method = m.group(1)
        method_by_line[idx] = current_method
    reserved_vars = {
        "public", "private", "protected", "void", "int", "float", "double", "char",
        "boolean", "class", "static", "final", "return", "error", "assignment",
        "type", "incompatible", "other",
    }
    ident_re = re.compile(r"^[a-zA-Z_]\w*$")
    var_lines_by_method: Dict[str, Dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for n in dfg.get("nodes", []):
        var_name = n.get("var")
        line_no = n.get("line")
        if not var_name or not isinstance(var_name, str):
            continue
        if var_name in reserved_vars or not ident_re.match(var_name):
            continue
        if not isinstance(line_no, int):
            continue
        method = method_by_line.get(line_no)
        if method is None:
            method = "<global>"
        var_lines_by_method[method][var_name].add(line_no)
    # Collect structural signals (variable declarations) from code lines
    structural_map: Dict[str, List[str]] = defaultdict(list)
    for idx, cl in enumerate(code_lines, start=1):
        stripped = cl.strip()
        if '=' in stripped and ('.insert(' in stripped or '.remove(' in stripped):
            method = method_by_line.get(idx)
            if method is None:
                method = "<global>"
            structural_map[method].append(stripped)
    # Build structural and data-flow information from sanitized code lines.
    # We parse the sanitized code to find annotation declarations and assignments,
    # and group them by the method they belong to.
    annotation_lines_by_method: Dict[str, List[str]] = defaultdict(list)
    assignment_lines_by_method: Dict[str, List[str]] = defaultdict(list)
    for idx, cl in enumerate(code_lines, start=1):
        stripped = cl.strip()
        if not stripped:
            continue
        # Detect annotation declarations in code (e.g., @MinLength(...))
        # Skip lines that are part of the annotation spec comment block (/* annotation ... */)
        if stripped.startswith("@") and not stripped.startswith("@interface"):
            method = method_by_line.get(idx)
            if method is None:
                method = "<global>"
            annotation_lines_by_method[method].append(stripped)
        # Detect insert/remove calls and assignments. We treat insert/remove calls as data-flow
        # regardless of whether they appear in an assignment, and we include simple assignments
        # (e.g., a = arg) that do not involve insert/remove. Exclude lines in the annotation spec
        # or other comment/spec lines.
        assign_pattern = re.compile(r"^\s*[A-Za-z_]\w*\s*=\s*")
        # Skip lines that clearly belong to the annotation specification or method signatures.
        skip_line = False
        # Exclude annotation spec parts and meta lines
        if (
            "annotation" in stripped
            or stripped.startswith(":")
            or stripped.startswith("for ")
            or stripped.startswith("*")
            or "§" in stripped
            or "predicate" in stripped
            or "wellformed" in stripped
            or "base" in stripped
            or "params" in stripped
            or "<==>" in stripped
            or "->" in stripped
        ):
            skip_line = True
        # We do not skip lines simply because they contain '@'.
        # Annotation declarations in code (e.g., @MinLength(...) l0 = ...) should be included in data-flow
        # because they contain assignments we want to capture. Specification lines from annotation blocks
        # are filtered by other conditions above.
        # Skip method declarations (containing '(' without assignment)
        if "(" in stripped and "=" not in stripped:
            skip_line = True
        if not skip_line:
            method = method_by_line.get(idx)
            if method is None:
                method = "<global>"
            # 1) Lines with insert/remove calls are always included as-is (for call flows)
            if ".insert(" in stripped or ".remove(" in stripped:
                assignment_lines_by_method[method].append(stripped)
                continue
            # 2) Handle assignments inside annotation declarations
            if "@" in stripped and "=" in stripped:
                last_eq = stripped.rfind('=')
                first_paren_close = stripped.find(')')
                # Only consider assignment if '=' appears after ')' (annotation parameters end)
                if last_eq > first_paren_close >= 0:
                    lhs_part, rhs_part = stripped.rsplit("=", 1)
                    lhs_var_tokens = lhs_part.strip().split()
                    lhs_var = lhs_var_tokens[-1] if lhs_var_tokens else lhs_part.strip()
                    if lhs_var not in {"other", "min", "max", "len", "remainder", "modulus"}:
                        # Store a processed assignment string rather than the original line
                        assignment_lines_by_method[method].append(f"{lhs_var} = {rhs_part.strip().rstrip(';')}")
                        continue
            # 3) General assignment anywhere in the line (lhs = rhs)
            assign_match = re.search(r"\b([A-Za-z_]\w*)\s*=\s*", stripped)
            if assign_match:
                lhs_var = assign_match.group(1)
                if lhs_var not in {"other", "min", "max", "len", "remainder", "modulus"}:
                    assignment_lines_by_method[method].append(stripped)
    # Print structural signals (annotations and GNN signals) with method context.
    print("[DEBUG] Structural signals (GNN):")
    # Use the GNN summary to print high-salience structural cues, but skip variable declarations
    if node_embs is not None:
        gnn_summary = summarize_gnn_structural_cues_compact(ast, node_embs, code)
    else:
        gnn_summary = ""
    # Extract annotation lines from GNN summary that aren't variable declarations or normal annotation expr duplicates
    for line in gnn_summary.strip().split("\n"):
        l = line.strip()
        if not l:
            continue
        # Skip any variable declaration lines (with or without bullet)
        if "Variable declaration" in l:
            continue
        # Skip generic annotation expr lines; we will print annotation lines with method context
        if "annotation" in l.lower():
            continue
        # Print other high-salience structural cues directly
        print("[DEBUG]" + l)
    # Print annotation declarations grouped by method
    for method in sorted(annotation_lines_by_method):
        for ann in annotation_lines_by_method[method]:
            prefix = f"{method}:" if method != "<global>" else ""
            print(f"[DEBUG]- {prefix}{ann}")
    # Build data-flow summary lines. For each assignment or call, construct a flow description
    flow_lines: List[str] = []
    for method in sorted(assignment_lines_by_method):
        for assign in assignment_lines_by_method[method]:
            prefix = f"{method}:" if method != "<global>" else ""
            # If there is an assignment (=) split into lhs and rhs
            if "=" in assign:
                parts = assign.split("=", 1)
                lhs_part = parts[0].strip()
                rhs_part = parts[1].strip().rstrip(";")
                # Extract lhs variable name as the last token before '='
                lhs_tokens = lhs_part.split()
                lhs_var = lhs_tokens[-1] if lhs_tokens else lhs_part
                flow_lines.append(f"[DEBUG]- {prefix}{lhs_var} <- {rhs_part}")
            else:
                # A call without assignment (e.g., l.remove(0))
                # Indicate that the call requires a non-empty receiver
                flow_lines.append(f"[DEBUG]- {prefix}{assign} requires receiver non-empty")
    if flow_lines:
        print("[DEBUG] Data-flow summary:")
        for fl in flow_lines:
            print(fl)
    # Print def-use chains with method context
    if var_lines_by_method:
        print("[DEBUG] Def-use chains (from DFG):")
        for method in sorted(var_lines_by_method):
            for var_name, lines_set in sorted(var_lines_by_method[method].items()):
                if len(lines_set) <= 1:
                    continue
                seq = " -> ".join(str(l) for l in sorted(lines_set))
                prefix = f"{method}:" if method != "<global>" else ""
                print(f"[DEBUG]- {prefix}{var_name}: lines {seq}")
    # Semantic anchors
    semantic_summary = build_semantic_cues(ast)
    print("[DEBUG] Semantic anchors (CodeBERT):")
    for line in semantic_summary.strip().split("\n"):
        line = line.strip()
        # Skip duplicate heading emitted by build_semantic_cues
        if not line or line.startswith("CodeBERT-based semantic anchors"):
            continue
        print("[DEBUG]" + line)

    print(f"[6] Generating candidate type rule with LLM.")
    rule = generate_rule_with_llm(prompt)
    print("=== CANDIDATE TYPE RULE ===")
    print(rule)
    annot = extract_annotation_spec(features["code"])
    check = check_rule(rule, annot)
    print("=== RULE Z3 CHECK ===")
    print("status:", check.get("status"))
    print("message:", check.get("message", ""))
    if check.get("model"):
        print("model:", check["model"])
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=str, default="", help="Path to a single .java testcase")
    ap.add_argument("--all", action="store_true", help="Run all .java files in testcases/")
    args = ap.parse_args()
    if args.all:
        java_files = sorted(TESTCASES_DIR.glob("*.java"))
        if not java_files:
            raise RuntimeError(f"No .java files found in {TESTCASES_DIR}")
        for jf in java_files:
            print("============================================================")
            print(f"TESTCASE: {jf.name}")
            print("============================================================")
            run_one(jf)
        return
    if args.file:
        run_one(Path(args.file))
        return
    run_one(DEFAULT_JAVA_FILE)


if __name__ == "__main__":
    main()