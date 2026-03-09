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
    # Structural signals
    if node_embs is not None:
        gnn_summary = summarize_gnn_structural_cues_compact(ast, node_embs, code)
    else:
        gnn_summary = "No high-salience structural signals from GNN."
    print("[DEBUG] Structural signals (GNN):")
    for line in gnn_summary.strip().split("\n"):
        line = line.strip()
        if line:
            print("[DEBUG]" + line)
    # Data-flow summary
    graph_facts_summary = summarize_graph_facts_compact(ast, cfg, dfg)
    print("[DEBUG] Data-flow summary:")
    for line in graph_facts_summary.strip().split("\n"):
        line = line.strip()
        # Skip duplicate heading emitted by summarize_graph_facts_compact
        if not line or line.startswith("Data-flow summary"):
            continue
        print("[DEBUG]" + line)
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