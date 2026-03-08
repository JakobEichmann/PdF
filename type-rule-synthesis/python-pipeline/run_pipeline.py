from pathlib import Path
import argparse
import os, tempfile

from config import DEFAULT_JAVA_FILE, TESTCASES_DIR
from run_java_analyzer import run_java_analyzer
from feature_extraction import build_feature_package
from graph_builder import load_feature_package, build_graph_from_features
from gnn_model import encode_graph_with_nodes
from prompt_builder import build_llm_prompt, extract_annotation_spec
from llm_stub import generate_rule_with_llm
from rule_check import check_rule

def strip_block_comments_except_annotation_spec(code: str) -> str:
    out = []
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
        block = code[j:k+2]
        if "annotation " in block:
            out.append(block)
        i = k + 2
    return "".join(out)
def make_sanitized_temp_java_file(src: Path) -> Path:
    code = src.read_text(encoding="utf-8")
    sanitized = strip_block_comments_except_annotation_spec(code)
    fd, tmp_path = tempfile.mkstemp(prefix="sanitized_", suffix=".java")
    os.close(fd)
    Path(tmp_path).write_text(sanitized, encoding="utf-8")
    return Path(tmp_path)

def run_one(java_file: Path):
    print(f"[1] Running Java analyzer on {java_file}.")
    sanitized_java = make_sanitized_temp_java_file(java_file)
    analysis_json_path = run_java_analyzer(sanitized_java)
#    analysis_json_path = run_java_analyzer(java_file)

    print(f"[2] Building feature package.")
    feature_package_path = build_feature_package(analysis_json_path)

    print(f"[3] Loading features and building graph.")
    features = load_feature_package(feature_package_path)
    graph_data, codebert_emb = build_graph_from_features(features)
    ast = features["ast"]
    cfg = features["cfg"]
    dfg = features["dfg"]

    print("[DEBUG] AST:", len(ast.get("nodes", [])), "nodes,", len(ast.get("edges", [])), "edges")
    print("[DEBUG] CFG:", len(cfg.get("nodes", [])), "nodes,", len(cfg.get("edges", [])), "edges")
    print("[DEBUG] DFG:", len(dfg.get("nodes", [])), "nodes,", len(dfg.get("edges", [])), "edges")
    print("[DEBUG] PyG graph:", graph_data.num_nodes, "nodes,", graph_data.num_edges, "edges")

    print(f"[4] Encoding graph with GNN.")
    gnn_emb, node_embs = encode_graph_with_nodes(graph_data)

    print(f"[5] Building LLM prompt.")
    prompt = build_llm_prompt(
        features,
        gnn_graph_embedding=gnn_emb,
        codebert_embedding=codebert_emb,
        gnn_node_embeddings=node_embs,
    )

    print(prompt)

    print(f"[6] Generating candidate type rule with LLM.")
    rule = generate_rule_with_llm(prompt)

    print("=== CANDIDATE TYPE RULE ===")
    print(rule)
    annot = extract_annotation_spec(features["code"])
    check = check_rule(rule, annot)
    print("=== RULE Z3 CHECK ===")
    print("status:", check["status"])
    print("message:", check.get("message", ""))
    if check.get("model"):
        print("model:", check["model"])
    print()


def main():
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
