from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import z3


# ---------------------------
# SMT mode (unchanged)
# ---------------------------

def _strip_rule_prefix(rule_text: str) -> str:
    s = (rule_text or "").strip()
    if s.upper().startswith("RULE:"):
        s = s.split(":", 1)[1].strip()
    return s


def _check_rule_smt(rule_text: str) -> Dict[str, Any]:
    expr = _strip_rule_prefix(rule_text)

    if not expr:
        return {"status": "PARSE_ERROR", "message": "empty rule"}

    if expr.strip().upper() == "UNKNOWN":
        return {"status": "UNKNOWN", "message": "LLM returned UNKNOWN"}

    smt2 = f"""
(set-logic ALL)
(set-option :produce-models true)
(assert (not {expr}))
"""

    try:
        constraints = z3.parse_smt2_string(smt2)
    except Exception as e:
        return {"status": "PARSE_ERROR", "message": f"SMT-LIB parse error: {e}", "rule": expr}

    s = z3.Solver()
    try:
        for c in constraints:
            s.add(c)
    except Exception as e:
        return {"status": "PARSE_ERROR", "message": f"failed to add constraints: {e}", "rule": expr}

    res = s.check()
    if res == z3.unsat:
        return {"status": "VALID", "message": "Z3: (not RULE) is UNSAT, RULE is valid", "rule": expr}

    if res == z3.sat:
        out: Dict[str, Any] = {"status": "INVALID", "message": "Z3: (not RULE) is SAT, RULE is not valid", "rule": expr}
        try:
            out["model"] = str(s.model())
        except Exception:
            pass
        return out

    return {"status": "Z3_UNKNOWN", "message": "Z3 returned unknown", "rule": expr}


# ---------------------------
# DSL mode (testcase format)
# ---------------------------

@dataclass
class AnnotSpec:
    name: str
    params: str
    base: str
    predicate: str
    wellformed: str


# Annotation line supports:
#   <term> : @Ann(args)
#   <term> : Ann(args)
# where <term> is either:
#   - identifier: v, v0, a, n, ...
#   - call: v1.insert(v2) / v1.remove(v2) / vX.anyName(vY)
ANNOT_LINE_RE = re.compile(
    r'^\s*(?P<term>[A-Za-z]\w*(?:\.[A-Za-z]\w*\([A-Za-z]\w*\))?)\s*:\s*(?P<ann>@?[A-Za-z]\w*)\((?P<args>.*)\)\s*$'
)

# Args like: min="a", max="b", other="v2"
ARG_RE = re.compile(r'([A-Za-z]\w*)\s*=\s*"?([A-Za-z]\w*)"?')

# Constraint line supports simple comparisons:
#   <expr> <= <expr> etc, where <expr> can contain:
#   names, numerals, +, -, parentheses
CONSTR_RE = re.compile(r'^\s*(?P<lhs>.+?)\s*(?P<op><=|>=|=|<|>)\s*(?P<rhs>.+?)\s*$')

# Allowed tokens inside arithmetic expressions (conservative)
ARITH_TOKEN_RE = re.compile(r'^[A-Za-z0-9_+\-() \t]+$')


def _to_spec(d: Optional[Dict[str, str]]) -> Optional[AnnotSpec]:
    if not d:
        return None
    return AnnotSpec(
        name=d.get("name", "").strip(),
        params=d.get("params", "").strip(),
        base=d.get("base", "").strip(),
        predicate=d.get("predicate", "").strip(),
        wellformed=d.get("wellformed", "").strip(),
    )


def _split_rule_blocks(text: str) -> List[Tuple[List[str], List[str]]]:
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not t:
        return []
    blocks = re.split(r"\n\s*\n", t)
    out: List[Tuple[List[str], List[str]]] = []
    for b in blocks:
        lines = [ln.strip() for ln in b.split("\n") if ln.strip() != ""]
        if not lines:
            continue
        if "---" not in lines:
            raise ValueError("block missing '---'")
        k = lines.index("---")
        premises = lines[:k]
        concls = lines[k + 1 :]
        if not concls:
            raise ValueError("block has no conclusions after '---'")
        out.append((premises, concls))
    return out


def _param_names_from_spec(spec: AnnotSpec) -> List[str]:
    ps = spec.params.strip()
    if not ps:
        return []
    names: List[str] = []
    for p in ps.split(","):
        p = p.strip()
        if not p:
            continue
        toks = p.split()
        names.append(toks[-1])
    return names


# ---------------------------
# Term model
# ---------------------------

@dataclass(frozen=True)
class Term:
    kind: str  # "id" or "call"
    name: str
    recv: Optional[str] = None
    meth: Optional[str] = None
    arg: Optional[str] = None


def _parse_term(s: str) -> Term:
    s = s.strip()
    m = re.match(r'^([A-Za-z]\w*)\.([A-Za-z]\w*)\(([A-Za-z]\w*)\)$', s)
    if m:
        return Term(kind="call", name=s, recv=m.group(1), meth=m.group(2), arg=m.group(3))
    return Term(kind="id", name=s)


# Decide whether the annotation base suggests Int-domain or Object-domain.
def _is_int_domain(spec: AnnotSpec) -> bool:
    return spec.base.lower() == "int"


# ---------------------------
# Z3 environment builders
# ---------------------------

@dataclass
class ZEnv:
    int_vars: Dict[str, z3.IntNumRef]
    obj_vars: Dict[str, z3.ExprRef]
    obj_sort: Optional[z3.SortRef]
    len_fun: Optional[z3.FuncDeclRef]
    call_fun: Dict[str, z3.FuncDeclRef]  # method name -> function


def _mk_zenv(int_domain: bool) -> ZEnv:
    if int_domain:
        return ZEnv(int_vars={}, obj_vars={}, obj_sort=None, len_fun=None, call_fun={})

    obj_sort = z3.DeclareSort("Obj")
    len_fun = z3.Function("len", obj_sort, z3.IntSort())

    # Generic method functions (Obj x Obj -> Obj) created lazily per method name
    return ZEnv(int_vars={}, obj_vars={}, obj_sort=obj_sort, len_fun=len_fun, call_fun={})


def _get_int_var(env: ZEnv, name: str) -> z3.IntNumRef:
    if name not in env.int_vars:
        env.int_vars[name] = z3.Int(name)
    return env.int_vars[name]


def _get_obj_var(env: ZEnv, name: str) -> z3.ExprRef:
    assert env.obj_sort is not None
    if name not in env.obj_vars:
        env.obj_vars[name] = z3.Const(name, env.obj_sort)
    return env.obj_vars[name]


def _get_call_fun(env: ZEnv, meth: str) -> z3.FuncDeclRef:
    assert env.obj_sort is not None
    if meth not in env.call_fun:
        env.call_fun[meth] = z3.Function(meth, env.obj_sort, env.obj_sort, env.obj_sort)
    return env.call_fun[meth]


def _term_to_z3(env: ZEnv, term: Term, int_domain: bool) -> Union[z3.IntNumRef, z3.ExprRef]:
    if int_domain:
        # In int-domain, we only allow id terms
        if term.kind != "id":
            raise ValueError(f"call terms are not allowed for int-based annotations: {term.name}")
        return _get_int_var(env, term.name)

    # object domain
    if term.kind == "id":
        return _get_obj_var(env, term.name)

    assert term.kind == "call"
    recv = _get_obj_var(env, term.recv or "")
    arg = _get_obj_var(env, term.arg or "")
    f = _get_call_fun(env, term.meth or "call")
    return f(recv, arg)


# ---------------------------
# Constraint parsing to Z3
# ---------------------------

def _parse_arith_expr_to_z3(env: ZEnv, expr: str) -> z3.ArithRef:
    e = expr.strip()
    if not ARITH_TOKEN_RE.match(e):
        raise ValueError(f"unsupported arithmetic tokens in: {expr}")

    # Collect variable-like tokens and declare them Int
    # (names are schema variables like a,b,c,d,n,m,k,v,v0..v9)
    names = set(re.findall(r"[A-Za-z]\w*", e))
    for nm in names:
        _get_int_var(env, nm)

    decls = "\n".join([f"(declare-const {nm} Int)" for nm in sorted(names)])
    # Use SMT parser for arithmetic expression itself by asserting equality to a fresh symbol
    smt2 = f"""
{decls}
(declare-const __tmp Int)
(assert (= __tmp {e}))
"""
    parsed = z3.parse_smt2_string(smt2)
    # Find the asserted equality and extract __tmp
    tmp = z3.Int("__tmp")
    # Conjoin constraints and then solve substitution? Too heavy.
    # Instead, we re-parse directly into a Z3 expression without constraints by using a second assert:
    # We will just return tmp and rely on constraints being added by caller.
    # To keep it simple: caller will add the parsed constraints and use tmp.
    #
    # So we return (tmp, parsed_constraints) is ideal, but we keep interface simple:
    raise ValueError("internal: use _constraint_line_to_z3 which handles constraints properly")


def _constraint_line_to_z3(env: ZEnv, line: str) -> z3.BoolRef:
    m = CONSTR_RE.match(line.strip())
    if not m:
        raise ValueError(f"not a constraint line: {line}")

    lhs_s = m.group("lhs").strip()
    rhs_s = m.group("rhs").strip()
    op = m.group("op").strip()

    for side in [lhs_s, rhs_s]:
        if not ARITH_TOKEN_RE.match(side):
            raise ValueError(f"unsupported arithmetic tokens in: {line}")

    # Declare all name tokens as Int
    names = set(re.findall(r"[A-Za-z]\w*", lhs_s + " " + rhs_s))
    for nm in names:
        _get_int_var(env, nm)

    decls = "\n".join([f"(declare-const {nm} Int)" for nm in sorted(names)])
    smt_expr = f"({op} {lhs_s} {rhs_s})"
    smt2 = f"""
{decls}
(assert {smt_expr})
"""
    parsed = z3.parse_smt2_string(smt2)
    if len(parsed) == 0:
        raise ValueError(f"could not parse constraint: {line}")
    return z3.And(parsed)


# ---------------------------
# Annotation line to Z3
# ---------------------------

def _parse_annotation_line(line: str) -> Tuple[Term, str, Dict[str, str]]:
    m = ANNOT_LINE_RE.match(line)
    if not m:
        raise ValueError(f"not an annotation line: {line}")

    term_s = m.group("term").strip()
    ann_raw = m.group("ann").strip()
    args_raw = (m.group("args") or "").strip()

    ann = ann_raw[1:] if ann_raw.startswith("@") else ann_raw

    args: Dict[str, str] = {}
    if args_raw:
        for am in ARG_RE.finditer(args_raw):
            args[am.group(1)] = am.group(2)

    return _parse_term(term_s), ann, args


def _normalize_predicate_for_object_domain(pred: str) -> str:
    """
    Heuristic normalization for common predicates:
    - replace '§subject§.length' with 'len(§subject§)'
    - allow both '.length' and '.size' if present
    """
    p = pred
    p = p.replace("§subject§.length", "len(§subject§)")
    p = p.replace("§subject§.size", "len(§subject§)")
    # also handle parameter occurrences like §other§.length if needed later
    p = re.sub(r"§([A-Za-z]\w*)§\.length", r"len(§\1§)", p)
    p = re.sub(r"§([A-Za-z]\w*)§\.size", r"len(§\1§)", p)
    return p


def _predicate_to_z3_from_infix_ints(env: ZEnv, pred_infix: str) -> z3.BoolRef:
    e = pred_infix.strip().replace("==", "=")

    op = None
    for cand in ["<=", ">=", "=", "<", ">"]:
        if _split_top_level(e, cand) is not None:
            op = cand
            break
    if op is None:
        raise ValueError(f"unsupported predicate form: {pred_infix}")

    left, right = _split_top_level(e, op)
    assert left is not None and right is not None

    # Declare all tokens as Int
    names = set(re.findall(r"[A-Za-z]\w*", left + " " + right))
    for nm in names:
        _get_int_var(env, nm)

    decls = "\n".join([f"(declare-const {nm} Int)" for nm in sorted(names)])
    smt_expr = f"({op} {left.strip()} {right.strip()})"
    smt2 = f"""
{decls}
(assert {smt_expr})
"""
    parsed = z3.parse_smt2_string(smt2)
    if len(parsed) == 0:
        raise ValueError(f"could not parse predicate: {pred_infix}")
    return z3.And(parsed)


def _predicate_to_z3_object_domain(env: ZEnv, pred_infix: str) -> z3.BoolRef:
    """
    For object domain we support predicates that reduce to Int comparisons using len(...).
    Example predicates:
      §subject§.length > 0
      §subject§.length >= §n§
    After normalization:
      len(subject) > 0
      len(subject) >= n
    """
    assert env.obj_sort is not None and env.len_fun is not None
    e = pred_infix.strip().replace("==", "=")

    # Replace len(x) occurrences by introducing tmp Int constants and adding equalities
    # We keep it simple: only allow patterns len(NAME)
    len_calls = re.findall(r"len\(\s*([A-Za-z]\w*)\s*\)", e)
    for nm in len_calls:
        _get_obj_var(env, nm)

    # Declare int vars from remaining tokens (excluding 'len')
    names = set(re.findall(r"[A-Za-z]\w*", e))
    names.discard("len")
    for nm in names:
        # Some names refer to objects (subject and maybe others). If they appear in len(...), they are objects.
        # Otherwise treat as Int schema var.
        if nm in len_calls:
            continue
        # "null" if ever appears, ignore it by mapping to 0? Better: forbid
        if nm.lower() == "null":
            raise ValueError("null is not supported in predicates; remove it from rules")
        _get_int_var(env, nm)

    # Build SMT with an uninterpreted function len: Obj -> Int and Obj declarations.
    obj_decls = "\n".join([f"(declare-const {nm} Obj)" for nm in sorted(set(len_calls))])
    int_decls = "\n".join([f"(declare-const {nm} Int)" for nm in sorted(set(env.int_vars.keys()))])

    # Declare len in SMT context
    smt2 = f"""
(declare-sort Obj 0)
(declare-fun len (Obj) Int)
{obj_decls}
{int_decls}
(assert { _infix_cmp_to_smt(e) })
"""
    parsed = z3.parse_smt2_string(smt2)
    if len(parsed) == 0:
        raise ValueError(f"could not parse object predicate: {pred_infix}")
    return z3.And(parsed)


def _infix_cmp_to_smt(e: str) -> str:
    s = e.strip()
    # find top-level comparator
    op = None
    for cand in ["<=", ">=", "=", "<", ">"]:
        if _split_top_level(s, cand) is not None:
            op = cand
            break
    if op is None:
        raise ValueError(f"unsupported predicate form: {e}")
    left, right = _split_top_level(s, op)
    assert left is not None and right is not None
    return f"({op} {left.strip()} {right.strip()})"


def _split_top_level(s: str, op: str) -> Optional[Tuple[str, str]]:
    depth = 0
    i = 0
    while i <= len(s) - len(op):
        ch = s[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)

        if depth == 0 and s.startswith(op, i):
            left = s[:i]
            right = s[i + len(op) :]
            return left, right
        i += 1
    return None


def _annotation_line_to_z3(line: str, spec: AnnotSpec, env: ZEnv, int_domain: bool) -> z3.BoolRef:
    term, ann, args = _parse_annotation_line(line)
    if ann != spec.name:
        raise ValueError(f"unexpected annotation {ann}, expected {spec.name}")

    # Subject is term; params are schema names/terms (we treat them as ids)
    subject_z3 = _term_to_z3(env, term, int_domain)

    repl: Dict[str, str] = {"§subject§": term.name if term.kind == "id" else term.name}

    pnames = _param_names_from_spec(spec)
    for pn in pnames:
        if pn not in args:
            raise ValueError(f"missing param '{pn}' in annotation line: {line}")
        repl[f"§{pn}§"] = args[pn]

        # declare param variable depending on domain
        if int_domain:
            _get_int_var(env, args[pn])
        else:
            # params could be ints (like n) or objects; we assume int params unless they appear in len(...)
            _get_int_var(env, args[pn])

    pred = spec.predicate
    if not int_domain:
        pred = _normalize_predicate_for_object_domain(pred)

    for k, v in repl.items():
        pred = pred.replace(k, v)

    if int_domain:
        # Ensure the subject is an Int symbol in parsing environment
        if term.kind != "id":
            raise ValueError(f"call subject not supported for int-domain: {term.name}")
        _get_int_var(env, term.name)
        return _predicate_to_z3_from_infix_ints(env, pred)

    # object domain: predicates must reduce to len(subject) comparisons
    return _predicate_to_z3_object_domain(env, pred)


def _line_to_z3(line: str, spec: AnnotSpec, env: ZEnv, int_domain: bool) -> z3.BoolRef:
    s = line.strip()

    # Annotation line (contains ':' and '(' after ann name)
    if ":" in s and "(" in s:
        return _annotation_line_to_z3(s, spec, env, int_domain)

    # Constraint line
    if CONSTR_RE.match(s):
        return _constraint_line_to_z3(env, s)

    raise ValueError(f"unrecognized rule line: {line}")


def _check_rule_dsl(rule_text: str, annot_spec: Optional[Dict[str, str]]) -> Dict[str, Any]:
    spec = _to_spec(annot_spec)
    if spec is None or not spec.name or not spec.predicate:
        return {
            "status": "PARSE_ERROR",
            "message": "annotation spec missing/incomplete (pass extract_annotation_spec(features['code']) into check_rule)",
        }

    int_domain = _is_int_domain(spec)
    blocks = _split_rule_blocks(rule_text)
    if not blocks:
        return {"status": "PARSE_ERROR", "message": "empty rule"}

    formulas: List[z3.BoolRef] = []
    for premises, concls in blocks:
        env = _mk_zenv(int_domain)

        prem_z3 = [_line_to_z3(ln, spec, env, int_domain) for ln in premises] if premises else []
        concl_z3 = [_line_to_z3(ln, spec, env, int_domain) for ln in concls]

        premise = z3.And(prem_z3) if prem_z3 else z3.BoolVal(True)
        conclusion = z3.And(concl_z3) if len(concl_z3) > 1 else concl_z3[0]

        # Quantify all symbols we introduced:
        qvars: List[z3.ExprRef] = []
        if int_domain:
            qvars.extend([env.int_vars[nm] for nm in sorted(env.int_vars.keys())])
        else:
            qvars.extend([env.obj_vars[nm] for nm in sorted(env.obj_vars.keys())])
            qvars.extend([env.int_vars[nm] for nm in sorted(env.int_vars.keys())])

        block_formula = z3.ForAll(qvars, z3.Implies(premise, conclusion)) if qvars else z3.Implies(premise, conclusion)
        formulas.append(block_formula)

    full = z3.And(formulas) if len(formulas) > 1 else formulas[0]

    s = z3.Solver()
    s.add(z3.Not(full))
    res = s.check()
    if res == z3.unsat:
        return {"status": "VALID", "message": "Z3: not(formula) UNSAT, rule is valid"}
    if res == z3.sat:
        out: Dict[str, Any] = {"status": "INVALID", "message": "Z3: not(formula) SAT, counterexample exists"}
        try:
            out["model"] = str(s.model())
        except Exception:
            pass
        return out
    return {"status": "Z3_UNKNOWN", "message": "Z3 returned unknown"}


# ---------------------------
# Public entry point
# ---------------------------

def check_rule(rule_text: str, annot_spec: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    try:
        t = (rule_text or "").strip()
        if not t:
            return {"status": "PARSE_ERROR", "message": "empty rule"}

        if t.upper().startswith("RULE:") or t.startswith("("):
            return _check_rule_smt(t)

        return _check_rule_dsl(t, annot_spec)

    except Exception as e:
        return {"status": "PARSE_ERROR", "message": str(e)}