from __future__ import annotations

import re
from typing import Optional, Tuple, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    PHI3_MODEL_NAME,
    PHI3_MODEL_PATH,
    PHI3_MAX_NEW_TOKENS,
    PHI3_DO_SAMPLE,
    PHI3_TEMPERATURE,
    PHI3_TOP_P,
    PHI3_REPETITION_PENALTY,
)

_model: Optional[AutoModelForCausalLM] = None
_tokenizer: Optional[AutoTokenizer] = None


def _pick_model_id() -> str:
    return PHI3_MODEL_PATH.strip() or PHI3_MODEL_NAME


def _load_phi3() -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    global _model, _tokenizer
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    model_id = _pick_model_id()
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id)

    model.eval()
    _model, _tokenizer = model, tok
    return model, tok


def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    t = re.sub(r"```[a-zA-Z0-9_-]*\n(.*?)```", r"\1", t, flags=re.DOTALL)
    t = re.sub(r"```(.*?)```", r"\1", t, flags=re.DOTALL)
    return t.strip()


def _extract_testcase_rules(text: str) -> str:
    """
    Extract full testcase-format rules (multi-line), not SMT.

    Expected output is one or more blocks:
      <0+ premise lines>
      ---
      <1+ conclusion lines>

    We therefore:
    - keep multiple lines
    - normalize excessive blank lines
    - if there is extra chatty text, try to cut to the first block containing a line '---'
    """
    t = _strip_code_fences(text)
    t = (t or "").strip()
    if not t:
        return ""

    t = t.replace("\r\n", "\n").replace("\r", "\n")
    lines = t.split("\n")

    first_sep = None
    for i, ln in enumerate(lines):
        if ln.strip() == "---":
            first_sep = i
            break

    if first_sep is not None:
        start = first_sep
        j = first_sep - 1
        while j >= 0:
            s = lines[j].strip()
            if s == "":
                j -= 1
                continue
            if ":" in s and ("@" in s or "(" in s):
                start = j
                j -= 1
                continue
            break
        t = "\n".join(lines[start:]).strip()

    cut_markers = ["Task:", "Output requirements", "Examples of the required OUTPUT SHAPE"]
    cut_pos = None
    for m in cut_markers:
        p = t.find(m)
        if p != -1:
            cut_pos = p if cut_pos is None else min(cut_pos, p)
    if cut_pos is not None:
        t = t[:cut_pos].strip()

    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t


# ---------------------------
# Normalization to DSL
# ---------------------------

_ANN_CALL_RE = re.compile(r'@(?P<ann>[A-Za-z]\w*)(?:\((?P<args>[^)]*)\))?')
_DSL_LINE_RE = re.compile(
    r'^\s*(?P<term>[A-Za-z]\w*(?:\.[A-Za-z]\w*\([A-Za-z]\w*\))?)\s*:\s*(?P<ann>@?[A-Za-z]\w*)'
    r'(?:\((?P<args>.*)\))?\s*$'
)
_CONSTR_RE = re.compile(r'^\s*[A-Za-z0-9_()+\-]+\s*(<=|>=|=|<|>)\s*[A-Za-z0-9_()+\-]+\s*$')

# key="x" required; model sometimes outputs key=x
_UNQUOTED_ARG_RE = re.compile(r'(\b[A-Za-z]\w*\b)\s*=\s*([A-Za-z]\w*)\b')

# Accept also key='x' (single quotes) and normalize to double quotes
_SINGLE_QUOTE_ARG_RE = re.compile(r'(\b[A-Za-z]\w*\b)\s*=\s*\'([^\']*)\'')


def _fix_args_quotes(arg_str: str) -> str:
    s = (arg_str or "").strip()
    if not s:
        return ""

    # normalize single quotes to double quotes
    s = _SINGLE_QUOTE_ARG_RE.sub(lambda m: f'{m.group(1)}="{m.group(2)}"', s)

    # add quotes around unquoted identifier RHS (other=v -> other="v")
    def repl(m):
        k = m.group(1)
        v = m.group(2)
        return f'{k}="{v}"'

    s = _UNQUOTED_ARG_RE.sub(repl, s)
    return s.strip()


def normalize_llm_output_to_dsl(text: str) -> str:
    """
    Normalize LLM output to the testcase DSL:
    - Keep '---' separators
    - Keep constraint lines
    - Convert Java-ish lines like '@Ann(args) v = v;' into 'v : @Ann(args)'
    - Enforce quotes in annotation args: other="v"
    - Drop everything else
    """
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.strip() for ln in t.split("\n")]

    out: List[str] = []
    for ln in lines:
        if not ln:
            continue

        if ln.strip() == "---":
            if not out or out[-1] != "---":
                out.append("---")
            continue

        # strip trailing semicolons
        ln = ln.rstrip(";").strip()

        # Already DSL line: normalize args quotes if present
        m_dsl = _DSL_LINE_RE.match(ln)
        if m_dsl:
            term = m_dsl.group("term")
            ann = m_dsl.group("ann")
            args = _fix_args_quotes(m_dsl.group("args") or "")
            if args:
                out.append(f'{term} : {ann}({args})')
            else:
                out.append(f"{term} : {ann}")
            continue

        # Constraint line
        if _CONSTR_RE.match(ln):
            out.append(ln)
            continue

        # Java-ish line: find @Ann(...) and then infer subject term after it
        m_ann = _ANN_CALL_RE.search(ln)
        if m_ann:
            ann_name = m_ann.group("ann")
            args = _fix_args_quotes(m_ann.group("args") or "")
            ann_txt = f"@{ann_name}({args})" if args else f"@{ann_name}"

            rest = ln[m_ann.end():].strip()

            # Heuristic term extraction: prefer call-term like v1.insert(v2) first
            m_call = re.search(r'([A-Za-z]\w*\.[A-Za-z]\w*\([A-Za-z]\w*\))', rest)
            if m_call:
                term = m_call.group(1)
                out.append(f"{term} : {ann_txt}")
                continue

            # Otherwise first identifier token
            m_id = re.search(r'([A-Za-z]\w*)', rest)
            if m_id:
                term = m_id.group(1)
                out.append(f"{term} : {ann_txt}")
                continue

        # Drop unknown lines
        continue

    if out and "---" not in out:
        out.insert(0, "---")

    s = "\n".join(out).strip()
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s
def canonicalize_rules(text: str, max_lines_total: int = 60, max_lines_per_block: int = 20) -> str:
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not t:
        return ""

    # split blocks by blank lines
    raw_blocks = re.split(r"\n\s*\n", t)
    cleaned_blocks: List[str] = []
    total_lines = 0

    for b in raw_blocks:
        lines = [ln.strip() for ln in b.split("\n") if ln.strip()]
        if not lines:
            continue

        # ensure there is exactly one '---'
        if "---" not in lines:
            # if model forgot, assume no premises
            lines = ["---"] + lines

        # keep only first separator
        sep_idx = lines.index("---")
        premises = lines[:sep_idx]
        concls = lines[sep_idx + 1 :]

        if not concls:
            continue

        # drop premise lines that repeat in conclusions (common spam)
        concl_set = set(concls)
        premises = [p for p in premises if p not in concl_set]

        # dedup while preserving order
        def dedup_keep_order(xs: List[str]) -> List[str]:
            seen = set()
            out = []
            for x in xs:
                if x in seen:
                    continue
                seen.add(x)
                out.append(x)
            return out

        premises = dedup_keep_order(premises)
        concls = dedup_keep_order(concls)

        # cap size
        premises = premises[: max_lines_per_block]
        concls = concls[: max_lines_per_block]

        block_lines = premises + ["---"] + concls
        total_lines += len(block_lines)
        if total_lines > max_lines_total:
            break

        cleaned_blocks.append("\n".join(block_lines))

    return "\n\n".join(cleaned_blocks).strip()

def generate_rule_with_llm(prompt: str) -> str:
    """
    Calls Phi-3 Mini (transformers).
    Returns testcase-format rules (multi-line), normalized to our DSL.
    """
    model, tok = _load_phi3()

    messages = [{"role": "user", "content": prompt}]

    if hasattr(tok, "apply_chat_template"):
        enc = tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    else:
        enc = tok(prompt, return_tensors="pt")

    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    gen_kwargs = dict(
        max_new_tokens=PHI3_MAX_NEW_TOKENS,
        do_sample=PHI3_DO_SAMPLE,
        temperature=PHI3_TEMPERATURE if PHI3_DO_SAMPLE else None,
        top_p=PHI3_TOP_P if PHI3_DO_SAMPLE else None,
        repetition_penalty=PHI3_REPETITION_PENALTY,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )

    gen_tokens = out[0, input_ids.shape[-1]:]
    raw = tok.decode(gen_tokens, skip_special_tokens=True)

    extracted = _extract_testcase_rules(raw)
    normalized = normalize_llm_output_to_dsl(extracted)
    canon = canonicalize_rules(normalized)
    return canon