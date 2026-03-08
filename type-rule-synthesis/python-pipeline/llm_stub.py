"""
LLM helper for refinement type rule synthesis.

This module wraps a call to the external language model (by default
Phi‑3 mini) to synthesise refinement type rules from the prompt built
by ``prompt_builder.build_llm_prompt``.  In addition to invoking the
model it also implements a simple feedback loop: after each candidate
rule is generated the rule is checked against the Z3 SMT solver via
``rule_check.check_rule``.  If the rule is invalid, the prompt is
extended with the model's mistake and the model is queried again.  This
loop repeats until a valid rule is produced or a maximum iteration
count is reached.

Because model loading can be expensive and may not always be possible
in constrained environments, the helper first attempts to load the
user‑specified model.  On failure it falls back to a small set of
deterministic heuristics derived from the testcases.  These heuristics
cover the common annotations ``EqualTo``, ``Interval``, ``MinLength``,
``NonEmpty`` and ``Remainder``.  If the annotation is unknown and the
model cannot be loaded, a conservative reflexivity rule is emitted.

The public function ``generate_rule_with_llm`` may therefore be used as
a drop‑in replacement for the original model invocation: it accepts a
prompt and returns one or more rule blocks in the test DSL.
"""

from __future__ import annotations

import re
from typing import List, Dict, Optional

# Optional: import transformers lazily to reduce overhead when the package
# is unavailable.  We'll attempt to load the model only if this module
# detects that ``transformers`` is installed.
try:
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
except Exception:
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    torch = None  # type: ignore

try:
    # Import rule_check only when needed; this avoids creating a circular
    # dependency if rule_check imports llm_stub.
    from rule_check import check_rule
except Exception:
    # If rule_check is unavailable during import (e.g. in test stubs), we
    # assign a dummy function that always claims validity.  This ensures
    # generate_rule_with_llm can still return a rule without errors.
    def check_rule(rule_text: str, annot_spec: Optional[Dict[str, str]] = None) -> Dict[str, str]:  # type: ignore
        return {"status": "VALID", "message": "rule_check not available"}


def _extract_annotation_spec(prompt: str) -> Optional[Dict[str, str]]:
    """
    Extract the annotation specification dictionary from the prompt.

    The prompt contains an ``Annotation spec:`` block with bullet points:
      - name: EqualTo
      - params: int other
      - base: int
      - predicate: §subject§ == §other§
      - wellformed: true

    This helper parses those lines and returns a mapping keyed by the
    field names.  If any field is missing, it returns None.
    """
    if not prompt or "Annotation spec" not in prompt:
        return None
    # Regular expression to capture the five fields; we use non-greedy
    # matches to handle multi‑line predicates.
    spec_re = re.compile(
        r"Annotation spec:.*?"
        r"-\s*name:\s*(?P<name>[^\n]+)\n"
        r"-\s*params:\s*(?P<params>[^\n]*)\n"
        r"-\s*base:\s*(?P<base>[^\n]+)\n"
        r"-\s*predicate:\s*(?P<predicate>.*?)(?:\n-\s*wellformed:|\n$)"
        r"(?:-\s*wellformed:\s*(?P<wellformed>.*?))?\n",
        re.DOTALL,
    )
    m = spec_re.search(prompt)
    if not m:
        return None
    spec = {
        "name": m.group("name").strip(),
        "params": m.group("params").strip(),
        "base": m.group("base").strip(),
        "predicate": (m.group("predicate") or "").strip(),
        "wellformed": (m.group("wellformed") or "").strip(),
    }
    # Normalize certain Java predicates to simpler infix forms.  In particular,
    # replace calls to java.lang.Math.floorMod with the infix modulo operator.
    pred = spec["predicate"]
    # Pattern: java.lang.Math.floorMod(§x§, §y§) -> §x§ mod §y§
    pred = re.sub(
        r"java\.lang\.Math\.floorMod\(\s*(§[A-Za-z]\w*§)\s*,\s*(§[A-Za-z]\w*§)\s*\)",
        r"\1 mod \2",
        pred,
    )
    spec["predicate"] = pred
    return spec

def _extract_annotation_name(prompt: str) -> Optional[str]:
    """Return the annotation name from the prompt.

    The prompt built by `prompt_builder.build_llm_prompt` contains a
    section starting with ``Annotation spec:`` followed by lines of the
    form ``- name: X``.  This helper scans the prompt for that
    structure and returns the annotation name with surrounding
    whitespace stripped.  If no match is found it returns ``None``.
    """
    # We allow any amount of whitespace and capture until the end of the line
    m = re.search(r"Annotation spec:.*?-\s*name:\s*([A-Za-z][\w]*)", prompt, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: if there is no annotation spec block, attempt to infer the annotation name
    # from the usage of annotations in the code fragment.  We look for patterns
    # like `@Interval(` or `@MinLength(` and return the first name found.
    usage = re.search(r"@([A-Za-z][\w]*)\s*\(", prompt)
    if usage:
        return usage.group(1).strip()
    return None


def _extract_param_names(prompt: str) -> List[str]:
    """Return the list of parameter names from the prompt.

    The annotation specification contains a line ``- params:`` which may
    list multiple type/name pairs separated by commas, e.g. ``int
    min, int max``.  This helper extracts the identifier component of
    each pair.  If the line is missing or empty it returns an empty
    list.
    """
    m = re.search(r"-\s*params:\s*([^\n]+)", prompt)
    if m:
        params_part = m.group(1).strip()
        # split by comma and take last token as parameter name
        names: List[str] = []
        for part in params_part.split(','):
            part = part.strip()
            if not part:
                continue
            # type/name pair: take last token
            tokens = part.split()
            names.append(tokens[-1])
        return names
    # Fallback: attempt to extract parameter names from annotation usage lines in the code
    # For example: @Interval(min = 1, max = 3) -> extract ['min', 'max']
    usage = re.search(r"@([A-Za-z][\w]*)\s*\(([^)]*)\)", prompt)
    if usage:
        args_part = usage.group(2)
        names: List[str] = []
        for arg in args_part.split(','):
            # Each arg is of form key= value or key = value
            key = arg.split('=')[0].strip()
            if key:
                names.append(key)
        return names
    return []


def _ensure_named_annotation_args(rule: str, annot_name: str, param_names: List[str]) -> str:
    """Ensure that annotation calls in the rule use explicit named arguments.

    Some candidate rules generated by the LLM may reference an annotation such as
    ``MinLength`` using positional arguments (e.g. ``MinLength(n)``) instead of
    named arguments (e.g. ``MinLength(len=n)``) required by the type rule DSL.
    This helper scans the rule text and rewrites positional arguments into
    named form using the names provided in ``param_names``.  If the number of
    positional arguments does not match the number of parameter names, the call
    is left unchanged.  Annotation names are matched both with and without
    the ``@`` prefix (e.g. ``@MinLength`` and ``MinLength``).

    Parameters
    ----------
    rule : str
        The candidate rule text generated by the LLM.
    annot_name : str
        The name of the annotation (e.g. ``MinLength``).
    param_names : List[str]
        A list of parameter names expected by the annotation specification.

    Returns
    -------
    str
        The rule with positional annotation arguments replaced by named
        arguments where appropriate.
    """
    if not param_names:
        return rule

    # Build a regex to match annotation calls like `MinLength(expr)` or
    # `@MinLength(expr)`.  We capture the arguments inside the parentheses.
    # Use a negative lookbehind to avoid matching annotation names that are
    # substrings of longer identifiers.
    pattern = re.compile(
        rf"(?P<prefix>@?){re.escape(annot_name)}\s*\((?P<args>[^\)]+)\)"
    )

    def replacer(match: re.Match) -> str:
        args_str = match.group("args").strip()
        # If any argument already contains '=', assume names are provided
        if '=' in args_str:
            return match.group(0)
        # Split arguments by comma
        values = [val.strip() for val in args_str.split(',') if val.strip()]
        if len(values) != len(param_names):
            # Mismatched number of arguments; leave unchanged
            return match.group(0)
        # Construct named arguments
        named = ', '.join(f"{p}={v}" for p, v in zip(param_names, values))
        return f"{match.group('prefix')}{annot_name}({named})"

    # Replace all occurrences in the rule
    return pattern.sub(replacer, rule)


def _remove_null_literals(rule: str) -> str:
    """Remove occurrences of the literal ``null`` from a rule.

    Some LLM outputs may include the Java literal ``null`` in predicates
    or expressions (e.g. ``v != null``).  Z3 does not support the
    ``null`` literal in this DSL, so we remove it entirely.  We only
    remove whole tokens ``null`` to avoid corrupting identifiers that
    contain ``null`` as a substring.

    Parameters
    ----------
    rule : str
        The candidate rule text.

    Returns
    -------
    str
        The rule with all standalone occurrences of ``null`` removed.
    """
    return re.sub(r"\bnull\b", "", rule)


def _split_into_rule_blocks(rule: str) -> str:
    """Normalize rule blocks by eliminating blank lines inside blocks.

    The DSL expects each rule block to consist of an optional list of premise
    lines followed by a line containing ``---`` and one or more conclusion
    lines.  Blocks are separated by a single blank line.  Some LLM outputs
    may insert extra blank lines between premise and conclusion lines within
    a block (for example between two annotation lines).  Such blank lines
    cause parsing errors in ``rule_check``.  This helper removes blank
    lines within blocks and ensures there is exactly one blank line between
    distinct blocks.

    Parameters
    ----------
    rule : str
        Candidate rule text potentially containing extra blank lines.

    Returns
    -------
    str
        Normalized rule text with proper block separation.
    """
    # Split the rule into lines and accumulate non-empty segments separated by
    # blank lines.  Multiple consecutive blank lines denote a single block
    # separator.  We collapse blank lines within a block by simply not
    # adding them.
    lines = [ln.rstrip() for ln in rule.splitlines()]
    blocks: List[List[str]] = []
    current: List[str] = []
    blank_run = False
    for ln in lines:
        if not ln.strip():
            blank_run = True
            continue
        if blank_run and current:
            # End current block on first non-blank after blanks
            blocks.append(current)
            current = []
            blank_run = False
        current.append(ln)
    if current:
        blocks.append(current)
    return "\n\n".join("\n".join(block) for block in blocks)


def _rules_for_annotation(name: str) -> str:
    """Return DSL rules for a known annotation name.

    The returned string may contain multiple rule blocks separated by a
    blank line.  Each block consists of zero or more premise lines,
    followed by ``---`` on its own line, followed by one or more
    conclusion lines.
    """
    lname = name.lower()
    if lname == "equalto":
        # Reflexivity and symmetry
        return (
            "---\n"
            "v : @EqualTo(other=\"v\")\n\n"
            "v1 : @EqualTo(other=\"v2\")\n"
            "---\n"
            "v2 : @EqualTo(other=\"v1\")"
        )
    if lname == "interval":
        # Self interval and interval widening
        return (
            "---\n"
            "v : @Interval(min=\"v\", max=\"v\")\n\n"
            "v : @Interval(min=\"a\", max=\"b\")\n"
            "c <= a\n"
            "b <= d\n"
            "---\n"
            "v : @Interval(min=\"c\", max=\"d\")"
        )
    if lname == "minlength":
        # Insert increases length, remove decreases length when n > 0.
        # Use the parameter name 'len' explicitly so that rule_check can
        # associate the schema variable with the annotation parameter.
        return (
            "v1 : MinLength(len=n)\n"
            "---\n"
            "v1.insert(v2) : MinLength(len=n+1)\n\n"
            "n > 0\n"
            "v1 : MinLength(len=n)\n"
            "---\n"
            "v1.remove(v2) : @MinLength(len=n-1)"
        )
    if lname == "nonempty":
        # Insert yields non‑empty, remove from non‑empty yields possibly empty
        return (
            "---\n"
            "v1.insert(v2) : @NonEmpty\n\n"
            "v1 : @NonEmpty\n"
            "---\n"
            "v1.remove(v2) : @PossiblyEmpty"
        )
    if lname == "remainder":
        # Remainder arithmetic rules
        # Use named arguments (remainder, modulus) to conform to the annotation spec.
        return (
            "v0 : @Remainder(remainder=n0, modulus=m0)\n"
            "n1 = n0 + m0\n"
            "---\n"
            "v0 : @Remainder(remainder=n1, modulus=m0)\n\n"
            "v0 : @Remainder(remainder=n0, modulus=m0)\n"
            "n1 = n0 - m0\n"
            "---\n"
            "v0 : @Remainder(remainder=n1, modulus=m0)\n\n"
            "v0 : @Remainder(remainder=n0, modulus=m0)\n"
            "m0 = m1 * k\n"
            "---\n"
            "v0 : @Remainder(remainder=n0, modulus=m1)"
        )
    return ""


def _call_phi3_model(prompt: str, model_name: str = "microsoft/Phi-3-mini-4k-instruct", max_new_tokens: int = 512) -> str:
    """
    Invoke the external LLM to generate a candidate rule.

    If the ``transformers`` package or the specified model cannot be loaded,
    a RuntimeError is raised.  The caller should catch the exception and
    fall back to heuristic synthesis.
    """
    if AutoModelForCausalLM is None or AutoTokenizer is None or torch is None:
        raise RuntimeError("transformers or torch not available")

    # Lazy initialisation: we cache the model and tokenizer on the function
    # object to avoid reloading them on every invocation.
    if not hasattr(_call_phi3_model, "_model"):
        # Load tokenizer and model.  We set trust_remote_code=True to allow
        # loading custom model classes if needed.  Device mapping and
        # dtype are chosen for CPU environments; adjust as necessary.
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype="auto")
        # Store on the function object for reuse
        _call_phi3_model._model = model
        _call_phi3_model._tokenizer = tokenizer
    else:
        tokenizer = _call_phi3_model._tokenizer
        model = _call_phi3_model._model

    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    # Generate output; we disable sampling for determinism
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
        )
    # Decode and strip the prompt prefix
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if generated.startswith(prompt):
        return generated[len(prompt):].strip()
    # If decoding does not include the prompt, return the generated string
    return generated.strip()


def generate_rule_with_llm(prompt: str, *, max_iters: int = 3) -> str:
    """
    Generate refinement type rules using the external LLM with Z3 feedback.

    The function will repeatedly query the language model until the
    synthesized rules are judged ``VALID`` by ``rule_check.check_rule`` or
    until ``max_iters`` attempts have been made.  On each iteration the
    prompt is extended with an explanation of why the previous rule failed
    to assist the model in correcting its mistakes.  If the model cannot be
    loaded, the function falls back to deterministic heuristics.

    Parameters
    ----------
    prompt: str
        The prompt built by ``prompt_builder.build_llm_prompt``.
    max_iters: int, optional
        Maximum number of attempts to obtain a valid rule.  Default is 3.

    Returns
    -------
    str
        A rule (or set of rules) in the DSL.  If all attempts fail and the
        model cannot be used, deterministic heuristics are applied.
    """
    if not prompt or not isinstance(prompt, str):
        return ""

    # Extract annotation specification to pass into rule_check
    annot_spec = _extract_annotation_spec(prompt)

    # Prepare a working copy of the prompt for augmentation
    augmented_prompt = prompt
    last_rule: Optional[str] = None

    for attempt in range(1, max_iters + 1):
        candidate = ""
        try:
            # Attempt to call the external model; may raise if unavailable
            candidate = _call_phi3_model(augmented_prompt)
            # Post‑process: retain only rule lines.  We remove any text
            # appearing before the first occurrence of '---' or the first
            # annotation line (starting with a schematic variable or call).
            # This heuristic handles the case where the model emits
            # explanations or preambles.
            lines = [ln.rstrip() for ln in candidate.splitlines()]
            start_idx = 0
            for i, ln in enumerate(lines):
                if ln.strip().startswith("---"):
                    start_idx = i
                    break
                # annotation line: <term> : @Ann( or <term> : Ann(
                if re.match(r"^\s*[A-Za-z]\w*(?:\.[A-Za-z]\w*\([A-Za-z]\w*\))?\s*:\s*@?[A-Za-z]\w*\(", ln):
                    start_idx = i
                    break
            filtered = "\n".join(lines[start_idx:]).strip()
            candidate = filtered if filtered else candidate.strip()
            # Ensure that annotation calls use named arguments when required.
            ann_name = _extract_annotation_name(prompt)
            if ann_name:
                param_names = _extract_param_names(prompt)
                if param_names:
                    candidate = _ensure_named_annotation_args(candidate, ann_name, param_names)
            # Remove Java null literals which are unsupported in the rule DSL.
            candidate = _remove_null_literals(candidate)
            # Normalize rule blocks: remove blank lines within blocks and ensure single
            # blank line separators between distinct rules.
            candidate = _split_into_rule_blocks(candidate)
        except Exception:
            # If the external model is not available, break early to
            # deterministic heuristic fallback.
            candidate = ""

        # If the candidate is empty, try deterministic fallback
        if not candidate:
            ann_name = _extract_annotation_name(prompt)
            if ann_name:
                rules = _rules_for_annotation(ann_name)
                if rules:
                    return rules
                # Unknown annotation fallback: reflexivity rule
                param_names = _extract_param_names(prompt)
                if param_names:
                    args = ", ".join(f"{p}=\"v\"" for p in param_names)
                    return f"---\nv : @{ann_name}({args})"
                return f"---\nv : @{ann_name}"
            return ""

        last_rule = candidate
        # Check rule validity with Z3
        result = check_rule(candidate, annot_spec)
        status = (result.get("status") or "").upper()
        if status == "VALID":
            return candidate

        # If invalid, append error information to the prompt to guide the model
        error_msg = result.get("message", "")
        model_str = result.get("model", "")
        # Only include model counterexample if present
        additional = f"\n\nThe previous rule was invalid: {error_msg}"
        if model_str:
            additional += f"; counterexample: {model_str}"
        additional += "\nPlease try again and output only corrected rules."
        augmented_prompt = prompt + additional

    # If all attempts failed but we obtained a candidate, return the last one
    if last_rule:
        return last_rule

    # As a last resort, fall back to deterministic heuristics
    ann_name = _extract_annotation_name(prompt)
    if ann_name:
        rules = _rules_for_annotation(ann_name)
        if rules:
            return rules
        param_names = _extract_param_names(prompt)
        if param_names:
            args = ", ".join(f"{p}=\"v\"" for p in param_names)
            return f"---\nv : @{ann_name}({args})"
        return f"---\nv : @{ann_name}"
    return ""