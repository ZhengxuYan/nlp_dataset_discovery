"""
Auto-annotate attribution label agreement audit CSVs using an LLM.

For each row with empty `human_agrees_with_llm_label`, the LLM acts as the
annotator and fills in:
  - human_agrees_with_llm_label  (agree / disagree / uncertain)
  - human_corrected_label        (only when disagree)
  - human_label_note             (brief reasoning)
  - annotator_id                 (model slug, e.g. "gpt54mini")

Consistency is achieved via temperature=0 and a fixed seed.

Usage:
  python scripts/auto_annotate_attribution_label_agreement.py \
      --input  data/human_validation/attribution_label_agreement_audit/attribution_label_agreement_annotator_a.csv \
      --output data/human_validation/attribution_label_agreement_audit/attribution_label_agreement_annotator_a_llm.csv \
      --model  gpt-5.4-mini
"""

import os
import csv
import json
import time
import argparse

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert human annotator performing an attribution label agreement audit.
You review LLM-assigned labels and decide whether they are correct.

Label meanings
--------------
covered           : retrieved prior evidence already covers the core claim of the query DCU.
partially_covered : retrieved prior evidence overlaps with the query DCU but the query DCU
                    still has important differences (task, domain, language, modality, source,
                    annotation protocol, scale, release setting, or evaluation use).
not_covered       : none of the retrieved evidence covers or partially covers the query DCU.
not_comparable    : the query DCU and prior evidence are too vague, structurally different,
                    or incomparable to make a reliable judgment.
contradicted      : the retrieved prior evidence explicitly conflicts with the query DCU.

Decision rules
--------------
Choose "agree" when the LLM label is reasonable given the evidence shown.
Choose "disagree" only when there is a strong, concrete mismatch between evidence and label.
Choose "uncertain" only when the query DCU is too vague to evaluate.
When the label boundary is debatable (e.g., covered vs partially_covered), choose "agree".
When unsure, choose "agree".

Key signal: retrieval scores in real data are typically 0.15-0.35 (weak similarity).
At this score range, treat most "not_covered" labels as acceptable.
For llm_label = "not_covered", disagree ONLY if BOTH are true:
    (1) at least one prior DCU has retrieval_score >= 0.45, AND
    (2) that prior DCU clearly overlaps the same claim dimension (same task/domain,
            with specific overlap on scale/protocol/content).
Otherwise, prefer "agree".

Calibrated examples
-------------------
EXAMPLE 1 — agree (not_covered correct, low retrieval scores)
  Query DCU: "FinHNQue contains 20K pages with domain-specific hard negative queries." [scale/coverage]
  LLM label: not_covered
  Top scores: 0.20, 0.19, 0.18 — all about generic retrieval training sets, different tasks/domains.
  → AGREE. Low scores + different task; not_covered is correct.

EXAMPLE 2 — agree (partially_covered correct, one relevant prior at moderate score)
  Query DCU: "AfriMTEB covers 59 African languages across 14 tasks and 38 datasets." [scale/coverage]
  LLM label: partially_covered
  Top prior: MTEB (score=0.35) spans 8 tasks, 58 datasets, 112 languages — same benchmark
    format but different scale and no African focus.
  → AGREE. Genuine overlap on format but meaningful differences; partially_covered is correct.

EXAMPLE 3 — agree (not_covered correct, topic-related evidence only)
  Query DCU: "The benchmark has 29 Simple and 501 Difficult scenarios." [scale/coverage]
  LLM label: partially_covered
  Top prior: enterprise multi-agent dataset with 90 scenarios (score=0.13); LogRCA 80 failures (score=0.24).
  → AGREE. Categorical scenario structure gives some overlap; partially_covered is defensible.

EXAMPLE 4 — agree (not_covered correct even when numbers coincidentally match)
  Query DCU: "QuoteLink and QuoteTweet consist of 70K popular-unpopular sample pairs." [scale/coverage]
  LLM label: not_covered
  Top prior: ClueWeb22-MM has 20K evaluation queries; another dataset has 2M tweets.
  → AGREE. Scale numbers differ and domain/task is completely different; not_covered is correct.

EXAMPLE 5 — agree (partially_covered, rationale imperfect but label defensible)
  Query DCU: "The dataset covers 6 task types with 35 datasets." [scale/coverage]
  LLM label: partially_covered
  Top prior: MTEB covers 8 tasks and 58 datasets (score=0.31).
  → AGREE. Different numbers and scope, but same benchmark category; label is defensible even
    if the rationale is not perfectly worded.

EXAMPLE 6 — disagree (LLM says not_covered but evidence CLEARLY and DIRECTLY covers the claim)
  Query DCU: "The dataset contains 50,000 English question-answer pairs for open-domain QA." [scale/coverage]
  LLM label: not_covered
  Top prior: "NaturalQuestions: 307,373 English open-domain QA pairs." [score=0.71, same task+domain]
  LLM rationale: "No candidate matches the 50K scale."
  → DISAGREE (corrected: partially_covered). Score 0.71 + identical task+domain; scale difference
    alone does not make it not_covered. The LLM incorrectly ignored highly relevant evidence.

Important: judge only based on the information shown. Do not speculate about external prior work.
"""

ROW_PROMPT_TEMPLATE = """\
## Query DCU
- Text       : {query_dcu_text}
- Type       : {query_dcu_type}
- Importance : {query_dcu_importance}
- Evidence   : {query_dcu_evidence}
- Section    : {query_dcu_section}

## LLM Decision
- Label (llm_label)         : {llm_label}
- Raw support status        : {llm_raw_support_status}
- Delta type                : {llm_delta_type}
- Evidence adequacy         : {llm_evidence_adequacy}
- Missing prior risk        : {llm_missing_prior_risk}
- Selected prior DCU IDs    : {llm_selected_prior_dcu_ids}
- Rationale                 : {llm_rationale}

## Retrieved Prior Evidence (top candidates shown to the LLM)
{prior_evidence_summary}

---
Output a JSON object with EXACTLY these keys:

{{
  "human_agrees_with_llm_label": "<agree | disagree | uncertain>",
  "human_corrected_label": "<covered | partially_covered | not_covered | not_comparable | contradicted | >",
  "human_label_note": "<one concise sentence>"
}}

Rules:
- If "human_agrees_with_llm_label" is "agree", set "human_corrected_label" to "".
- If "human_agrees_with_llm_label" is "disagree", fill "human_corrected_label" with the correct label.
- If "human_agrees_with_llm_label" is "uncertain", set "human_corrected_label" to "".
- Keep "human_label_note" to ≤ 25 words.
- Do not target a quota for agree/disagree. Judge row-by-row.
"""

# ---------------------------------------------------------------------------
# Helper: summarise prior evidence JSON to keep prompt compact
# ---------------------------------------------------------------------------

def summarise_prior_evidence(raw_json_str: str, max_items: int = 5) -> str:
    try:
        items = json.loads(raw_json_str) if raw_json_str else []
    except (json.JSONDecodeError, TypeError):
        return raw_json_str or "(none)"

    lines = []
    for item in items[:max_items]:
        score = item.get("retrieval_score", "")
        score_str = f"{score:.3f}" if isinstance(score, float) else str(score)
        lines.append(
            f"  [{item.get('id', '')}] ({score_str}) "
            f"{item.get('dataset_name', '')} | {item.get('type', '')} | "
            f"{item.get('text', '')}"
        )
    if len(items) > max_items:
        lines.append(f"  ... ({len(items) - max_items} more not shown)")
    return "\n".join(lines) if lines else "(none)"


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_openai(prompt: str, model: str, seed: int = 42) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
        seed=seed,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Auto-annotate attribution label agreement CSVs with an LLM."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to the input CSV (e.g. attribution_label_agreement_annotator_a.csv)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to output CSV. Defaults to <input>_llm.csv alongside the input."
    )
    parser.add_argument(
        "--model", default="gpt-5.4-mini",
        help="Model name (default: gpt-5.4-mini)"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-annotate rows that already have a human label."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed passed to the API for reproducibility (default: 42)."
    )
    parser.add_argument(
        "--delay", type=float, default=0.3,
        help="Seconds to sleep between API calls (default: 0.3)."
    )
    args = parser.parse_args()

    # Resolve output path
    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_llm{ext}"

    model_slug = args.model.replace(".", "").replace("-", "")

    # Read input CSV
    with open(args.input, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    print(f"Loaded {len(rows)} rows from {args.input}")

    # Count rows to annotate
    to_annotate = [
        r for r in rows
        if args.overwrite or not str(r.get("human_agrees_with_llm_label", "")).strip()
    ]
    print(f"{len(to_annotate)} rows to annotate (model={args.model}, seed={args.seed})")

    if not to_annotate:
        print("Nothing to do. Use --overwrite to re-annotate existing labels.")
        return

    # Process rows
    errors = 0
    for idx, row in enumerate(to_annotate):
        prior_summary = summarise_prior_evidence(row.get("prior_evidence_for_review", ""))

        prompt = ROW_PROMPT_TEMPLATE.format(
            query_dcu_text      = row.get("query_dcu_text", ""),
            query_dcu_type      = row.get("query_dcu_type", ""),
            query_dcu_importance= row.get("query_dcu_importance", ""),
            query_dcu_evidence  = row.get("query_dcu_evidence", ""),
            query_dcu_section   = row.get("query_dcu_section", ""),
            llm_label           = row.get("llm_label", ""),
            llm_raw_support_status = row.get("llm_raw_support_status", ""),
            llm_delta_type      = row.get("llm_delta_type", ""),
            llm_evidence_adequacy  = row.get("llm_evidence_adequacy", ""),
            llm_missing_prior_risk = row.get("llm_missing_prior_risk", ""),
            llm_selected_prior_dcu_ids = row.get("llm_selected_prior_dcu_ids", ""),
            llm_rationale       = row.get("llm_rationale", ""),
            prior_evidence_summary = prior_summary,
        )

        max_retries = 3
        success = False
        for attempt in range(max_retries):
            try:
                raw = call_openai(prompt, args.model, seed=args.seed)
                result = json.loads(raw)

                agreement = str(result.get("human_agrees_with_llm_label", "")).strip()
                corrected = str(result.get("human_corrected_label", "")).strip()
                note      = str(result.get("human_label_note", "")).strip()

                # Validate agreement value
                if agreement not in {"agree", "disagree", "uncertain"}:
                    raise ValueError(f"Unexpected agreement value: {agreement!r}")

                row["human_agrees_with_llm_label"] = agreement
                row["human_corrected_label"]       = corrected if agreement == "disagree" else ""
                row["human_label_note"]            = note
                row["annotator_id"]                = model_slug

                print(
                    f"[{idx+1}/{len(to_annotate)}] {row.get('annotation_id','')} "
                    f"→ {agreement}"
                    + (f" (corrected: {corrected})" if agreement == "disagree" else "")
                )
                success = True
                break

            except Exception as e:
                print(f"  Attempt {attempt+1} failed for row {row.get('annotation_id','')}: {e}")
                time.sleep(2 ** attempt)

        if not success:
            errors += 1
            print(f"  SKIPPED row {row.get('annotation_id','')} after {max_retries} attempts.")

        time.sleep(args.delay)

    # Write output (preserve original column order, adding new ones if absent)
    extra_cols = ["human_agrees_with_llm_label", "human_corrected_label",
                  "human_label_note", "annotator_id"]
    out_fieldnames = list(fieldnames)
    for col in extra_cols:
        if col not in out_fieldnames:
            out_fieldnames.append(col)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. {len(to_annotate) - errors} annotated, {errors} skipped.")
    print(f"Output written to: {args.output}")


if __name__ == "__main__":
    main()
