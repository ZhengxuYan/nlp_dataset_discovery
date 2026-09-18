#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_BIB_GZ = "data/census/acl_anthology/anthology+abstracts.bib.gz"
DEFAULT_CATALOG = "data/census/acl_anthology/acl_anthology_2023_2025_core.jsonl"
DEFAULT_OUTPUT_DIR = "data/census/acl_anthology"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_bib_text(path: str | Path) -> str:
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    return path.read_text(encoding="utf-8", errors="replace")


def iter_bib_entries(text: str):
    index = 0
    while True:
        at = text.find("@", index)
        if at < 0:
            return
        open_brace = text.find("{", at)
        if open_brace < 0:
            return
        entry_type = text[at + 1:open_brace].strip().lower()
        depth = 0
        pos = open_brace
        while pos < len(text):
            char = text[pos]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    yield entry_type, text[open_brace + 1:pos]
                    index = pos + 1
                    break
            pos += 1
        else:
            return


def parse_bib_fields(body: str) -> tuple[str, dict[str, str]]:
    comma = body.find(",")
    if comma < 0:
        return body.strip(), {}
    key = body[:comma].strip()
    rest = body[comma + 1:]
    fields: dict[str, str] = {}
    pos = 0
    while pos < len(rest):
        while pos < len(rest) and rest[pos] in " \n\r\t,":
            pos += 1
        name_start = pos
        while pos < len(rest) and (rest[pos].isalnum() or rest[pos] in "_-"):
            pos += 1
        if pos == name_start:
            break
        name = rest[name_start:pos].lower()
        while pos < len(rest) and rest[pos].isspace():
            pos += 1
        if pos >= len(rest) or rest[pos] != "=":
            break
        pos += 1
        while pos < len(rest) and rest[pos].isspace():
            pos += 1
        value, pos = parse_bib_value(rest, pos)
        fields[name] = clean_latex(value)
    return key, fields


def parse_bib_value(text: str, pos: int) -> tuple[str, int]:
    if pos >= len(text):
        return "", pos
    if text[pos] == "{":
        depth = 0
        start = pos + 1
        pos += 1
        while pos < len(text):
            char = text[pos]
            if char == "{":
                depth += 1
            elif char == "}":
                if depth == 0:
                    return text[start:pos], pos + 1
                depth -= 1
            pos += 1
        return text[start:], pos
    if text[pos] == '"':
        start = pos + 1
        pos += 1
        escaped = False
        while pos < len(text):
            char = text[pos]
            if char == '"' and not escaped:
                return text[start:pos], pos + 1
            escaped = char == "\\" and not escaped
            if char != "\\":
                escaped = False
            pos += 1
        return text[start:], pos
    start = pos
    while pos < len(text) and text[pos] not in ",\n":
        pos += 1
    return text[start:pos].strip(), pos


LATEX_REPLACEMENTS = {
    r"{\'e}": "é",
    r"{\`e}": "è",
    r"{\'a}": "á",
    r"{\`a}": "à",
    r"{\'i}": "í",
    r"{\'o}": "ó",
    r"{\"o}": "ö",
    r"{\"u}": "ü",
    r"{\~n}": "ñ",
    r"{\c c}": "ç",
    r"{'}": "'",
    "---": "-",
    "--": "-",
}


def clean_latex(value: str) -> str:
    value = value.replace("\n", " ")
    for src, dst in LATEX_REPLACEMENTS.items():
        value = value.replace(src, dst)
    value = re.sub(r"\\([&%$_#{}])", r"\1", value)
    value = re.sub(r"\\[a-zA-Z]+\s*", "", value)
    value = value.replace("{", "").replace("}", "")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def acl_id_from_url(url: str) -> str | None:
    match = re.search(r"aclanthology\.org/([^/\s]+)/?", url or "")
    if not match:
        return None
    acl_id = match.group(1)
    return acl_id.removesuffix(".pdf")


def build_bib_index(path: str | Path) -> dict[str, dict[str, Any]]:
    text = load_bib_text(path)
    index: dict[str, dict[str, Any]] = {}
    for entry_type, body in iter_bib_entries(text):
        key, fields = parse_bib_fields(body)
        acl_id = acl_id_from_url(fields.get("url", ""))
        if not acl_id:
            continue
        index[acl_id] = {
            "bib_key": key,
            "bib_entry_type": entry_type,
            "title": fields.get("title", ""),
            "abstract": fields.get("abstract", ""),
            "author": fields.get("author", ""),
            "booktitle": fields.get("booktitle", ""),
            "journal": fields.get("journal", ""),
            "year": fields.get("year", ""),
            "month": fields.get("month", ""),
            "doi": fields.get("doi", ""),
            "pages": fields.get("pages", ""),
            "url": fields.get("url", ""),
        }
    return index


def merge(catalog_rows: list[dict[str, Any]], bib_index: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    merged = []
    for row in catalog_rows:
        acl_id = row.get("acl_id")
        bib = bib_index.get(str(acl_id), {})
        merged.append({
            **row,
            "title": bib.get("title") or row.get("title") or "",
            "authors": bib.get("author") or row.get("authors") or "",
            "abstract": bib.get("abstract") or "",
            "booktitle": bib.get("booktitle") or "",
            "journal": bib.get("journal") or "",
            "doi": bib.get("doi") or "",
            "pages": bib.get("pages") or "",
            "bib_key": bib.get("bib_key") or "",
            "bib_entry_type": bib.get("bib_entry_type") or "",
            "has_abstract": bool(bib.get("abstract")),
            "metadata_source": "acl_anthology_bib_with_abstracts" if bib else row.get("source", "acl_anthology"),
        })
    return merged


def summarize(rows: list[dict[str, Any]], bib_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    missing = [row for row in rows if not row.get("bib_key")]
    no_abstract = [row for row in rows if not row.get("has_abstract")]
    return {
        "bib_entries_indexed": len(bib_index),
        "catalog_rows": len(rows),
        "matched_bib_entries": sum(1 for row in rows if row.get("bib_key")),
        "rows_with_abstract": sum(1 for row in rows if row.get("has_abstract")),
        "rows_without_abstract": len(no_abstract),
        "missing_bib_entries": len(missing),
        "rows_by_year": dict(sorted(Counter(row.get("year") for row in rows).items())),
        "abstract_coverage_by_year": dict(sorted(
            (year, {
                "rows": sum(1 for row in rows if row.get("year") == year),
                "with_abstract": sum(1 for row in rows if row.get("year") == year and row.get("has_abstract")),
            })
            for year in sorted({row.get("year") for row in rows})
        )),
        "missing_examples": [
            {"acl_id": row.get("acl_id"), "title": row.get("title"), "venue_prefix": row.get("venue_prefix")}
            for row in missing[:20]
        ],
        "no_abstract_examples": [
            {"acl_id": row.get("acl_id"), "title": row.get("title"), "venue_prefix": row.get("venue_prefix")}
            for row in no_abstract[:20]
        ],
    }


def markdown(summary: dict[str, Any], outputs: dict[str, str]) -> str:
    lines = [
        "# ACL Anthology Abstract Merge",
        "",
        f"- Bib entries indexed: {summary['bib_entries_indexed']}",
        f"- Catalog rows: {summary['catalog_rows']}",
        f"- Matched BibTeX entries: {summary['matched_bib_entries']}",
        f"- Rows with abstract: {summary['rows_with_abstract']}",
        f"- Rows without abstract: {summary['rows_without_abstract']}",
        "",
        "## Abstract Coverage By Year",
        "",
        "| Year | Rows | With Abstract |",
        "| --- | ---: | ---: |",
    ]
    for year, row in summary["abstract_coverage_by_year"].items():
        lines.append(f"| {year} | {row['rows']} | {row['with_abstract']} |")
    lines.extend(["", "## Outputs", ""])
    for label, path in outputs.items():
        lines.append(f"- {label}: `{path}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge ACL Anthology official abstracts BibTeX into a normalized ACL catalog.")
    parser.add_argument("--bib-gz", default=DEFAULT_BIB_GZ)
    parser.add_argument("--catalog-jsonl", default=DEFAULT_CATALOG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    catalog_path = Path(args.catalog_jsonl)
    name = args.name or catalog_path.stem + "_with_abstracts"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "jsonl": str(output_dir / f"{name}.jsonl"),
        "summary_json": str(output_dir / f"{name}_summary.json"),
        "summary_md": str(output_dir / f"{name}_summary.md"),
    }

    bib_index = build_bib_index(args.bib_gz)
    catalog_rows = read_jsonl(args.catalog_jsonl)
    merged_rows = merge(catalog_rows, bib_index)
    summary = summarize(merged_rows, bib_index)
    write_jsonl(outputs["jsonl"], merged_rows)
    write_json(outputs["summary_json"], summary)
    Path(outputs["summary_md"]).write_text(markdown(summary, outputs), encoding="utf-8")
    print(json.dumps({"summary": summary, "outputs": outputs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
