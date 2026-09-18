#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import requests


DEFAULT_CENSUS = "data/census/acl_gemini_flashlite_all.clean.jsonl"
DEFAULT_ACL_CORPUS = os.environ.get("ACL_ANTHOLOGY_CORPUS", "ACL-anthology-corpus")
DEFAULT_ERROR_JSONL = "data/census/acl_fulltext_materialization_errors.jsonl"
GROBID_URL = "http://localhost:8070/api/processFulltextDocument"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def acl_id_from_paper_id(paper_id: str) -> str:
    return paper_id.removeprefix("ACL:")


def tei_path_for(row: dict[str, Any], acl_corpus_dir: Path) -> Path:
    acl_id = acl_id_from_paper_id(str(row.get("paper_id") or ""))
    return acl_corpus_dir / "data" / "grobid_xml" / f"{acl_id}.tei.xml"


def pdf_path_for(row: dict[str, Any], acl_corpus_dir: Path) -> Path:
    acl_id = acl_id_from_paper_id(str(row.get("paper_id") or ""))
    return acl_corpus_dir / "data" / "pdfs" / f"{acl_id}.pdf"


def pdf_url_for(row: dict[str, Any]) -> str:
    acl_id = acl_id_from_paper_id(str(row.get("paper_id") or ""))
    return f"https://aclanthology.org/{acl_id}.pdf"


def positive_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("is_dataset_introducing") and row.get("datasets")]


def has_tei(row: dict[str, Any], acl_corpus_dir: Path) -> bool:
    path = tei_path_for(row, acl_corpus_dir)
    return path.exists() and path.stat().st_size > 500


def download_pdf(row: dict[str, Any], pdf_path: Path, timeout: int) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    if pdf_path.exists() and pdf_path.stat().st_size > 500:
        return
    url = pdf_url_for(row)
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        tmp_path = pdf_path.with_suffix(".pdf.tmp")
        with tmp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if chunk:
                    handle.write(chunk)
        tmp_path.replace(pdf_path)


def run_grobid(pdf_path: Path, tei_path: Path, timeout: int) -> None:
    tei_path.parent.mkdir(parents=True, exist_ok=True)
    if tei_path.exists() and tei_path.stat().st_size > 500:
        return
    with pdf_path.open("rb") as handle:
        response = requests.post(GROBID_URL, files={"input": handle}, timeout=timeout)
    response.raise_for_status()
    tmp_path = tei_path.with_suffix(".tei.xml.tmp")
    tmp_path.write_text(response.text, encoding="utf-8")
    tmp_path.replace(tei_path)


def materialize_one(
    row: dict[str, Any],
    *,
    acl_corpus_dir: Path,
    download_timeout: int,
    grobid_timeout: int,
    keep_pdf: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    started = time.monotonic()
    paper_id = row.get("paper_id")
    pdf_path = pdf_path_for(row, acl_corpus_dir)
    tei_path = tei_path_for(row, acl_corpus_dir)
    try:
        if has_tei(row, acl_corpus_dir):
            return {
                "paper_id": paper_id,
                "status": "already_has_tei",
                "tei_path": str(tei_path),
                "runtime_seconds": round(time.monotonic() - started, 3),
            }, None
        download_pdf(row, pdf_path, download_timeout)
        run_grobid(pdf_path, tei_path, grobid_timeout)
        if not keep_pdf and pdf_path.exists():
            pdf_path.unlink()
        return {
            "paper_id": paper_id,
            "status": "materialized",
            "tei_path": str(tei_path),
            "pdf_deleted": not keep_pdf,
            "runtime_seconds": round(time.monotonic() - started, 3),
        }, None
    except Exception as exc:  # noqa: BLE001
        if not keep_pdf and pdf_path.exists() and pdf_path.suffix == ".pdf":
            try:
                pdf_path.unlink()
            except OSError:
                pass
        return None, {
            "paper_id": paper_id,
            "title": row.get("title"),
            "pdf_url": pdf_url_for(row),
            "error": str(exc),
            "error_type": type(exc).__name__,
            "failed_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ACL PDFs, run local GROBID, and keep TEI XML for dataset-introducing papers.")
    parser.add_argument("--census-jsonl", default=DEFAULT_CENSUS)
    parser.add_argument("--acl-corpus-dir", default=DEFAULT_ACL_CORPUS)
    parser.add_argument("--error-jsonl", default=DEFAULT_ERROR_JSONL)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sample-mode", choices=["first", "random"], default="first")
    parser.add_argument("--sample-seed", type=int, default=17)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--download-timeout", type=int, default=60)
    parser.add_argument("--grobid-timeout", type=int, default=300)
    parser.add_argument("--keep-pdfs", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    acl_corpus_dir = Path(args.acl_corpus_dir)
    rows = positive_rows(read_jsonl(args.census_jsonl))
    total_positive = len(rows)
    missing_rows = [row for row in rows if not has_tei(row, acl_corpus_dir)]
    if args.sample_mode == "random":
        rng = random.Random(args.sample_seed)
        rng.shuffle(missing_rows)
    if args.offset:
        missing_rows = missing_rows[args.offset :]
    if args.limit is not None:
        missing_rows = missing_rows[: args.limit]

    status = {
        "census_jsonl": args.census_jsonl,
        "acl_corpus_dir": args.acl_corpus_dir,
        "total_positive_rows": total_positive,
        "missing_tei_rows_before_selection": len([row for row in rows if not has_tei(row, acl_corpus_dir)]),
        "selected_rows": len(missing_rows),
        "workers": args.workers,
        "keep_pdfs": args.keep_pdfs,
        "dry_run": args.dry_run,
    }
    print(json.dumps(status, indent=2))

    if args.dry_run:
        for row in missing_rows[:10]:
            print(json.dumps({
                "paper_id": row.get("paper_id"),
                "title": row.get("title"),
                "pdf_url": pdf_url_for(row),
                "pdf_path": str(pdf_path_for(row, acl_corpus_dir)),
                "tei_path": str(tei_path_for(row, acl_corpus_dir)),
            }, ensure_ascii=False))
        return

    completed = 0
    failed = 0
    runtime_total = 0.0
    if args.workers <= 1:
        for row in missing_rows:
            output, error = materialize_one(
                row,
                acl_corpus_dir=acl_corpus_dir,
                download_timeout=args.download_timeout,
                grobid_timeout=args.grobid_timeout,
                keep_pdf=args.keep_pdfs,
            )
            if output:
                completed += 1
                runtime_total += float(output.get("runtime_seconds") or 0)
            if error:
                failed += 1
                append_jsonl(args.error_jsonl, [error])
            print(json.dumps({
                "completed": completed,
                "failed": failed,
                "last_paper_id": row.get("paper_id"),
                "avg_runtime_seconds": round(runtime_total / completed, 3) if completed else None,
            }, indent=2))
        return

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                materialize_one,
                row,
                acl_corpus_dir=acl_corpus_dir,
                download_timeout=args.download_timeout,
                grobid_timeout=args.grobid_timeout,
                keep_pdf=args.keep_pdfs,
            )
            for row in missing_rows
        ]
        for future in as_completed(futures):
            output, error = future.result()
            if output:
                completed += 1
                runtime_total += float(output.get("runtime_seconds") or 0)
            if error:
                failed += 1
                append_jsonl(args.error_jsonl, [error])
            print(json.dumps({
                "completed": completed,
                "failed": failed,
                "avg_runtime_seconds": round(runtime_total / completed, 3) if completed else None,
            }, indent=2))


if __name__ == "__main__":
    main()
