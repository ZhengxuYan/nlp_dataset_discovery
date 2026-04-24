#!/usr/bin/env python3
import argparse
import cgi
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from scv.benchmark_models import JobRecord
from scv.benchmark_store import (
    DEFAULT_JOB_QUEUE_PATH,
    canonical_paper_id,
    load_benchmark_drafts,
    load_job_queue,
    load_paper_bank,
    load_previous_work_candidates,
    load_processed_query_papers,
    now_iso,
    summarize_query_status,
    upsert_job_queue,
)
from scv.prior_work_builder import (
    attach_pdf_to_paper,
    create_arxiv_paper_record,
    create_manual_paper_record,
    link_candidate_to_paper,
    patch_candidate,
    patch_draft,
    patch_paper,
    serialize_query_view,
)


ROOT = Path(__file__).resolve().parents[1]
HIGH_PRIORITY_COMPLETION_TASKS_PATH = ROOT / "data" / "benchmark" / "high_priority_prior_completion_tasks.jsonl"
PRIOR_COMPLETION_TASKS_PATH = ROOT / "data" / "benchmark" / "prior_completion_tasks.jsonl"
QUEUE_LOCK = threading.Lock()
LLM_MODEL = "gpt-5.4"
LLM_BACKEND = None
LLM_BACKEND_PARAMS = None


def tail_text(text: str, limit: int = 4000) -> str:
    return text[-limit:] if text else ""


def update_job(job_id: str, **patch) -> JobRecord:
    with QUEUE_LOCK:
        jobs = load_job_queue()
        updated = None
        rows = []
        for job in jobs:
            if job.job_id == job_id:
                data = job.model_dump()
                data.update(patch)
                updated = JobRecord(**data)
                rows.append(updated)
            else:
                rows.append(job)
        if updated is None:
            raise KeyError(f"Job {job_id} not found")
        upsert_job_queue(rows)
        return updated


def enqueue_job(job_type: str, query_paper_id=None, paper_ids=None) -> JobRecord:
    job = JobRecord(
        job_id=f"job-{uuid.uuid4().hex[:12]}",
        job_type=job_type,
        query_paper_id=query_paper_id,
        paper_ids=paper_ids or [],
        created_at=now_iso(),
    )
    with QUEUE_LOCK:
        upsert_job_queue([job])
    return job


def mark_stale_running_jobs() -> None:
    jobs = load_job_queue()
    changed = []
    for job in jobs:
        if job.status == "running":
            changed.append(JobRecord(**{
                **job.model_dump(),
                "status": "stale",
                "finished_at": now_iso(),
                "error_message": "Server restarted before job completion.",
            }))
        else:
            changed.append(job)
    if changed:
        upsert_job_queue(changed)


def add_llm_args(command):
    if LLM_MODEL:
        command.extend(["--model", LLM_MODEL])
    if LLM_BACKEND:
        command.extend(["--backend", LLM_BACKEND])
    if LLM_BACKEND_PARAMS:
        command.extend(["--backend-params", LLM_BACKEND_PARAMS])
    return command


def build_command(job: JobRecord, query_input: str):
    if job.job_type == "extract_candidates":
        cmd = add_llm_args(["python3", "scripts/extract_previous_work_candidates.py", "--input", query_input])
        if job.query_paper_id:
            cmd.extend(["--paper-ids", job.query_paper_id])
        return [cmd]
    if job.job_type == "resolve_candidates":
        cmd = ["python3", "scripts/resolve_previous_work_candidates.py"]
        if job.query_paper_id:
            candidate_ids = [
                candidate.candidate_id
                for candidate in load_previous_work_candidates()
                if candidate.query_paper_id == job.query_paper_id
            ]
            if candidate_ids:
                cmd.extend(["--candidate-ids", *candidate_ids])
        return [cmd]
    if job.job_type == "fetch_papers":
        cmd = ["python3", "scripts/fetch_prior_papers.py"]
        if job.paper_ids:
            cmd.extend(["--paper-ids", *job.paper_ids])
        elif job.query_paper_id:
            paper_ids = sorted({
                candidate.resolved_paper_id
                for candidate in load_previous_work_candidates()
                if candidate.query_paper_id == job.query_paper_id and candidate.resolved_paper_id
            })
            if paper_ids:
                cmd.extend(["--paper-ids", *paper_ids])
        return [cmd]
    if job.job_type == "process_papers":
        cmd = add_llm_args(["python3", "scripts/process_prior_papers.py"])
        if job.paper_ids:
            cmd.extend(["--paper-ids", *job.paper_ids])
        elif job.query_paper_id:
            paper_ids = sorted({
                candidate.resolved_paper_id
                for candidate in load_previous_work_candidates()
                if candidate.query_paper_id == job.query_paper_id and candidate.resolved_paper_id
            })
            if paper_ids:
                cmd.extend(["--paper-ids", *paper_ids])
        return [cmd]
    if job.job_type == "build_drafts":
        cmd = ["python3", "scripts/build_benchmark_drafts.py", "--queries", query_input]
        if job.query_paper_id:
            cmd.extend(["--paper-ids", job.query_paper_id])
        return [cmd]
    if job.job_type == "bulk_extract_missing":
        query_rows = load_processed_query_papers(query_input)
        candidates_by_query = {candidate.query_paper_id for candidate in load_previous_work_candidates()}
        missing = [
            canonical_paper_id(row.get("arxiv_id"), row.get("title", ""))
            for row in query_rows
            if canonical_paper_id(row.get("arxiv_id"), row.get("title", "")) not in candidates_by_query
        ]
        return [add_llm_args(["python3", "scripts/extract_previous_work_candidates.py", "--input", query_input, "--paper-ids", *missing])] if missing else []
    if job.job_type == "bulk_resolve_all":
        unresolved = [
            candidate.candidate_id for candidate in load_previous_work_candidates()
            if candidate.resolution_status in {"needs_resolution", "ambiguous", "unresolved"}
        ]
        return [["python3", "scripts/resolve_previous_work_candidates.py", "--candidate-ids", *unresolved]] if unresolved else []
    if job.job_type == "bulk_fetch_all":
        paper_ids = [
            paper.paper_id for paper in load_paper_bank()
            if paper.status in {"resolved_metadata", "fetch_failed"}
        ]
        return [["python3", "scripts/fetch_prior_papers.py", "--paper-ids", *paper_ids]] if paper_ids else []
    if job.job_type == "bulk_process_all":
        paper_ids = [
            paper.paper_id for paper in load_paper_bank()
            if paper.status == "fetched"
        ]
        return [add_llm_args(["python3", "scripts/process_prior_papers.py", "--paper-ids", *paper_ids])] if paper_ids else []
    if job.job_type == "bulk_build_all":
        return [["python3", "scripts/build_benchmark_drafts.py", "--queries", query_input]]
    raise ValueError(f"Unknown job type: {job.job_type}")


def run_command(command):
    return subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True)


def load_prior_completion_tasks():
    task_path = HIGH_PRIORITY_COMPLETION_TASKS_PATH if HIGH_PRIORITY_COMPLETION_TASKS_PATH.exists() else PRIOR_COMPLETION_TASKS_PATH
    if not task_path.exists():
        return []
    rows = []
    with open(task_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def extract_arxiv_id(value: str):
    value = (value or "").strip()
    if not value:
        return None
    match = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)", value, re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"\b([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)\b", value)
    if match:
        return match.group(1)
    return None


def run_pipeline_for_query(job: JobRecord, query_input: str):
    combined_stdout = []
    combined_stderr = []

    commands = [
        add_llm_args(["python3", "scripts/extract_previous_work_candidates.py", "--input", query_input, "--paper-ids", job.query_paper_id]),
    ]
    for command in commands:
        result = run_command(command)
        combined_stdout.append(result.stdout)
        combined_stderr.append(result.stderr)
        if result.returncode != 0:
            return result.returncode, combined_stdout, combined_stderr, command

    candidate_ids = [
        candidate.candidate_id
        for candidate in load_previous_work_candidates()
        if candidate.query_paper_id == job.query_paper_id
    ]
    if candidate_ids:
        result = run_command(["python3", "scripts/resolve_previous_work_candidates.py", "--candidate-ids", *candidate_ids])
        combined_stdout.append(result.stdout)
        combined_stderr.append(result.stderr)
        if result.returncode != 0:
            return result.returncode, combined_stdout, combined_stderr, ["python3", "scripts/resolve_previous_work_candidates.py", "--candidate-ids", *candidate_ids]

    paper_ids = sorted({
        candidate.resolved_paper_id
        for candidate in load_previous_work_candidates()
        if candidate.query_paper_id == job.query_paper_id and candidate.resolved_paper_id
    })
    if paper_ids:
        result = run_command(["python3", "scripts/fetch_prior_papers.py", "--paper-ids", *paper_ids])
        combined_stdout.append(result.stdout)
        combined_stderr.append(result.stderr)
        if result.returncode != 0:
            return result.returncode, combined_stdout, combined_stderr, ["python3", "scripts/fetch_prior_papers.py", "--paper-ids", *paper_ids]

        process_command = add_llm_args(["python3", "scripts/process_prior_papers.py", "--paper-ids", *paper_ids])
        result = run_command(process_command)
        combined_stdout.append(result.stdout)
        combined_stderr.append(result.stderr)
        if result.returncode != 0:
            return result.returncode, combined_stdout, combined_stderr, process_command

    result = run_command(["python3", "scripts/build_benchmark_drafts.py", "--queries", query_input, "--paper-ids", job.query_paper_id])
    combined_stdout.append(result.stdout)
    combined_stderr.append(result.stderr)
    return result.returncode, combined_stdout, combined_stderr, ["python3", "scripts/build_benchmark_drafts.py", "--queries", query_input, "--paper-ids", job.query_paper_id]


def job_worker(query_input: str):
    while True:
        time.sleep(1)
        with QUEUE_LOCK:
            jobs = load_job_queue()
            queued = [job for job in jobs if job.status == "queued"]
            if not queued:
                continue
            job = queued[0]
        update_job(job.job_id, status="running", started_at=now_iso())
        try:
            if job.job_type == "run_pipeline_for_query":
                returncode, combined_stdout, combined_stderr, failed_command = run_pipeline_for_query(job, query_input)
                if returncode != 0:
                    update_job(
                        job.job_id,
                        status="failed",
                        finished_at=now_iso(),
                        stdout_tail=tail_text("\n".join(combined_stdout)),
                        stderr_tail=tail_text("\n".join(combined_stderr)),
                        error_message=f"Command failed with exit code {returncode}: {' '.join(failed_command)}",
                    )
                    continue
                update_job(
                    job.job_id,
                    status="succeeded",
                    finished_at=now_iso(),
                    stdout_tail=tail_text("\n".join(combined_stdout)),
                    stderr_tail=tail_text("\n".join(combined_stderr)),
                )
                continue

            commands = build_command(job, query_input)
            if not commands:
                update_job(job.job_id, status="succeeded", started_at=now_iso(), finished_at=now_iso(), stdout_tail="No-op job: nothing to process.")
                continue
            combined_stdout = []
            combined_stderr = []
            for command in commands:
                result = run_command(command)
                combined_stdout.append(result.stdout)
                combined_stderr.append(result.stderr)
                if result.returncode != 0:
                    update_job(
                        job.job_id,
                        status="failed",
                        finished_at=now_iso(),
                        stdout_tail=tail_text("\n".join(combined_stdout)),
                        stderr_tail=tail_text("\n".join(combined_stderr)),
                        error_message=f"Command failed with exit code {result.returncode}: {' '.join(command)}",
                    )
                    break
            else:
                update_job(
                    job.job_id,
                    status="succeeded",
                    finished_at=now_iso(),
                    stdout_tail=tail_text("\n".join(combined_stdout)),
                    stderr_tail=tail_text("\n".join(combined_stderr)),
                )
        except Exception as exc:
            update_job(
                job.job_id,
                status="failed",
                finished_at=now_iso(),
                error_message=str(exc),
            )


class BenchmarkBuilderHandler(BaseHTTPRequestHandler):
    query_input = "data/processed/final_scv_200.jsonl"

    def _send_json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,PATCH,OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    def do_OPTIONS(self):
        self._send_json({}, status=HTTPStatus.NO_CONTENT)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/queries":
            self.handle_queries()
            return
        if parsed.path == "/api/jobs":
            self.handle_jobs()
            return
        if parsed.path.startswith("/api/query/"):
            self.handle_query_detail(unquote(parsed.path[len("/api/query/"):]))
            return
        if parsed.path.startswith("/pdf/"):
            self.handle_pdf(unquote(parsed.path[len("/pdf/"):]))
            return
        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_PATCH(self):
        parsed = urlparse(self.path)
        body = self._read_json()
        if parsed.path.startswith("/api/candidate/"):
            row = patch_candidate(unquote(parsed.path[len("/api/candidate/"):]), body)
            self._send_json(row.model_dump())
            return
        if parsed.path.startswith("/api/draft/"):
            row = patch_draft(unquote(parsed.path[len("/api/draft/"):]), body)
            self._send_json(row.model_dump())
            return
        if parsed.path.startswith("/api/paper/"):
            row = patch_paper(unquote(parsed.path[len("/api/paper/"):]), body)
            self._send_json(row.model_dump())
            return
        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/upload-pdf":
            self.handle_upload_pdf()
            return
        body = self._read_json()
        if parsed.path == "/api/run-job":
            self.handle_run_job(body)
            return
        if parsed.path == "/api/run-bulk-job":
            self.handle_run_bulk_job(body)
            return
        if parsed.path == "/api/create-paper-record":
            self.handle_create_paper_record(body)
            return
        if parsed.path == "/api/link-arxiv":
            self.handle_link_arxiv(body)
            return
        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def handle_queries(self):
        query_rows = load_processed_query_papers(self.query_input)
        drafts = {row.query_paper_id: row for row in load_benchmark_drafts()}
        candidates = load_previous_work_candidates()
        jobs = load_job_queue()
        priority_tasks = load_prior_completion_tasks()
        priority_tasks_by_query = {}
        priority_task_names_by_query = {}
        for task in priority_tasks:
            priority_tasks_by_query.setdefault(task.get("query_paper_id"), 0)
            priority_tasks_by_query[task.get("query_paper_id")] += 1
            priority_task_names_by_query.setdefault(task.get("query_paper_id"), [])
            task_name = task.get("candidate_name") or task.get("reference_title") or task.get("candidate_id")
            if task_name:
                priority_task_names_by_query[task.get("query_paper_id")].append(task_name)
        running_by_query = {}
        for job in jobs:
            if job.status in {"queued", "running"} and job.query_paper_id:
                running_by_query[job.query_paper_id] = job.status
        candidates_by_query = {}
        for candidate in candidates:
            candidates_by_query.setdefault(candidate.query_paper_id, 0)
            candidates_by_query[candidate.query_paper_id] += 1

        items = []
        for row in query_rows:
            paper_id = canonical_paper_id(row.get("arxiv_id"), row.get("title", ""))
            introduced = [dataset.get("info", dataset) for dataset in row.get("datasets", []) if dataset.get("info", dataset).get("is_introduced")]
            items.append({
                "paper_id": paper_id,
                "title": row.get("metadata", {}).get("title", row.get("title", "")),
                "arxiv_id": row.get("arxiv_id"),
                "query_dataset_names": [dataset.get("name", "") for dataset in introduced],
                "candidate_count": candidates_by_query.get(paper_id, 0),
                "priority_task_count": priority_tasks_by_query.get(paper_id, 0),
                "priority_task_names": priority_task_names_by_query.get(paper_id, []),
                "status": summarize_query_status(paper_id),
                "job_status": running_by_query.get(paper_id),
            })

        unresolved_candidates = sum(1 for c in candidates if c.resolution_status in {"needs_resolution", "ambiguous", "unresolved"})
        papers = load_paper_bank()
        counts = {
            "llm_model": LLM_MODEL,
            "needs_extraction": sum(1 for item in items if item["status"] == "needs_extraction"),
            "ready_for_review": sum(1 for item in items if item["status"] == "needs_annotation"),
            "complete": sum(1 for item in items if item["status"] == "complete"),
            "unresolved_candidates": unresolved_candidates,
            "waiting_fetch": sum(1 for p in papers if p.status in {"resolved_metadata", "fetch_failed"}),
            "waiting_processing": sum(1 for p in papers if p.status == "fetched"),
            "priority_attention_rows": sum(1 for item in items if item["priority_task_count"] > 0 and item["status"] != "complete"),
        }
        self._send_json({"items": items, "counts": counts})

    def handle_jobs(self):
        jobs = [job.model_dump() for job in load_job_queue()]
        self._send_json({"items": jobs})

    def handle_query_detail(self, paper_id: str):
        payload = serialize_query_view(paper_id, self.query_input)
        if payload is None:
            self._send_json({"error": "Query paper not found"}, status=HTTPStatus.NOT_FOUND)
            return
        payload["priority_tasks"] = [
            task for task in load_prior_completion_tasks()
            if task.get("query_paper_id") == paper_id
        ]
        payload["jobs"] = [
            job.model_dump()
            for job in load_job_queue()
            if job.query_paper_id == paper_id and job.status in {"queued", "running", "failed"}
        ]
        self._send_json(payload)

    def handle_pdf(self, relative_path: str):
        file_path = (ROOT / relative_path).resolve()
        if not str(file_path).startswith(str((ROOT / "data/benchmark/pdfs").resolve())) or not file_path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "PDF not found")
            return
        data = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/pdf")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def handle_run_job(self, body):
        job = enqueue_job(body.get("job"), query_paper_id=body.get("paper_id"), paper_ids=body.get("paper_ids"))
        self._send_json(job.model_dump(), status=HTTPStatus.ACCEPTED)

    def handle_run_bulk_job(self, body):
        job = enqueue_job(body.get("job"))
        self._send_json(job.model_dump(), status=HTTPStatus.ACCEPTED)

    def handle_create_paper_record(self, body):
        paper = create_manual_paper_record(
            title=body.get("title") or "Untitled prior paper",
            query_paper_id=body.get("query_paper_id"),
            year=body.get("year"),
            authors=body.get("authors") or [],
            notes=body.get("notes"),
        )
        candidate_id = body.get("candidate_id")
        if candidate_id:
            link_candidate_to_paper(candidate_id, paper.paper_id)
        self._send_json(paper.model_dump(), status=HTTPStatus.CREATED)

    def handle_link_arxiv(self, body):
        arxiv_id = extract_arxiv_id(body.get("arxiv") or body.get("arxiv_id") or body.get("url") or "")
        if not arxiv_id:
            self._send_json({"error": "Could not parse arXiv ID. Use formats like 1611.09268 or https://arxiv.org/abs/1611.09268"}, status=HTTPStatus.BAD_REQUEST)
            return
        title = body.get("title") or body.get("candidate_name") or arxiv_id
        paper = create_arxiv_paper_record(
            arxiv_id=arxiv_id,
            title=title,
            year=body.get("year"),
            authors=body.get("authors") or [],
            notes=body.get("notes") or "Linked manually from arXiv input.",
        )
        candidate = None
        candidate_id = body.get("candidate_id")
        if candidate_id:
            candidate = link_candidate_to_paper(candidate_id, paper.paper_id)
        job = enqueue_job("fetch_papers", paper_ids=[paper.paper_id]) if body.get("fetch", True) else None
        self._send_json({
            "paper": paper.model_dump(),
            "candidate": candidate.model_dump() if candidate else None,
            "job": job.model_dump() if job else None,
        }, status=HTTPStatus.CREATED)

    def handle_upload_pdf(self):
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type"),
            },
        )
        candidate_id = form.getvalue("candidate_id")
        paper_id = form.getvalue("paper_id")
        title = form.getvalue("title")
        query_paper_id = form.getvalue("query_paper_id")
        if not paper_id:
            paper = create_manual_paper_record(title=title or "Manual prior paper", query_paper_id=query_paper_id, notes="Created via PDF upload")
            paper_id = paper.paper_id
        file_item = form["file"] if "file" in form else None
        if file_item is None or not getattr(file_item, "filename", ""):
            self._send_json({"error": "Missing PDF upload"}, status=HTTPStatus.BAD_REQUEST)
            return
        if not file_item.filename.lower().endswith(".pdf"):
            self._send_json({"error": "Only PDF upload is supported"}, status=HTTPStatus.BAD_REQUEST)
            return
        temp_path = ROOT / "data" / "benchmark" / "_upload_temp.pdf"
        with open(temp_path, "wb") as handle:
            handle.write(file_item.file.read())
        paper = attach_pdf_to_paper(paper_id, str(temp_path))
        temp_path.unlink(missing_ok=True)
        if candidate_id:
            link_candidate_to_paper(candidate_id, paper.paper_id)
        self._send_json(paper.model_dump(), status=HTTPStatus.CREATED)


def main() -> None:
    global LLM_MODEL, LLM_BACKEND, LLM_BACKEND_PARAMS
    load_dotenv()
    parser = argparse.ArgumentParser(description="Serve the local benchmark-builder review API.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--queries", type=str, default="data/processed/final_scv_200.jsonl")
    parser.add_argument("--model", type=str, default=os.environ.get("BENCHMARK_BUILDER_MODEL", "gpt-5.4"))
    parser.add_argument("--backend", type=str, default=os.environ.get("BACKEND"))
    parser.add_argument("--backend-params", type=str, default=os.environ.get("BACKEND_PARAMS"))
    args = parser.parse_args()

    LLM_MODEL = args.model
    LLM_BACKEND = args.backend
    LLM_BACKEND_PARAMS = args.backend_params

    mark_stale_running_jobs()
    worker = threading.Thread(target=job_worker, args=(args.queries,), daemon=True)
    worker.start()

    BenchmarkBuilderHandler.query_input = args.queries
    server = ThreadingHTTPServer((args.host, args.port), BenchmarkBuilderHandler)
    print(f"Serving benchmark builder API on http://{args.host}:{args.port} with LLM model {LLM_MODEL}")
    server.serve_forever()


if __name__ == "__main__":
    main()
