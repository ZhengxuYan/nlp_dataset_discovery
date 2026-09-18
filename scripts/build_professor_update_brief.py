#!/usr/bin/env python3
"""Build a short professor-facing update brief from current expansion artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_RUN_PLAN = Path("artifacts/corpus_expansion_2020_2025_run_plan.json")
DEFAULT_HYDRATION_STATUS = Path("artifacts/hydration_status_2020_2025.json")
DEFAULT_LOCAL_COVERAGE = Path("artifacts/local_smoke_2020_2025/metadata_coverage_2020_2025.json")
DEFAULT_LOCAL_SCHEMA = Path("artifacts/local_smoke_2020_2025/metadata_schema_audit_2020_2025.json")
DEFAULT_OUTPUT = Path("artifacts/professor_update_brief_2020_2025.md")
DEFAULT_ZH_OUTPUT = Path("artifacts/professor_update_brief_zh_2020_2025.md")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def pct(value: Any) -> str:
    try:
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def rate(rates: Mapping[str, Any], preferred: str, fallback: str) -> Any:
    return rates.get(preferred, rates.get(fallback))


def render_brief(
    run_plan: Mapping[str, Any],
    hydration_status: Mapping[str, Any],
    local_coverage: Mapping[str, Any],
    local_schema: Mapping[str, Any],
) -> str:
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    placeholder_count = hydration_status.get("placeholder_count", run_plan.get("placeholder_count", "unknown"))
    high_priority = (hydration_status.get("by_priority") or {}).get("high", "unknown")
    step_count = len(run_plan.get("steps") or [])
    rates = local_coverage.get("coverage_rates") or {}
    schema_rows = local_schema.get("rows", 0)
    return f"""# Professor Update Brief: 2020-2025 Dataset Discovery Expansion

Generated: {generated_at}

## Short Version
I have implemented the 2020-2025 expansion interface and the public-metadata enrichment layer. The pipeline now has a reproducible `{step_count}`-step run plan covering corpus expansion, integrated dataset/ACU banks, citation/download/star enrichment, stable-ID preservation auditing, metadata coverage reporting, metadata schema/provenance auditing, and a 2023-2025 overlap regression audit.

## Progress Since Last Update
- Generalized the 2023-2025 workflow to explicit 2020-2025 year/date parameters while preserving old 2023-2025 artifacts.
- Added public metadata enrichment using OpenAlex, Semantic Scholar, Hugging Face, GitHub, Papers with Code, and URL-health checks with cached API calls.
- Added scored title fuzzy matching for paper metadata fallback when DOI/arXiv identifiers are unavailable, with low-similarity matches rejected.
- Added dataset-name fallback for Hugging Face downloads and Papers with Code links when direct resource URLs are missing.
- Added cached URL-health validation for resource status, resolved URL, and downloadable flags.
- Added local fixture smoke evidence for the integrated bank -> enrichment -> coverage -> schema-audit path.
- Added a staged 2020-2021 smoke run plan to use before launching the full 2020-2025 run.
- Added coverage reporting for citation counts, OpenAlex/Semantic Scholar IDs, HF downloads, and GitHub stars.
- Added a stable-ID audit so enriched metadata is matched back to papers/datasets by record ID rather than row position.
- Added a metadata review sample for manually checking high/low citation papers and HF/GitHub-linked datasets.
- Added quality gates for metadata schema completeness, resource match provenance, and old-vs-new 2023-2025 overlap regression.

## Current Evidence
- Local smoke citation coverage: `{pct(rates.get('citation_count_pct'))}`
- Local smoke OpenAlex ID coverage: `{pct(rates.get('openalex_id_pct'))}`
- Local smoke Semantic Scholar ID coverage: `{pct(rates.get('semantic_scholar_id_pct'))}`
- Local smoke HF download coverage per HF resource: `{pct(rate(rates, 'hf_download_count_per_hf_resource_pct', 'hf_download_count_per_hf_link_pct'))}`
- Local smoke GitHub star coverage per GitHub resource: `{pct(rate(rates, 'github_star_count_per_github_resource_pct', 'github_star_count_per_github_link_pct'))}`
- Local smoke schema-audit rows: `{schema_rows}`

## Current Blocker
Full-data execution is gated on hydrating macOS cloud placeholder files. Current total placeholder count is `{placeholder_count}`, with `{high_priority}` high-priority files listed for first-pass hydration.

## Next Commands
```bash
python scripts/check_cloud_placeholders.py --paths-file artifacts/high_priority_hydration_files_2020_2025.txt --summary-only
python scripts/run_corpus_expansion_2020_2025.py --mode full --execute --allow-network-steps
```

## Key Artifacts
- `artifacts/hydration_status_2020_2025.md`
- `artifacts/corpus_expansion_2020_2021_smoke_run_plan.json`
- `artifacts/local_smoke_2020_2025/enrichment_stable_id_audit_2020_2025.md`
- `artifacts/local_smoke_2020_2025/metadata_coverage_2020_2025.md`
- `artifacts/local_smoke_2020_2025/metadata_schema_audit_2020_2025.md`
- `artifacts/local_smoke_2020_2025/metadata_review_sample_2020_2025.md`
- `artifacts/expansion_regression_audit_2020_2025.md`
- `artifacts/corpus_expansion_2020_2025_run_plan.json`
"""


def render_brief_zh(
    run_plan: Mapping[str, Any],
    hydration_status: Mapping[str, Any],
    local_coverage: Mapping[str, Any],
    local_schema: Mapping[str, Any],
) -> str:
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    placeholder_count = hydration_status.get("placeholder_count", run_plan.get("placeholder_count", "unknown"))
    high_priority = (hydration_status.get("by_priority") or {}).get("high", "unknown")
    step_count = len(run_plan.get("steps") or [])
    rates = local_coverage.get("coverage_rates") or {}
    schema_rows = local_schema.get("rows", 0)
    return f"""# 教授汇报简版：2020-2025 Dataset Discovery Expansion

生成时间：{generated_at}

## 一句话状态
我已经把原来偏 2023-2025 hardcoded 的流程改成了可复现的 2020-2025 expansion workflow，并加上了 public metadata enrichment。现在 pipeline 有一个 `{step_count}` 步 run plan，覆盖 corpus expansion、integrated dataset/ACU banks、citation/download/star enrichment、stable-ID preservation audit、metadata coverage report、metadata schema/provenance audit，以及 2023-2025 overlap regression audit。

## 本阶段完成的进展
- 把 2023-2025 workflow 参数化为明确的 2020-2025 year/date range，同时保留旧的 2023-2025 artifacts。
- 增加 public metadata enrichment：OpenAlex、Semantic Scholar、Hugging Face、GitHub、Papers with Code 和 URL health checks，并使用 cache 避免重复 API 请求。
- 增加 scored title fuzzy matching：当 DOI/arXiv identifier 缺失时用 title fallback，但会记录 confidence score，并拒绝低相似度 match。
- 增加 dataset-name fallback：当数据集没有直接 HF/Papers with Code URL 时，也能用数据集名称去补 HF downloads 和 PWC links，并记录 match confidence。
- 增加 cached URL-health validation，用于记录 resource status code、resolved URL 和 downloadable flag。
- 增加 local fixture smoke，验证 integrated bank -> enrichment -> coverage -> schema audit 这条链路可以本地跑通。
- 增加 2020-2021 staged smoke run plan，用于 full 2020-2025 run 前的小范围验证。
- 增加 citation count、OpenAlex/Semantic Scholar ID、HF downloads、GitHub stars 的 coverage report。
- 增加 stable-ID audit，确保 enriched metadata 是按 paper/dataset record ID 对齐，而不是按 row position 贴回去。
- 增加 metadata review sample，用于人工检查 high/low citation papers 和带 HF/GitHub 链接的数据集。
- 增加 metadata schema completeness、resource match provenance 和 old-vs-new 2023-2025 overlap regression 质量门。

## 当前证据
- Local smoke citation coverage: `{pct(rates.get('citation_count_pct'))}`
- Local smoke OpenAlex ID coverage: `{pct(rates.get('openalex_id_pct'))}`
- Local smoke Semantic Scholar ID coverage: `{pct(rates.get('semantic_scholar_id_pct'))}`
- Local smoke HF download coverage per HF resource: `{pct(rate(rates, 'hf_download_count_per_hf_resource_pct', 'hf_download_count_per_hf_link_pct'))}`
- Local smoke GitHub star coverage per GitHub resource: `{pct(rate(rates, 'github_star_count_per_github_resource_pct', 'github_star_count_per_github_link_pct'))}`
- Local smoke schema-audit rows: `{schema_rows}`

## 当前 blocker
full-data execution 现在只卡在本地 cloud placeholder hydration。当前总 placeholder 数是 `{placeholder_count}`，其中 `{high_priority}` 个 high-priority 文件需要优先下载/本地化。

## 下一步
```bash
python scripts/check_cloud_placeholders.py --paths-file artifacts/high_priority_hydration_files_2020_2025.txt --summary-only
python scripts/refresh_expansion_status_2020_2025.py
python scripts/validate_expansion_readiness.py --allow-blocked
```

high-priority hydration 清零后，先跑 2020-2021 staged smoke，再跑 full 2020-2025 pipeline：

```bash
python scripts/run_corpus_expansion_2020_2025.py --start-year 2020 --end-year 2021 --mode smoke --plan-output artifacts/corpus_expansion_2020_2021_smoke_run_plan.json
python scripts/run_corpus_expansion_2020_2025.py --mode full --execute --allow-network-steps
```

## 关键 artifacts
- `artifacts/hydration_action_guide_2020_2025.md`
- `artifacts/hydration_status_2020_2025.md`
- `artifacts/corpus_expansion_2020_2021_smoke_run_plan.json`
- `artifacts/local_smoke_2020_2025/enrichment_stable_id_audit_2020_2025.md`
- `artifacts/local_smoke_2020_2025/metadata_coverage_2020_2025.md`
- `artifacts/local_smoke_2020_2025/metadata_schema_audit_2020_2025.md`
- `artifacts/local_smoke_2020_2025/metadata_review_sample_2020_2025.md`
- `artifacts/expansion_regression_audit_2020_2025.md`
- `artifacts/corpus_expansion_2020_2025_run_plan.json`
"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a concise professor update brief.")
    parser.add_argument("--run-plan", type=Path, default=DEFAULT_RUN_PLAN)
    parser.add_argument("--hydration-status", type=Path, default=DEFAULT_HYDRATION_STATUS)
    parser.add_argument("--local-coverage", type=Path, default=DEFAULT_LOCAL_COVERAGE)
    parser.add_argument("--local-schema", type=Path, default=DEFAULT_LOCAL_SCHEMA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--zh-output", type=Path, default=DEFAULT_ZH_OUTPUT)
    parser.add_argument("--zh-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_plan = load_json(args.run_plan)
    hydration_status = load_json(args.hydration_status)
    local_coverage = load_json(args.local_coverage)
    local_schema = load_json(args.local_schema)
    if not args.zh_only:
        markdown = render_brief(run_plan, hydration_status, local_coverage, local_schema)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
        print(args.output)
    zh_markdown = render_brief_zh(run_plan, hydration_status, local_coverage, local_schema)
    args.zh_output.parent.mkdir(parents=True, exist_ok=True)
    args.zh_output.write_text(zh_markdown, encoding="utf-8")
    print(args.zh_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
