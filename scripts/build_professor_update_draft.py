#!/usr/bin/env python3
"""Build a send-ready Chinese professor update draft for the expansion work."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_STATUS = Path("artifacts/expansion_status_packet_2020_2025.json")
DEFAULT_COMPLETION_AUDIT = Path("artifacts/expansion_completion_audit_2020_2025.json")
DEFAULT_HANDOFF = Path("artifacts/next_action_handoff_2020_2025.json")
DEFAULT_MEETING_PACKET = Path("artifacts/professor_meeting_packet_2020_2025.json")
DEFAULT_OUTPUT_JSON = Path("artifacts/professor_update_draft_zh_2020_2025.json")
DEFAULT_OUTPUT_MD = Path("artifacts/professor_update_draft_zh_2020_2025.md")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def pct(value: Any) -> str:
    try:
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def requirement_counts(completion: Mapping[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in completion.get("requirements") or []:
        if not isinstance(item, Mapping):
            continue
        status = str(item.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def build_draft(
    status: Mapping[str, Any],
    completion: Mapping[str, Any],
    handoff: Mapping[str, Any],
    meeting_packet: Mapping[str, Any],
) -> dict[str, Any]:
    done = meeting_packet.get("done_summary") or {}
    rates = meeting_packet.get("coverage_rates") or {}
    counts = requirement_counts(completion)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "subject": "2020-2025 dataset discovery expansion progress update",
        "status": status.get("status"),
        "current_gate": handoff.get("current_gate"),
        "placeholder_count": status.get("placeholder_count"),
        "high_priority_placeholder_count": status.get("high_priority_placeholder_count"),
        "artifact_count": status.get("artifact_count"),
        "missing_artifact_count": status.get("missing_artifact_count"),
        "run_plan_steps": status.get("run_plan_steps"),
        "local_smoke": {
            "papers": done.get("local_smoke_papers"),
            "datasets": done.get("local_smoke_datasets"),
            "acus": done.get("local_smoke_acus"),
            "review_sample_count": done.get("review_sample_count"),
            "stable_id_audit_status": done.get("stable_id_audit_status"),
            "full_plan_url_health_enabled": done.get("full_plan_url_health_enabled"),
        },
        "coverage_rates": {
            "citation_count_pct": rates.get("citation_count_pct"),
            "openalex_id_pct": rates.get("openalex_id_pct"),
            "semantic_scholar_id_pct": rates.get("semantic_scholar_id_pct"),
            "hf_download_count_per_hf_resource_pct": rates.get("hf_download_count_per_hf_resource_pct"),
            "github_star_count_per_github_resource_pct": rates.get("github_star_count_per_github_resource_pct"),
            "healthy_url_pct": rates.get("healthy_url_pct"),
        },
        "completion_requirement_counts": counts,
        "next_commands": handoff.get("next_commands") or [],
        "supporting_artifacts": {
            "meeting_packet": "artifacts/professor_meeting_packet_2020_2025.md",
            "next_action_handoff": "artifacts/next_action_handoff_2020_2025.md",
            "completion_audit": "artifacts/expansion_completion_audit_2020_2025.md",
            "artifact_index": "artifacts/expansion_artifact_index_2020_2025.md",
        },
    }


def render_markdown(draft: Mapping[str, Any]) -> str:
    smoke = draft.get("local_smoke") or {}
    rates = draft.get("coverage_rates") or {}
    counts = draft.get("completion_requirement_counts") or {}
    artifacts = draft.get("supporting_artifacts") or {}
    lines = [
        "# 教授 Update Draft: 2020-2025 Dataset Discovery Expansion",
        "",
        f"- Generated: {draft.get('generated_at')}",
        f"- Suggested subject: {draft.get('subject')}",
        "",
        "## 可直接发送版本",
        "",
        "老师您好，我这边已经把 dataset discovery 的 expansion 和 enrichment 部分推进到一个可汇报的状态。",
        "",
        (
            f"目前我已经把原先偏 2023-2025 的流程参数化成 2020-2025 workflow，生成了 `{draft.get('run_plan_steps')}` 步 run plan，"
            "并且保留旧的 2023-2025 artifacts，不会覆盖之前结果。新的流程里也加了 public metadata enrichment："
            "paper 层面包括 DOI/arXiv/ACL/OpenAlex/Semantic Scholar identifiers、citation counts、venue/authors 等；"
            "dataset/resource 层面包括 Hugging Face downloads/likes/tags/license、GitHub stars/forks/issues/license、"
            "Papers with Code links，以及 URL health/status/resolved/downloadable checks。"
        ),
        "",
        (
            f"我已经做了一个本地 smoke validation，覆盖 `{smoke.get('papers')}` 篇 paper、`{smoke.get('datasets')}` 个 dataset、"
            f"`{smoke.get('acus')}` 个 ACU。local smoke 里 citation/OpenAlex/Semantic Scholar coverage 都是 "
            f"`{pct(rates.get('citation_count_pct'))}`/`{pct(rates.get('openalex_id_pct'))}`/`{pct(rates.get('semantic_scholar_id_pct'))}`，"
            f"GitHub stars coverage 是 `{pct(rates.get('github_star_count_per_github_resource_pct'))}`，URL health 是 `{pct(rates.get('healthy_url_pct'))}`。"
            f"我也加了 stable-ID audit，结果是 `{smoke.get('stable_id_audit_status')}`，确保 enrichment 是按 paper/dataset ID merge，"
            "不是按 row position 贴回去。"
        ),
        "",
        (
            f"现在 completion audit 里有 `{counts.get('done', 0)}` 项已经完成，"
            f"`{counts.get('ready_after_hydration', 0)}` 项 hydration 后可以继续，"
            f"`{counts.get('ready_after_full_run', 0)}` 项 full run 后验证，"
            f"`{counts.get('blocked_by_hydration', 0)}` 项当前被 hydration blocker 卡住。"
            f"当前唯一主要 blocker 是本地 cloud placeholder files：总数 `{draft.get('placeholder_count')}`，"
            f"其中 `{draft.get('high_priority_placeholder_count')}` 个 high-priority 文件需要先下载到本地。"
        ),
        "",
        "下一步我会先清掉 high-priority hydration queue，然后跑 readiness gate；通过后再跑 2020-2021 staged smoke 和 full 2020-2025 pipeline。相关 evidence packet、completion audit 和 next-action handoff 我已经整理好了，可以直接给您看。",
        "",
        "## 当前下一步命令",
        "",
        "```bash",
        *draft.get("next_commands", []),
        "```",
        "",
        "## Supporting Artifacts",
        "",
    ]
    for label, path in artifacts.items():
        lines.append(f"- `{label}`: `{path}`")
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a send-ready professor update draft.")
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--completion-audit", type=Path, default=DEFAULT_COMPLETION_AUDIT)
    parser.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--meeting-packet", type=Path, default=DEFAULT_MEETING_PACKET)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    draft = build_draft(
        load_json(args.status),
        load_json(args.completion_audit),
        load_json(args.handoff),
        load_json(args.meeting_packet),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(draft, indent=2, sort_keys=True), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(draft), encoding="utf-8")
    print(args.output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
