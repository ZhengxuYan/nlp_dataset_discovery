from scripts import refresh_expansion_status_2020_2025


def test_build_steps_includes_local_smoke_by_default():
    steps = refresh_expansion_status_2020_2025.build_steps()
    names = [step.name for step in steps]

    assert names[0] == "smoke_run_plan"
    assert names[1] == "full_run_plan"
    assert names[2] == "local_smoke"
    assert names[3] == "staged_2020_2021_smoke_plan"
    assert "local_smoke" in names
    assert names.index("remaining_hydration_queue") > names.index("hydration_manifest")
    assert names.index("hydration_action_guide") > names.index("hydration_manifest")
    assert "readiness" in names
    assert names[-12] == "progress_update"
    assert names[-11] == "professor_meeting_packet_seed"
    assert names[-10] == "artifact_index_seed"
    assert names[-9] == "status_packet_seed"
    assert names[-8] == "completion_audit"
    assert names[-7] == "next_action_handoff_seed"
    assert names[-6] == "professor_update_draft_seed"
    assert names[-5] == "artifact_index"
    assert names[-4] == "status_packet"
    assert names[-3] == "next_action_handoff"
    assert names[-2] == "professor_meeting_packet"
    assert names[-1] == "professor_update_draft"


def test_build_steps_can_skip_local_smoke():
    steps = refresh_expansion_status_2020_2025.build_steps(include_local_smoke=False)

    assert "local_smoke" not in [step.name for step in steps]


def test_run_steps_dry_run_marks_steps_planned(tmp_path):
    steps = [refresh_expansion_status_2020_2025.RefreshStep("example", ["python", "--version"])]

    results = refresh_expansion_status_2020_2025.run_steps(steps, tmp_path, dry_run=True)

    assert results == [
        {
            "name": "example",
            "command": ["python", "--version"],
            "command_string": "python --version",
            "status": "planned",
        }
    ]
