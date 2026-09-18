from scripts import run_post_hydration_expansion_sequence


def test_build_steps_orders_readiness_staged_full_refresh():
    steps = run_post_hydration_expansion_sequence.build_steps(allow_network_steps=True)
    names = [step.name for step in steps]

    assert names == [
        "refresh_status",
        "readiness_gate",
        "staged_2020_2021_smoke",
        "full_2020_2025_expansion",
        "refresh_after_full",
        "final_readiness_report",
    ]
    assert "--allow-network-steps" in steps[3].command


def test_run_sequence_plans_execution_steps_without_execute(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, cwd, check):
        calls.append(command)

        class Completed:
            returncode = 0

        return Completed()

    monkeypatch.setattr(run_post_hydration_expansion_sequence.subprocess, "run", fake_run)
    steps = run_post_hydration_expansion_sequence.build_steps()

    results = run_post_hydration_expansion_sequence.run_sequence(steps, tmp_path, execute=False)

    assert [result["status"] for result in results] == ["passed", "passed", "planned", "planned", "planned", "planned"]
    assert calls == [steps[0].command, steps[1].command]


def test_run_sequence_stops_on_failed_readiness(monkeypatch, tmp_path):
    def fake_run(command, cwd, check):
        class Completed:
            returncode = 2 if any("validate_expansion_readiness.py" in part for part in command) else 0

        return Completed()

    monkeypatch.setattr(run_post_hydration_expansion_sequence.subprocess, "run", fake_run)
    steps = run_post_hydration_expansion_sequence.build_steps()

    results = run_post_hydration_expansion_sequence.run_sequence(steps, tmp_path, execute=True)

    assert [result["name"] for result in results] == ["refresh_status", "readiness_gate"]
    assert results[-1]["status"] == "failed"
