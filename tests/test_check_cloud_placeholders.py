from pathlib import Path

from scripts import check_cloud_placeholders


def test_summary_only_prints_count(tmp_path: Path, capsys):
    path = tmp_path / "local.txt"
    path.write_text("local", encoding="utf-8")
    code = check_cloud_placeholders.main(["--summary-only", str(tmp_path)])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.strip() == "placeholder_count=0"


def test_limit_omits_extra_paths(monkeypatch, tmp_path: Path, capsys):
    files = [tmp_path / f"file_{idx}.txt" for idx in range(3)]
    for path in files:
        path.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(check_cloud_placeholders, "is_cloud_placeholder", lambda path: path in files)
    code = check_cloud_placeholders.main(["--limit", "1", str(tmp_path)])
    captured = capsys.readouterr()
    assert code == 0
    assert "file_" in captured.out
    assert "additional placeholders omitted" in captured.out
    assert "placeholder_count=3" in captured.out


def test_paths_file_checks_only_listed_paths(monkeypatch, tmp_path: Path, capsys):
    listed = tmp_path / "listed.txt"
    unlisted = tmp_path / "unlisted.txt"
    listed.write_text("placeholder", encoding="utf-8")
    unlisted.write_text("placeholder", encoding="utf-8")
    paths_file = tmp_path / "paths.txt"
    paths_file.write_text(f"{listed}\n", encoding="utf-8")

    monkeypatch.setattr(check_cloud_placeholders, "is_cloud_placeholder", lambda path: path in {listed, unlisted})

    code = check_cloud_placeholders.main(["--paths-file", str(paths_file)])
    captured = capsys.readouterr()

    assert code == 0
    assert str(listed) in captured.out
    assert str(unlisted) not in captured.out
    assert "placeholder_count=1" in captured.out


def test_no_paths_uses_default_paths(monkeypatch):
    monkeypatch.setattr(check_cloud_placeholders, "iter_files", lambda paths: paths)
    monkeypatch.setattr(check_cloud_placeholders, "is_cloud_placeholder", lambda path: False)

    code = check_cloud_placeholders.main(["--summary-only"])

    assert code == 0
