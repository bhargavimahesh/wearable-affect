from wearable_affect.data import load_self_reports


def test_self_reports_follow_condition_order(tmp_path):
    folder = tmp_path / "S99"
    folder.mkdir()
    (folder / "S99_quest.csv").write_text(
        "# ORDER;Base;TSST;Fun;bRead;;\n# DIM;5;2;;\n# DIM;3;7;;\n# DIM;7;3;;\n"
    )
    reports = load_self_reports(tmp_path, "S99")

    assert list(reports["condition"]) == ["Base", "TSST", "Fun"]
    assert reports.loc[reports["condition"] == "TSST", "arousal"].item() == 7