from xert.eval.ledger import append_row, new_row

from scripts.resume_numbers import main


def _row(value, time, metric="success_rate", **extra):
    return new_row(time_utc=time, git_commit="0" * 40, git_dirty=False, slice="P9.9",
                   experiment="exp", backend="mujoco", embodiment="so101", metric=metric,
                   value=value, ci95=None, n=3, seeds={"train": [], "eval": []},
                   config_path="configs/x.json", artifact_paths=["results/x.json"],
                   notes="Simulation only.", **extra)


def test_supersession_label_and_values_come_from_rows(capsys):
    old = _row(0.25, "2026-01-01T00:00:00+00:00")
    newer = _row(0.5, "2026-01-02T00:00:00+00:00")
    wrong = _row(0.125, "2026-01-01T00:00:00+00:00", metric="ik_err_mm_mean")
    fix = _row(0.75, "2026-01-03T00:00:00+00:00", metric="ik_err_mm_p95", superseded_id=wrong["id"])
    for row in (old, newer, wrong, fix):
        append_row(row)

    main()
    out = capsys.readouterr().out

    assert f"superseded {wrong['id']} by {fix['id']}" in out
    assert f"id {newer['id']}" in out and f"id {fix['id']}" in out
    assert f"id {old['id']}" not in out and f"id {wrong['id']}" not in out
    assert "value 0.5 " in out and "value 0.75 " in out
    assert "0.25" not in out and "0.125" not in out
    assert out.count("backend mujoco (simulation)") == 2
