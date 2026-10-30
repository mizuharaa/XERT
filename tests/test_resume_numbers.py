from xert.eval.ledger import append_row, new_row

from scripts.resume_numbers import main


def _row(value, time, metric="success_rate", dirty=False, notes="Simulation only.", train=(), **extra):
    return new_row(time_utc=f"2026-01-0{time}T00:00:00+00:00", git_commit="0" * 40, git_dirty=dirty,
                   slice="P9.9", experiment="exp", backend="mujoco", embodiment="so101", metric=metric,
                   value=value, ci95=None, n=3, seeds={"train": list(train), "eval": []},
                   config_path="configs/x.json", artifact_paths=["results/x.json"], notes=notes, **extra)


def _run(capsys, rows):
    for row in rows:
        append_row(row)
    main()
    return capsys.readouterr().out


def test_older_repeat_and_superseded_id_are_dropped_values_print_exactly(capsys):
    old = _row(0.25, 1)
    newer = _row(0.123456789, 2)
    wrong = _row(0.125, 1, metric="ik_err_mm_mean")
    fix = _row(0.75, 3, metric="ik_err_mm_p95", superseded_id=wrong["id"])
    out = _run(capsys, [old, newer, wrong, fix])

    assert f"superseded {wrong['id']} by {fix['id']}" in out
    assert f"older repeat {old['id']} dropped; kept {newer['id']}" in out
    assert f"id {newer['id']}" in out and f"id {fix['id']}" in out
    assert f"id {old['id']}" not in out and f"id {wrong['id']}" not in out
    assert "value 0.123456789 " in out and "value 0.75 " in out
    assert "value 0.25 " not in out and "value 0.125 " not in out
    assert out.count("backend mujoco (simulation)") == 2


def test_correction_note_supersedes_but_a_bare_mention_does_not(capsys):
    wrong = _row(0.5, 1, metric="a")
    mentioned = _row(0.5, 1, metric="b")
    fix = _row(0.5, 2, metric="c", notes=f"Rerun. Corrects earlier row IDs: {wrong['id']}")
    aside = _row(0.5, 2, metric="d", notes=f"Same states as row {mentioned['id']}.")
    out = _run(capsys, [wrong, mentioned, fix, aside])

    assert f"superseded {wrong['id']} by {fix['id']}" in out and f"id {wrong['id']}" not in out
    assert f"id {mentioned['id']}" in out


def test_rows_differing_in_seed_phase_or_condition_are_all_kept(capsys):
    rows = [_row(0.5, 1, train=[0]), _row(0.5, 2, train=[1]),
            _row(0.5, 1, metric="mae", phase="sweep_10"), _row(0.5, 2, metric="mae", phase="sweep_20"),
            *(_row(0.5, i + 1, metric="gate", condition=c) for i, c in enumerate("ABC"))]
    out = _run(capsys, rows)

    assert "0 older repeats dropped" in out
    assert all(f"id {row['id']}" in out for row in rows)
    for label in ("train_seeds=[0]", "train_seeds=[1]", "phase=sweep_10", "phase=sweep_20",
                  "condition=A", "condition=B", "condition=C"):
        assert label in out


def test_dirty_rows_are_labelled(capsys):
    out = _run(capsys, [_row(0.5, 1, metric="clean"), _row(0.5, 1, metric="dirty", dirty=True)])

    assert out.count("tree DIRTY") == 1 and out.count("tree clean") == 1
