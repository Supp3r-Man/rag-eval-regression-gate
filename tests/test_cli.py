import json

import pytest

from raggate import cli
from raggate.cli import main
from raggate.corpus import build_golden
from raggate.golden import save_golden


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("runs")
    for name in ("baseline", "regressed_generator", "tight_context"):
        assert main(["run", name, "--out", str(d / f"{name}.json")]) == 0
    return d


def test_gate_exit_0_for_pass(files):
    assert main(["gate", str(files / "baseline.json"), str(files / "baseline.json")]) == 0


def test_gate_exit_1_for_a_regression(files, capsys):
    assert main(["gate", str(files / "baseline.json"), str(files / "regressed_generator.json")]) == 1
    out = capsys.readouterr().out
    assert out.startswith("GATE FAIL") and "critical slice qtype:unanswerable" in out


def test_gate_json_output_carries_the_exit_code(files, capsys):
    rc = main(["gate", str(files / "baseline.json"), str(files / "tight_context.json"), "--json"])
    d = json.loads(capsys.readouterr().out)
    assert rc == d["exit_code"] == 1 and d["decision"] == "FAIL"


def test_gate_exit_2_when_a_file_is_missing_not_1(files, capsys):
    assert main(["gate", str(files / "baseline.json"), str(files / "nope.json")]) == 2
    assert "could not judge" in capsys.readouterr().err


def test_gate_exit_2_for_a_corrupt_run_file(files, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert main(["gate", str(files / "baseline.json"), str(bad)]) == 2


def test_gate_exit_2_when_the_candidate_run_has_errors(files, tmp_path, capsys):
    d = json.loads((files / "baseline.json").read_text(encoding="utf-8"))
    d["results"][0]["error"] = "TimeoutError: retriever"
    bad = tmp_path / "errored.json"
    bad.write_text(json.dumps(d), encoding="utf-8")
    assert main(["gate", str(files / "baseline.json"), str(bad)]) == 2
    assert "INCONCLUSIVE" in capsys.readouterr().out


def test_a_crash_inside_a_command_exits_2_never_1(monkeypatch, files):
    def boom(*a, **k):
        raise RuntimeError("scorer exploded")
    monkeypatch.setattr(cli, "run_gate", boom)
    assert main(["gate", str(files / "baseline.json"), str(files / "baseline.json")]) == 2


def test_unknown_pipeline_exits_2(tmp_path):
    assert main(["run", "no-such-pipeline", "--out", str(tmp_path / "x.json")]) == 2


def test_hygiene_exit_codes(tmp_path, capsys):
    assert main(["hygiene"]) == 0
    gs = build_golden()
    gs.items[0] = type(gs.items[0])(**{**gs.items[0].__dict__, "entity": ""})
    p = tmp_path / "g.json"
    save_golden(gs, p)
    assert main(["hygiene", "--golden", str(p)]) == 2
    assert "missing_entity" in capsys.readouterr().out


def test_gate_against_a_broken_golden_file_is_exit_2(files, tmp_path):
    gs = build_golden()
    gs.items[0] = type(gs.items[0])(**{**gs.items[0].__dict__, "entity": ""})
    p = tmp_path / "g.json"
    save_golden(gs, p)
    assert main(["gate", "--golden", str(p), str(files / "baseline.json"), str(files / "baseline.json")]) == 2


def test_judge_and_power_and_tune_commands_run(capsys):
    assert main(["judge"]) == 0
    out = capsys.readouterr().out
    assert "rule " in out and "always-pass" in out and "trustworthy=True" in out
    assert main(["power", "--n", "60", "--trials", "8"]) == 0
    assert "false-alarm rate" in capsys.readouterr().out
    assert main(["tune"]) == 0
    assert "dev split only" in capsys.readouterr().out


def test_demo_runs_end_to_end(capsys):
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "regressed_retriever: FAIL (exit 1)" in out and "A/A: baseline vs itself -> PASS" in out
