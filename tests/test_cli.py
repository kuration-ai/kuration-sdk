"""`kai` CLI integration tests: list, describe, error paths."""
import subprocess


def _run(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["kai", *args],
        capture_output=True,
        text=True,
        check=False,
    )


class TestKitList:
    def test_shows_all_three_builtin_operations(self):
        r = _run("list")
        assert r.returncode == 0
        assert "BalancedSelector" in r.stdout
        assert "DiverseSubset" in r.stdout
        assert "RareEventCoverage" in r.stdout

    def test_annotates_extras_for_diverse_subset(self):
        r = _run("list")
        assert r.returncode == 0
        # DiverseSubset requires [methods] extra
        lines = [ln for ln in r.stdout.splitlines() if "DiverseSubset" in ln]
        assert len(lines) == 1
        assert "needs kuration-ai[methods]" in lines[0]

    def test_reports_zero_user_registered(self):
        r = _run("list")
        assert "3 built-in, 0 user-registered" in r.stdout


class TestKitDescribe:
    def test_shows_parameter_contract(self):
        r = _run("describe", "BalancedSelector")
        assert r.returncode == 0
        assert "feature" in r.stdout
        assert "of" in r.stdout
        assert "bins" in r.stdout
        assert "count" in r.stdout
        # each field labelled as required or optional
        assert "(required)" in r.stdout
        assert "(optional)" in r.stdout

    def test_annotates_extras_for_diverse_subset(self):
        r = _run("describe", "DiverseSubset")
        assert r.returncode == 0
        assert "Needs kuration-ai[methods]" in r.stdout

    def test_unknown_name_exits_nonzero_with_actionable_error(self):
        r = _run("describe", "TotallyMadeUp")
        assert r.returncode == 1
        assert "unknown operation" in r.stderr.lower()
        assert "Available:" in r.stderr
        assert "BalancedSelector" in r.stderr


class TestKitBareInvocation:
    def test_bare_kai_prints_help_and_exits_nonzero(self):
        r = _run()
        assert r.returncode == 1
        # help output goes to stdout, lists both subcommands
        combined = r.stdout + r.stderr
        assert "list" in combined
        assert "describe" in combined
