from markov_lean import cli


def test_help_parses(capsys):
    try:
        cli.main(["--help"])
    except SystemExit as e:
        assert e.code == 0
    assert "pilot" in capsys.readouterr().out
