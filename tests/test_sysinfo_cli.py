import json

from bench import cli, sysinfo


def test_collect_is_json_serialisable_and_private():
    info = sysinfo.collect()
    text = json.dumps(info)
    for key in ("cpu", "memory", "os", "python", "packages", "git", "container"):
        assert key in info
    assert info["cpu"]["logical_cpus"] >= 1
    import platform
    node = platform.node()
    assert not node or node not in text  # no hostname in committed results
    assert len(sysinfo.summary_lines(info)) >= 6


def test_cli_parser_knows_all_commands():
    p = cli.build_parser()
    for argv in (["sysinfo"], ["models", "list"], ["stream", "--threads", "1,2"], ["smoke"],
                 ["plots"], ["tiny-model"]):
        assert p.parse_args(argv).func


def test_models_list_command(capsys):
    assert cli.main(["models", "list"]) == 0
    out = capsys.readouterr().out
    assert "smoke" in out and "full" in out
