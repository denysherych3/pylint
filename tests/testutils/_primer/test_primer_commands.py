# Licensed under the GPL: https://www.gnu.org/licenses/old-licenses/gpl-2.0.html
# For details: https://github.com/pylint-dev/pylint/blob/main/LICENSE
# Copyright (c) https://github.com/pylint-dev/pylint/blob/main/CONTRIBUTORS.txt

"""Unit tests for the prepare and run commands of the pylint primer."""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, NoReturn
from unittest.mock import Mock, patch

import pytest
from _pytest.capture import CaptureFixture

from pylint.reporters.json_reporter import JSON2Reporter, JSONMessage
from pylint.testutils._primer.package_to_lint import PackageToLint
from pylint.testutils._primer.primer_prepare_command import PrepareCommand
from pylint.testutils._primer.primer_run_command import (
    GITHUB_CRASH_TEMPLATE_LOCATION,
    RunCommand,
)

VERSION = ".".join(str(i) for i in sys.version_info[:2])
FIRST_SHA = "aaaaaaaa11111111"
SECOND_SHA = "bbbbbbbb22222222"


def _fake_package(name: str, sha: str) -> Mock:
    package = Mock(spec=PackageToLint)
    package.clone_directory = Path("/primer") / name
    package.commit = sha
    package.lazy_clone.return_value = sha
    package.pylint_args = [f"/primer/{name}"]
    return package


def _packages() -> dict[str, Any]:
    return {
        "first": _fake_package("first", FIRST_SHA),
        "second": _fake_package("second", SECOND_SHA),
    }


def _prepare_config(**action: bool) -> Namespace:
    config = {
        "clone": False,
        "check": False,
        "make_commit_string": False,
        "read_commit_string": False,
    }
    config.update(action)
    return Namespace(**config)


def _json_message(message_id: str, text: str) -> JSONMessage:
    return JSONMessage(
        type="fatal" if message_id.startswith("F") else "convention",
        message=text,
        messageId=message_id,
        symbol="astroid-error" if message_id.startswith("F") else "missing-docstring",
        confidence="UNDEFINED",
        module="first",
        path="first/__init__.py",
        absolutePath="/primer/first/__init__.py",
        line=1,
        endLine=None,
        column=0,
        endColumn=None,
        obj="",
    )


def _fake_run(messages: list[JSONMessage], exit_code: int) -> Mock:
    def run(_arguments: list[str], reporter: JSON2Reporter) -> NoReturn:
        reporter.out.write(json.dumps({"messages": messages, "statistics": {}}))
        raise SystemExit(exit_code)

    return Mock(side_effect=run)


def _fake_repo(sha: str) -> Mock:
    repo = Mock()
    repo.head.object.hexsha = sha
    return repo


def test_prepare_clone_writes_short_commits(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    packages = _packages()
    PrepareCommand(tmp_path, packages, _prepare_config(clone=True)).run()

    commit_file = tmp_path / f"commit_string_{VERSION}.txt"
    assert commit_file.read_text(encoding="utf-8") == "aaaaaaaa_bbbbbbbb_"
    assert f"Cloned 'first' at commit '{FIRST_SHA}'." in capsys.readouterr().out
    for package in packages.values():
        package.lazy_clone.assert_called_once_with()


def test_prepare_check_reads_the_local_head(tmp_path: Path) -> None:
    with patch(
        "pylint.testutils._primer.primer_prepare_command.Repo",
        side_effect=[_fake_repo("cccccccc0"), _fake_repo("dddddddd0")],
    ) as repo:
        PrepareCommand(tmp_path, _packages(), _prepare_config(check=True)).run()

    assert [call.args[0] for call in repo.call_args_list] == [
        Path("/primer/first"),
        Path("/primer/second"),
    ]
    commit_file = tmp_path / f"commit_string_{VERSION}.txt"
    assert commit_file.read_text(encoding="utf-8") == "cccccccc_dddddddd_"


def test_prepare_make_commit_string_does_not_clone(tmp_path: Path) -> None:
    packages = _packages()
    PrepareCommand(tmp_path, packages, _prepare_config(make_commit_string=True)).run()

    commit_file = tmp_path / f"commit_string_{VERSION}.txt"
    assert commit_file.read_text(encoding="utf-8") == "aaaaaaaa_bbbbbbbb_"
    for package in packages.values():
        package.lazy_clone.assert_not_called()


def test_prepare_read_commit_string_prints_without_rewriting(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    commit_file = tmp_path / f"commit_string_{VERSION}.txt"
    commit_file.write_text("12345678_", encoding="utf-8")

    PrepareCommand(
        tmp_path, _packages(), _prepare_config(read_commit_string=True)
    ).run()

    assert capsys.readouterr().out.strip() == "12345678_"
    assert commit_file.read_text(encoding="utf-8") == "12345678_"


def test_prepare_without_an_action_writes_nothing(tmp_path: Path) -> None:
    PrepareCommand(tmp_path, _packages(), _prepare_config()).run()

    assert not list(tmp_path.iterdir())


def test_run_writes_messages_and_commit_of_each_package(tmp_path: Path) -> None:
    message = _json_message("C0114", "Missing module docstring")
    config = Namespace(type="main", batches=None, batchIdx=None)
    with (
        patch(
            "pylint.testutils._primer.primer_run_command.Run",
            _fake_run([message], exit_code=16),
        ) as run,
        patch(
            "pylint.testutils._primer.primer_run_command.Repo",
            side_effect=[_fake_repo(FIRST_SHA), _fake_repo(SECOND_SHA)],
        ),
    ):
        RunCommand(tmp_path, _packages(), config).run()

    output = json.loads(
        (tmp_path / f"output_{VERSION}_main.txt").read_text(encoding="utf-8")
    )
    assert output == {
        "first": {"commit": FIRST_SHA, "messages": [message]},
        "second": {"commit": SECOND_SHA, "messages": [message]},
    }
    first_arguments = run.call_args_list[0].args[0]
    assert first_arguments[0] == "/primer/first"
    assert "--disable=duplicate-code,cyclic-import" in first_arguments


def test_run_lints_only_the_packages_of_its_batch(tmp_path: Path) -> None:
    config = Namespace(type="pr", batches=2, batchIdx=1)
    with (
        patch(
            "pylint.testutils._primer.primer_run_command.Run",
            _fake_run([], exit_code=0),
        ) as run,
        patch(
            "pylint.testutils._primer.primer_run_command.Repo",
            return_value=_fake_repo(SECOND_SHA),
        ),
    ):
        RunCommand(tmp_path, _packages(), config).run()

    assert run.call_count == 1
    assert run.call_args.args[0][0] == "/primer/second"
    output = json.loads(
        (tmp_path / f"output_{VERSION}_pr_batch1.txt").read_text(encoding="utf-8")
    )
    assert list(output) == ["second"]


def test_run_on_a_pr_fails_when_pylint_crashes(tmp_path: Path) -> None:
    fatal = _json_message("F0002", "Astroid crashed")
    config = Namespace(type="pr", batches=None, batchIdx=None)
    with (
        patch(
            "pylint.testutils._primer.primer_run_command.Run",
            _fake_run([fatal], exit_code=1),
        ),
        patch(
            "pylint.testutils._primer.primer_run_command.Repo",
            return_value=_fake_repo(FIRST_SHA),
        ),
    ):
        with pytest.warns(UserWarning, match="Encountered fatal errors"):
            with pytest.raises(AssertionError, match="2 fatal error messages"):
                RunCommand(tmp_path, _packages(), config).run()


def test_run_on_main_only_warns_when_pylint_crashes(tmp_path: Path) -> None:
    fatal = _json_message("F0002", "Astroid crashed")
    config = Namespace(type="main", batches=None, batchIdx=None)
    with (
        patch(
            "pylint.testutils._primer.primer_run_command.Run",
            _fake_run([fatal], exit_code=1),
        ),
        patch(
            "pylint.testutils._primer.primer_run_command.Repo",
            return_value=_fake_repo(FIRST_SHA),
        ),
    ):
        with pytest.warns(UserWarning, match="Encountered fatal errors"):
            RunCommand(tmp_path, _packages(), config).run()

    assert (tmp_path / f"output_{VERSION}_main.txt").exists()


def test_filter_fatal_errors_drops_the_github_crash_template() -> None:
    crash = _json_message(
        "F0002",
        f"Crashed. There is a pre-filled template at {GITHUB_CRASH_TEMPLATE_LOCATION}/x",
    )
    convention = _json_message("C0114", "Missing module docstring")

    fatal_msgs = RunCommand._filter_fatal_errors([crash, convention])

    assert [message.msg for message in fatal_msgs] == ["Crashed. "]
