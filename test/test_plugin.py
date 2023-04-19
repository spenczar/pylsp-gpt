from unittest.mock import ANY
import pytest
from pylsp_gpt import plugin
from test.conftest import *
import ast


def test_definitions(config, workspace, document):
    position = {"line": 3, "character": 6}

    response = plugin.pylsp_definitions(
        config=config,
        workspace=workspace,
        document=document,
        position=position,
    )

    expected = [
        {
            "uri": ANY,
            "range": {
                "start": {
                    "line": ANY,
                    "character": ANY,
                },
                "end": {
                    "line": ANY,
                    "character": ANY,
                },
            },
        },
    ]

    assert response == expected


def test_code_action_replaces_docstrings(config, workspace, docstring_cases_document, code_action_context):
    selection = {
        "start": {
            "line": 1,
            "character": 3,
        },
        "end": {
            "line": 1,
            "character": 7,
        },
    }

    response = plugin.pylsp_code_actions(
        config=config,
        workspace=workspace,
        document=docstring_cases_document,
        range=selection,
        context=code_action_context,
    )

    expected = [
        {
            "title": "gpt-docstring",
            "kind": "refactor",
            "command": {
                "command": "refactor.gpt.docstring",
                "arguments": [docstring_cases_document.uri, selection],
            },
        },
    ]

    assert response == expected

    command = response[0]["command"]["command"]
    arguments = response[0]["command"]["arguments"]

    response = plugin.pylsp_execute_command(
        config=config,
        workspace=workspace,
        command=command,
        arguments=arguments,
    )

    workspace._endpoint.request.assert_called_once_with(
        "workspace/applyEdit",
        {
            "edit": {
                "changes": {
                    docstring_cases_document.uri: [
                        {
                            "range": {
                                "start": {"line": 2, "character": 4},
                                "end": {"line": 2, "character": 33},
                            },
                            "newText": '"""This is a new docstring"""',
                        },
                    ],
                },
            },
        },
    )
