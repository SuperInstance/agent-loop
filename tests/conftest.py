"""Shared pytest fixtures for the kimi-agent-loop test suite.

Every fixture runs inside a private temporary directory and (optionally)
calls autonomous.ensure_files() so each test gets a clean, predictable
workspace without touching the developer's real files.
"""

import os

import pytest

import autonomous


@pytest.fixture
def in_tmp_path(monkeypatch, tmp_path):
    """Chdir into a fresh tmp directory. Returns the tmp_path.

    Does NOT call ensure_files() — caller decides whether the standard
    file set is needed. Use this when a test wants a totally empty
    workspace or wants to write files manually.
    """
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def in_workspace(monkeypatch, tmp_path):
    """Chdir into tmp_path AND call autonomous.ensure_files().

    Returns tmp_path. All tests of functions that read module-level
    file constants (goals.md, stream.md, etc.) should use this fixture.
    """
    monkeypatch.chdir(tmp_path)
    autonomous.ensure_files()
    return tmp_path
