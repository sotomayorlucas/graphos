"""Tests for GraphOS Shell REPL (mocked runtime, no ONNX needed)."""

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.constants import (
    TENSOR_DIM, NUM_CLASSES, NUM_ROUTES, DEFAULT_BATCH_SIZE,
)
from kernel.repl import GraphOSShell


def _make_mock_runtime():
    """Create a mock KernelRuntime."""
    rt = MagicMock()
    rt.programs = []
    rt.device = "CPU"
    rt.health.return_value = {
        "device": "CPU",
        "programs": [],
        "exec_count": 0,
        "mean_latency_us": 0.0,
        "last_latency_us": 0.0,
        "errors": 0,
        "healthy": True,
    }
    return rt


def _make_shell(runtime=None):
    """Create a shell with a mock runtime."""
    rt = runtime or _make_mock_runtime()
    return GraphOSShell(runtime=rt, batch_size=DEFAULT_BATCH_SIZE)


class TestREPLPrograms:
    def test_programs_empty(self, capsys):
        shell = _make_shell()
        shell.do_programs("")
        out = capsys.readouterr().out
        assert "No programs loaded" in out

    def test_programs_listed(self, capsys):
        rt = _make_mock_runtime()
        rt.programs = ["classifier"]
        mock_prog = MagicMock()
        mock_prog.spec.input_shape = (64, 64)
        mock_prog.spec.output_shape = (64, 3)
        mock_prog.spec.description = "test"
        rt.get.return_value = mock_prog
        shell = _make_shell(rt)
        shell.do_programs("")
        out = capsys.readouterr().out
        assert "classifier" in out


class TestREPLHealth:
    def test_health(self, capsys):
        shell = _make_shell()
        shell.do_health("")
        out = capsys.readouterr().out
        assert "CPU" in out
        assert "Healthy: True" in out


class TestREPLSend:
    def test_send_no_args(self, capsys):
        shell = _make_shell()
        shell.do_send("")
        out = capsys.readouterr().out
        assert "Usage" in out

    def test_send_generates_packets(self, capsys):
        shell = _make_shell()
        shell.do_send("http 4")
        out = capsys.readouterr().out
        assert "Generated 4" in out
        assert len(shell._packets) == 4
        assert all(l == "HTTP" for l in shell._labels)


class TestREPLRun:
    def test_run_no_packets(self, capsys):
        shell = _make_shell()
        shell.do_run("classifier")
        out = capsys.readouterr().out
        assert "No packets" in out

    def test_run_no_program(self, capsys):
        rt = _make_mock_runtime()
        rt.programs = []
        shell = _make_shell(rt)
        shell._packets = [b"\x00" * 64]
        shell._labels = ["test"]
        shell.do_run("classifier")
        out = capsys.readouterr().out
        assert "not loaded" in out


class TestREPLStats:
    def test_stats_empty(self, capsys):
        shell = _make_shell()
        shell.do_stats("")
        out = capsys.readouterr().out
        assert "No executions" in out

    def test_stats_after_run(self, capsys):
        shell = _make_shell()
        shell._exec_history = [{"program": "classifier", "count": 8}]
        shell.do_stats("")
        out = capsys.readouterr().out
        assert "classifier" in out
        assert "8" in out


class TestREPLQuit:
    def test_quit(self, capsys):
        shell = _make_shell()
        result = shell.do_quit("")
        assert result is True
        out = capsys.readouterr().out
        assert "Goodbye" in out


class TestREPLPipe:
    def test_pipe_no_args(self, capsys):
        shell = _make_shell()
        shell.do_pipe("")
        out = capsys.readouterr().out
        assert "Usage" in out

    def test_pipe_unknown_program(self, capsys):
        shell = _make_shell()
        shell.do_pipe("foo bar")
        out = capsys.readouterr().out
        assert "Unknown" in out


class TestREPLInspect:
    def test_inspect_no_pipeline(self, capsys):
        shell = _make_shell()
        shell.do_inspect("")
        out = capsys.readouterr().out
        assert "No pipeline" in out
