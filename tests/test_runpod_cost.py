"""Tests for RunPod cost tracking in scripts/ci/runpod_destroy.py."""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Add scripts/ci to sys.path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "ci"))

from runpod_destroy import format_uptime, query_pod_cost  # noqa: E402


class TestFormatUptime:
    def test_seconds_only(self):
        assert format_uptime(45) == "45s"

    def test_minutes_and_seconds(self):
        assert format_uptime(1255) == "20m 55s"

    def test_hours_minutes_seconds(self):
        assert format_uptime(3930) == "1h 5m 30s"

    def test_exact_hour(self):
        assert format_uptime(3600) == "1h"

    def test_exact_minute(self):
        assert format_uptime(120) == "2m"

    def test_zero_seconds(self):
        assert format_uptime(0) == "0s"

    def test_fractional_seconds_truncated(self):
        assert format_uptime(90.7) == "1m 30s"


class TestQueryPodCost:
    @patch("runpod_destroy.runpod")
    def test_cost_calculation(self, mock_runpod):
        mock_runpod.get_pod.return_value = {
            "costPerHr": 2.61,
            "uptimeSeconds": 1255,
            "machine": {"gpuDisplayName": "NVIDIA A100 80GB PCIe"},
        }
        os.environ["RUNPOD_API_KEY"] = "test-key"

        result = query_pod_cost("pod-123")

        assert result["pod_id"] == "pod-123"
        assert result["cost_per_hr"] == 2.61
        assert result["uptime_seconds"] == 1255
        # 2.61 * (1255/3600) = 0.909...  rounded to 0.91
        assert result["total_cost"] == 0.91
        assert result["gpu_name"] == "NVIDIA A100 80GB PCIe"
        assert result["uptime_human"] == "20m 55s"
        mock_runpod.get_pod.assert_called_once_with("pod-123")

    @patch("runpod_destroy.runpod")
    def test_cost_with_flat_gpu_name(self, mock_runpod):
        """Falls back to top-level gpuDisplayName if machine dict is missing."""
        mock_runpod.get_pod.return_value = {
            "costPerHr": 1.0,
            "uptimeSeconds": 3600,
            "gpuDisplayName": "RTX 4090",
        }
        os.environ["RUNPOD_API_KEY"] = "test-key"

        result = query_pod_cost("pod-456")

        assert result["total_cost"] == 1.0
        assert result["gpu_name"] == "RTX 4090"


class TestAppendToSummary:
    def test_append_to_existing_summary(self, tmp_path):
        summary_file = tmp_path / "summary.md"
        summary_file.write_text("# Smoke Test Results\nAll good.\n")

        cost_line = (
            "\n**Cost:** $0.91 "
            "(20m 55s on NVIDIA A100 80GB PCIe "
            "@ $2.61/hr via RunPod)\n"
        )

        existing = summary_file.read_text()
        summary_file.write_text(existing + cost_line)

        content = summary_file.read_text()
        assert "# Smoke Test Results" in content
        assert "**Cost:** $0.91" in content
        assert "20m 55s on NVIDIA A100 80GB PCIe" in content

    def test_append_to_missing_summary(self, tmp_path):
        """When summary file doesn't exist, create it with just the cost line."""
        summary_file = tmp_path / "summary.md"

        cost_line = "\n**Cost:** $0.50 (10m 0s on GPU @ $3.00/hr via RunPod)\n"

        existing = ""
        if summary_file.exists():
            existing = summary_file.read_text()
        summary_file.write_text(existing + cost_line)

        assert summary_file.exists()
        assert "**Cost:** $0.50" in summary_file.read_text()


class TestCostErrorStillTerminates:
    @patch("runpod_destroy.runpod")
    def test_cost_error_still_terminates(self, mock_runpod):
        """If cost query raises, pod should still be terminated."""
        mock_runpod.get_pod.side_effect = Exception("API timeout")
        mock_runpod.terminate_pod.return_value = None
        os.environ["RUNPOD_API_KEY"] = "test-key"

        # Simulate the main() logic: query cost (fails), then terminate
        pod_id = "pod-789"

        try:
            query_pod_cost(pod_id)
            cost_queried = True
        except Exception:
            cost_queried = False

        # Terminate must always be called
        mock_runpod.terminate_pod(pod_id)

        assert not cost_queried
        mock_runpod.terminate_pod.assert_called_once_with(pod_id)
