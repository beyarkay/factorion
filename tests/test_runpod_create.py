"""Tests for RunPod pod creation retry logic in scripts/ci/runpod_create.py."""

import os
import sys

import pytest
from unittest.mock import patch

# Add scripts/ci to sys.path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "ci"))

import runpod_create  # noqa: E402

RUNNING_POD = {
    "desiredStatus": "RUNNING",
    "runtime": {
        "ports": [
            {"privatePort": 22, "isIpPublic": True, "ip": "1.2.3.4", "publicPort": 12345},
        ],
    },
    "machine": {"gpuDisplayName": "NVIDIA RTX A6000"},
}


class TestCreatePodRetry:
    @patch("runpod_create.time.sleep")
    @patch("runpod_create.runpod")
    def test_retries_same_gpu_until_available(self, mock_runpod, mock_sleep):
        """Retries the same GPU when it is transiently unavailable."""
        os.environ["RUNPOD_API_KEY"] = "test-key"
        mock_runpod.create_pod.side_effect = [
            Exception("no longer any instances available"),
            Exception("no longer any instances available"),
            {"id": "pod-1"},
        ]
        mock_runpod.get_pod.return_value = RUNNING_POD

        result = runpod_create.create_pod(
            gpu_type="some-custom-gpu", max_retries=5, retry_delay=5.0
        )

        assert result["pod_id"] == "pod-1"
        assert result["ssh_host"] == "1.2.3.4"
        assert result["ssh_port"] == 12345
        assert mock_runpod.create_pod.call_count == 3
        # Slept once after each of the two failed attempts.
        assert mock_sleep.call_args_list == [((5.0,),), ((5.0,),)]
        # All three attempts targeted the same GPU.
        gpus = [c.kwargs["gpu_type_id"] for c in mock_runpod.create_pod.call_args_list]
        assert gpus == ["some-custom-gpu"] * 3

    @patch("runpod_create.time.sleep")
    @patch("runpod_create.runpod")
    def test_falls_back_to_next_gpu_after_exhausting_retries(self, mock_runpod, mock_sleep):
        """Falls back to the next GPU only after exhausting retries on the first."""
        os.environ["RUNPOD_API_KEY"] = "test-key"
        # First GPU fails both attempts; second GPU succeeds on its first.
        mock_runpod.create_pod.side_effect = [
            Exception("no longer any instances available"),
            Exception("no longer any instances available"),
            {"id": "pod-2"},
        ]
        mock_runpod.get_pod.return_value = RUNNING_POD

        result = runpod_create.create_pod(
            gpu_type=runpod_create.GPU_FALLBACKS[0], max_retries=2, retry_delay=5.0
        )

        assert result["pod_id"] == "pod-2"
        assert mock_runpod.create_pod.call_count == 3
        gpus = [c.kwargs["gpu_type_id"] for c in mock_runpod.create_pod.call_args_list]
        assert gpus == [
            runpod_create.GPU_FALLBACKS[0],
            runpod_create.GPU_FALLBACKS[0],
            runpod_create.GPU_FALLBACKS[1],
        ]
        # One sleep between the two attempts on the first GPU; none before the
        # immediate fallback to the second GPU.
        assert mock_sleep.call_count == 1

    @patch("runpod_create.time.sleep")
    @patch("runpod_create.runpod")
    def test_gives_up_after_max_retries(self, mock_runpod, mock_sleep):
        """Exits non-zero once every retry round is exhausted."""
        os.environ["RUNPOD_API_KEY"] = "test-key"
        mock_runpod.create_pod.side_effect = Exception("no instances available")

        with pytest.raises(SystemExit) as exc:
            runpod_create.create_pod(
                gpu_type="some-custom-gpu", max_retries=3, retry_delay=2.0
            )

        assert exc.value.code == 1
        assert mock_runpod.create_pod.call_count == 3
        # Sleeps between rounds, but not after the final failed round.
        assert mock_sleep.call_count == 2

    @patch("runpod_create.time.sleep")
    @patch("runpod_create.runpod")
    def test_succeeds_first_try_no_sleep(self, mock_runpod, mock_sleep):
        """No retry delay when the first attempt succeeds."""
        os.environ["RUNPOD_API_KEY"] = "test-key"
        mock_runpod.create_pod.return_value = {"id": "pod-9"}
        mock_runpod.get_pod.return_value = RUNNING_POD

        result = runpod_create.create_pod(gpu_type="some-custom-gpu")

        assert result["pod_id"] == "pod-9"
        assert mock_runpod.create_pod.call_count == 1
        mock_sleep.assert_not_called()
