"""
End-to-end tests for the compiled PyInstaller binary.

These tests spawn the actual binary as a subprocess and verify:
- It processes real audio files correctly
- JSON output is valid and complete
- Results match the Python module output
"""

import json
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import soundfile as sf

from sas_processor.processor import process_audio
from tests.fixtures import create_test_wav_from_sample


def get_binary_path() -> Optional[Path]:
    """Get the path to the compiled binary, or None if it doesn't exist."""
    project_root = Path(__file__).parent.parent
    binary = project_root / "dist" / "sas-processor" / "sas-processor"
    return binary if binary.exists() else None


# Skip all tests in this module if binary doesn't exist
pytestmark = [
    pytest.mark.binary,
    pytest.mark.skipif(
        get_binary_path() is None,
        reason="Binary not found. Run 'pyinstaller sas-processor.spec' first."
    ),
]


# =============================================================================
# Binary Basic Tests
# =============================================================================

@pytest.mark.binary
class TestBinaryBasic:
    """Basic functionality tests for the compiled binary."""

    def test_binary_exists(self, binary_path):
        """Verify the binary exists and is executable."""
        assert binary_path is not None, "Binary not found"
        binary = Path(binary_path)
        assert binary.exists()
        assert binary.is_file()

    def test_binary_help(self, binary_path):
        """Test that --help works."""
        result = subprocess.run(
            [binary_path, '--help'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0
        assert 'trim' in result.stdout.lower()
        assert 'analyze' in result.stdout.lower()

    def test_binary_missing_args(self, binary_path):
        """Test that missing required arguments produce an error."""
        result = subprocess.run(
            [binary_path, 'trim'],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode != 0

    def test_binary_invalid_input_file(self, binary_path, tmp_path):
        """Test handling of non-existent input file."""
        output_path = str(tmp_path / "output.wav")

        result = subprocess.run(
            [binary_path, 'trim',
             '--input', '/nonexistent/file.wav',
             '--output', output_path,
             '--bpm', '120',
             '--bars', '4'],
            capture_output=True,
            text=True,
            timeout=30
        )

        assert result.returncode != 0

        # Should have error JSON on stderr
        assert result.stderr.strip()
        try:
            error_json = json.loads(result.stderr.strip().split('\n')[-1])
            assert error_json.get('type') == 'error'
        except json.JSONDecodeError:
            # May not be JSON, but should still indicate error
            assert 'error' in result.stderr.lower() or 'not found' in result.stderr.lower()


# =============================================================================
# Binary with Real Audio Tests
# =============================================================================

@pytest.mark.binary
@pytest.mark.real_audio
class TestBinaryRealAudio:
    """Test the binary with real audio files from librosa."""

    def test_binary_nutcracker(self, binary_path, nutcracker_wav, tmp_path):
        """Test binary processing of classical music."""
        input_path, bpm = nutcracker_wav
        output_path = str(tmp_path / "output.wav")

        result = subprocess.run(
            [binary_path, 'trim',
             '--input', input_path,
             '--output', output_path,
             '--bpm', str(bpm),
             '--bars', '4',
             '--json'],
            capture_output=True,
            text=True,
            timeout=120
        )

        assert result.returncode == 0, f"Binary failed: {result.stderr}"
        assert Path(output_path).exists()

        # Parse JSON output
        lines = result.stdout.strip().split('\n')
        json_objects = [json.loads(line) for line in lines if line.strip()]

        complete_events = [j for j in json_objects if j.get('type') == 'trim']
        assert len(complete_events) == 1
        assert complete_events[0]['success'] is True

        # Verify output audio
        output_audio, out_sr = sf.read(output_path)
        assert len(output_audio) > 0
        assert out_sr == 44100

    def test_binary_choice(self, binary_path, choice_wav, tmp_path):
        """Test binary processing of electronic music."""
        input_path, bpm = choice_wav
        output_path = str(tmp_path / "output.wav")

        result = subprocess.run(
            [binary_path, 'trim',
             '--input', input_path,
             '--output', output_path,
             '--bpm', str(bpm),
             '--bars', '4',
             '--json'],
            capture_output=True,
            text=True,
            timeout=120
        )

        assert result.returncode == 0, f"Binary failed: {result.stderr}"
        assert Path(output_path).exists()

        # Verify output
        output_audio, _ = sf.read(output_path)
        expected_duration = 4 * 4 * (60.0 / bpm)
        actual_duration = len(output_audio) / 44100
        assert abs(actual_duration - expected_duration) < 0.5

    def test_binary_verbose_mode(self, binary_path, nutcracker_wav, tmp_path):
        """Test binary with verbose output (human-readable)."""
        input_path, bpm = nutcracker_wav
        output_path = str(tmp_path / "output.wav")

        result = subprocess.run(
            [binary_path, 'trim',
             '--input', input_path,
             '--output', output_path,
             '--bpm', str(bpm),
             '--bars', '2',
             '--verbose'],
            capture_output=True,
            text=True,
            timeout=120
        )

        assert result.returncode == 0, f"Binary failed: {result.stderr}"
        assert Path(output_path).exists()

    def test_binary_stereo_preservation(self, binary_path, tmp_path):
        """Test that binary preserves stereo audio."""
        # Create a synthetic stereo test file
        sr = 44100
        duration = 10.0
        bpm = 120.0

        t = np.linspace(0, duration, int(sr * duration))
        left_channel = np.sin(2 * np.pi * 440 * t)
        right_channel = np.sin(2 * np.pi * 880 * t)
        stereo_audio = np.column_stack([left_channel, right_channel])

        input_path = str(tmp_path / "stereo_test.wav")
        sf.write(input_path, stereo_audio, sr, subtype='PCM_24')

        output_path = str(tmp_path / "output.wav")

        result = subprocess.run(
            [binary_path, 'trim',
             '--input', input_path,
             '--output', output_path,
             '--bpm', str(int(bpm)),
             '--bars', '2',
             '--json'],
            capture_output=True,
            text=True,
            timeout=120
        )

        assert result.returncode == 0, f"Binary failed: {result.stderr}"

        # Verify stereo preserved
        output_audio, _ = sf.read(output_path)
        assert output_audio.ndim == 2, "Stereo should be preserved"
        assert output_audio.shape[1] == 2, "Should have 2 channels"


# =============================================================================
# Binary vs Python Module Comparison
# =============================================================================

@pytest.mark.binary
@pytest.mark.real_audio
class TestBinaryModuleComparison:
    """Compare binary output to Python module output."""

    def test_binary_matches_module_output(self, binary_path, tmp_path):
        """Verify binary produces same results as Python module."""
        # Create test audio
        input_path, bpm = create_test_wav_from_sample('choice', tmp_path)
        binary_output = str(tmp_path / "binary_output.wav")
        module_output = str(tmp_path / "module_output.wav")

        # Process with binary
        result = subprocess.run(
            [binary_path, 'trim',
             '--input', input_path,
             '--output', binary_output,
             '--bpm', str(bpm),
             '--bars', '4',
             '--json'],
            capture_output=True,
            text=True,
            timeout=120
        )
        assert result.returncode == 0, f"Binary failed: {result.stderr}"

        # Parse binary result
        lines = result.stdout.strip().split('\n')
        complete_events = [json.loads(line) for line in lines
                         if line.strip() and json.loads(line).get('type') == 'trim']
        binary_result = complete_events[0]

        # Process with module
        module_result = process_audio(
            input_path=input_path,
            output_path=module_output,
            bpm=bpm,
            bars=4,
            meter=4
        )

        assert module_result.success

        # Compare results
        assert binary_result['success'] == module_result.success
        assert binary_result['sample_rate'] == module_result.sample_rate

        # Downbeat time should be identical (same algorithm)
        assert abs(binary_result['downbeat_time'] - module_result.downbeat_time) < 0.001

        # Output duration should be identical
        assert abs(binary_result['output_duration'] - module_result.output_duration) < 0.01

        # Compare actual audio output
        binary_audio, _ = sf.read(binary_output)
        module_audio, _ = sf.read(module_output)

        assert len(binary_audio) == len(module_audio), "Audio lengths should match"
        np.testing.assert_array_almost_equal(
            binary_audio, module_audio, decimal=6,
            err_msg="Audio content should be identical"
        )

    def test_binary_consistency_across_runs(self, binary_path, tmp_path):
        """Verify binary produces consistent results across multiple runs."""
        input_path, bpm = create_test_wav_from_sample('nutcracker', tmp_path)

        results = []
        for i in range(3):
            output_path = str(tmp_path / f"output_{i}.wav")

            result = subprocess.run(
                [binary_path, 'trim',
                 '--input', input_path,
                 '--output', output_path,
                 '--bpm', str(bpm),
                 '--bars', '4',
                 '--json'],
                capture_output=True,
                text=True,
                timeout=120
            )
            assert result.returncode == 0

            lines = result.stdout.strip().split('\n')
            complete = [json.loads(line) for line in lines
                       if line.strip() and json.loads(line).get('type') == 'trim'][0]
            results.append(complete)

        # All runs should produce identical downbeat times
        downbeat_times = [r['downbeat_time'] for r in results]
        assert all(t == downbeat_times[0] for t in downbeat_times), \
            f"Inconsistent downbeat times: {downbeat_times}"

        # All runs should produce identical output durations
        durations = [r['output_duration'] for r in results]
        assert all(d == durations[0] for d in durations), \
            f"Inconsistent durations: {durations}"


# =============================================================================
# Binary Progress Events Tests
# =============================================================================

@pytest.mark.binary
@pytest.mark.real_audio
class TestBinaryProgressEvents:
    """Test JSON progress event output from binary."""

    def test_progress_events_complete(self, binary_path, nutcracker_wav, tmp_path):
        """Verify all expected progress events are emitted."""
        input_path, bpm = nutcracker_wav
        output_path = str(tmp_path / "output.wav")

        result = subprocess.run(
            [binary_path, 'trim',
             '--input', input_path,
             '--output', output_path,
             '--bpm', str(bpm),
             '--bars', '4',
             '--json'],
            capture_output=True,
            text=True,
            timeout=120
        )

        assert result.returncode == 0

        lines = result.stdout.strip().split('\n')
        json_objects = [json.loads(line) for line in lines if line.strip()]

        # Get all progress events
        progress_events = [j for j in json_objects if j.get('type') == 'progress']

        # Should have progress events for each stage
        stages = [e['stage'] for e in progress_events]
        assert 'loading' in stages
        assert 'detecting' in stages
        assert 'trimming' in stages
        assert 'writing' in stages

        # Should have a complete event
        complete_events = [j for j in json_objects if j.get('type') == 'trim']
        assert len(complete_events) == 1

    def test_progress_percentages_increase(self, binary_path, nutcracker_wav, tmp_path):
        """Verify progress percentages increase over time."""
        input_path, bpm = nutcracker_wav
        output_path = str(tmp_path / "output.wav")

        result = subprocess.run(
            [binary_path, 'trim',
             '--input', input_path,
             '--output', output_path,
             '--bpm', str(bpm),
             '--bars', '4',
             '--json'],
            capture_output=True,
            text=True,
            timeout=120
        )

        lines = result.stdout.strip().split('\n')
        json_objects = [json.loads(line) for line in lines if line.strip()]

        progress_events = [j for j in json_objects if j.get('type') == 'progress']
        percentages = [e['percent'] for e in progress_events]

        # Percentages should generally increase (or stay same for same stage)
        for i in range(1, len(percentages)):
            assert percentages[i] >= percentages[i-1], \
                f"Progress went backwards: {percentages}"


# =============================================================================
# Binary Performance Tests
# =============================================================================

@pytest.mark.binary
@pytest.mark.real_audio
@pytest.mark.slow
class TestBinaryPerformance:
    """Performance tests for the compiled binary."""

    def test_binary_processing_time(self, binary_path, nutcracker_wav, tmp_path):
        """Test that binary completes in reasonable time."""
        import time

        input_path, bpm = nutcracker_wav
        output_path = str(tmp_path / "output.wav")

        start = time.time()
        result = subprocess.run(
            [binary_path, 'trim',
             '--input', input_path,
             '--output', output_path,
             '--bpm', str(bpm),
             '--bars', '8',
             '--json'],
            capture_output=True,
            text=True,
            timeout=120
        )
        elapsed = time.time() - start

        assert result.returncode == 0
        # Binary should complete in under 30 seconds for normal audio
        # (includes startup overhead which is higher than module)
        assert elapsed < 30.0, f"Binary took too long: {elapsed:.2f}s"

    def test_binary_startup_time(self, binary_path):
        """Test that binary starts up quickly (--help)."""
        import time

        start = time.time()
        result = subprocess.run(
            [binary_path, '--help'],
            capture_output=True,
            text=True,
            timeout=30
        )
        elapsed = time.time() - start

        assert result.returncode == 0
        # Help should display in under 5 seconds (PyInstaller startup)
        assert elapsed < 5.0, f"Binary startup too slow: {elapsed:.2f}s"


# =============================================================================
# Binary Edge Cases
# =============================================================================

@pytest.mark.binary
@pytest.mark.real_audio
class TestBinaryEdgeCases:
    """Edge case tests for the compiled binary."""

    def test_binary_single_bar(self, binary_path, choice_wav, tmp_path):
        """Test extracting just 1 bar."""
        input_path, bpm = choice_wav
        output_path = str(tmp_path / "output.wav")

        result = subprocess.run(
            [binary_path, 'trim',
             '--input', input_path,
             '--output', output_path,
             '--bpm', str(bpm),
             '--bars', '1',
             '--json'],
            capture_output=True,
            text=True,
            timeout=120
        )

        assert result.returncode == 0
        assert Path(output_path).exists()

        # Verify duration
        output_audio, _ = sf.read(output_path)
        expected_duration = 1 * 4 * (60.0 / bpm)
        actual_duration = len(output_audio) / 44100
        assert abs(actual_duration - expected_duration) < 0.5

    def test_binary_many_bars(self, binary_path, choice_wav, tmp_path):
        """Test extracting many bars."""
        input_path, bpm = choice_wav
        output_path = str(tmp_path / "output.wav")

        result = subprocess.run(
            [binary_path, 'trim',
             '--input', input_path,
             '--output', output_path,
             '--bpm', str(bpm),
             '--bars', '16',
             '--json'],
            capture_output=True,
            text=True,
            timeout=120
        )

        assert result.returncode == 0
        assert Path(output_path).exists()

    def test_binary_custom_meter(self, binary_path, choice_wav, tmp_path):
        """Test with non-4/4 meter."""
        input_path, bpm = choice_wav
        output_path = str(tmp_path / "output.wav")

        result = subprocess.run(
            [binary_path, 'trim',
             '--input', input_path,
             '--output', output_path,
             '--bpm', str(bpm),
             '--bars', '4',
             '--meter', '3',  # 3/4 time
             '--json'],
            capture_output=True,
            text=True,
            timeout=120
        )

        assert result.returncode == 0
        assert Path(output_path).exists()

        # Duration should reflect 3/4 time
        output_audio, _ = sf.read(output_path)
        expected_duration = 4 * 3 * (60.0 / bpm)  # 3 beats per bar
        actual_duration = len(output_audio) / 44100
        assert abs(actual_duration - expected_duration) < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
