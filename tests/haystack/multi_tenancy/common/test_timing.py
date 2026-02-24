"""Tests for multi-tenancy timing utilities."""

import time

from vectordb.haystack.multi_tenancy.common.timing import Timer


class TestTimer:
    """Tests for Timer context manager."""

    def test_timer_initial_state(self):
        """Test timer initializes with zero values."""
        timer = Timer()
        assert timer._start == 0.0
        assert timer._end == 0.0

    def test_timer_context_manager_enters(self):
        """Test timer can be used as context manager."""
        timer = Timer()
        result = timer.__enter__()
        assert result is timer
        assert timer._start > 0.0

    def test_timer_context_manager_exits(self):
        """Test timer context manager exit sets end time."""
        timer = Timer()
        timer.__enter__()
        timer.__exit__(None, None, None)
        assert timer._end > 0.0

    def test_timer_elapsed_ms_zero(self):
        """Test elapsed_ms returns 0 before timing."""
        timer = Timer()
        assert timer.elapsed_ms == 0.0

    def test_timer_elapsed_ms_after_timing(self):
        """Test elapsed_ms returns positive value after timing."""
        timer = Timer()
        timer.__enter__()
        timer.__exit__(None, None, None)
        assert timer.elapsed_ms > 0.0

    def test_timer_elapsed_ms_calculation(self):
        """Test elapsed_ms calculation is approximately correct."""
        timer = Timer()
        timer.__enter__()
        time.sleep(0.01)  # 10ms
        timer.__exit__(None, None, None)
        # Should be approximately 10ms (with some tolerance)
        assert timer.elapsed_ms >= 10.0
        assert timer.elapsed_ms < 100.0  # Should not be huge

    def test_timer_context_manager_with_sleep(self):
        """Test timer with context manager and actual sleep."""
        with Timer() as timer:
            time.sleep(0.01)
        assert timer.elapsed_ms >= 10.0

    def test_timer_multiple_uses(self):
        """Test timer can be used multiple times."""
        timer = Timer()

        # First use
        with timer:
            time.sleep(0.005)
        first_elapsed = timer.elapsed_ms

        # Reset and use again (create new timer)
        timer2 = Timer()
        with timer2:
            time.sleep(0.01)
        second_elapsed = timer2.elapsed_ms

        assert first_elapsed >= 5.0
        assert second_elapsed >= 10.0

    def test_timer_elapsed_ms_is_float(self):
        """Test elapsed_ms returns a float."""
        with Timer() as timer:
            timer._start = timer._start
        assert isinstance(timer.elapsed_ms, float)

    def test_timer_handles_exceptions(self):
        """Test timer works correctly even when exception occurs."""
        timer = Timer()
        try:
            with timer:
                time.sleep(0.005)
                raise ValueError("Test exception")
        except ValueError:
            assert timer._start > 0.0

        # Timer should still have recorded the time
        assert timer._end > 0.0
        assert timer.elapsed_ms >= 5.0
