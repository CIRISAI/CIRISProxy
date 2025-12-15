"""
Tests for sdk/logshipper.py - CIRISLens log shipping.

Uses:
- pytest fixtures for LogShipper instances
- hypothesis for property-based testing
- mocking for HTTP requests
"""

import json
import logging
import os
import queue
import time
from unittest.mock import patch, MagicMock, Mock
from urllib.error import URLError, HTTPError
from io import BytesIO

import pytest
from hypothesis import given, strategies as st, settings

from sdk.logshipper import (
    LogShipper,
    LogShipperHandler,
    setup_logging,
    from_env,
    DEFAULT_CIRISLENS_ENDPOINT,
)


# =============================================================================
# LogShipper Tests
# =============================================================================


class TestLogShipperInit:
    """Tests for LogShipper initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        shipper = LogShipper(
            service_name="test",
            token="token123",
        )

        try:
            assert shipper.service_name == "test"
            assert shipper.token == "token123"
            assert shipper.endpoint == DEFAULT_CIRISLENS_ENDPOINT
            assert shipper.batch_size == 100
            assert shipper.flush_interval == 5.0
            assert shipper.max_retries == 3
        finally:
            shipper.shutdown()

    def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        shipper = LogShipper(
            service_name="custom",
            token="custom_token",
            endpoint="https://custom.endpoint.com",
            batch_size=50,
            flush_interval=10.0,
            server_id="server-1",
            max_retries=5,
            timeout=30.0,
        )

        try:
            assert shipper.service_name == "custom"
            assert shipper.endpoint == "https://custom.endpoint.com"
            assert shipper.batch_size == 50
            assert shipper.flush_interval == 10.0
            assert shipper.server_id == "server-1"
            assert shipper.max_retries == 5
            assert shipper.timeout == 30.0
        finally:
            shipper.shutdown()


class TestLogShipperLogging:
    """Tests for LogShipper logging methods."""

    @pytest.fixture
    def shipper(self):
        """Create a shipper with auto-flush disabled."""
        s = LogShipper(
            service_name="test",
            token="token",
            flush_interval=9999,  # Disable auto-flush
        )
        yield s
        s.shutdown()

    def test_debug_adds_to_buffer(self, shipper):
        """Test debug() adds entry to buffer."""
        shipper.debug("Debug message")
        assert shipper._buffer.qsize() == 1

    def test_info_adds_to_buffer(self, shipper):
        """Test info() adds entry to buffer."""
        shipper.info("Info message")
        assert shipper._buffer.qsize() == 1

    def test_warning_adds_to_buffer(self, shipper):
        """Test warning() adds entry to buffer."""
        shipper.warning("Warning message")
        assert shipper._buffer.qsize() == 1

    def test_error_adds_to_buffer(self, shipper):
        """Test error() adds entry to buffer."""
        shipper.error("Error message")
        assert shipper._buffer.qsize() == 1

    def test_critical_adds_to_buffer(self, shipper):
        """Test critical() adds entry to buffer."""
        shipper.critical("Critical message")
        assert shipper._buffer.qsize() == 1

    def test_log_entry_structure(self, shipper):
        """Test log entry has expected structure."""
        shipper.info("Test message", event="test_event", user_id="u123")

        entry = shipper._buffer.get_nowait()
        assert entry["level"] == "INFO"
        assert entry["message"] == "Test message"
        assert entry["event"] == "test_event"
        assert entry["user_id"] == "u123"
        assert "timestamp" in entry
        assert "server_id" in entry

    def test_log_with_attributes(self, shipper):
        """Test log entry includes custom attributes."""
        shipper.info("Test", custom_field="value", number=42)

        entry = shipper._buffer.get_nowait()
        assert entry["attributes"]["custom_field"] == "value"
        assert entry["attributes"]["number"] == 42

    @given(message=st.text(min_size=1, max_size=1000))
    @settings(max_examples=50)
    def test_handles_various_messages(self, message):
        """Property: handles various message strings."""
        shipper = LogShipper(
            service_name="test",
            token="token",
            flush_interval=9999,
        )
        try:
            shipper.info(message)
            entry = shipper._buffer.get_nowait()
            assert entry["message"] == message
        finally:
            shipper.shutdown()


class TestLogShipperFlush:
    """Tests for LogShipper flush functionality."""

    @pytest.fixture
    def shipper(self):
        """Create a shipper with auto-flush disabled."""
        s = LogShipper(
            service_name="test",
            token="token",
            flush_interval=9999,
        )
        yield s
        s.shutdown()

    def test_flush_empty_buffer_returns_true(self, shipper):
        """Test flushing empty buffer returns True."""
        result = shipper.flush()
        assert result is True

    def test_flush_clears_buffer(self, shipper):
        """Test flush clears the buffer on success."""
        shipper.info("Message 1")
        shipper.info("Message 2")
        assert shipper._buffer.qsize() == 2

        # Mock successful send
        with patch.object(shipper, "_send_logs", return_value=True):
            shipper.flush()

        assert shipper._buffer.qsize() == 0

    def test_flush_requeues_on_failure(self, shipper):
        """Test flush re-queues logs on send failure."""
        shipper.info("Message 1")
        shipper.info("Message 2")

        # Mock failed send
        with patch.object(shipper, "_send_logs", return_value=False):
            shipper.flush()

        # Logs should be re-queued
        assert shipper._buffer.qsize() == 2


class TestLogShipperSendLogs:
    """Tests for LogShipper._send_logs method."""

    @pytest.fixture
    def shipper(self):
        """Create a shipper with auto-flush disabled."""
        s = LogShipper(
            service_name="test",
            token="token",
            flush_interval=9999,
            max_retries=2,
        )
        yield s
        s.shutdown()

    def test_successful_send(self, shipper):
        """Test successful log send."""
        logs = [{"level": "INFO", "message": "Test"}]

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("sdk.logshipper.urlopen", return_value=mock_response):
            result = shipper._send_logs(logs)

        assert result is True
        assert shipper._sent_count == 1

    def test_http_error_increments_error_count(self, shipper):
        """Test HTTP error increments error count."""
        logs = [{"level": "INFO", "message": "Test"}]

        error = HTTPError(
            url="http://test",
            code=500,
            msg="Server Error",
            hdrs={},
            fp=BytesIO(b""),
        )

        with patch("sdk.logshipper.urlopen", side_effect=error):
            with patch("sdk.logshipper.time.sleep"):  # Skip backoff
                result = shipper._send_logs(logs)

        assert result is False
        assert shipper._error_count > 0
        assert "500" in shipper._last_error

    def test_auth_error_no_retry(self, shipper):
        """Test auth errors don't retry."""
        logs = [{"level": "INFO", "message": "Test"}]

        error = HTTPError(
            url="http://test",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=BytesIO(b""),
        )

        call_count = 0

        def mock_urlopen(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise error

        with patch("sdk.logshipper.urlopen", side_effect=mock_urlopen):
            result = shipper._send_logs(logs)

        assert result is False
        assert call_count == 1  # No retries for auth errors

    def test_url_error_retries(self, shipper):
        """Test URL errors are retried."""
        logs = [{"level": "INFO", "message": "Test"}]

        call_count = 0

        def mock_urlopen(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise URLError("Connection refused")

        with patch("sdk.logshipper.urlopen", side_effect=mock_urlopen):
            with patch("sdk.logshipper.time.sleep"):  # Skip backoff
                result = shipper._send_logs(logs)

        assert result is False
        assert call_count == shipper.max_retries


class TestLogShipperStats:
    """Tests for LogShipper.get_stats method."""

    def test_get_stats_structure(self):
        """Test stats have expected structure."""
        shipper = LogShipper(
            service_name="test",
            token="token",
            flush_interval=9999,
        )

        try:
            stats = shipper.get_stats()

            assert "sent_count" in stats
            assert "error_count" in stats
            assert "buffer_size" in stats
            assert "last_error" in stats
        finally:
            shipper.shutdown()


class TestLogShipperShutdown:
    """Tests for LogShipper.shutdown method."""

    def test_shutdown_sets_event(self):
        """Test shutdown sets the shutdown event."""
        shipper = LogShipper(
            service_name="test",
            token="token",
        )

        shipper.shutdown()

        assert shipper._shutdown.is_set()

    def test_shutdown_flushes_buffer(self):
        """Test shutdown flushes remaining logs."""
        shipper = LogShipper(
            service_name="test",
            token="token",
            flush_interval=9999,
        )

        shipper.info("Test message")

        with patch.object(shipper, "_send_logs", return_value=True) as mock:
            shipper.shutdown()

        mock.assert_called_once()


# =============================================================================
# LogShipperHandler Tests
# =============================================================================


class TestLogShipperHandler:
    """Tests for LogShipperHandler logging handler."""

    @pytest.fixture
    def shipper(self):
        """Create a mock shipper."""
        s = MagicMock(spec=LogShipper)
        return s

    def test_emit_calls_shipper_log(self, shipper):
        """Test emit calls shipper._log."""
        handler = LogShipperHandler(shipper)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        handler.emit(record)

        shipper._log.assert_called_once()

    def test_extracts_extra_fields(self, shipper):
        """Test extracts extra fields from log record."""
        handler = LogShipperHandler(shipper)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.event = "custom_event"
        record.user_id = "u123"

        handler.emit(record)

        call_kwargs = shipper._log.call_args[1]
        assert call_kwargs["event"] == "custom_event"
        assert call_kwargs["user_id"] == "u123"


# =============================================================================
# setup_logging Tests
# =============================================================================


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_returns_shipper(self):
        """Test returns a LogShipper instance."""
        shipper = setup_logging(
            service_name="test",
            token="token",
            also_console=False,
        )

        try:
            assert isinstance(shipper, LogShipper)
        finally:
            shipper.shutdown()
            # Clean up handlers
            logging.getLogger().handlers.clear()

    def test_adds_handler_to_root_logger(self):
        """Test adds handler to root logger."""
        root = logging.getLogger()
        initial_count = len(root.handlers)

        shipper = setup_logging(
            service_name="test",
            token="token",
            also_console=False,
        )

        try:
            assert len(root.handlers) > initial_count
        finally:
            shipper.shutdown()
            root.handlers.clear()


# =============================================================================
# from_env Tests
# =============================================================================


class TestFromEnv:
    """Tests for from_env function."""

    def test_creates_from_env_vars(self):
        """Test creates shipper from environment variables."""
        with patch.dict(os.environ, {
            "CIRISLENS_SERVICE_NAME": "test_service",
            "CIRISLENS_TOKEN": "test_token",
        }):
            shipper = from_env()

        try:
            assert shipper.service_name == "test_service"
            assert shipper.token == "test_token"
        finally:
            shipper.shutdown()

    def test_service_name_override(self):
        """Test service_name parameter overrides env var."""
        with patch.dict(os.environ, {
            "CIRISLENS_SERVICE_NAME": "env_name",
            "CIRISLENS_TOKEN": "test_token",
        }):
            shipper = from_env(service_name="override_name")

        try:
            assert shipper.service_name == "override_name"
        finally:
            shipper.shutdown()

    def test_raises_without_service_name(self):
        """Test raises ValueError when service_name missing."""
        with patch.dict(os.environ, {"CIRISLENS_TOKEN": "token"}, clear=True):
            with pytest.raises(ValueError, match="service_name required"):
                from_env()

    def test_raises_without_token(self):
        """Test raises ValueError when token missing."""
        with patch.dict(os.environ, {"CIRISLENS_SERVICE_NAME": "test"}, clear=True):
            with pytest.raises(ValueError, match="CIRISLENS_TOKEN"):
                from_env()

    def test_uses_custom_endpoint(self):
        """Test uses custom endpoint from env."""
        with patch.dict(os.environ, {
            "CIRISLENS_SERVICE_NAME": "test",
            "CIRISLENS_TOKEN": "token",
            "CIRISLENS_ENDPOINT": "https://custom.endpoint.com",
        }):
            shipper = from_env()

        try:
            assert shipper.endpoint == "https://custom.endpoint.com"
        finally:
            shipper.shutdown()


# =============================================================================
# Auto-Flush Tests
# =============================================================================


class TestAutoFlush:
    """Tests for auto-flush behavior."""

    def test_auto_flush_on_batch_size(self):
        """Test auto-flushes when batch size reached."""
        shipper = LogShipper(
            service_name="test",
            token="token",
            batch_size=3,
            flush_interval=9999,
        )

        flush_called = False

        def mock_flush():
            nonlocal flush_called
            flush_called = True
            return True

        try:
            with patch.object(shipper, "flush", side_effect=mock_flush):
                shipper.info("Message 1")
                shipper.info("Message 2")
                assert not flush_called

                shipper.info("Message 3")  # This should trigger flush
                assert flush_called
        finally:
            shipper.shutdown()
