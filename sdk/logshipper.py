"""
CIRISLens LogShipper - Drop-in log shipping for CIRIS services.

Copy this file to your project and configure it to send logs to CIRISLens.

Usage:
    from logshipper import LogShipper, setup_logging

    # Option 1: As a logging handler (recommended)
    setup_logging(
        service_name="cirisbilling",
        token="svc_xxx",
        endpoint="https://agents.ciris.ai/lens/api/v1/logs/ingest"
    )

    # Then use standard logging
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Payment processed", extra={"user_id": "u123", "amount": 99.99})

    # Option 2: Direct API
    shipper = LogShipper(
        service_name="cirisbilling",
        token="svc_xxx",
        endpoint="https://agents.ciris.ai/lens/api/v1/logs/ingest"
    )
    shipper.info("Payment processed", event="payment_completed", user_id="u123")
    shipper.flush()  # Send buffered logs
"""

import atexit
import json
import logging
import os
import queue
import socket
import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


__version__ = "1.0.0"


class LogShipper:
    """
    Batched log shipper for CIRISLens.

    Buffers logs and sends them in batches to reduce network overhead.
    Thread-safe and handles failures gracefully.
    """

    def __init__(
        self,
        service_name: str,
        token: str,
        endpoint: str = "https://agents.ciris.ai/lens/api/v1/logs/ingest",
        batch_size: int = 100,
        flush_interval: float = 5.0,
        server_id: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 10.0,
    ):
        """
        Initialize the LogShipper.

        Args:
            service_name: Name of the service (e.g., "cirisbilling")
            token: Service token from CIRISLens admin
            endpoint: CIRISLens log ingestion endpoint
            batch_size: Max logs to buffer before auto-flush
            flush_interval: Seconds between auto-flushes
            server_id: Optional server identifier (defaults to hostname)
            max_retries: Number of retry attempts on failure
            timeout: HTTP request timeout in seconds
        """
        self.service_name = service_name
        self.token = token
        self.endpoint = endpoint
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.server_id = server_id or socket.gethostname()
        self.max_retries = max_retries
        self.timeout = timeout

        self._buffer: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._shutdown = threading.Event()
        self._flush_thread: Optional[threading.Thread] = None

        # Stats
        self._sent_count = 0
        self._error_count = 0
        self._last_error: Optional[str] = None

        # Start background flush thread
        self._start_flush_thread()

        # Register shutdown handler
        atexit.register(self.shutdown)

    def _start_flush_thread(self):
        """Start the background flush thread."""
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()

    def _flush_loop(self):
        """Background thread that periodically flushes logs."""
        while not self._shutdown.is_set():
            self._shutdown.wait(self.flush_interval)
            if not self._shutdown.is_set():
                self.flush()

    def _log(
        self,
        level: str,
        message: str,
        event: Optional[str] = None,
        logger_name: Optional[str] = None,
        request_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **attributes,
    ):
        """Add a log entry to the buffer."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "message": message,
            "server_id": self.server_id,
        }

        if event:
            entry["event"] = event
        if logger_name:
            entry["logger"] = logger_name
        if request_id:
            entry["request_id"] = request_id
        if trace_id:
            entry["trace_id"] = trace_id
        if user_id:
            entry["user_id"] = user_id
        if attributes:
            entry["attributes"] = attributes

        self._buffer.put(entry)

        # Auto-flush if buffer is full
        if self._buffer.qsize() >= self.batch_size:
            self.flush()

    def debug(self, message: str, **kwargs):
        """Log a DEBUG message."""
        self._log("DEBUG", message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log an INFO message."""
        self._log("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log a WARNING message."""
        self._log("WARNING", message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log an ERROR message."""
        self._log("ERROR", message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log a CRITICAL message."""
        self._log("CRITICAL", message, **kwargs)

    def flush(self) -> bool:
        """
        Flush buffered logs to CIRISLens.

        Returns:
            True if successful, False otherwise.
        """
        logs = []

        # Drain the buffer
        while True:
            try:
                logs.append(self._buffer.get_nowait())
            except queue.Empty:
                break

        if not logs:
            return True

        # Send logs
        success = self._send_logs(logs)

        if not success:
            # Re-queue failed logs (at the front)
            for log in reversed(logs):
                try:
                    self._buffer.put_nowait(log)
                except queue.Full:
                    break  # Drop oldest logs if buffer is full

        return success

    def _send_logs(self, logs: list) -> bool:
        """Send logs to CIRISLens with retry logic."""
        payload = "\n".join(json.dumps(log) for log in logs)

        for attempt in range(self.max_retries):
            try:
                request = Request(
                    self.endpoint,
                    data=payload.encode("utf-8"),
                    headers={
                        "Authorization": f"Bearer {self.token}",
                        "Content-Type": "application/x-ndjson",
                    },
                    method="POST",
                )

                with urlopen(request, timeout=self.timeout) as response:
                    if response.status == 200:
                        with self._lock:
                            self._sent_count += len(logs)
                        return True

            except HTTPError as e:
                with self._lock:
                    self._last_error = f"HTTP {e.code}: {e.reason}"
                    self._error_count += 1

                # Don't retry on auth errors
                if e.code in (401, 403):
                    break

            except URLError as e:
                with self._lock:
                    self._last_error = str(e.reason)
                    self._error_count += 1

            except Exception as e:
                with self._lock:
                    self._last_error = str(e)
                    self._error_count += 1

            # Exponential backoff
            if attempt < self.max_retries - 1:
                time.sleep(2 ** attempt)

        return False

    def get_stats(self) -> dict:
        """Get shipping statistics."""
        with self._lock:
            return {
                "sent_count": self._sent_count,
                "error_count": self._error_count,
                "buffer_size": self._buffer.qsize(),
                "last_error": self._last_error,
            }

    def shutdown(self):
        """Gracefully shutdown the shipper."""
        self._shutdown.set()
        self.flush()  # Final flush


class LogShipperHandler(logging.Handler):
    """
    Python logging handler that ships logs to CIRISLens.

    Integrates with standard Python logging so existing code
    works without modification.
    """

    def __init__(self, shipper: LogShipper, min_level: int = logging.INFO):
        """
        Initialize the handler.

        Args:
            shipper: LogShipper instance
            min_level: Minimum log level to ship (default: INFO)
        """
        super().__init__(level=min_level)
        self.shipper = shipper

    def emit(self, record: logging.LogRecord):
        """Emit a log record."""
        try:
            # Extract extra attributes
            attributes = {}
            for key, value in record.__dict__.items():
                if key not in (
                    "name", "msg", "args", "created", "filename", "funcName",
                    "levelname", "levelno", "lineno", "module", "msecs",
                    "pathname", "process", "processName", "relativeCreated",
                    "stack_info", "exc_info", "exc_text", "thread", "threadName",
                    "message", "asctime",
                ):
                    # Only include serializable values
                    if isinstance(value, (str, int, float, bool, type(None))):
                        attributes[key] = value
                    elif isinstance(value, (list, dict)):
                        try:
                            json.dumps(value)  # Test serializability
                            attributes[key] = value
                        except (TypeError, ValueError):
                            pass

            # Extract special fields from attributes
            event = attributes.pop("event", None)
            request_id = attributes.pop("request_id", None)
            trace_id = attributes.pop("trace_id", None)
            user_id = attributes.pop("user_id", None)

            self.shipper._log(
                level=record.levelname,
                message=self.format(record),
                event=event,
                logger_name=record.name,
                request_id=request_id,
                trace_id=trace_id,
                user_id=user_id,
                **attributes,
            )

        except Exception:
            self.handleError(record)


def setup_logging(
    service_name: str,
    token: str,
    endpoint: str = "https://agents.ciris.ai/lens/api/v1/logs/ingest",
    min_level: int = logging.INFO,
    also_console: bool = True,
    **shipper_kwargs,
) -> LogShipper:
    """
    Set up Python logging to ship logs to CIRISLens.

    This is the easiest way to integrate - just call this once at startup
    and all your existing logging calls will automatically ship to CIRISLens.

    Args:
        service_name: Name of the service
        token: Service token from CIRISLens admin
        endpoint: CIRISLens endpoint
        min_level: Minimum log level to ship
        also_console: Also log to console (default: True)
        **shipper_kwargs: Additional args passed to LogShipper

    Returns:
        LogShipper instance (for stats/manual flush)

    Example:
        from logshipper import setup_logging
        import logging

        shipper = setup_logging(
            service_name="cirisbilling",
            token=os.environ["CIRISLENS_TOKEN"]
        )

        logger = logging.getLogger(__name__)
        logger.info("Service started")
        logger.error("Payment failed", extra={"user_id": "u123", "event": "payment_failed"})
    """
    # Create shipper
    shipper = LogShipper(
        service_name=service_name,
        token=token,
        endpoint=endpoint,
        **shipper_kwargs,
    )

    # Create handler
    handler = LogShipperHandler(shipper, min_level=min_level)
    handler.setFormatter(logging.Formatter("%(message)s"))

    # Add to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    # Optionally add console handler
    if also_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        root_logger.addHandler(console_handler)

    # Set level if not already set
    if root_logger.level == logging.NOTSET:
        root_logger.setLevel(min_level)

    return shipper


# Convenience for environment-based configuration
def from_env(service_name: Optional[str] = None) -> LogShipper:
    """
    Create a LogShipper from environment variables.

    Environment variables:
        CIRISLENS_SERVICE_NAME: Service name (required if not passed)
        CIRISLENS_TOKEN: Service token (required)
        CIRISLENS_ENDPOINT: API endpoint (optional)

    Args:
        service_name: Override service name from env

    Returns:
        Configured LogShipper instance
    """
    name = service_name or os.environ.get("CIRISLENS_SERVICE_NAME")
    token = os.environ.get("CIRISLENS_TOKEN")
    endpoint = os.environ.get(
        "CIRISLENS_ENDPOINT",
        "https://agents.ciris.ai/lens/api/v1/logs/ingest"
    )

    if not name:
        raise ValueError("service_name required or set CIRISLENS_SERVICE_NAME")
    if not token:
        raise ValueError("CIRISLENS_TOKEN environment variable required")

    return LogShipper(service_name=name, token=token, endpoint=endpoint)
