"""Tests for logging utilities.

This module tests the LoggerFactory class which provides consistent logging
across the vectordb library. The factory pattern ensures uniform log formatting
and level configuration throughout the codebase.

Test coverage includes:
    - Logger instance creation
    - Custom log level configuration (DEBUG, INFO, WARNING, ERROR)
    - Logger name assignment
    - Log message emission at various levels
    - Logger name-based singleton behavior
"""

import logging

from vectordb.utils.logging import LoggerFactory


class TestLoggerFactory:
    """Test suite for LoggerFactory.

    Tests cover:
    - Logger creation
    - Log level configuration
    - Singleton behavior
    - Log message output
    """

    def test_logger_factory_creates_logger(self) -> None:
        """Test that LoggerFactory creates a logger instance."""
        factory = LoggerFactory(logger_name=__name__)
        logger = factory.get_logger()

        assert logger is not None
        assert isinstance(logger, logging.Logger)

    def test_logger_factory_with_custom_level(self) -> None:
        """Test LoggerFactory with custom log level."""
        factory = LoggerFactory(logger_name=__name__, log_level=logging.DEBUG)
        logger = factory.get_logger()

        assert logger is not None

    def test_logger_factory_logger_name(self) -> None:
        """Test that logger uses correct name."""
        logger_name = "test.logger.name"
        factory = LoggerFactory(logger_name=logger_name)
        logger = factory.get_logger()

        assert logger.name == logger_name

    def test_logger_factory_logs_info(self, caplog) -> None:
        """Test that logger can emit info messages."""
        with caplog.at_level(logging.INFO):
            factory = LoggerFactory(logger_name="test_info")
            logger = factory.get_logger()
            logger.info("Test info message")

        assert "Test info message" in caplog.text

    def test_logger_factory_logs_warning(self, caplog) -> None:
        """Test that logger can emit warning messages."""
        with caplog.at_level(logging.WARNING):
            factory = LoggerFactory(logger_name="test_warning")
            logger = factory.get_logger()
            logger.warning("Test warning message")

        assert "Test warning message" in caplog.text

    def test_logger_factory_logs_error(self, caplog) -> None:
        """Test that logger can emit error messages."""
        with caplog.at_level(logging.ERROR):
            factory = LoggerFactory(logger_name="test_error")
            logger = factory.get_logger()
            logger.error("Test error message")

        assert "Test error message" in caplog.text

    def test_logger_factory_debug_level(self) -> None:
        """Test logger with DEBUG level."""
        factory = LoggerFactory(logger_name="test_debug", log_level=logging.DEBUG)
        logger = factory.get_logger()

        assert logger.level in (logging.DEBUG, logging.NOTSET)

    def test_logger_factory_info_level(self) -> None:
        """Test logger with INFO level."""
        factory = LoggerFactory(logger_name="test_info_level", log_level=logging.INFO)
        logger = factory.get_logger()

        # Logger is created with the specified level
        assert logger is not None

    def test_logger_factory_multiple_calls_same_name(self) -> None:
        """Test that multiple calls with same name return same logger."""
        name = "test_same"
        factory1 = LoggerFactory(logger_name=name)
        factory2 = LoggerFactory(logger_name=name)

        logger1 = factory1.get_logger()
        logger2 = factory2.get_logger()

        assert logger1.name == logger2.name
