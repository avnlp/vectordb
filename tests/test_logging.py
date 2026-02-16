"""Tests for the logging utility module."""

import logging
import os
from unittest import mock

from vectordb.utils.logging import LoggerFactory


class TestLoggerFactory:
    """Test cases for LoggerFactory class."""

    def test_init_with_default_parameters(self):
        """Test LoggerFactory initialization with default parameters."""
        factory = LoggerFactory("test_logger")
        assert factory.logger_name == "test_logger"
        assert factory.log_level == logging.INFO
        assert factory.log_format == "%(asctime)s - %(levelname)s - %(message)s"
        assert isinstance(factory.logger, logging.Logger)

    def test_init_with_custom_parameters(self):
        """Test LoggerFactory initialization with custom parameters."""
        custom_format = "%(name)s - %(message)s"
        factory = LoggerFactory(
            "custom_logger", log_level=logging.DEBUG, log_format=custom_format
        )
        assert factory.logger_name == "custom_logger"
        assert factory.log_level == logging.DEBUG
        assert factory.log_format == custom_format

    def test_get_logger_returns_logger_instance(self):
        """Test that get_logger returns a Logger instance."""
        factory = LoggerFactory("test_get_logger")
        logger = factory.get_logger()
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_get_logger"

    def test_logger_singleton_behavior(self):
        """Test that LoggerFactory creates unique loggers for different names."""
        factory1 = LoggerFactory("logger_one")
        factory2 = LoggerFactory("logger_two")

        logger1 = factory1.get_logger()
        logger2 = factory2.get_logger()

        assert logger1.name == "logger_one"
        assert logger2.name == "logger_two"
        assert logger1 is not logger2

    def test_configure_from_env_with_default(self):
        """Test configure_from_env with default LOG_LEVEL."""
        with mock.patch.dict(os.environ, {}, clear=True):
            factory = LoggerFactory.configure_from_env("env_logger")
            assert factory.log_level == logging.INFO

    def test_configure_from_env_with_debug(self):
        """Test configure_from_env with DEBUG log level."""
        with mock.patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}):
            factory = LoggerFactory.configure_from_env("env_logger_debug")
            assert factory.log_level == logging.DEBUG

    def test_configure_from_env_with_warning(self):
        """Test configure_from_env with WARNING log level."""
        with mock.patch.dict(os.environ, {"LOG_LEVEL": "WARNING"}):
            factory = LoggerFactory.configure_from_env("env_logger_warning")
            assert factory.log_level == logging.WARNING

    def test_configure_from_env_with_error(self):
        """Test configure_from_env with ERROR log level."""
        with mock.patch.dict(os.environ, {"LOG_LEVEL": "ERROR"}):
            factory = LoggerFactory.configure_from_env("env_logger_error")
            assert factory.log_level == logging.ERROR

    def test_configure_from_env_with_invalid_level_defaults_to_info(self):
        """Test that invalid log level defaults to INFO."""
        with mock.patch.dict(os.environ, {"LOG_LEVEL": "INVALID"}):
            factory = LoggerFactory.configure_from_env("env_logger_invalid")
            assert factory.log_level == logging.INFO

    def test_configure_from_env_with_custom_env_var(self):
        """Test configure_from_env with custom environment variable name."""
        with mock.patch.dict(os.environ, {"CUSTOM_LOG_LEVEL": "DEBUG"}):
            factory = LoggerFactory.configure_from_env(
                "custom_env_logger", env_var="CUSTOM_LOG_LEVEL"
            )
            assert factory.log_level == logging.DEBUG

    def test_logger_level_is_set_correctly(self):
        """Test that the logger level is set correctly."""
        factory = LoggerFactory("level_test", log_level=logging.ERROR)
        assert factory.log_level == logging.ERROR

    def test_multiple_factories_same_name_share_logger(self):
        """Test that factories with same name share the same logger instance."""
        factory1 = LoggerFactory("shared_logger")
        factory2 = LoggerFactory("shared_logger")

        logger1 = factory1.get_logger()
        logger2 = factory2.get_logger()

        assert logger1 is logger2
