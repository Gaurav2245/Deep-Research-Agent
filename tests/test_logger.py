import logging

from utils.logger import get_logger


def test_get_logger_does_not_propagate_to_root():
    logger = get_logger("test.no.propagate")
    assert logger.propagate is False


def test_get_logger_reuses_existing_handler():
    logger1 = get_logger("test.reuse.handler")
    handler_count_1 = len(logger1.handlers)
    logger2 = get_logger("test.reuse.handler")
    assert len(logger2.handlers) == handler_count_1  # no duplicate handler added


def test_get_logger_sets_level():
    logger = get_logger("test.level", level="DEBUG")
    assert logger.level == logging.DEBUG
