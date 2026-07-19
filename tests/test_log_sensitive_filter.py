"""Tests for the sensitive-data masking filter (RFC G2)."""

import logging
from collections.abc import Generator
from pathlib import Path

import pytest
from akquant.log import (
    LogConfig,
    SensitiveFilter,
    configure_logging,
    get_logger,
    mask_sensitive_text,
    mask_sensitive_value,
)


class TestMaskSensitiveValue:
    """Unit coverage for the per-value masking helper."""

    def test_full_mask_keys_are_wholly_redacted(self) -> None:
        """Credential-class keys are replaced wholesale."""
        assert mask_sensitive_value("password", "hunter2") == "****"
        assert mask_sensitive_value("api_key", "sk-abcdef") == "****"
        assert mask_sensitive_value("auth_code", "9999") == "****"

    def test_tail_mask_keeps_last_four_chars(self) -> None:
        """Account-class keys keep only their trailing 4 chars for reconciliation."""
        assert mask_sensitive_value("user_id", "1234567890") == "****7890"
        assert mask_sensitive_value("account", "SIM-ABCD-4321") == "****4321"

    def test_tail_mask_short_value_is_fully_masked(self) -> None:
        """A value no longer than the retained tail leaks nothing useful."""
        assert mask_sensitive_value("user_id", "12") == "****"
        assert mask_sensitive_value("user_id", "1234") == "****"

    def test_key_matching_is_case_insensitive(self) -> None:
        """Key lookup normalizes case before matching."""
        assert mask_sensitive_value("PASSWORD", "x") == "****"
        assert mask_sensitive_value("BrokerID", "99887766") == "****7766"

    def test_non_sensitive_keys_pass_through_unchanged(self) -> None:
        """Business fields are never altered."""
        assert mask_sensitive_value("symbol", "600000.SH") == "600000.SH"
        assert mask_sensitive_value("quantity", 100) == 100


class TestMaskSensitiveText:
    """Unit coverage for inline `key=value` message masking."""

    def test_masks_inline_pairs(self) -> None:
        """Bare `key=value` sensitive pairs in the message are redacted."""
        assert (
            mask_sensitive_text("login password=hunter2 ok") == "login password=**** ok"
        )
        assert "****7890" in mask_sensitive_text("user_id=1234567890 connected")

    def test_masks_quoted_and_colon_forms(self) -> None:
        """Quoted values and colon separators are handled too."""
        assert mask_sensitive_text('token="abc123"') == 'token="****"'
        assert mask_sensitive_text("api_key: sk-xyz") == "api_key: ****"

    def test_leaves_non_sensitive_pairs(self) -> None:
        """Non-sensitive inline pairs are untouched."""
        assert (
            mask_sensitive_text("symbol=600000.SH qty=100")
            == "symbol=600000.SH qty=100"
        )

    def test_empty_message_is_safe(self) -> None:
        """Empty messages short-circuit without error."""
        assert mask_sensitive_text("") == ""


class TestSensitiveFilterRecord:
    """The filter mutates records regardless of formatter."""

    def _record(self, msg: str, **extra: object) -> logging.LogRecord:
        """Build a bare LogRecord with optional structured extras."""
        record = logging.LogRecord(
            name="akquant.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg=msg,
            args=(),
            exc_info=None,
        )
        for key, value in extra.items():
            setattr(record, key, value)
        return record

    def test_filter_masks_structured_extra(self) -> None:
        """Structured extra fields are masked in place on the record."""
        record = self._record("login", password="hunter2", user_id="1234567890")
        SensitiveFilter().filter(record)
        assert getattr(record, "password") == "****"
        assert getattr(record, "user_id") == "****7890"

    def test_filter_masks_inline_message(self) -> None:
        """Inline message pairs are masked in place on the record."""
        record = self._record("connect token=abcdef done")
        SensitiveFilter().filter(record)
        assert record.getMessage() == "connect token=**** done"

    def test_filter_returns_true(self) -> None:
        """Filters must not drop records — masking never suppresses."""
        assert SensitiveFilter().filter(self._record("hi")) is True


class TestConfiguredLoggingMasks:
    """End-to-end: configure_logging wires the filter onto handlers."""

    @pytest.fixture(autouse=True)
    def _reset(self) -> Generator[None, None, None]:
        """Restore the library-silent default after each test."""
        yield
        configure_logging(LogConfig(console=False, filename=None, reset_handlers=True))

    def test_file_output_is_masked_by_default(self, tmp_path: Path) -> None:
        """Sensitive values never reach the log file under default config."""
        log_file = tmp_path / "audit.log"
        configure_logging(
            LogConfig(console=False, filename=str(log_file), level="INFO")
        )
        logger = get_logger("gateway.live")
        logger.info("login user_id=1234567890 password=hunter2")
        for handler in logging.getLogger("akquant").handlers:
            handler.flush()
        content = log_file.read_text(encoding="utf-8")
        assert "1234567890" not in content
        assert "hunter2" not in content
        assert "****7890" in content

    def test_masking_can_be_disabled(self, tmp_path: Path) -> None:
        """Setting mask_sensitive=False opts out of redaction."""
        log_file = tmp_path / "raw.log"
        configure_logging(
            LogConfig(
                console=False,
                filename=str(log_file),
                level="INFO",
                mask_sensitive=False,
            )
        )
        logger = get_logger("gateway.live")
        logger.info("login user_id=1234567890")
        for handler in logging.getLogger("akquant").handlers:
            handler.flush()
        assert "1234567890" in log_file.read_text(encoding="utf-8")
