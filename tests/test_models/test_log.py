"""Tests for DecisionLogger and StructLogAdapter."""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from stateguard.models.enums import ValidationDimension
from stateguard.models.log import DecisionEntry, DecisionLogger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def logger() -> DecisionLogger:
    return DecisionLogger()


@pytest.fixture
def entry_a() -> DecisionEntry:
    return DecisionEntry(
        agent_id="agent-1",
        step_id="tier_1",
        dimension=ValidationDimension.STRUCTURAL,
        score=85.0,
        decision="pass",
        details={"key": "value"},
    )


@pytest.fixture
def entry_b() -> DecisionEntry:
    return DecisionEntry(
        agent_id="agent-2",
        step_id="tier_2",
        dimension=ValidationDimension.SEMANTIC,
        score=45.0,
        decision="fail",
        details={"reason": "threshold"},
    )


@pytest.fixture
def entry_c() -> DecisionEntry:
    return DecisionEntry(
        agent_id="agent-1",
        step_id="tier_3",
        dimension=ValidationDimension.STRUCTURAL,
        score=70.0,
        decision="retry",
        details={},
    )


# ---------------------------------------------------------------------------
# DecisionLogger Tests — AC 1
# ---------------------------------------------------------------------------

class TestDecisionLogger:
    """DecisionLogger — log and query."""

    def test_log_single_entry(self, logger: DecisionLogger, entry_a: DecisionEntry):
        """Tek kayıt eklenir ve sorgulanır."""
        logger.log(entry_a)
        results = logger.query()
        assert len(results) == 1
        assert results[0].agent_id == "agent-1"
        assert results[0].score == 85.0
        assert results[0].decision == "pass"

    def test_log_multiple_entries(
        self,
        logger: DecisionLogger,
        entry_a: DecisionEntry,
        entry_b: DecisionEntry,
        entry_c: DecisionEntry,
    ):
        """Birden çok kayıt eklenir ve sıra korunur."""
        logger.log(entry_a)
        logger.log(entry_b)
        logger.log(entry_c)
        results = logger.query()
        assert len(results) == 3
        assert results[0].agent_id == "agent-1"
        assert results[1].agent_id == "agent-2"
        assert results[2].agent_id == "agent-1"

    def test_query_empty(self, logger: DecisionLogger):
        """Boş logger boş liste döndürür."""
        assert logger.query() == []

    def test_query_filter_agent_id(
        self,
        logger: DecisionLogger,
        entry_a: DecisionEntry,
        entry_b: DecisionEntry,
        entry_c: DecisionEntry,
    ):
        """agent_id filtresi çalışır."""
        logger.log(entry_a)
        logger.log(entry_b)
        logger.log(entry_c)
        results = logger.query(agent_id="agent-1")
        assert len(results) == 2
        assert all(r.agent_id == "agent-1" for r in results)

    def test_query_filter_agent_id_no_match(
        self,
        logger: DecisionLogger,
        entry_a: DecisionEntry,
    ):
        """Eşleşmeyen agent_id → boş liste."""
        logger.log(entry_a)
        results = logger.query(agent_id="nonexistent")
        assert len(results) == 0

    def test_query_filter_result(
        self,
        logger: DecisionLogger,
        entry_a: DecisionEntry,
        entry_b: DecisionEntry,
        entry_c: DecisionEntry,
    ):
        """result (decision) filtresi çalışır."""
        logger.log(entry_a)  # pass
        logger.log(entry_b)  # fail
        logger.log(entry_c)  # retry
        results = logger.query(result="pass")
        assert len(results) == 1
        assert results[0].decision == "pass"

    def test_query_filter_result_case_insensitive(
        self,
        logger: DecisionLogger,
        entry_a: DecisionEntry,
    ):
        """result filtresi case-insensitive."""
        logger.log(entry_a)
        results = logger.query(result="PASS")
        assert len(results) == 1

    def test_query_filter_time_range(
        self,
        logger: DecisionLogger,
    ):
        """time_range filtresi çalışır."""
        now = datetime.now(timezone.utc)
        old_entry = DecisionEntry(
            agent_id="old", step_id="s1",
            dimension=ValidationDimension.STRUCTURAL,
            score=50.0, decision="pass",
            timestamp=now - timedelta(hours=2),
        )
        new_entry = DecisionEntry(
            agent_id="new", step_id="s2",
            dimension=ValidationDimension.STRUCTURAL,
            score=80.0, decision="pass",
            timestamp=now,
        )
        logger.log(old_entry)
        logger.log(new_entry)

        # Last 1 hour → only new
        results = logger.query(time_range=(now - timedelta(hours=1), now))
        assert len(results) == 1
        assert results[0].agent_id == "new"

        # All time (include old) → both
        results = logger.query(time_range=(now - timedelta(days=1), now))
        assert len(results) == 2

    def test_query_multiple_filters_and(
        self,
        logger: DecisionLogger,
        entry_a: DecisionEntry,
        entry_c: DecisionEntry,
    ):
        """Birden çok filtre AND ile birleşir."""
        # entry_a: agent-1, STRUCTURAL, pass
        # entry_c: agent-1, STRUCTURAL, retry
        logger.log(entry_a)
        logger.log(entry_c)
        results = logger.query(agent_id="agent-1", result="pass")
        assert len(results) == 1
        assert results[0].decision == "pass"

    def test_thread_safe(self, logger: DecisionLogger):
        """Thread-safe: çoklu thread'den log ekleme."""
        entry = DecisionEntry(
            agent_id="t", step_id="s1",
            dimension=ValidationDimension.STRUCTURAL,
            score=100.0, decision="pass",
        )

        def add_100():
            for _ in range(100):
                logger.log(entry)

        threads = [threading.Thread(target=add_100) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        results = logger.query()
        assert len(results) == 400

    def test_query_returns_copy(self, logger: DecisionLogger, entry_a: DecisionEntry):
        """query() kopya döndürür, orijinal liste değişmez."""
        logger.log(entry_a)
        results = logger.query()
        results.clear()
        assert len(logger.query()) == 1


# ---------------------------------------------------------------------------
# StructLogAdapter Tests — AC 2
# ---------------------------------------------------------------------------

class TestStructLogAdapter:
    """StructLogAdapter — configure, bind, log methods."""

    def test_import(self):
        """Module import edilebilir."""
        from stateguard.utils.logging import StructLogAdapter
        assert StructLogAdapter is not None

    def test_configure_default(self):
        """configure() varsayılan parametrelerle çağrılabilir."""
        from stateguard.utils.logging import StructLogAdapter
        # Should not raise
        StructLogAdapter.configure()

    def test_configure_custom_level(self):
        """configure() özel level ile çağrılabilir."""
        from stateguard.utils.logging import StructLogAdapter
        StructLogAdapter.configure(level="DEBUG", json_format=True)

    def test_create_adapter(self):
        """StructLogAdapter oluşturulabilir."""
        from stateguard.utils.logging import StructLogAdapter
        log = StructLogAdapter("test-logger")
        assert log.name == "test-logger"

    def test_create_root_adapter(self):
        """İsimsiz adapter root logger kullanır."""
        from stateguard.utils.logging import StructLogAdapter
        log = StructLogAdapter()
        assert log.name == "stateguard"

    def test_bind_returns_new_adapter(self):
        """bind() yeni adapter döndürür."""
        from stateguard.utils.logging import StructLogAdapter
        log = StructLogAdapter("test")
        bound = log.bind(request_id="abc")
        assert bound is not log
        assert isinstance(bound, StructLogAdapter)

    def test_log_methods_do_not_raise(self):
        """Tüm log metodları çağrılabilir (exception fırlatmaz)."""
        from stateguard.utils.logging import StructLogAdapter
        log = StructLogAdapter("test")
        # Should not raise regardless of structlog availability
        log.debug("test event", key="value")
        log.info("test event")
        log.warning("test warning")
        log.error("test error")
        log.critical("test critical")


# ---------------------------------------------------------------------------
# Integration: DecisionLogger + Engine — AC 3
# ---------------------------------------------------------------------------

class TestDecisionLoggerEngineIntegration:
    """Engine'in DecisionLogger ile entegrasyonu."""

    def test_engine_accepts_decision_logger(self):
        """Engine DecisionLogger parametresini kabul eder."""
        from stateguard.core.engine import ValidationEngine
        from stateguard.models.log import DecisionLogger

        dlog = DecisionLogger()
        engine = ValidationEngine(decision_logger=dlog)
        assert engine._decision_logger is dlog

    def test_engine_creates_default_logger(self):
        """Engine varsayılan DecisionLogger oluşturur."""
        from stateguard.core.engine import ValidationEngine

        engine = ValidationEngine()
        assert engine._decision_logger is not None
        assert isinstance(engine._decision_logger, DecisionLogger)
