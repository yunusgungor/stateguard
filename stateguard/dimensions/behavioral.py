"""Behavioral validation — Finite-state machine (FSM) state compliance
and snapshot drift detection.

Provides :class:`BehavioralValidator` that validates system behaviour
against a declared finite-state machine and compares pipeline snapshots
for cumulative drift.
"""

from __future__ import annotations

import json
import math
from typing import Any

from ..models.enums import ValidationDimension, ValidationTier
from ..models.result import ValidationResult
from ..plugin.base import BaseValidator
from ..state.machine import ValidationStateMachine


class BehavioralValidator(BaseValidator):
    """Validator that checks FSM state-transition compliance and snapshot drift.

    Two independent validation modes:

    1. **FSM State Compliance** — validates that a state transition
       described in *output* (e.g. ``{"current": "IDLE", "next": "VALIDATING"}``)
       is legal according to the declared FSM graph.
    2. **Snapshot Comparison** — compares two pipeline snapshots and
       computes a combined structural + cosine drift score.

    Configuration is passed through the *context* dict (modes can be
    combined — the final score is the average of both results).
    """

    # --- Required class attributes (BaseValidator __init_subclass__) ---
    name: str = "behavioral"
    dimension: ValidationDimension = ValidationDimension.BEHAVIORAL
    tier: ValidationTier = ValidationTier.TIER_3

    # ------------------------------------------------------------------
    def __init__(self, drift_threshold: float = 0.15) -> None:
        """Initialise the behavioural validator.

        Args:
            drift_threshold: Max allowed cumulative drift ratio
                (0.0 – 1.0).  The default 0.15 means 15 %.

        Raises:
            ValueError: If *drift_threshold* is not positive.
        """
        super().__init__()
        if not isinstance(drift_threshold, (int, float)) or isinstance(drift_threshold, bool) or drift_threshold <= 0:
            raise ValueError(
                f"drift_threshold must be a positive number, got {drift_threshold!r}"
            )
        self._drift_threshold = drift_threshold

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def drift_threshold(self) -> float:
        """Current drift threshold."""
        return self._drift_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        output: Any,
        context: dict | None = None,
    ) -> ValidationResult:
        """Validate *output* for FSM compliance and/or snapshot drift.

        Args:
            output:  A dict with ``"current"`` / ``"next"`` keys (or
                     ``"state"`` / ``"next_state"`` aliases), a tuple
                     ``(current, next)``, or arbitrary text (when only
                     snapshot comparison is wanted).
            context: May contain ``states``, ``transitions`` overrides
                     for FSM validation and/or ``snapshot_a``,
                     ``snapshot_b``, ``drift_threshold`` for snapshot
                     comparison.

        Returns:
            A :class:`ValidationResult` reflecting the combined
            behavioural validation outcome.
        """
        # Patch 1: context type guard
        ctx = context if isinstance(context, dict) else {}
        details: dict[str, Any] = {}
        scores: list[float] = []

        # --- 1. FSM State Compliance ----------------------------------
        state_score, state_details = self._check_state_compliance(output, ctx)
        scores.append(state_score)
        details["fsm"] = state_details

        # --- 2. Snapshot Comparison -----------------------------------
        snap_score, snap_details = self._check_snapshot(ctx)
        details["snapshot"] = snap_details or {}
        if snap_score is not None:
            scores.append(snap_score)

        # --- 3. Combined score ----------------------------------------
        final_score = sum(scores) / len(scores) if scores else 100.0
        if not math.isfinite(final_score):
            final_score = 0.0
        final_score = max(0.0, min(100.0, final_score))
        passed = final_score >= 50.0

        return ValidationResult(
            score=final_score,
            passed=passed,
            dimension=self.dimension,
            details=details,
        )

    # ------------------------------------------------------------------
    # Internal helpers — FSM State Compliance
    # ------------------------------------------------------------------

    def _parse_state_info(self, output: Any, _from_json: bool = False) -> tuple[str, str] | None:
        """Extract ``(current, next)`` from *output*.

        Supports:
        * A dict with ``"current"`` / ``"next"`` or ``"state"`` /
          ``"next_state"`` keys.
        * A tuple ``(current, next)``.
        * A JSON string that, when parsed, matches one of the above.

        Returns ``None`` when no state info can be extracted.
        """
        if output is None:
            return None

        # String → try JSON parse
        if isinstance(output, str):
            output = output.strip()
            if not output:
                return None
            try:
                parsed = json.loads(output)
            except (json.JSONDecodeError, ValueError):
                return None
            return self._parse_state_info(parsed, _from_json=True)

        # Tuple/list — only for direct callers, not JSON-parsed arrays
        if not _from_json and isinstance(output, (tuple, list)) and len(output) == 2:
            return (str(output[0]).upper(), str(output[1]).upper())

        # Dict
        if isinstance(output, dict):
            # Try primary keys
            cur = output.get("current")
            if cur is None:
                cur = output.get("state")
            nxt = output.get("next")
            if nxt is None:
                nxt = output.get("next_state")
            if cur is not None and nxt is not None:
                return (str(cur).upper(), str(nxt).upper())

        return None

    def _check_state_compliance(
        self,
        output: Any,
        ctx: dict[str, Any],
    ) -> tuple[float, dict[str, Any]]:
        """Validate FSM state transition.

        Returns ``(score, details)``.  If no state info is found in
        *output*, returns ``(100.0, {"warning": ...})``.
        """
        state_info = self._parse_state_info(output)
        if state_info is None:
            return (100.0, {"warning": "No state info in output — FSM check skipped"})

        current_state, next_state = state_info

        # Resolve transitions from context override or FSM dict
        raw_transitions = ctx.get("transitions")
        if isinstance(raw_transitions, set) and raw_transitions:
            transitions = raw_transitions
        elif raw_transitions is not None:
            return (100.0, {"warning": "Invalid transitions type in context — FSM check skipped"})
        else:
            transitions = set(
                (k, v)
                for k, vals in ValidationStateMachine.VALID_TRANSITIONS.items()
                for v in vals
            )

        # Validate states (if context override)
        states_override = ctx.get("states")
        if states_override is not None and isinstance(states_override, (set, frozenset, list, tuple)):
            if current_state not in states_override:
                return (0.0, {
                    "invalid_transition": f"State {current_state!r} not in allowed states",
                    "expected_states": sorted(states_override),
                    "from": current_state,
                    "to": next_state,
                })
        else:
            # Default: validate against FSM states
            all_states = set(ValidationStateMachine.VALID_TRANSITIONS.keys())
            if current_state not in all_states:
                return (0.0, {
                    "invalid_transition": f"State {current_state!r} not in FSM",
                    "expected_states": sorted(all_states),
                    "from": current_state,
                    "to": next_state,
                })

        # Check transition validity
        if (current_state, next_state) in transitions:
            return (100.0, {
                "transition": "valid",
                "from": current_state,
                "to": next_state,
            })

        # Invalid transition — show valid targets
        valid_targets = [
            v for k, v in transitions if k == current_state
        ]
        return (0.0, {
            "invalid_transition": (
                f"Invalid transition {current_state!r} → {next_state!r}"
            ),
            "from": current_state,
            "to": next_state,
            "expected_states": sorted(set(valid_targets)),
        })

    # ------------------------------------------------------------------
    # Internal helpers — Snapshot Comparison
    # ------------------------------------------------------------------

    def _check_snapshot(
        self,
        ctx: dict[str, Any],
    ) -> tuple[float | None, dict[str, Any]]:
        """Compare two snapshots for drift.

        Returns ``(score, details)`` or ``(None, {})`` when snapshots
        are not provided (skipped gracefully).
        """
        snap_a = ctx.get("snapshot_a")
        snap_b = ctx.get("snapshot_b")

        # Skip if either snapshot is missing
        if snap_a is None or snap_b is None:
            return (None, {"warning": "No snapshots provided — snapshot comparison skipped"})

        # Skip if both snapshots are non-dict or empty
        if not isinstance(snap_a, dict) or not isinstance(snap_b, dict):
            return (None, {"warning": "Snapshots must be dicts — comparison skipped"})
        if not snap_a or not snap_b:
            return (None, {"warning": "Empty snapshot — comparison skipped"})

        # Resolve drift threshold from context
        raw_threshold = ctx.get("drift_threshold", self._drift_threshold)
        threshold = (
            raw_threshold
            if isinstance(raw_threshold, (int, float)) and raw_threshold > 0
            else self._drift_threshold
        )

        # Calculate diff
        diff_result = self._calculate_snapshot_diff(snap_a, snap_b)
        drift = diff_result.get("drift", 1.0)

        # Score: inverse of drift relative to threshold
        if drift >= threshold:
            score = max(0.0, 100.0 - (drift / threshold) * 100.0)
        else:
            score = 100.0
        details: dict[str, Any] = {
            "drift": drift,
            "drift_threshold": threshold,
            "structural_diff": diff_result.get("structural_diff", {}),
        }
        if "cosine_similarity" in diff_result:
            details["cosine_similarity"] = diff_result["cosine_similarity"]

        return (score, details)

    def _calculate_snapshot_diff(
        self,
        a: dict[str, Any],
        b: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute structural diff and optional cosine similarity.

        Returns a dict with:
        * ``structural_diff`` — keys added, removed, changed
        * ``drift`` — combined 0-1 drift score
        * ``cosine_similarity`` — present only when both snapshots
          have common numeric values
        """
        keys_a = set(a.keys())
        keys_b = set(b.keys())

        added = keys_b - keys_a
        removed = keys_a - keys_b
        common = keys_a & keys_b

        # Changed keys: same key, different value
        changed: set[str] = set()
        for k in common:
            try:
                if json.dumps(a[k], sort_keys=True) != json.dumps(b[k], sort_keys=True):
                    changed.add(k)
            except (TypeError, ValueError):
                changed.add(k)

        structural_diff = {
            "added": sorted(added),
            "removed": sorted(removed),
            "changed": sorted(changed),
        }

        total_keys = max(len(keys_a), len(keys_b))
        diff_count = len(added) + len(removed) + len(changed)
        
        # Cosine similarity on common numeric values
        cos_sim: float | None = None
        numeric_common = [
            k for k in common
            if isinstance(a.get(k), (int, float)) and isinstance(b.get(k), (int, float))
            and not isinstance(a.get(k), bool) and not isinstance(b.get(k), bool)
            and math.isfinite(a[k]) and math.isfinite(b[k])
        ]

        if len(numeric_common) >= 2:
            vec_a = [float(a[k]) for k in numeric_common]
            vec_b = [float(b[k]) for k in numeric_common]
            # Manual cosine similarity (no numpy dependency)
            dot = sum(va * vb for va, vb in zip(vec_a, vec_b))
            norm_a = sum(v * v for v in vec_a) ** 0.5
            norm_b = sum(v * v for v in vec_b) ** 0.5
            if norm_a > 0 and norm_b > 0:
                cos_sim = max(-1.0, min(1.0, dot / (norm_a * norm_b)))
            else:
                cos_sim = 0.0

        # Drift score: structural ratio * 0.6 + (1 - cosine) * 0.4
        structural_ratio = diff_count / total_keys if total_keys > 0 else 0.0

        if cos_sim is not None:
            drift = structural_ratio * 0.6 + (1.0 - cos_sim) * 0.4
        else:
            drift = structural_ratio  # pure structural

        if not math.isfinite(drift):
            drift = 0.0
        drift = max(0.0, min(1.0, drift))

        result: dict[str, Any] = {
            "structural_diff": structural_diff,
            "drift": drift,
        }
        if cos_sim is not None:
            result["cosine_similarity"] = cos_sim

        return result
