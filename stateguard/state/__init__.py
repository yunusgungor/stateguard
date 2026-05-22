"""State module — Pipeline state tracking and drift detection.

Modules:
    machine:   ValidationStateMachine — 7-state FSM (IDLE → VALIDATING → ... → COMPLETED)
    snapshot:  SnapshotManager — JSON state serialization & diff
    drift:     DriftDetector — cumulative drift detection with weighted cosine + structural diff
"""
