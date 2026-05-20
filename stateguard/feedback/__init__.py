"""Feedback module — Error recovery and human-in-the-loop.

Modules:
    retry:    RetryHandler — auto-retry with feedback (max 2 retries)
    hitl:     HITLHandler — human-in-the-loop approval with timeout
    rollback: RollbackHandler — safe-snapshot restoration
"""
