# Edge Case & Boundary Condition Analysis — SnapshotManager

**Review date:** 2026-05-21
**Scope:** `stateguard/state/snapshot.py`, `tests/test_state/test_snapshot.py`
**Test run:** 45/45 passed ✅ (100% line coverage)

---

## 🔴 CRITICAL (0 findings)

No critical issues found. The core implementation is sound.

---

## 🟡 WARNING (3 findings)

### W1. Unhandled `TypeError` for non-JSON-serializable dict keys

| Aspect | Detail |
|--------|--------|
| **File** | `snapshot.py:53-58` |
| **Code** | `except (ValueError, RecursionError) as exc:` |
| **Risk** | Medium |

The size-guard `json.dumps(data, default=str)` on line 54 uses `default=str` to handle non-serializable **values**, but this does **not** cover non-serializable **keys** (tuples, frozensets, custom objects, complex numbers, etc.). When such keys are present, `json.dumps` raises `TypeError`, which is **not caught** by the `except (ValueError, RecursionError)` handler.

```python
m.take_snapshot({(1,2): "value"})  # → unhandled TypeError ✗
```

The error propagates as an undocumented `TypeError` indistinguishable from the "not a dict" input validation error on line 38-41.

**Impact:** Unusual but plausible input crashes the caller with an unhandled exception and a confusing error message instead of a friendly `ValueError`.

**Recommendation:** Either:
1. Add `TypeError` to the except clause and re-raise as `ValueError`:
   ```python
   except (ValueError, RecursionError, TypeError) as exc:
       raise ValueError(f"state_data cannot be serialized: {exc}") from exc
   ```
2. Or add a pre-check that all dict keys are JSON-serializable types:
   ```python
   for key in state_data:
       if not isinstance(key, (str, int, float, bool, type(None))):
           raise TypeError(f"dict key {key!r} is not JSON-serializable")
   ```

---

### W2. Test duplication between `TestTakeSnapshot` and `TestEdgeCases`

| Aspect | Detail |
|--------|--------|
| **File** | `test_snapshot.py` |
| **Risk** | Low (maintenance burden) |

The following test scenarios are tested in **both** classes with nearly identical code:

| Scenario | TestTakeSnapshot | TestEdgeCases |
|----------|-----------------|---------------|
| Empty dict | `test_snapshot_empty_dict` (line 105) | `test_empty_state_data` (line 314) |
| None input | `test_snapshot_none_input` (line 110) | `test_none_state_data` (line 319) |
| String input | `test_snapshot_string_input` (line 114) | `test_string_state_data` (line 323) |
| List input | `test_snapshot_list_input` (line 118) | `test_list_state_data` (line 327) |

**Impact:** If the behavior changes, both tests need updating. Test maintenance overhead doubles for no additional coverage benefit.

**Recommendation:** Remove the 4 duplicated tests from `TestEdgeCases` (lines 314-329), or keep only one set and cross-reference with comments.

---

### W3. Unused fixture `manager_with_snapshot`

| Aspect | Detail |
|--------|--------|
| **File** | `test_snapshot.py:37-41` |
| **Risk** | Low |

The fixture `manager_with_snapshot` is defined but never referenced by any test. This is dead test code that adds confusion.

**Recommendation:** Remove the fixture or add tests that use it.

---

## 🔵 INFO (10 findings)

### I1. Exact 1MB boundary not tested

| Aspect | Detail |
|--------|--------|
| **File** | `test_snapshot.py` |
| **Code** | `test_snapshot_size_guard_near_limit` (500K chars) and `test_size_guard_exact_boundary` (1.1M chars) |

The boundary test uses 500K chars (well under) and 1.1M chars (well over), but never tests **exactly at** `MAX_SNAPSHOT_BYTES`. The actual boundary (verified manually) works correctly at N=1,048,514 chars for the `data` value (plus `_timestamp` overhead of ~32 chars), but no automated test asserts this.

**Recommendation:** Add a test that verifies behavior at exactly the 1MB boundary (one char under, exactly at, one char over).

---

### I2. `diff()` with only `_timestamp` keys not tested

| Aspect | Detail |
|--------|--------|
| **File** | `test_snapshot.py` |
| **Scenario** | `diff({"_timestamp": "t1"}, {"_timestamp": "t2"})` |

After timestamp exclusion, both dicts are empty, so the result should be all-empty diffs. This is a trivial path but uncovered.

---

### I3. `diff()` with `None` values not tested

| Aspect | Detail |
|--------|--------|
| **File** | `test_snapshot.py` |
| **Scenario** | `diff({"x": None}, {"x": "val"})` and vice versa |

The `type` field in changed entries would show `"NoneType"` for a None → value change. This isn't tested.

---

### I4. `diff()` with boolean/`True==1` semantic subtlety

| Aspect | Detail |
|--------|--------|
| **File** | `snapshot.py:127` |
| **Code** | `if snapshot_a[k] != snapshot_b[k]:` |
| **Risk** | Very Low |

`True` and `1` are considered equal in Python (`True == 1` is `True`), so `diff({"x": True}, {"x": 1})` reports **no change**. In JSON semantics, these are distinct. This is technically correct Python behavior, but the implicit JSON serialization context makes it potentially surprising.

Similarly, `float('nan')` always shows as "changed" because `nan != nan` in IEEE 754.

---

### I5. FIFO eviction: no test for empty `_snapshots` path

| Aspect | Detail |
|--------|--------|
| **File** | `test_snapshot.py` |
| **Code** | `snapshot.py:75` — `if len(self._snapshots) >= self.MAX_SNAPSHOTS:` |

While the guard (`>=`) prevents accessing an empty dict, no test explicitly verifies the behavior when `_snapshots` is empty and the condition is False (first 1024 snapshots).

---

### I6. No stress/performance test for FIFO eviction

| Aspect | Detail |
|--------|--------|
| **File** | `test_snapshot.py` |

All FIFO tests use ≤1026 snapshots. No test exercises large-scale eviction (e.g., 10,000+ snapshots) which could surface performance issues with `next(iter(self._snapshots))` for oldest lookup on a large dict.

---

### I7. `diff()` with deeply nested/changed dict values not tested

| Aspect | Detail |
|--------|--------|
| **File** | `test_snapshot.py` |

The `changed` value could be a large nested dict. The test only uses primitive types (str, int, float). No test for:
- Nested dict values that change
- List values that change
- Dict-within-dict structural changes

---

### I8. Double deepcopy in `take_snapshot()`

| Aspect | Detail |
|--------|--------|
| **File** | `snapshot.py:49,80` |

Two deep copies are made per successful `take_snapshot()`:
1. `data = copy.deepcopy(state_data)` (line 49) — used for size check + returned to caller
2. `stored = copy.deepcopy(data)` (line 80) — kept internally

This is **correct** behavior (caller mutations don't affect internal state), but doubles memory for each snapshot. For very large state data near 1MB, this could be a performance concern under heavy usage.

---

### I9. Collision counter starts at `_1`, not `_0`

| Aspect | Detail |
|--------|--------|
| **File** | `snapshot.py:67-72` |

Same-millisecond ID collision produces `base_id_1`, `base_id_2`, ... (never `_0`). This is consistent and tested, but unconventional — most systems start at `_0`. Not a bug, but worth documenting.

---

### I10. `diff()` with `True`/`1` semantic edge case

The `diff` comparison uses Python's `!=` operator. This causes `True == 1` to be considered equal (no change detected), and `float('nan')` to always appear as "changed" even between identical NaN values. Neither case is documented or tested.

---

## ✅ Summary

| Severity | Count | Key |
|----------|-------|-----|
| 🔴 CRITICAL | 0 | — |
| 🟡 WARNING | 3 | W1: Unhandled TypeError for non-JSON keys; W2: Test duplication; W3: Unused fixture |
| 🔵 INFO | 10 | Boundary coverage gaps, edge case test gaps, double deepcopy note, NaN/True==1 semantics |
| **Total** | **13** | |

### Key Action Items

1. **🟡 W1 (highest priority):** Fix the `except` clause in `take_snapshot()` to catch `TypeError` from non-JSON-serializable dict keys, wrapping it as a user-friendly `ValueError`.
2. **🟡 W2:** Remove duplicated test cases from `TestEdgeCases` that are already covered by `TestTakeSnapshot`.
3. **🟡 W3:** Remove or use the unused `manager_with_snapshot` fixture.
4. **🔵 I1:** Add an exact-boundary test at `MAX_SNAPSHOT_BYTES`.
5. **🔵 I2-I4:** Add `diff()` edge case tests for `_timestamp`-only dicts, `None` values, and boolean/NaN semantics.
