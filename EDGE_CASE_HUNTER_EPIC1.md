# Edge Case Hunter Report — StateGuard Epic 1

**Review date:** 2026-05-22
**Review scope:** Stories 1-1 through 1-5 (5 commits)
**Commit IDs:** 1675449, 7a1f81b, 6e73234, f99a75c, cef400d
**Reviewer:** Edge Case Hunter agent

---

## Methodology

Each file was reviewed line-by-line for:
- Null/boundary inputs
- Type mismatches and implicit coercions
- Concurrency / thread safety
- Failure modes and exception handling
- Resource leaks
- Logic errors in conditional branches
- ReDoS / injection vectors
- Silent data corruption paths

---

## File-by-File Findings

---

### Story 1-2: `stateguard/plugin/base.py` — Plugin System Hardening (W1+W2 fix)

#### ECH-01 🔵 INFO: `hasattr(cls, "__abstractmethods__")` early-return gate — RESOLVED (works correctly)

**File:** `stateguard/plugin/base.py:75`
**Code:** `if ABC in cls.__bases__ or hasattr(cls, "__abstractmethods__"): return`

**Analysis (empirically verified):** During `__init_subclass__`, `ABCMeta.__new__` has NOT yet set `cls.__abstractmethods__` on the newly-created class. And critically, for metaclass-instance attribute lookup, the class's own MRO (BaseValidator → ABC → object) is NOT traversed — only the metaclass's MRO is checked. Therefore `hasattr(cls, "__abstractmethods__")` returns **False** for ALL classes during `__init_subclass__` regardless of whether they are abstract or concrete.

```
ConcreteValidator during __init_subclass__:
  __abstractmethods__ in cls.__dict__: NOT_IN_OWN_DICT
  hasattr(__abstractmethods__): False         ← because ABCMeta hasn't set it yet
```

This means the `hasattr` branch is essentially dead code — it never triggers. The ONLY working mechanism for abstract-class bypass is `ABC in cls.__bases__`, which requires the abstract intermediate to explicitly list `abc.ABC` in its bases (which the test `test_abstract_intermediate_still_skips_check` does correctly).

**Gap:** Abstract intermediate classes created WITHOUT `abc.ABC` in direct bases (e.g., `class Abs(BaseValidator): @abstractmethod ...`) will be subject to the attribute enforcement and cannot skip it. This is a minor edge case — if you're creating an abstract intermediate with `@abstractmethod` but forgot to list `abc.ABC` in bases, you'll get a `TypeError` about missing attributes instead of being allowed to defer them.

**Recommendation:** Add a comment explaining that `hasattr(__abstractmethods__)` always returns False during `__init_subclass__` and is preserved only for forward-compatibility. Or remove it to simplify the code.

---

#### ECH-02 🟡 WARNING: Validate callable check only checks `cls.__dict__`, ignoring inherited non-callable values

**File:** `stateguard/plugin/base.py:117`
**Code:** `validate_attr = cls.__dict__.get("validate")`

**Issue:** The callable check only looks at `cls.__dict__`, not inherited values. If a parent class defines `validate = "not-callable"`, and the child inherits it unchanged, the child's `cls.__dict__` won't have `validate` at all. The check returns `None`, so no TypeError is raised. But `child.validate("test")` would fail at runtime.

This is a much narrower edge case — a parent would have to define a non-callable `validate` class attribute, which itself would be caught by the W2 enforcement on that parent. However, if the parent was an abstract intermediate class (which skips enforcement via ECH-01), the non-callable could slip through.

**Impact:** Low-to-medium. Requires: abstract intermediate + non-callable validate attribute in the abstract class + child inherits without override.

**Recommendation:** Check both `cls.__dict__` and inherited values:
```python
validate_attr = cls.__dict__.get("validate")
if validate_attr is None:
    validate_attr = getattr(cls, "validate", None)
if validate_attr is not None and not callable(validate_attr):
    raise TypeError(...)
```

---

#### ECH-03 🔵 INFO: `__init_subclass__` MRO traversal assumes `BaseValidator` is in MRO

**File:** `stateguard/plugin/base.py:89`
**Code:** `base_idx = mro.index(BaseValidator)`

**Issue:** If someone creates a class hierarchy where `BaseValidator` is not in the direct MRO (e.g., a mixin that happens to implement the same interface), this raises `ValueError`. This is extremely unlikely in normal usage but could be triggered by advanced metaclass/mixin patterns.

**Impact:** Very low — misuse pattern.

**Recommendation:** Consider a guard:
```python
if BaseValidator not in mro:
    return  # or raise
```

---

#### ECH-04 🔵 INFO: `super().__init_subclass__(**kwargs)` — metaclass conflicts

**File:** `stateguard/plugin/base.py:72`
**Code:** `super().__init_subclass__(**kwargs)`

**Issue:** If a subclass uses a custom metaclass that also overrides `__init_subclass__`, the `**kwargs` forwarding works, but multiple `__init_subclass__` implementations might interfere. No tests cover custom metaclass scenarios.

**Impact:** Low — advanced usage only.

---

### Story 1-3: `stateguard/core/llm_client.py` — LLM Client Abstraction

#### ECH-05 🟡 WARNING: Protocol default `build_prompt` is English; `HTTPLLMClient` uses Turkish — silent language switch for existing users

**File:** `stateguard/core/llm_client.py:49`
**Code:** `return f"Evaluate the following output:\n{output}"`

**Issue:** Before this story, `LLMValidator` directly called `HTTPLLMClient._build_prompt(output)`, which always used the Turkish `JUDGE_PROMPT_TEMPLATE`. Now, `LLMValidator` calls `self._llm_client.build_prompt(output)`. If a user injected a custom `LLMClient` that doesn't override `build_prompt`, the protocol's **default English prompt** is used instead of Turkish.

This is the correct abstraction behavior, but it's a **silent breaking change** for anyone who:
1. Previously relied on `LLMValidator` always using the Turkish prompt, AND
2. Uses a custom `LLMClient` that doesn't override `build_prompt`

Previously, the Turkish prompt was hardcoded in `LLMValidator`'s logic. Now it's in `HTTPLLMClient.build_prompt()`. If a custom client doesn't provide `build_prompt`, it silently falls to English.

**Impact:** Medium — behavioral change for users with custom clients. The test `test_custom_client_build_prompt` only tests a custom client that DOES override `build_prompt`. No test covers the "custom client WITHOUT override" case.

**Recommendation:** Either:
1. Add a test that verifies the default prompt is used when `build_prompt` is not overridden, and DOCUMENT this behavior, or
2. Make `HTTPLLMClient` the default prompt source for all clients (less clean)

---

#### ECH-06 🔴 CRITICAL: `httpx` lazy import — missing dependency at runtime

**File:** `stateguard/core/llm_client.py:135`
**Code:** `import httpx  # lazy import — httpx is optional`

**Issue:** The comment says "httpx is optional" but `HTTPLLMClient.ask()` **cannot function without httpx**. If `httpx` is not installed, the lazy import raises `ImportError`, which is caught by the bare `except Exception` on line 138 and re-raised as `LLMError("LLM request failed: ...")`. The error message is misleading — it looks like a network error when it's actually a missing dependency.

Additionally, the single lazy import is inside a try/except that catches **all** exceptions:

```python
try:
    import httpx
    with httpx.Client(timeout=self.timeout_seconds) as client:
        response = client.post(url, headers=headers, json=payload)
except Exception as e:
    raise LLMError(f"LLM request failed: {e}") from e
```

This conflates:
- `ImportError` (missing httpx)
- `httpx.ConnectError` (network down)
- `httpx.TimeoutException` (timeout)
- Any other unexpected error

**Impact:** High — poor error reporting in production, misleading debug output.

**Recommendation:** Separate the import from the network call:

```python
try:
    import httpx
except ImportError as e:
    raise LLMError(
        "httpx is required for HTTPLLMClient. "
        "Install it with: pip install httpx"
    ) from e

try:
    with httpx.Client(timeout=self.timeout_seconds) as client:
        response = client.post(url, headers=headers, json=payload)
except httpx.RequestError as e:
    raise LLMError(f"LLM request failed: {e}") from e
```

---

#### ECH-07 🟡 WARNING: `build_prompt` in `LLMClient` protocol — `output: str` type, but `LLMValidator.validate()` calls with `Any`

**File:** `stateguard/core/llm_client.py:40` and `stateguard/core/tier3.py:62`
**Code:** `def build_prompt(self, output: str) -> str:`

**Issue:** `LLMValidator.validate()` accepts `output: Any` (from base interface), checks `isinstance(output, str)`, and calls `self._llm_client.build_prompt(output)`. This is correct — the isinstance guard ensures `output` is `str`. However, the `LLMClient` protocol itself (which could be used by other code) assumes `str` input without any validation.

**Impact:** Low — the guard in `LLMValidator` protects the common path.

---

#### ECH-08 🔵 INFO: `ask()` returns `content.strip()` — loss of whitespace-significant responses

**File:** `stateguard/core/llm_client.py:161`
**Code:** `return content.strip()`

**Issue:** If an LLM response is meaningful whitespace-only (edge case), `strip()` could produce an empty string. The response is then used in `LLMValidator.validate()` which checks for EVET/HAYIR, so an empty string would be caught by the "unrecognised response" error path.

**Impact:** Negligible — correct failure mode.

---

### Story 1-1: `stateguard/core/tier2.py` — Ensemble Training/Calibration

#### ECH-09 🟡 WARNING: `_extract_features` returns `None` for 0-samples or 0-features, but `fit()` raises ValueError — inconsistent behavior with `validate()`

**File:** `stateguard/core/tier2.py` lines 287-291 (fit) vs 491-498 (validate)

**Issue:** When `_extract_features` returns `None`:
- `fit()` → raises `ValueError`
- `validate()` → returns `ValidationResult(score=0.0, passed=False, error=...)`

This inconsistency means the user gets different failure modes for the same invalid input depending on whether they called `fit()` or `validate()`.

Specifically, empty data:
```python
v.fit([])           # ValueError
v.validate([])      # ValidationResult with score=0
```

**Impact:** Medium — API inconsistency, could confuse users.

**Recommendation:** Make both paths consistent. Either both raise `ValueError`, or both return a validation result.

---

#### ECH-10 🟡 WARNING: `NaN`/`inf` values in features — silent corruption

**File:** `stateguard/core/tier2.py:450-454`
**Code:** `arr = np.asarray(features, dtype=np.float64)`

**Issue:** `np.asarray` happily converts lists containing `NaN`, `inf`, or `-inf` to float64 arrays. These propagate through the analyzers:
- `ZScoreAnalyzer.fit()`: `np.mean()` + `np.std()` produce `NaN` when input contains `NaN` (unless `np.nanmean`/`np.nanstd` are used)
- `np.std(data, axis=0)` with `NaN` → `NaN`
- Division by zero at `predict()` line 82: `(data - self.mean_) / self.std_` — if std has NaN, result is NaN
- `IsolationForest.fit()` with NaN: sklearn may produce NaN or raise
- `OneClassSVM.fit()` with NaN: similar

The validation result may show `score=0.0` or `score=100.0` depending on how NaN propagates, without indicating that the data was malformed.

**Impact:** Medium — silent data corruption, unpredictable results.

**Recommendation:** Add a NaN/inf guard in `_extract_features`:
```python
arr = np.asarray(features, dtype=np.float64)
if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
    return None  # or raise ValueError with clear message
```

---

#### ECH-11 🔵 INFO: `_analyzers` lazy init is NOT thread-safe

**File:** `stateguard/core/tier2.py:416-435`
**Code:**
```python
def _ensure_analyzers(self) -> list[Any]:
    if self._analyzers is not None:
        return self._analyzers
    self._analyzers = [...]
    return self._analyzers
```

**Issue:** Double-checked locking anti-pattern. If two threads call `_ensure_analyzers()` simultaneously, both could enter the `if` check, and two sets of analyzers could be created. One set would overwrite the other, leaking the first set. This is a classic race condition.

**Impact:** Low, unless `EnsembleValidator` is used in a multi-threaded validation pipeline (e.g., web server handling concurrent requests).

**Recommendation:** Add a threading lock:
```python
import threading
self._analyzers_lock = threading.Lock()

def _ensure_analyzers(self) -> list[Any]:
    if self._analyzers is not None:
        return self._analyzers
    with self._analyzers_lock:
        if self._analyzers is not None:  # double-check
            return self._analyzers
        self._analyzers = [...]
        return self._analyzers
```

---

#### ECH-12 🟡 WARNING: Atomic write — `os.replace` failure leaves temp file

**File:** `stateguard/core/tier2.py:333-347`
**Code:**
```python
tmp = tempfile.NamedTemporaryFile(...)
try:
    _joblib.dump(state, tmp.name)
    os.replace(tmp.name, str(path))
except Exception:
    if os.path.exists(tmp.name):
        os.unlink(tmp.name)
    raise
```

**Issue:** If `os.replace` fails (e.g., cross-filesystem rename, permissions error, or disk full), the except block cleans up the temp file. However, there's a **gap**: if `_joblib.dump` succeeds but `os.replace` fails:
1. The temp file is deleted (good)
2. The original file at `path` remains unchanged (good)
3. But the exception propagates as a generic `Exception`

If `os.replace` fails because of a **permissions error on the target** (common on read-only filesystems or containers), the user sees an opaque error with no indication of _why_. The error message only says "LLM request failed" or similar if caught upstream.

Additionally, there's a subtle race: between `os.replace` and the existence check, another process could delete the temp file.

**Recommendation:** Catch specific exceptions:
```python
import errno
try:
    _joblib.dump(state, tmp.name)
    os.replace(tmp.name, str(path))
except OSError as e:
    if os.path.exists(tmp.name):
        os.unlink(tmp.name)
    raise ValueError(f"Failed to save model to {path}: {e}") from e
except Exception:
    if os.path.exists(tmp.name):
        os.unlink(tmp.name)
    raise
```

---

#### ECH-13 🔴 CRITICAL: `load()` reads arbitrary pickled data — remote code execution vector

**File:** `stateguard/core/tier2.py:370`
**Code:** `state = _joblib.load(str(path))`

**Issue:** `joblib.load` is a wrapper around `pickle.load`. Loading untrusted `.joblib` files can execute arbitrary Python code. The version check on line 374 runs AFTER deserialization, so an attacker can craft a malicious file with the correct `_model_version` key and arbitrary code.

For a CLI-based validation tool, this is a significant security concern if models are shared or downloaded.

**Impact:** High — RCE via crafted `.joblib` file.

**Recommendation:** Document that `.joblib` files must come from trusted sources. If practical, consider:
1. Validating the file's integrity (checksum comparison)
2. Running `joblib.load` in a restricted environment (sandbox, subprocess)
3. Using `pickle.Unpickler` with a restricted `find_class`

---

#### ECH-14 🟡 WARNING: `load()` uses `state[]` with `KeyError` on missing keys

**File:** `stateguard/core/tier2.py:380-390`
**Code:** `state["_contamination"]`, `state["_svm_nu"]`, etc.

**Issue:** If a saved model was produced by a different version of the code with different state keys, the `state[key]` access raises `KeyError`. Only `_model_version` is checked via `.get()` (line 373). If `_model_version` is "1.0.0" but the state dict is missing keys (e.g., partial save, truncated file), the error is an opaque `KeyError`.

**Impact:** Low — version check should prevent most cases, but truncated/corrupt files could bypass it.

**Recommendation:** Use `.get()` with defaults or catch `KeyError`:

```python
try:
    v = cls(
        contamination=state["_contamination"],
        ...
    )
except KeyError as e:
    raise ValueError(f"Saved model is corrupt or incompatible: missing key {e}") from e
```

---

#### ECH-15 🔵 INFO: `save()` with path that's a symlink to a directory passes `is_dir()` check

**File:** `stateguard/core/tier2.py:317`
**Code:** `if path.is_dir(): raise ValueError(...)`

**Issue:** `Path.is_dir()` follows symlinks by default. If `path` is a symlink to a directory, `is_dir()` returns `True` and raises. If `path` is a symlink to a non-existent file, `is_dir()` returns `False` and proceeds. This is correct behavior, but a symlink to a regular file would pass the check silently.

**Impact:** Negligible — standard POSIX behavior.

---

#### ECH-16 🟡 WARNING: `setup()` catches bare `Exception` — swallows all errors silently

**File:** `stateguard/core/tier2.py:410-412`
**Code:**
```python
except Exception:
    # Config not available or model not found — no-op is fine.
    pass
```

**Issue:** `setup()` catches `Exception` and silently swallows it. This means:
- `ConfigManager()` could fail (e.g., corrupted config file) → silently ignored
- `self.__class__.load()` could fail with `FileNotFoundError` → silently ignored
- `self.__dict__.update(loaded.__dict__)` could fail with `TypeError` → silently ignored
- Any `ImportError` (missing joblib) → silently ignored
- `KeyboardInterrupt` is NOT caught (it inherits from `BaseException`, not `Exception`) — good

**Impact:** Medium — bugs in config/model loading are hidden.

**Recommendation:** Log the exception instead of silently swallowing:

```python
except Exception:
    logger.debug("setup() failed to load model from config", exc_info=True)
```

---

#### ECH-17 🟡 WARNING: Auto-fit `UserWarning` `stacklevel=2` may point to wrong frame

**File:** `stateguard/core/tier2.py:511`
**Code:** `warnings.warn(..., UserWarning, stacklevel=2)`

**Issue:** `stacklevel=2` means the warning points to the caller of the function that calls `warnings.warn`. If `validate()` is called directly by user code:

```
user code → validate() → warnings.warn()
```

Then `stacklevel=2` would point to `validate()` itself (the caller of `warn`), not the user. The correct `stacklevel` to point to user code would be:

```
user code → validate() → warnings.warn()  # stacklevel=2 points to validate(), stacklevel=1 points to warn()
```

Wait, actually:
- `stacklevel=1` (default): points to the line where `warnings.warn()` is called (inside `validate()`)
- `stacklevel=2`: points to the caller of the function that called `warnings.warn` — that's the caller of `validate()`, which IS the user code

So `stacklevel=2` IS correct for pointing to user code. Good.

**Impact:** Very low — correct behavior.

---

#### ECH-18 🔵 INFO: `ZScoreAnalyzer` zero-std guard modifies `self.std_` in-place

**File:** `stateguard/core/tier2.py:74`
**Code:** `self.std_[self.std_ == 0] = 1.0`

**Issue:** When a feature has zero standard deviation (all training samples identical), the std is set to 1.0 to prevent division by zero. This means the feature is treated as though it has unit variance, which may produce misleading z-scores for that feature.

**Impact:** Low — mathematically reasonable behavior for constant features. Documented behavior.

---

#### ECH-19 🟡 WARNING: `fit()` with 2D array of shape (n_samples, 0) — edge case

**File:** `stateguard/core/tier2.py:286` and `_extract_features` lines 464-465

**Issue:** 
```python
if arr.shape[1] == 0 or arr.shape[0] == 0:
    return None
```

A 2D array with shape (5, 0) passes `ndim == 2` check, but `arr.shape[1] == 0` catches it. Similarly, shape (0, 3) is caught by `arr.shape[0] == 0`. But what about shape (0, 0)? Both checks catch it. Good.

However, `np.asarray([])` produces shape (0,) — a 1D array. After `reshape(1, -1)` on line 460, it becomes shape (1, 0). Then `shape[1] == 0` catches it. Let me verify:

```python
arr = np.asarray([])  # shape (0,)
arr.ndim == 1 → True
arr = arr.reshape(1, -1)  # shape (1, 0)
arr.shape[1] == 0 → True → return None
```

But what about `np.asarray([[]])`? This gives shape (1, 0). Still caught.

What about `np.asarray([[1, 2], []])`? This would fail in `np.asarray` because of ragged lists. Let me check — actually, Python's `np.asarray` with ragged lists creates a 1D array of list objects, not a 2D array. So it would hit the `ndim == 1` branch and get reshaped. If the inner lists are different lengths, you get an object array.

**Impact:** Low — mostly correct, but ragged lists produce object arrays which would fail at `astype(np.float64)` with a confusing error.

---

#### ECH-20 🟡 WARNING: `save()` re-imports `joblib` on every call

**File:** `stateguard/core/tier2.py:314`
**Code:** `import joblib as _joblib`

**Issue:** The `import joblib` is inside the method, so it's executed every time `save()` is called. While Python caches module imports (so it's not actually re-loading), there's still a `sys.modules` lookup and local name binding. Minor performance concern. Same for `load()` at line 364.

**Impact:** Negligible performance — style concern.

**Recommendation:** Move `import joblib` to module level or use a cached reference.

---

### Story 1-5: `stateguard/plugin/examples/regex.py` — Regex Validator

#### ECH-21 🟡 WARNING: ReDoS via evil regex pattern (catastrophic backtracking)

**File:** `stateguard/plugin/examples/regex.py:63`
**Code:** `compiled = re.compile(raw_pattern)`

**Issue:** `re.compile` compiles but does NOT validate complexity. A user-supplied pattern like `(a+)+b` against input "aaaaaaaaaaaaac" causes catastrophic backtracking (exponential time). This can freeze the validation process.

Since `context` is user-supplied, an attacker could craft a ReDoS pattern.

**Recommendation:** Use `re.compile(raw_pattern, re.DOTALL)` with a timeout. In Python 3.12+, `re.compile` accepts `timeout` parameter:
```python
compiled = re.compile(raw_pattern, timeout=5)
```

Or use a signal/alarm-based timeout for older Python versions. Document that patterns are user-supplied and should be trusted.

---

#### ECH-22 🔵 INFO: Empty pattern string `""` — matches everything (zero-width match)

**File:** `stateguard/plugin/examples/regex.py:63`
**Code:** `compiled = re.compile(raw_pattern)`

**Issue:** If `raw_pattern = ""`, `re.compile("")` succeeds and `compiled.search("anything")` returns a match at position 0 with an empty string. The validator returns `score=100.0, passed=True`. This may be unexpected — an empty pattern is semantically equivalent to "no constraint," but `raw_pattern is None` (line 44) already handles that case. Empty string and None have different behavior.

**Impact:** Low — empty pattern passes everything, which may surprise users.

**Recommendation:** Consider treating empty string the same as None:
```python
if not raw_pattern:  # catches None, "", and other falsy values
    ...
```

---

#### ECH-23 🔵 INFO: Non-string `output` is `str()`-coerced — may produce unexpected matches

**File:** `stateguard/plugin/examples/regex.py:73`
**Code:** `output_str = str(output) if output is not None else ""`

**Issue:** Non-string outputs like `bytes`, `list`, `dict` are converted with `str()`. This means:
- `str(b"hello")` → `"b'hello'"` (with the `b'` prefix)
- `str([1, 2, 3])` → `"[1, 2, 3]"` (includes brackets)
- `str({"a": 1})` → `"{'a': 1}"` (includes braces and quotes)

A user crafting a regex for "hello" wouldn't expect `b"hello"` to show up as `b'hello'`. The match might fail or produce confusing results.

**Impact:** Low — acceptable for a structural validator, but should be documented.

---

#### ECH-24 🔵 INFO: `compiled.search()` returns only the first match — no global matching

**File:** `stateguard/plugin/examples/regex.py:74`
**Code:** `match = compiled.search(output_str)`

**Issue:** `search()` finds the first match. There's no option for full-match (`fullmatch()`), match-at-start (`match()`), or global matching (`findall()`/`finditer()`). Users who want "entire string must match" would need `^(...)$` in their pattern.

**Impact:** Very low — documented behavior.

---

### Story 1-4: Snapshot/State Robustness (cross-cutting)

#### ECH-25 🔵 INFO: No `__init_subclass__` test for diamond inheritance

**No specific code location** — test gap.

**Issue:** The MRO traversal in `__init_subclass__` theoretically handles diamond inheritance (`GrandParent` with two intermediate classes both inheriting from `BaseValidator`), but there's no test for it.

**Test coverage gap:** `test_plugin/test_base.py` covers linear 2-level and 3-level inheritance, but not diamond.

---

## Summary Table

| ID | Severity | Story | File | Issue |
|----|----------|-------|------|-------|
| ECH-06 | 🔴 CRITICAL | 1-3 | llm_client.py:135 | `httpx` lazy import conflated with network errors |
| ECH-13 | 🔴 CRITICAL | 1-1 | tier2.py:370 | `joblib.load` RCE via untrusted pickle |
| ECH-01 | 🔵 INFO | 1-2 | base.py:75 | `hasattr(__abstractmethods__)` — dead code (always False), verified empirically |
| ECH-02 | 🟡 WARNING | 1-2 | base.py:117 | Non-callable validate from parent not caught |
| ECH-05 | 🟡 WARNING | 1-3 | llm_client.py:49 | Silent English→Turkish prompt switch for custom clients |
| ECH-07 | 🟡 WARNING | 1-3 | llm_client.py:40 | Protocol type narrowing not enforced |
| ECH-09 | 🟡 WARNING | 1-1 | tier2.py:287/491 | `fit()` vs `validate()` inconsistent error handling |
| ECH-10 | 🟡 WARNING | 1-1 | tier2.py:450 | NaN/inf in features silently corrupt results |
| ECH-12 | 🟡 WARNING | 1-1 | tier2.py:333 | Atomic write error handling gaps |
| ECH-14 | 🟡 WARNING | 1-1 | tier2.py:380 | `KeyError` on corrupt model loading |
| ECH-16 | 🟡 WARNING | 1-1 | tier2.py:410 | `setup()` swallows all exceptions |
| ECH-17 | 🟡 WARNING | 1-1 | tier2.py:511 | `stacklevel=2` verified correct (no issue) |
| ECH-19 | 🟡 WARNING | 1-1 | tier2.py:464 | Empty feature arrays handled (edge verified OK) |
| ECH-21 | 🟡 WARNING | 1-5 | regex.py:63 | ReDoS via evil regex pattern |
| ECH-11 | 🔵 INFO | 1-1 | tier2.py:416 | Thread-unsafe lazy init |
| ECH-03 | 🔵 INFO | 1-2 | base.py:89 | MRO traversal assumes BaseValidator in MRO |
| ECH-04 | 🔵 INFO | 1-2 | base.py:72 | Custom metaclass interactions untested |
| ECH-08 | 🔵 INFO | 1-3 | llm_client.py:161 | `strip()` edge case documented |
| ECH-15 | 🔵 INFO | 1-1 | tier2.py:317 | Symlink behavior documented |
| ECH-18 | 🔵 INFO | 1-1 | tier2.py:74 | Zero-std guard documented behavior |
| ECH-20 | 🔵 INFO | 1-1 | tier2.py:314 | Repeated import style concern |
| ECH-22 | 🔵 INFO | 1-5 | regex.py:63 | Empty pattern matches everything |
| ECH-23 | 🔵 INFO | 1-5 | regex.py:73 | Non-string output str()-coercion |
| ECH-24 | 🔵 INFO | 1-5 | regex.py:74 | Only `search()`, no fullmatch |
| ECH-25 | 🔵 INFO | 1-2 | test_base.py | Diamond inheritance not tested |

---

## Severity Distribution

| Severity | Count |
|----------|-------|
| 🔴 CRITICAL | 2 |
| 🟡 WARNING | 9 |
| 🔵 INFO | 14 |
| **Total** | **25** |

---

## Top 3 Action Items

1. **ECH-06** (CRITICAL): Fix httpx lazy import in `llm_client.py:135` — separate import error from network errors. This causes misleading error messages in production.

2. **ECH-13** (CRITICAL): Document RCE risk in `joblib.load()` at `tier2.py:370`. Add integrity verification or sandboxing guidance for production deployments that load externally-supplied models.

3. **ECH-01** (WARNING): Investigate the `hasattr(cls, "__abstractmethods__")` gate in `base.py:75` empirically. If it short-circuits enforcement for all subclasses, the W1+W2 hardening is effectively a no-op and needs to be redesigned. The fact that existing tests pass suggests either correct timing behavior (ABCMeta hasn't set `__abstractmethods__` yet when `__init_subclass__` runs) or tests are catching ABCMeta's TypeError instead. Write a dedicated test to verify.
