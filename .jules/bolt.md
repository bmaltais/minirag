## 2025-05-14 - [Top-K Ranking Optimization]
**Learning:** For retrieval tasks where we only need the top-K results, using `numpy.argpartition` is significantly faster than a full `argsort` or Python `sorted()`. The complexity drops from O(N log N) to O(N + k log k).
**Action:** Always prefer `argpartition` for ranking large sets of scores in search/retrieval paths.

## 2025-05-14 - [Regex Pre-compilation]
**Learning:** Pre-compiling regex at the module level in hot paths (like tokenization) avoids repeated compilation/cache lookups, providing a measurable micro-optimization.
**Action:** Pre-compile all regex objects used in loops or frequently called functions.
