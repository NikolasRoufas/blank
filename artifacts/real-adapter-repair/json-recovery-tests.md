# JSON recovery tests (§3, §8)

`tests/unit/test_structured_json.py` (12 tests) for `recover_json_object`:

| Case | Expected |
|------|----------|
| strict valid JSON object | ok, `recovered=False` |
| trailing whitespace | ok, `recovered=False` (strip + whole-string) |
| JSON then prose suffix | ok, `recovered=True` |
| prose prefix then one object | ok, `recovered=True` |
| braces inside strings (`"a }{ b"`, `"{not json}"`) | handled, one object |
| two competing valid objects | **rejected** (ambiguous) |
| one object + a stray non-JSON `{…}` group | ok (only one *valid* object) |
| truncated object | **rejected** |
| top-level array | **rejected** (not an object) |
| scalar (`42`) | **rejected** |
| empty/whitespace | **rejected** |
| pure prose | **rejected** |

The scanner is brace-aware and string-aware (respects `\"` escapes) — no naive
`\{.*\}` regex. Recovery is surfaced, not hidden: `ParsedAnswer.recovered` (generation)
and an extraction warning ("structured output required JSON recovery …").

Integration reproduction (`tests/integration/test_real_adapter_regression.py`):
- extractor recovers one JSON object from prose and extracts a grounded claim
  (formerly 0/4), with the recovery warning present;
- extractor still raises on pure prose;
- generator output "valid JSON + trailing prose" parses to the correct answer;
- `parse_generation` marks `recovered` True/False correctly.

Boundary respected: output is only accepted when it contains exactly one
unambiguous top-level JSON object that passes the full schema; malformed/ambiguous
output still fails.
