# Baseline management

A baseline is an explicitly saved retrieval report used as the reference for later checks. It lives in the SQLite database selected by `baseline.db_path`, normally `.longprobe/baselines.db`.

## Save a trusted reference

```bash
longprobe check --no-compare
longprobe baseline save --label latest
```

`baseline save` performs a fresh evaluation of approved questions. By default every evaluated question must meet its effective recall threshold. A strong average cannot hide an individual failure. Empty selections and drafts cannot be promoted. A rejected promotion leaves the existing labeled reference intact.

The first command is optional inspection; its result is not silently reused by the second. Neither command edits `goldens.yaml`. Review any question or required-evidence changes against the source documents before approving them.

A passing check never replaces a baseline automatically. Saving an existing label explicitly replaces that label's report. Saving `release-a` does not update `latest`: `latest` is a literal name, not a newest-report lookup.

## Policy and threshold overrides

```yaml
scoring:
  recall_threshold: 0.8
  question_thresholds:
    cancellation-policy: 1.0
baseline:
  quality_gate: all_questions
```

An invocation-level `--threshold` overrides the configured global threshold. Explicit per-question thresholds take precedence over that global override. Override IDs are validated against the complete loaded suite, so a typo fails while a valid override for an excluded question is allowed.

For an intentional average-based baseline policy:

```yaml
baseline:
  quality_gate: aggregate
```

This explicit alternative permits individual failures when overall recall meets the effective global threshold. The report records that choice and retains individual failure details. It only changes baseline eligibility; regular checks still assess every selected question and regressions.

`fail_on_regression: false` does not disable baseline promotion gates, configuration validation, empty-selection rejection, or comparison compatibility checks.

## Select the reference

```bash
longprobe check --baseline release-a
longprobe diff --baseline release-a
longprobe diff release-a release-b
longprobe check --no-compare
longprobe baseline list --db-path .longprobe/baselines.db
```

An explicit missing label fails before retrieval. The default automatic comparison looks for optional `latest`; if it does not exist, quality can still be assessed, but the report says no historical comparison occurred. `--no-compare` skips the configured comparison; it cannot be combined with an explicit `--baseline`.

The report includes requested/resolved label, reference timestamp, and comparison status. Stored `diff A B` treats A as the reference and B as the target, without running retrieval.

## Understand differences and compatibility

Version 0.2.0 reports carry effective matching, retrieval-limit, threshold, question, and evidence contracts. A strict comparison requires the same evaluated question coverage, suite name/version, and contracts. Reordering questions or required evidence does not make otherwise identical reports incompatible.

Comparisons surface added/removed question IDs, changed contract fields, and lost/gained required evidence. Losing previously found required evidence is a regression even if another requirement is found and total recall stays the same.

Changed coverage or contracts produce an incompatible outcome instead of a clean regression verdict. Raw scores remain available, but a like-for-like overall delta is unavailable. Review intentional changes, then explicitly establish a new reference if appropriate.

## Upgrade existing references

Unversioned v0.1.x rows remain readable/listable; adoption does not rewrite the database. Missing historical settings cannot be inferred safely, so strict comparisons against those rows are marked unverified and exit unsuccessfully.

1. Close baseline writers and back up your database.
2. Review the current approved questions, evidence, and settings.
3. Inspect a fresh evaluation with `longprobe check --no-compare`.
4. Explicitly save a new reference with `longprobe baseline save --label latest` when eligible.
5. Retain the old backup or historical labels as needed.

A metadata-free old reference does not prevent independent baseline saving. No SQL table migration is required; new metadata is stored in report JSON. An older executable replacing a label may discard metadata it does not understand, so preserve backups before downgrading.

## Python promotion

```python
report = probe.run()
probe.save_baseline(label="release-a")
```

The Python facade applies the same policy. It saves a retained evaluation snapshot: modifying the returned report or configuration afterward does not weaken that candidate. Run again to adopt new settings. Failed reruns invalidate prior candidates. A passing candidate can be explicitly promoted even when its comparison with an older incompatible reference is unavailable.

`BaselineStore.save()` is low-level storage, not a quality certification operation. Use the facade or CLI to promote trusted baselines.
