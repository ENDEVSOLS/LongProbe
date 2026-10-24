# Python interface

Use your existing retriever through an adapter. LongProbe tests retrieval evidence; it does not index documents or evaluate the generated answer.

```python
from longprobe import LongProbe
from longprobe.adapters import LangChainRetrieverAdapter

adapter = LangChainRetrieverAdapter(your_application_retriever)
probe = LongProbe(
    adapter=adapter,
    goldens_path="goldens.yaml",
    config_path="longprobe.yaml",
)
report = probe.run()
assert report.quality["passed"], report.quality["reasons"]
```

Replace `your_application_retriever` with the retriever already constructed by your application. HTTP and other supported adapters can be created through `create_adapter_from_config(config)`; optional backend SDKs remain lazy.

## Evaluation and references

```python
report = probe.run(top_k_override=10, tags=["critical"])
report = probe.run(baseline_label="release-a")
differences = probe.diff(baseline_label="release-a")
```

`run()` returns `ProbeReport`, including per-question results/contracts, aggregate scores, selected baseline identity, structured comparison, and the quality decision. Quality failure is returned as data; invalid input and failed execution raise exceptions. Explicit missing labels fail. The default automatic reference is optional `latest`.

To deliberately evaluate without the default comparison, set `probe.config.baseline.auto_compare = False` before running. An explicit `baseline_label` still requests that reference.

CLI and Python share evaluation and policy resolution. A global `recall_threshold` constructor override supersedes the configured global value; explicit `scoring.question_thresholds` still take precedence for their question IDs. Unknown override IDs fail before retrieval.

## Save a baseline

```python
probe.run()
probe.save_baseline(label="latest")
```

Every evaluated question must pass under the default policy. Draft-containing reports cannot be promoted. The optional aggregate policy must be selected explicitly in configuration and is captured with the evaluation.

The facade retains a defensive snapshot after a completed run. Editing the returned report or changing settings afterward cannot weaken its promotion policy. A failed rerun clears the old candidate. Use another `run()` to evaluate changed settings, and catch `BaselineQualityError` from `longprobe.core.policy` if you need to handle a rejected promotion.

## Configuration and in-memory suites

You may supply `golden_set=GoldenSet(...)` instead of a golden YAML path. The facade applies approved-only selection by default; use `include_drafts=True` for analysis. Tag filtering requires all selected tags. Empty selections fail.

When you supply an adapter and the conventional `longprobe.yaml` is absent, documented Python defaults are supported. Explicit missing custom paths and existing invalid files fail. Unused built-in retriever credentials are not required for your supplied adapter.

## Reports and comparisons

New reports use schema version 2. Old unversioned reports remain readable but cannot establish strict compatibility when historical settings are missing. Changed requirements, retrieval limits, thresholds, suite identity, or selected question coverage are surfaced rather than silently compared as the same test.

`probe.diff()` returns the same comparison fields used by CLI output, including `compatible`, `compatibility_reasons`, coverage/contract/evidence changes, regressions, improvements, and a nullable `overall_delta`. Check compatibility before interpreting the delta.

See [baseline management](baseline-management.md), [pytest integration](../integrations/pytest.md), and [release migration notes](../releases/v0.2.0.md).
