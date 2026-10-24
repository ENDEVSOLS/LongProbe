# Pytest integration

The installed LongProbe package registers its pytest plugin automatically. Install pytest separately in your test environment. Tests can request a shared evaluation report or enable automatic quality enforcement.

## Evaluate once and inspect the report

Configure your retriever in `longprobe.yaml` and review the approved questions in `goldens.yaml`:

```python
def test_retrieval(longprobe_report):
    assert longprobe_report.quality["passed"], longprobe_report.quality["reasons"]
```

```bash
pytest --longprobe-config tests/longprobe.yaml --longprobe-goldens tests/goldens.yaml
```

The selected configuration path is forwarded to the probe. The session-scoped `longprobe_report` fixture evaluates once per pytest process, even when several tests request it. The terminal summary uses that same report. You do not need to assign private attributes on pytest's configuration.

Without an assertion or enforcement option, the report fixture returns failed quality results for inspection; requesting a report alone does not automatically assert its quality. Invalid setup and failed execution remain test errors.

## Enforce quality automatically

```bash
pytest --longprobe-fail-threshold 0.85
```

This option activates the evaluation even when no user test requests `longprobe_report`. Every selected question must pass, and any detected regression or incompatible comparison fails the gate. Configured `scoring.question_thresholds` take precedence over the invocation's global threshold. A setting that makes CLI quality failures nonblocking does not disable explicit pytest enforcement.

Version 0.2.0 changes the old option's aggregate-oriented description: a good average cannot hide an individually failing question. Invalid thresholds and invalid requested configuration fail rather than skip tests. Pytest retains its own exit-code conventions.

Without requested LongProbe fixtures or the enforcement option, the plugin does not load configuration or call a retriever. If no tests are collected, normal pytest no-tests behavior applies.

## Use an application-owned retriever

Override the adapter fixture in `conftest.py`:

```python
import pytest
from longprobe.adapters import LangChainRetrieverAdapter

@pytest.fixture(scope="session")
def longprobe_adapter():
    return LangChainRetrieverAdapter(build_application_retriever())
```

`build_application_retriever()` represents your application's setup function. A custom adapter does not require credentials for an unused built-in backend. Evaluation/scoring/baseline settings still come from the selected configuration.

## Fixtures and options

| Fixture | Purpose |
|---|---|
| `longprobe_report` | One completed session evaluation |
| `longprobe_probe` | Configured Python facade; manual `run()` calls deliberately evaluate again |
| `longprobe_adapter` | Default configured adapter or your override |
| `longprobe_config_path` | Selected configuration path |
| `longprobe_goldens_path` | Selected golden-set path |
| `longprobe_fail_threshold` | Optional invocation threshold |
| `longprobe_include_drafts` | Whether analysis includes drafts |

Use `--longprobe-include-drafts` to analyze drafts explicitly. Draft-containing results cannot be promoted to a trusted baseline. Review candidate questions and evidence before approval.

See [baseline management](../guide/baseline-management.md) and the [quick start](../getting-started/quick-start.md).
