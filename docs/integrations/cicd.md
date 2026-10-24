# CI/CD integration

LongProbe v0.2.0 evaluates once and derives JSON, GitHub annotations, and PR feedback from that saved report. Setup, incompatible comparisons, invalid reports, and execution failures always fail the workflow. You can explicitly make completed quality failures nonblocking.

## Call the reusable workflow

```yaml
name: Retrieval regression check
on: [push, pull_request]
permissions:
  contents: read
  actions: read
  pull-requests: write
jobs:
  retrieval:
    uses: ENDEVSOLS/LongProbe/.github/workflows/longprobe.yml@main
    with:
      python-version: "3.12"
      longprobe-version: "0.2.0"
      config-path: longprobe.yaml
      goldens-path: goldens.yaml
      fail-on-regression: true
    secrets:
      retriever-api-key: ${{ secrets.RAG_RETRIEVER_API_KEY }}
```

Use a reviewed release tag or commit when available instead of a moving branch. The requested package version must be published before this installation path can succeed. The workflow requires v0.2.0 or newer. An empty version input installs the available default and then checks compatibility.

`install-extras` is a separate input, for example `chroma`. HTTP retrieval is included in the base package; there is no `http` extra.

If required, reference the optional credential in your configuration:

```yaml
retriever:
  type: http
  http:
    url: https://your-application.example/retrieve
    headers:
      Authorization: "Bearer ${LONGPROBE_RETRIEVER_API_KEY}"
```

Replace the URL and configure the response mapping for your endpoint. Supply other application-specific authentication through your own wrapper workflow if needed. Never put secrets in golden files or report artifacts.

## Restore a trusted baseline

Fresh GitHub runners do not retain a previous job's SQLite file. Configure an artifact and source run explicitly:

```yaml
with:
  longprobe-version: "0.2.0"
  baseline-artifact-name: longprobe-baseline
  baseline-run-id: "123456789"
  baseline-label: latest
  baseline-path: .longprobe/baselines.db
```

Replace the example run ID with the actual source run. The source must be a successful `push`, `schedule`, or `workflow_dispatch` run on the caller repository's default branch. The caller must allow artifact read permissions. A PR-produced report is not selected automatically as trusted ground truth.

The artifact must contain exactly one regular `.db` file from a closed standalone LongProbe SQLite database. Its requested label must exist, its contracts must match the selected approved suite/settings, and `baseline-path` must match `baseline.db_path` in configuration. Legacy metadata, missing artifacts/labels, journal sidecars, and incompatible suites fail setup; they are not treated as passing comparisons.

Without restoration inputs, the helper runs `check --no-compare`. The report explicitly says no historical comparison was requested. It does not silently use an unrelated checked-out database.

## Publish a baseline deliberately

After reviewing the golden set and deciding to establish a new reference:

```bash
longprobe baseline save --label latest
```

This command performs a fresh approved-only evaluation and enforces the selected baseline quality policy. A successful check does not update the reference automatically.

A maintainer-controlled workflow can retain that explicitly promoted reference:

```yaml
- uses: actions/upload-artifact@v7
  with:
    name: longprobe-baseline
    path: .longprobe/baselines.db
    include-hidden-files: true
    if-no-files-found: error
```

Upload only the intended database. The hidden-file option is needed for a path under `.longprobe`; do not upload an entire configuration directory. See `examples/ci_example/save-reviewed-baseline.yml` in the repository for a manually invoked example.

## Reports and verdicts

The reusable workflow retains `longprobe-result.json`, `longprobe-diagnostics.txt`, and the derived `longprobe-comment.md`. It does not upload a new trusted baseline automatically.

The final verdict comes from the saved report and process outcome together. `fail-on-regression: false` only relaxes completed quality failures. It cannot hide invalid configuration, missing/incompatible references, invalid JSON, or a crashed command. A report's quality status remains failed even if the process is allowed to exit successfully.

Fork PRs receive annotations/artifacts without attempted write comments. A denied same-repository comment is reported separately; it does not change the evaluation result.

## Run the same workflow locally

```bash
python -m longprobe.ci run --config longprobe.yaml --goldens goldens.yaml
longprobe report longprobe-result.json --output github
```

The first command evaluates once. The second only renders existing data. Use `--baseline LABEL` with the helper for an explicitly prepared local reference, or `--nonblocking` to relax quality failures. GitHub artifact download and PR publication still require the hosted workflow.

## Migration and validation

Version 0.2.0 adds strict report compatibility and per-question baseline gates. Re-establish a reviewed baseline when adopting an old metadata-free reference; do not invent missing historical settings. See [release notes](../releases/v0.2.0.md) and [baseline management](../guide/baseline-management.md).

Implementation references: [artifact download inputs](https://github.com/actions/download-artifact), [artifact uploads and hidden files](https://github.com/actions/upload-artifact). Hosted execution must still be verified after the maintainer pushes the changes.
